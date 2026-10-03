"""The principal a shadow evaluation's own LLM calls run as: the admin who created the job.

Evaluation calls are spend the creator chose to incur, so they are attributed, routed and
budget-checked as that admin's own requests, through the same owners an authenticated
request uses. Nothing the sampled request carried decides who pays for them.
"""

from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Final

from pydantic import TypeAdapter

from litellm._logging import verbose_proxy_logger
from litellm.exceptions import BudgetExceededError
from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.auth.auth_checks import effective_user_role, get_user_object
from litellm.proxy.auth.fallback_budget import is_token_within_budget_for_model
from litellm.proxy.litellm_pre_call_utils import LiteLLMProxyRequestSetup
from litellm.types.proxy.auth.auth_checks import UserNotFoundError

_MODEL_BUDGETS: Final[TypeAdapter[Mapping[str, object] | None]] = TypeAdapter(Mapping[str, object] | None)


async def evaluation_principal(creator_user_id: str) -> UserAPIKeyAuth:
    """The creator as a principal carrying their current budgets.

    A creator with no user row is the master key's admin, which has no personal budget to
    enforce: the spend is still attributed to that id. Any other read failure propagates, so
    the caller withholds work it cannot verify the creator can pay for.
    """
    from litellm.proxy.proxy_server import prisma_client, proxy_logging_obj, user_api_key_cache

    try:
        user: Final = await get_user_object(
            user_id=creator_user_id,
            prisma_client=prisma_client,
            user_api_key_cache=user_api_key_cache,
            user_id_upsert=False,
            proxy_logging_obj=proxy_logging_obj,
        )
    except UserNotFoundError:
        return UserAPIKeyAuth(user_id=creator_user_id, user_role=LitellmUserRoles.PROXY_ADMIN)
    if user is None:
        return UserAPIKeyAuth(user_id=creator_user_id, user_role=LitellmUserRoles.PROXY_ADMIN)
    return UserAPIKeyAuth(
        user_id=user.user_id,
        user_role=effective_user_role(user.user_role),
        user_email=user.user_email,
        user_spend=user.spend,
        user_max_budget=user.max_budget,
        user_model_max_budget=_MODEL_BUDGETS.validate_python(user.model_max_budget),
    )


async def principal_can_pay_for(principal: UserAPIKeyAuth, models: Sequence[str]) -> bool:
    """Whether the principal is within its total budget and every per-model budget for
    ``models``, read through the owners the request path enforces them with."""
    from litellm.proxy.proxy_server import llm_router, model_max_budget_limiter

    if llm_router is None:
        return False
    try:
        for model in models:
            if not await is_token_within_budget_for_model(model=model, valid_token=principal, llm_router=llm_router):
                return False
            if principal.user_id is not None and principal.user_model_max_budget:
                await model_max_budget_limiter.is_user_within_model_budget(
                    user_id=principal.user_id, user_model_max_budget=principal.user_model_max_budget, model=model
                )
    except BudgetExceededError as e:
        verbose_proxy_logger.debug("shadow_eval: creator %s is over budget: %s", principal.user_id, e)
        return False
    return True


def principal_call_metadata(principal: UserAPIKeyAuth) -> Mapping[str, object]:
    """Request metadata naming ``principal`` exactly as auth stamps it on a request the
    principal sent, so every spend writer, limiter and router filter reads one identity."""
    stamped: Final[dict[str, object]] = {}  # mutable-ok: the stampers write into a request dict
    request: Final = {"metadata": stamped}
    LiteLLMProxyRequestSetup.add_user_api_key_auth_to_request_metadata(
        data=request, user_api_key_dict=principal, _metadata_variable_name="metadata"
    )
    LiteLLMProxyRequestSetup.add_budget_metadata_to_request_metadata(
        data=request, user_api_key_dict=principal, _metadata_variable_name="metadata"
    )
    return MappingProxyType(stamped)
