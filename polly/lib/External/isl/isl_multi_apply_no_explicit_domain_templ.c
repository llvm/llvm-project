/*
 * Copyright 2022      Cerebras Systems
 *
 * Use of this software is governed by the MIT license
 *
 * Written by Sven Verdoolaege,
 * Cerebras Systems, 1237 E Arques Ave, Sunnyvale, CA, USA
 */

/* Transform the explicit domain of "multi" by applying "fn_domain" or
 * "fn_params" to it with extra argument "domain".
 * In particular, if the explicit domain is a parameter set,
 * then apply "fn_params".  Otherwise, apply "fn_domain".
 *
 * Do this for a type MULTI(BASE) that cannot have an explicit domain.
 * That is, this function is never called.
 */

static __isl_give MULTI(BASE) *FN(FN(MULTI(BASE),apply_domain),APPLY_DOMBASE)(
	__isl_take MULTI(BASE) *multi, __isl_take isl_set *domain,
	__isl_give DOM *(*fn_domain)(DOM *domain, __isl_take APPLY_DOM *set),
	__isl_give DOM *(*fn_params)(DOM *domain, __isl_take isl_set *set))
{
	isl_set_free(domain);

	return multi;
}

#include <isl_multi_apply_templ.c>
