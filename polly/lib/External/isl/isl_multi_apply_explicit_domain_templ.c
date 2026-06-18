/*
 * Copyright 2017      Sven Verdoolaege
 *
 * Use of this software is governed by the MIT license
 *
 * Written by Sven Verdoolaege.
 */

/* Transform the explicit domain of "multi" by applying "fn_domain" or
 * "fn_params" to it with extra argument "domain".
 * In particular, if the explicit domain is a parameter set,
 * then apply "fn_params".  Otherwise, apply "fn_domain".
 *
 * The parameters of "multi" and "domain" are assumed to have been aligned.
 */
static __isl_give MULTI(BASE) *FN(FN(MULTI(BASE),apply_domain),APPLY_DOMBASE)(
	__isl_take MULTI(BASE) *multi, __isl_take APPLY_DOM *domain,
	__isl_give DOM *(*fn_domain)(DOM *domain, __isl_take APPLY_DOM *set),
	__isl_give DOM *(*fn_params)(DOM *domain, __isl_take APPLY_DOM *set))
{
	isl_bool is_params;
	DOM *multi_dom;

	multi_dom = FN(MULTI(BASE),get_explicit_domain)(multi);
	is_params = FN(DOM,is_params)(multi_dom);
	if (is_params < 0) {
		FN(APPLY_DOM,free)(domain);
		multi_dom = FN(DOM,free)(multi_dom);
	} else if (!is_params) {
		multi_dom = fn_domain(multi_dom, domain);
	} else {
		multi_dom = fn_params(multi_dom, domain);
	}
	multi = FN(MULTI(BASE),set_explicit_domain)(multi, multi_dom);
	return multi;
}

#include <isl_multi_apply_templ.c>
