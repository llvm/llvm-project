/*
 * Copyright 2017      Sven Verdoolaege
 *
 * Use of this software is governed by the MIT license
 *
 * Written by Sven Verdoolaege.
 */

#include <isl_multi_macro.h>

/* Compute the sum of "multi1" and "multi2" on the union of their domains,
 * with the actual sum on the shared domain and
 * the defined expression on the symmetric difference of the domains.
 *
 * We simply iterate over the elements in both arguments and
 * call isl_union_pw_aff_union_add on each of them, if there is
 * at least one element.
 *
 * Otherwise, the two expressions have an explicit domain and
 * the union of these explicit domains is computed.
 * This assumes that the explicit domains are either both in terms
 * of specific domains elements or both in terms of parameters.
 * However, if one of the expressions does not have any constraints
 * on its explicit domain, then this is allowed as well and the result
 * is the expression with no constraints on its explicit domain.
 */
__isl_give MULTI(BASE) *FN(MULTI(BASE),union_add)(
	__isl_take MULTI(BASE) *multi1, __isl_take MULTI(BASE) *multi2)
{
	isl_bool has_domain, is_params1, is_params2;

	if (!multi1)
		goto error;
	if (multi1->n > 0)
		return FN(MULTI(BASE),bin_op)(multi1, multi2,
					    &FN(EL,union_add));
	FN(MULTI(BASE),align_params_bin)(&multi1, &multi2);
	if (FN(MULTI(BASE),check_equal_space)(multi1, multi2) < 0)
		goto error;
	if (FN(MULTI(BASE),check_has_explicit_domain)(multi1) < 0 ||
	    FN(MULTI(BASE),check_has_explicit_domain)(multi2) < 0)
		goto error;

	has_domain = FN(MULTI(BASE),has_non_trivial_domain)(multi1);
	if (has_domain < 0)
		goto error;
	if (!has_domain) {
		FN(MULTI(BASE),free)(multi2);
		return multi1;
	}
	has_domain = FN(MULTI(BASE),has_non_trivial_domain)(multi2);
	if (has_domain < 0)
		goto error;
	if (!has_domain) {
		FN(MULTI(BASE),free)(multi1);
		return multi2;
	}

	is_params1 = FN(DOM,is_params)(multi1->u.dom);
	is_params2 = FN(DOM,is_params)(multi2->u.dom);
	if (is_params1 < 0 || is_params2 < 0)
		goto error;
	if (is_params1 != is_params2)
		isl_die(FN(MULTI(BASE),get_ctx)(multi1),
			isl_error_invalid,
			"cannot compute union of concrete domain and "
			"parameter constraints", goto error);
	multi1 = FN(MULTI(BASE),cow)(multi1);
	if (!multi1)
		goto error;
	multi1->u.dom = FN(DOM,union)(multi1->u.dom,
					FN(DOM,copy)(multi2->u.dom));
	if (!multi1->u.dom)
		goto error;
	FN(MULTI(BASE),free)(multi2);
	return multi1;
error:
	FN(MULTI(BASE),free)(multi1);
	FN(MULTI(BASE),free)(multi2);
	return NULL;
}
