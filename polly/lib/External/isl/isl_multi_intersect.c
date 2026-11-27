/*
 * Copyright 2011      Sven Verdoolaege
 * Copyright 2012-2013 Ecole Normale Superieure
 *
 * Use of this software is governed by the MIT license
 *
 * Written by Sven Verdoolaege,
 * Ecole Normale Superieure, 45 rue d’Ulm, 75230 Paris, France
 */

#include <isl_multi_macro.h>

/* Intersect the parameter domain "dom1" with "dom2".
 * That is, intersect the parameters of "dom2" with "dom1".
 *
 * Even though "dom1" is known to only involve parameter constraints,
 * it may be of type isl_union_set, so explicitly convert it
 * to an isl_set first.
 */
static __isl_give DOM *FN(MULTI(BASE),params_domain_intersect)(DOM *dom1,
	__isl_take DOM *dom2)
{
	isl_set *params;

	params = FN(DOM,params)(dom1);
	dom2 = FN(DOM,intersect_params)(dom2, params);

	return dom2;
}

/* Intersect the domain of "multi" with "domain".
 *
 * If "multi" has an explicit domain, then only this domain
 * needs to be intersected.
 */
__isl_give MULTI(BASE) *FN(MULTI(BASE),intersect_domain)(
	__isl_take MULTI(BASE) *multi, __isl_take DOM *domain)
{
	if (FN(MULTI(BASE),check_compatible_domain)(multi, domain) < 0)
		domain = FN(DOM,free)(domain);
	return FN(FN(MULTI(BASE),apply),DOMBASE)(multi, domain,
				&FN(EL,intersect_domain),
				&FN(DOM,intersect),
				&FN(MULTI(BASE),params_domain_intersect));
}

/* Intersect the parameter domain of "multi" with "domain".
 *
 * If "multi" has an explicit domain, then only this domain
 * needs to be intersected.
 */
__isl_give MULTI(BASE) *FN(MULTI(BASE),intersect_params)(
	__isl_take MULTI(BASE) *multi, __isl_take isl_set *domain)
{
	return FN(MULTI(BASE),apply_set)(multi, domain,
					&FN(EL,intersect_params),
					&FN(DOM,intersect_params),
					&FN(DOM,intersect_params));
}
