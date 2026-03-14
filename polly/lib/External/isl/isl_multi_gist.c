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

/* Compute the gist of the parameter domain "dom1" with respect to "dom2".
 *
 * Since "dom2" may not be a parameter domain, explicitly convert it
 * to a parameter domain first.
 */
static __isl_give DOM *FN(MULTI(BASE),domain_gist_params)(DOM *dom1,
	__isl_take DOM *dom2)
{
	isl_set *params;

	params = FN(DOM,params)(dom2);
	dom1 = FN(DOM,gist_params)(dom1, params);

	return dom1;
}

/* Compute the gist of "multi" with respect to the domain constraints
 * of "context".
 */
__isl_give MULTI(BASE) *FN(MULTI(BASE),gist)(__isl_take MULTI(BASE) *multi,
	__isl_take DOM *context)
{
	if (FN(MULTI(BASE),check_compatible_domain)(multi, context) < 0)
		context = FN(DOM,free)(context);
	return FN(FN(MULTI(BASE),apply),DOMBASE)(multi, context, &FN(EL,gist),
					&FN(DOM,gist),
					&FN(MULTI(BASE),domain_gist_params));
}

/* Compute the gist of "multi" with respect to the parameter constraints
 * of "context".
 */
__isl_give MULTI(BASE) *FN(MULTI(BASE),gist_params)(
	__isl_take MULTI(BASE) *multi, __isl_take isl_set *context)
{
	return FN(MULTI(BASE),apply_set)(multi, context, &FN(EL,gist_params),
				&FN(DOM,gist_params), &FN(DOM,gist_params));
}
