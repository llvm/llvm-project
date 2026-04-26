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

/* Transform the elements of "multi" by applying "fn" to them
 * with extra argument "set".
 * If "multi" has an explicit domain, then apply "fn_domain" or
 * "fn_params" to this explicit domain instead.
 * In particular, if the explicit domain is a parameter set,
 * then apply "fn_params".  Otherwise, apply "fn_domain".
 */
static __isl_give MULTI(BASE) *FN(FN(MULTI(BASE),apply),APPLY_DOMBASE)(
	__isl_take MULTI(BASE) *multi, __isl_take APPLY_DOM *set,
	__isl_give EL *(*fn)(EL *el, __isl_take APPLY_DOM *set),
	__isl_give DOM *(*fn_domain)(DOM *domain, __isl_take APPLY_DOM *set),
	__isl_give DOM *(*fn_params)(DOM *domain, __isl_take APPLY_DOM *set))
{
	isl_size n;
	int i;

	FN(FN(MULTI(BASE),align_params),APPLY_DOMBASE)(&multi, &set);

	if (FN(MULTI(BASE),has_explicit_domain)(multi))
		return FN(FN(MULTI(BASE),apply_domain),APPLY_DOMBASE)(multi,
						set, fn_domain, fn_params);

	n = FN(MULTI(BASE),size)(multi);
	if (n < 0 || !set)
		goto error;

	for (i = 0; i < n; ++i) {
		EL *el;

		el = FN(MULTI(BASE),take_at)(multi, i);
		el = fn(el, FN(APPLY_DOM,copy)(set));
		multi = FN(MULTI(BASE),restore_at)(multi, i, el);
	}

	FN(APPLY_DOM,free)(set);
	return multi;
error:
	FN(APPLY_DOM,free)(set);
	FN(MULTI(BASE),free)(multi);
	return NULL;
}
