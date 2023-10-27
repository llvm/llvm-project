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
 *
 * The parameters of "multi" and "set" are assumed to have been aligned.
 */
__isl_give MULTI(BASE) *FN(FN(MULTI(BASE),apply_aligned),APPLY_DOMBASE)(
	__isl_take MULTI(BASE) *multi, __isl_take APPLY_DOM *set,
	__isl_give EL *(*fn)(EL *el, __isl_take APPLY_DOM *set))
{
	isl_size n;
	int i;

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

/* Transform the elements of "multi" by applying "fn" to them
 * with extra argument "set".
 *
 * Align the parameters if needed and call apply_set_aligned.
 */
static __isl_give MULTI(BASE) *FN(FN(MULTI(BASE),apply),APPLY_DOMBASE)(
	__isl_take MULTI(BASE) *multi, __isl_take APPLY_DOM *set,
	__isl_give EL *(*fn)(EL *el, __isl_take APPLY_DOM *set))
{
	isl_bool aligned;
	isl_ctx *ctx;

	if (!multi || !set)
		goto error;

	aligned = FN(APPLY_DOM,space_has_equal_params)(set, multi->space);
	if (aligned < 0)
		goto error;
	if (aligned)
		return FN(FN(MULTI(BASE),apply_aligned),APPLY_DOMBASE)(multi,
								    set, fn);
	ctx = FN(MULTI(BASE),get_ctx)(multi);
	if (!isl_space_has_named_params(multi->space) ||
	    !isl_space_has_named_params(set->dim))
		isl_die(ctx, isl_error_invalid,
			"unaligned unnamed parameters", goto error);
	multi = FN(MULTI(BASE),align_params)(multi,
						FN(APPLY_DOM,get_space)(set));
	set = FN(APPLY_DOM,align_params)(set, FN(MULTI(BASE),get_space)(multi));
	return FN(FN(MULTI(BASE),apply_aligned),APPLY_DOMBASE)(multi, set, fn);
error:
	FN(MULTI(BASE),free)(multi);
	FN(APPLY_DOM,free)(set);
	return NULL;
}
