/*
 * Copyright 2013      Ecole Normale Superieure
 *
 * Use of this software is governed by the MIT license
 *
 * Written by Sven Verdoolaege,
 * Ecole Normale Superieure, 45 rue d'Ulm, 75230 Paris, France
 */

/* Apply "fn" to each of the elements of "multi" with as second argument "v".
 */
static __isl_give MULTI(BASE) *FN(MULTI(BASE),fn_val)(
	__isl_take MULTI(BASE) *multi,
	__isl_give EL *(*fn)(__isl_take EL *el, __isl_take isl_val *v),
	 __isl_take isl_val *v)
{
	isl_size n;
	int i;

	n = FN(MULTI(BASE),size)(multi);
	if (n < 0 || !v)
		goto error;

	for (i = 0; i < n; ++i) {
		EL *el;

		el = FN(MULTI(BASE),take_at)(multi, i);
		el = fn(el, isl_val_copy(v));
		multi = FN(MULTI(BASE),restore_at)(multi, i, el);
	}

	isl_val_free(v);
	return multi;
error:
	isl_val_free(v);
	FN(MULTI(BASE),free)(multi);
	return NULL;
}

#undef TYPE
#define TYPE	MULTI(BASE)
#include "isl_type_check_match_range_multi_val.c"

/* Elementwise apply "fn" to "multi" and "mv".
 */
static __isl_give MULTI(BASE) *FN(MULTI(BASE),fn_multi_val)(
	__isl_take MULTI(BASE) *multi,
	__isl_give EL *(*fn)(__isl_take EL *el, __isl_take isl_val *v),
	__isl_take isl_multi_val *mv)
{
	isl_size n;
	int i;

	n = FN(MULTI(BASE),size)(multi);
	if (n < 0 || FN(MULTI(BASE),check_match_range_multi_val)(multi, mv) < 0)
		goto error;

	for (i = 0; i < n; ++i) {
		isl_val *v;
		EL *el;

		v = isl_multi_val_get_val(mv, i);
		el = FN(MULTI(BASE),take_at)(multi, i);
		el = fn(el, v);
		multi = FN(MULTI(BASE),restore_at)(multi, i, el);
	}

	isl_multi_val_free(mv);
	return multi;
error:
	isl_multi_val_free(mv);
	return FN(MULTI(BASE),free)(multi);
}
