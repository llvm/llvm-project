/*
 * Copyright 2019      Cerebras Systems
 *
 * Use of this software is governed by the MIT license
 *
 * Written by Sven Verdoolaege,
 * Cerebras Systems, 175 S San Antonio Rd, Los Altos, CA, USA
 */

#include <isl_multi_macro.h>

/* Add "v" to the constant terms of all the base expressions of "multi".
 */
__isl_give MULTI(BASE) *FN(MULTI(BASE),add_constant_val)(
	__isl_take MULTI(BASE) *multi, __isl_take isl_val *v)
{
	isl_bool zero;
	isl_size n;
	int i;

	zero = isl_val_is_zero(v);
	n = FN(MULTI(BASE),size)(multi);
	if (zero < 0 || n < 0)
		goto error;
	if (zero || n == 0) {
		isl_val_free(v);
		return multi;
	}

	multi = FN(MULTI(BASE),cow)(multi);
	if (!multi)
		goto error;

	for (i = 0; i < n; ++i) {
		multi->u.p[i] = FN(EL,add_constant_val)(multi->u.p[i],
							    isl_val_copy(v));
		if (!multi->u.p[i])
			goto error;
	}

	isl_val_free(v);
	return multi;
error:
	FN(MULTI(BASE),free)(multi);
	isl_val_free(v);
	return NULL;
}

/* Add the elements of "mv" to the constant terms of
 * the corresponding base expressions of "multi".
 */
__isl_give MULTI(BASE) *FN(MULTI(BASE),add_constant_multi_val)(
	__isl_take MULTI(BASE) *multi, __isl_take isl_multi_val *mv)
{
	isl_space *multi_space, *mv_space;
	isl_bool zero, equal;
	isl_size n;
	int i;

	zero = isl_multi_val_is_zero(mv);
	n = FN(MULTI(BASE),size)(multi);
	multi_space = FN(MULTI(BASE),peek_space)(multi);
	mv_space = isl_multi_val_peek_space(mv);
	equal = isl_space_tuple_is_equal(multi_space, isl_dim_out,
					mv_space, isl_dim_out);
	if (zero < 0 || n < 0 || equal < 0)
		goto error;
	if (!equal)
		isl_die(isl_multi_val_get_ctx(mv), isl_error_invalid,
			"spaces don't match", goto error);
	if (zero || n == 0) {
		isl_multi_val_free(mv);
		return multi;
	}

	multi = FN(MULTI(BASE),cow)(multi);
	if (!multi)
		goto error;

	for (i = 0; i < n; ++i) {
		isl_val *v = isl_multi_val_get_at(mv, i);
		multi->u.p[i] = FN(EL,add_constant_val)(multi->u.p[i], v);
		if (!multi->u.p[i])
			goto error;
	}

	isl_multi_val_free(mv);
	return multi;
error:
	FN(MULTI(BASE),free)(multi);
	isl_multi_val_free(mv);
	return NULL;
}
