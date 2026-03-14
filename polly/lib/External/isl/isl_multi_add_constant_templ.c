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

	zero = isl_val_is_zero(v);
	if (zero < 0)
		goto error;
	if (zero) {
		isl_val_free(v);
		return multi;
	}

	return FN(MULTI(BASE),fn_val)(multi, &FN(EL,add_constant_val), v);
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
	isl_bool zero;

	zero = isl_multi_val_is_zero(mv);
	if (zero < 0)
		goto error;
	if (zero) {
		isl_multi_val_free(mv);
		return multi;
	}

	return FN(MULTI(BASE),fn_multi_val)(multi, &FN(EL,add_constant_val),
						mv);

error:
	FN(MULTI(BASE),free)(multi);
	isl_multi_val_free(mv);
	return NULL;
}
