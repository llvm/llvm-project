/*
 * Copyright 2019      Cerebras Systems
 *
 * Use of this software is governed by the MIT license
 *
 * Written by Sven Verdoolaege,
 * Cerebras Systems, 175 S San Antonio Rd, Los Altos, CA, USA
 */

#include <isl_pw_macro.h>

#undef VAL
#define VAL	CAT(isl_,VAL_BASE)

/* Add "v" to the constant term of "pw" over its entire definition domain.
 */
__isl_give PW *FN(FN(PW,add_constant),VAL_BASE)(__isl_take PW *pw,
	__isl_take VAL *v)
{
	isl_bool zero;
	isl_size n;
	int i;

	zero = FN(VAL,is_zero)(v);
	n = FN(PW,n_piece)(pw);
	if (zero < 0 || n < 0)
		goto error;
	if (zero || n == 0) {
		FN(VAL,free)(v);
		return pw;
	}

	for (i = 0; i < n; ++i) {
		EL *el;

		el = FN(PW,take_base_at)(pw, i);
		el = FN(FN(EL,add_constant),VAL_BASE)(el, FN(VAL,copy)(v));
		pw = FN(PW,restore_base_at)(pw, i, el);
	}

	FN(VAL,free)(v);
	return pw;
error:
	FN(PW,free)(pw);
	FN(VAL,free)(v);
	return NULL;
}
