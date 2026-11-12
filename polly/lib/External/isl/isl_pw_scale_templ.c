/*
 * Copyright 2010      INRIA Saclay
 *
 * Use of this software is governed by the MIT license
 *
 * Written by Sven Verdoolaege, INRIA Saclay - Ile-de-France,
 * Parc Club Orsay Universite, ZAC des vignes, 4 rue Jacques Monod,
 * 91893 Orsay, France
 */

#include <isl_pw_macro.h>

__isl_give PW *FN(PW,scale)(__isl_take PW *pw, isl_int v)
{
	int i;
	isl_size n;

	if (isl_int_is_one(v))
		return pw;
	if (pw && DEFAULT_IS_ZERO && isl_int_is_zero(v)) {
		PW *zero;
		isl_space *space = FN(PW,get_space)(pw);
		zero = FN(PW,ZERO)(space OPT_TYPE_ARG(pw->));
		FN(PW,free)(pw);
		return zero;
	}
	if (isl_int_is_neg(v))
		pw = FN(PW,negate_type)(pw);

	n = FN(PW,n_piece)(pw);
	if (n < 0)
		return FN(PW,free)(pw);
	for (i = 0; i < n; ++i) {
		EL *el;

		el = FN(PW,take_base_at)(pw, i);
		el = FN(EL,scale)(el, v);
		pw = FN(PW,restore_base_at)(pw, i, el);
	}

	return pw;
}
