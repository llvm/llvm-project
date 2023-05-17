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

/* Apply "fn" to each of the base expressions of "pw".
 * The function is assumed to have no effect on the default value
 * (i.e., zero for those objects with a default value).
 */
static __isl_give PW *FN(PW,un_op)(__isl_take PW *pw,
	__isl_give EL *(*fn)(__isl_take EL *el))
{
	isl_size n;
	int i;

	n = FN(PW,n_piece)(pw);
	if (n < 0)
		return FN(PW,free)(pw);

	for (i = 0; i < n; ++i) {
		EL *el;

		el = FN(PW,take_base_at)(pw, i);
		el = fn(el);
		pw = FN(PW,restore_base_at)(pw, i, el);
	}

	return pw;
}
