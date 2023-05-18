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

__isl_give PW *FN(PW,split_dims)(__isl_take PW *pw,
	enum isl_dim_type type, unsigned first, unsigned n)
{
	int i;
	isl_size n_piece;

	n_piece = FN(PW,n_piece)(pw);
	if (n_piece < 0)
		return FN(PW,free)(pw);
	if (n == 0)
		return pw;

	if (type == isl_dim_in)
		type = isl_dim_set;

	for (i = 0; i < n; ++i) {
		isl_set *domain;

		domain = FN(PW,take_domain_at)(pw, i);
		domain = isl_set_split_dims(domain, type, first, n);
		pw = FN(PW,restore_domain_at)(pw, i, domain);
	}

	return pw;
}
