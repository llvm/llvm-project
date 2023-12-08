/*
 * Copyright 2010      INRIA Saclay
 *
 * Use of this software is governed by the MIT license
 *
 * Written by Sven Verdoolaege, INRIA Saclay - Ile-de-France,
 * Parc Club Orsay Universite, ZAC des vignes, 4 rue Jacques Monod,
 * 91893 Orsay, France
 */

__isl_give PW *FN(PW,move_dims)(__isl_take PW *pw,
	enum isl_dim_type dst_type, unsigned dst_pos,
	enum isl_dim_type src_type, unsigned src_pos, unsigned n)
{
	int i;
	isl_size n_piece;
	isl_space *space;

	space = FN(PW,take_space)(pw);
	space = isl_space_move_dims(space, dst_type, dst_pos,
				    src_type, src_pos, n);
	pw = FN(PW,restore_space)(pw, space);

	n_piece = FN(PW,n_piece)(pw);
	if (n_piece < 0)
		return FN(PW,free)(pw);

	for (i = 0; i < n_piece; ++i) {
		EL *el;

		el = FN(PW,take_base_at)(pw, i);
		el = FN(EL,move_dims)(el,
					dst_type, dst_pos, src_type, src_pos, n);
		pw = FN(PW,restore_base_at)(pw, i, el);
	}

	if (dst_type == isl_dim_in)
		dst_type = isl_dim_set;
	if (src_type == isl_dim_in)
		src_type = isl_dim_set;

	for (i = 0; i < n_piece; ++i) {
		isl_set *domain;

		domain = FN(PW,take_domain_at)(pw, i);
		domain = isl_set_move_dims(domain, dst_type, dst_pos,
						src_type, src_pos, n);
		pw = FN(PW,restore_domain_at)(pw, i, domain);
	}

	return pw;
}
