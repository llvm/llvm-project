/*
 * Copyright 2010      INRIA Saclay
 *
 * Use of this software is governed by the MIT license
 *
 * Written by Sven Verdoolaege, INRIA Saclay - Ile-de-France,
 * Parc Club Orsay Universite, ZAC des vignes, 4 rue Jacques Monod,
 * 91893 Orsay, France
 */

__isl_give PW *FN(PW,morph_domain)(__isl_take PW *pw,
	__isl_take isl_morph *morph)
{
	int i;
	isl_size n;
	isl_ctx *ctx;
	isl_space *space;

	n = FN(PW,n_piece)(pw);
	if (n < 0 || !morph)
		goto error;

	ctx = isl_space_get_ctx(pw->dim);
	isl_assert(ctx, isl_space_is_domain_internal(morph->dom->dim, pw->dim),
		goto error);

	space = FN(PW,take_space)(pw);
	space = isl_space_extend_domain_with_range(
			isl_space_copy(morph->ran->dim), space);
	pw = FN(PW,restore_space)(pw, space);

	for (i = 0; i < n; ++i) {
		isl_set *domain;
		EL *el;

		domain = FN(PW,take_domain_at)(pw, i);
		domain = isl_morph_set(isl_morph_copy(morph), domain);
		pw = FN(PW,restore_domain_at)(pw, i, domain);
		el = FN(PW,take_base_at)(pw, i);
		el = FN(EL,morph_domain)(el, isl_morph_copy(morph));
		pw = FN(PW,restore_base_at)(pw, i, el);
	}

	isl_morph_free(morph);

	return pw;
error:
	FN(PW,free)(pw);
	isl_morph_free(morph);
	return NULL;
}
