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
	isl_ctx *ctx;

	if (!pw || !morph)
		goto error;

	ctx = isl_space_get_ctx(pw->dim);
	isl_assert(ctx, isl_space_is_domain_internal(morph->dom->dim, pw->dim),
		goto error);

	pw = FN(PW,cow)(pw);
	if (!pw)
		goto error;
	pw->dim = isl_space_extend_domain_with_range(
			isl_space_copy(morph->ran->dim), pw->dim);
	if (!pw->dim)
		goto error;

	for (i = 0; i < pw->n; ++i) {
		pw->p[i].set = isl_morph_set(isl_morph_copy(morph), pw->p[i].set);
		if (!pw->p[i].set)
			goto error;
		pw->p[i].FIELD = FN(EL,morph_domain)(pw->p[i].FIELD,
						isl_morph_copy(morph));
		if (!pw->p[i].FIELD)
			goto error;
	}

	isl_morph_free(morph);

	return pw;
error:
	FN(PW,free)(pw);
	isl_morph_free(morph);
	return NULL;
}
