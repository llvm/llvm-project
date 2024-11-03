/*
 * Copyright 2010      INRIA Saclay
 * Copyright 2011      Sven Verdoolaege
 *
 * Use of this software is governed by the MIT license
 *
 * Written by Sven Verdoolaege, INRIA Saclay - Ile-de-France,
 * Parc Club Orsay Universite, ZAC des vignes, 4 rue Jacques Monod,
 * 91893 Orsay, France
 */

#include <isl_pw_macro.h>

/* Make sure "pw" has room for at least "n" more pieces.
 *
 * If there is only one reference to pw, we extend it in place.
 * Otherwise, we create a new PW and copy the pieces.
 */
static __isl_give PW *FN(PW,grow)(__isl_take PW *pw, int n)
{
	int i;
	isl_ctx *ctx;
	PW *res;

	if (!pw)
		return NULL;
	if (pw->n + n <= pw->size)
		return pw;
	ctx = FN(PW,get_ctx)(pw);
	n += pw->n;
	if (pw->ref == 1) {
		res = isl_realloc(ctx, pw, struct PW,
			    sizeof(struct PW) + (n - 1) * sizeof(S(PW,piece)));
		if (!res)
			return FN(PW,free)(pw);
		res->size = n;
		return res;
	}
	res = FN(PW,alloc_size)(isl_space_copy(pw->dim) OPT_TYPE_ARG(pw->), n);
	if (!res)
		return FN(PW,free)(pw);
	for (i = 0; i < pw->n; ++i)
		res = FN(PW,add_piece)(res, isl_set_copy(pw->p[i].set),
					    FN(EL,copy)(pw->p[i].FIELD));
	FN(PW,free)(pw);
	return res;
}

__isl_give PW *FN(PW,add_disjoint)(__isl_take PW *pw1, __isl_take PW *pw2)
{
	int i;
	isl_ctx *ctx;

	if (FN(PW,align_params_bin)(&pw1, &pw2) < 0)
		goto error;

	if (pw1->size < pw1->n + pw2->n && pw1->n < pw2->n)
		return FN(PW,add_disjoint)(pw2, pw1);

	ctx = isl_space_get_ctx(pw1->dim);
	if (!OPT_EQUAL_TYPES(pw1->, pw2->))
		isl_die(ctx, isl_error_invalid,
			"fold types don't match", goto error);
	if (FN(PW,check_equal_space)(pw1, pw2) < 0)
		goto error;

	if (FN(PW,IS_ZERO)(pw1)) {
		FN(PW,free)(pw1);
		return pw2;
	}

	if (FN(PW,IS_ZERO)(pw2)) {
		FN(PW,free)(pw2);
		return pw1;
	}

	pw1 = FN(PW,grow)(pw1, pw2->n);
	if (!pw1)
		goto error;

	for (i = 0; i < pw2->n; ++i)
		pw1 = FN(PW,add_piece)(pw1,
				isl_set_copy(pw2->p[i].set),
				FN(EL,copy)(pw2->p[i].FIELD));

	FN(PW,free)(pw2);

	return pw1;
error:
	FN(PW,free)(pw1);
	FN(PW,free)(pw2);
	return NULL;
}
