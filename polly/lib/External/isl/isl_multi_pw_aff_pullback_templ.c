/*
 * Copyright 2013      Ecole Normale Superieure
 *
 * Use of this software is governed by the MIT license
 *
 * Written by Sven Verdoolaege,
 * Ecole Normale Superieure, 45 rue d'Ulm, 75230 Paris, France
 */

#define xCAT(A,B) A ## B
#define CAT(A,B) xCAT(A,B)
#undef TYPE
#define TYPE CAT(isl_,BASE)
#define xFN(TYPE,NAME) TYPE ## _ ## NAME
#define FN(TYPE,NAME) xFN(TYPE,NAME)

#undef SUFFIX
#define SUFFIX	BASE
#undef ARG1
#define ARG1	isl_multi_pw_aff
#undef ARG2
#define ARG2	TYPE

static
#include "isl_align_params_templ.c"

/* Compute the pullback of "mpa" by the function represented by "fn".
 * In other words, plug in "fn" in "mpa".
 *
 * If "mpa" has an explicit domain, then it is this domain
 * that needs to undergo a pullback, i.e., a preimage.
 */
__isl_give isl_multi_pw_aff *FN(isl_multi_pw_aff_pullback,BASE)(
	__isl_take isl_multi_pw_aff *mpa, __isl_take TYPE *fn)
{
	int i;
	isl_size n;
	isl_space *space = NULL;

	FN(isl_multi_pw_aff_align_params,BASE)(&mpa, &fn);
	mpa = isl_multi_pw_aff_cow(mpa);
	n = isl_multi_pw_aff_size(mpa);
	if (n < 0 || !fn)
		goto error;

	space = isl_space_join(FN(TYPE,get_space)(fn),
				isl_multi_pw_aff_get_space(mpa));

	for (i = 0; i < n; ++i) {
		isl_pw_aff *pa;

		pa = isl_multi_pw_aff_take_at(mpa, i);
		pa = FN(isl_pw_aff_pullback,BASE)(pa, FN(TYPE,copy)(fn));
		mpa = isl_multi_pw_aff_restore_at(mpa, i, pa);
		if (!mpa)
			goto error;
	}
	if (isl_multi_pw_aff_has_explicit_domain(mpa)) {
		mpa->u.dom = FN(isl_set_preimage,BASE)(mpa->u.dom,
							FN(TYPE,copy)(fn));
		if (!mpa->u.dom)
			goto error;
	}

	FN(TYPE,free)(fn);
	isl_multi_pw_aff_restore_space(mpa, space);
	return mpa;
error:
	isl_space_free(space);
	isl_multi_pw_aff_free(mpa);
	FN(TYPE,free)(fn);
	return NULL;
}
