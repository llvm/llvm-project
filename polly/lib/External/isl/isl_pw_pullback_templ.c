/*
 * Copyright 2012      Ecole Normale Superieure
 *
 * Use of this software is governed by the MIT license
 *
 * Written by Sven Verdoolaege,
 * Ecole Normale Superieure, 45 rue d'Ulm, 75230 Paris, France
 */

#include <isl_pw_macro.h>

#undef SUFFIX
#define SUFFIX	multi_aff
#undef ARG1
#define ARG1	PW
#undef ARG2
#define ARG2	isl_multi_aff

static
#include "isl_align_params_templ.c"

#undef SUFFIX
#define SUFFIX	pw_multi_aff
#undef ARG1
#define ARG1	PW
#undef ARG2
#define ARG2	isl_pw_multi_aff

static
#include "isl_align_params_templ.c"

/* Compute the pullback of "pw" by the function represented by "ma".
 * In other words, plug in "ma" in "pw".
 */
__isl_give PW *FN(PW,pullback_multi_aff)(__isl_take PW *pw,
	__isl_take isl_multi_aff *ma)
{
	int i;
	isl_size n;
	isl_space *space = NULL;

	FN(PW,align_params_multi_aff)(&pw, &ma);
	ma = isl_multi_aff_align_divs(ma);
	n = FN(PW,n_piece)(pw);
	if (n < 0 || !ma)
		goto error;

	space = isl_space_join(isl_multi_aff_get_space(ma),
				FN(PW,get_space)(pw));

	for (i = 0; i < n; ++i) {
		isl_set *domain;
		EL *el;

		domain = FN(PW,take_domain_at)(pw, i);
		domain = isl_set_preimage_multi_aff(domain,
						    isl_multi_aff_copy(ma));
		pw = FN(PW,restore_domain_at)(pw, i, domain);
		el = FN(PW,take_base_at)(pw, i);
		el = FN(EL,pullback_multi_aff)(el, isl_multi_aff_copy(ma));
		pw = FN(PW,restore_base_at)(pw, i, el);
	}

	pw = FN(PW,reset_space)(pw, space);
	isl_multi_aff_free(ma);
	return pw;
error:
	isl_space_free(space);
	isl_multi_aff_free(ma);
	FN(PW,free)(pw);
	return NULL;
}

/* Compute the pullback of "pw" by the function represented by "pma".
 * In other words, plug in "pma" in "pw".
 */
static __isl_give PW *FN(PW,pullback_pw_multi_aff_aligned)(__isl_take PW *pw,
	__isl_take isl_pw_multi_aff *pma)
{
	int i;
	PW *res;

	if (!pma)
		goto error;

	if (pma->n == 0) {
		isl_space *space;
		space = isl_space_join(isl_pw_multi_aff_get_space(pma),
					FN(PW,get_space)(pw));
		isl_pw_multi_aff_free(pma);
		res = FN(PW,empty)(space);
		FN(PW,free)(pw);
		return res;
	}

	res = FN(PW,pullback_multi_aff)(FN(PW,copy)(pw),
					isl_multi_aff_copy(pma->p[0].maff));
	res = FN(PW,intersect_domain)(res, isl_set_copy(pma->p[0].set));

	for (i = 1; i < pma->n; ++i) {
		PW *res_i;

		res_i = FN(PW,pullback_multi_aff)(FN(PW,copy)(pw),
					isl_multi_aff_copy(pma->p[i].maff));
		res_i = FN(PW,intersect_domain)(res_i,
					isl_set_copy(pma->p[i].set));
		res = FN(PW,add_disjoint)(res, res_i);
	}

	isl_pw_multi_aff_free(pma);
	FN(PW,free)(pw);
	return res;
error:
	isl_pw_multi_aff_free(pma);
	FN(PW,free)(pw);
	return NULL;
}

__isl_give PW *FN(PW,pullback_pw_multi_aff)(__isl_take PW *pw,
	__isl_take isl_pw_multi_aff *pma)
{
	FN(PW,align_params_pw_multi_aff)(&pw, &pma);
	return FN(PW,pullback_pw_multi_aff_aligned)(pw, pma);
}
