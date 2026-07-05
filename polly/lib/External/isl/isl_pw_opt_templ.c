/*
 * Copyright 2010      INRIA Saclay
 *
 * Use of this software is governed by the MIT license
 *
 * Written by Sven Verdoolaege, INRIA Saclay - Ile-de-France,
 * Parc Club Orsay Universite, ZAC des vignes, 4 rue Jacques Monod,
 * 91893 Orsay, France
 */

/* Compute the maximal value attained by the piecewise quasipolynomial
 * on its domain or zero if the domain is empty.
 * In the worst case, the domain is scanned completely,
 * so the domain is assumed to be bounded.
 */
__isl_give isl_val *FN(PW,opt)(__isl_take PW *pw, int max)
{
	int i;
	isl_val *opt;

	if (!pw)
		return NULL;

	if (pw->n == 0) {
		opt = isl_val_zero(FN(PW,get_ctx)(pw));
		FN(PW,free)(pw);
		return opt;
	}

	opt = FN(EL,opt_on_domain)(FN(EL,copy)(pw->p[0].FIELD),
					isl_set_copy(pw->p[0].set), max);
	for (i = 1; i < pw->n; ++i) {
		isl_val *opt_i;
		opt_i = FN(EL,opt_on_domain)(FN(EL,copy)(pw->p[i].FIELD),
						isl_set_copy(pw->p[i].set), max);
		if (max)
			opt = isl_val_max(opt, opt_i);
		else
			opt = isl_val_min(opt, opt_i);
	}

	FN(PW,free)(pw);
	return opt;
}

__isl_give isl_val *FN(PW,max)(__isl_take PW *pw)
{
	return FN(PW,opt)(pw, 1);
}

__isl_give isl_val *FN(PW,min)(__isl_take PW *pw)
{
	return FN(PW,opt)(pw, 0);
}
