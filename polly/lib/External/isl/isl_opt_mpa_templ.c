/*
 * Copyright 2018      Cerebras Systems
 *
 * Use of this software is governed by the MIT license
 *
 * Written by Sven Verdoolaege,
 * Cerebras Systems, 175 S San Antonio Rd, Los Altos, CA, USA
 */

#undef TYPE
#define TYPE CAT(isl_,BASE)
#define xFN(TYPE,NAME) TYPE ## _ ## NAME
#define FN(TYPE,NAME) xFN(TYPE,NAME)

/* Compute the optima of the set or output dimensions as a function of the
 * parameters (and input dimensions), but independently of
 * the other set or output dimensions,
 * given a function "opt" that computes this optimum
 * for a single dimension.
 *
 * If the resulting multi piecewise affine expression has
 * an explicit domain, then assign it the parameter domain of the input.
 * In other cases, the parameter domain is stored in the individual elements.
 */
static __isl_give isl_multi_pw_aff *FN(BASE,opt_mpa)(__isl_take TYPE *obj,
	__isl_give isl_pw_aff *(*opt)(__isl_take TYPE *obj, int pos))
{
	int i;
	isl_size n;
	isl_multi_pw_aff *mpa;

	mpa = isl_multi_pw_aff_alloc(FN(TYPE,get_space)(obj));
	n = isl_multi_pw_aff_size(mpa);
	if (n < 0)
		mpa = isl_multi_pw_aff_free(mpa);
	for (i = 0; i < n; ++i) {
		isl_pw_aff *pa;

		pa = opt(FN(TYPE,copy)(obj), i);
		mpa = isl_multi_pw_aff_set_pw_aff(mpa, i, pa);
	}
	if (isl_multi_pw_aff_has_explicit_domain(mpa)) {
		isl_set *dom;

		dom = FN(TYPE,params)(FN(TYPE,copy)(obj));
		mpa = isl_multi_pw_aff_intersect_params(mpa, dom);
	}
	FN(TYPE,free)(obj);

	return mpa;
}
