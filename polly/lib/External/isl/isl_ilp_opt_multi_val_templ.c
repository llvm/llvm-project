/*
 * Copyright 2019      Cerebras Systems
 *
 * Use of this software is governed by the MIT license
 *
 * Written by Sven Verdoolaege,
 * Cerebras Systems, 175 S San Antonio Rd, Los Altos, CA, USA
 */

#define xFN(TYPE,NAME) TYPE ## _ ## NAME
#define FN(TYPE,NAME) xFN(TYPE,NAME)

/* Return a list of minima (maxima if "max" is set)
 * for each of the expressions in "f" over their (shared) domain.
 *
 * An element in the list is infinity or negative infinity if the optimal
 * value of the corresponding expression is unbounded and
 * NaN if the domain of the expression is empty.
 *
 * Iterate over all the expressions in "f" and collect the results.
 */
static __isl_give isl_multi_val *FN(TYPE,opt_multi_val)(__isl_take TYPE *f,
	int max)
{
	int i;
	isl_size n;
	isl_space *space;
	isl_multi_val *mv;

	n = FN(TYPE,dim)(f, isl_dim_out);
	if (n < 0)
		f = FN(TYPE,free)(f);
	if (!f)
		return NULL;

	space = isl_space_range(FN(TYPE,get_space)(f));
	space = isl_space_drop_all_params(space);
	mv = isl_multi_val_zero(space);

	for (i = 0; i < n; ++i) {
		isl_val *v;
		isl_pw_aff *pa;

		pa = FN(TYPE,get_pw_aff)(f, i);
		v = isl_pw_aff_opt_val(pa, max);
		mv = isl_multi_val_set_val(mv, i, v);
	}

	FN(TYPE,free)(f);
	return mv;
}

/* Return a list of minima
 * for each of the expressions in "f" over their (shared) domain.
 *
 * An element in the list is negative infinity if the optimal
 * value of the corresponding expression is unbounded and
 * NaN if the domain of the expression is empty.
 */
__isl_give isl_multi_val *FN(TYPE,min_multi_val)(__isl_take TYPE *f)
{
	return FN(TYPE,opt_multi_val)(f, 0);
}

/* Return a list of maxima
 * for each of the expressions in "f" over their (shared) domain.
 *
 * An element in the list is infinity if the optimal
 * value of the corresponding expression is unbounded and
 * NaN if the domain of the expression is empty.
 */
__isl_give isl_multi_val *FN(TYPE,max_multi_val)(__isl_take TYPE *f)
{
	return FN(TYPE,opt_multi_val)(f, 1);
}
