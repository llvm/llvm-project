/*
 * Copyright 2018      Sven Verdoolaege
 *
 * Use of this software is governed by the MIT license
 *
 * Written by Sven Verdoolaege.
 */

#define xFN(TYPE,NAME) TYPE ## _ ## NAME
#define FN(TYPE,NAME) xFN(TYPE,NAME)

/* Return the minimum of the integer piecewise affine
 * expression "f" over its definition domain.
 *
 * Return negative infinity if the optimal value is unbounded and
 * NaN if the domain of the expression is empty.
 */
__isl_give isl_val *FN(TYPE,min_val)(__isl_take TYPE *f)
{
	return FN(TYPE,opt_val)(f, 0);
}

/* Return the maximum of the integer piecewise affine
 * expression "f" over its definition domain.
 *
 * Return infinity if the optimal value is unbounded and
 * NaN if the domain of the expression is empty.
 */
__isl_give isl_val *FN(TYPE,max_val)(__isl_take TYPE *f)
{
	return FN(TYPE,opt_val)(f, 1);
}
