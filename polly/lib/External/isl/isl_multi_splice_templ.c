/*
 * Copyright 2012      Ecole Normale Superieure
 *
 * Use of this software is governed by the MIT license
 *
 * Written by Sven Verdoolaege,
 * Ecole Normale Superieure, 45 rue d'Ulm, 75230 Paris, France
 */

#include <isl/ctx.h>
#include <isl/space.h>

#include <isl_multi_macro.h>

/* Given two multi expressions, "multi1"
 *
 *	[A1 A2] -> [B1 B2]
 *
 * where A2 starts at position "in_pos" and B2 starts at position "out_pos",
 * and "multi2"
 *
 *	[C] -> [D]
 *
 * return the multi expression
 *
 *	[A1 C A2] -> [B1 D B2]
 *
 * We first insert input dimensions to obtain
 *
 *	[A1 C A2] -> [B1 B2]
 *
 * and
 *
 *	[A1 C A2] -> [D]
 *
 * and then apply range_splice.
 */
__isl_give MULTI(BASE) *FN(MULTI(BASE),splice)(
	__isl_take MULTI(BASE) *multi1, unsigned in_pos, unsigned out_pos,
	__isl_take MULTI(BASE) *multi2)
{
	isl_size n_in1;
	isl_size n_in2;

	n_in1 = FN(MULTI(BASE),dim)(multi1, isl_dim_in);
	n_in2 = FN(MULTI(BASE),dim)(multi2, isl_dim_in);
	if (n_in1 < 0 || n_in2 < 0)
		goto error;

	if (FN(MULTI(BASE),check_range)(multi1, isl_dim_in, in_pos, 0) < 0)
		goto error;

	multi1 = FN(MULTI(BASE),insert_dims)(multi1, isl_dim_in, in_pos, n_in2);
	multi2 = FN(MULTI(BASE),insert_dims)(multi2, isl_dim_in, n_in2,
						n_in1 - in_pos);
	multi2 = FN(MULTI(BASE),insert_dims)(multi2, isl_dim_in, 0, in_pos);

	return FN(MULTI(BASE),range_splice)(multi1, out_pos, multi2);
error:
	FN(MULTI(BASE),free)(multi1);
	FN(MULTI(BASE),free)(multi2);
	return NULL;
}
