/*
 * Copyright 2010      INRIA Saclay
 *
 * Use of this software is governed by the MIT license
 *
 * Written by Sven Verdoolaege, INRIA Saclay - Ile-de-France,
 * Parc Club Orsay Universite, ZAC des vignes, 4 rue Jacques Monod,
 * 91893 Orsay, France
 */

#include <isl_union_macro.h>

/* Evaluate "u" in the void point "pnt".
 * In particular, return the value NaN.
 */
static __isl_give isl_val *FN(UNION,eval_void)(__isl_take UNION *u,
	__isl_take isl_point *pnt)
{
	isl_ctx *ctx;

	ctx = isl_point_get_ctx(pnt);
	FN(UNION,free)(u);
	isl_point_free(pnt);
	return isl_val_nan(ctx);
}

/* Internal data structure for isl_union_*_eval.
 *
 * "pnt" is the point in which the function is evaluated.
 * "v" stores the result and is initialized to zero.
 */
S(UNION,eval_data) {
	isl_point *pnt;
	isl_val *v;
};

/* Update the evaluation in data->v based on the evaluation of "part".
 *
 * Only (at most) a single part on which this function is called
 * is assumed to evaluate to anything other than zero.
 * Since the value is initialized to zero, the evaluation of "part"
 * can simply be added.
 */
static isl_stat FN(UNION,eval_entry)(__isl_take PART *part, void *user)
{
	S(UNION,eval_data) *data = user;
	isl_val *v;

	v = FN(PART,eval)(part, isl_point_copy(data->pnt));
	data->v = isl_val_add(data->v, v);

	return isl_stat_non_null(data->v);
}

/* Evaluate "u" in the point "pnt".
 */
__isl_give isl_val *FN(UNION,eval)(__isl_take UNION *u,
	__isl_take isl_point *pnt)
{
	S(UNION,eval_data) data = { pnt };
	isl_bool is_void;
	isl_space *space;

	is_void = isl_point_is_void(pnt);
	if (is_void < 0)
		goto error;
	if (is_void)
		return FN(UNION,eval_void)(u, pnt);

	data.v = isl_val_zero(isl_point_get_ctx(pnt));
	space = isl_point_peek_space(pnt);
	if (FN(UNION,foreach_on_domain)(u, space,
					&FN(UNION,eval_entry), &data) < 0)
		data.v = isl_val_free(data.v);
	FN(UNION,free)(u);
	isl_point_free(pnt);
	return data.v;
error:
	FN(UNION,free)(u);
	isl_point_free(pnt);
	return NULL;
}
