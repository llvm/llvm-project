/*
 * Copyright 2010      INRIA Saclay
 * Copyright 2013      Ecole Normale Superieure
 *
 * Use of this software is governed by the MIT license
 *
 * Written by Sven Verdoolaege, INRIA Saclay - Ile-de-France,
 * Parc Club Orsay Universite, ZAC des vignes, 4 rue Jacques Monod,
 * 91893 Orsay, France
 * and Ecole Normale Superieure, 45 rue d'Ulm, 75230 Paris, France
 */

#include <isl/val.h>
#include <isl_space_private.h>
#include <isl_point_private.h>

#include <isl_pw_macro.h>

#undef SUFFIX
#define SUFFIX	point
#undef ARG1
#define ARG1	PW
#undef ARG2
#define ARG2	isl_point

static
#include "isl_align_params_templ.c"

/* Evaluate "pw" in the void point "pnt".
 * In particular, return the value NaN.
 */
static __isl_give isl_val *FN(PW,eval_void)(__isl_take PW *pw,
	__isl_take isl_point *pnt)
{
	isl_ctx *ctx;

	ctx = isl_point_get_ctx(pnt);
	FN(PW,free)(pw);
	isl_point_free(pnt);
	return isl_val_nan(ctx);
}

/* Evaluate the piecewise function "pw" in "pnt".
 * If the point is void, then return NaN.
 * If the point lies outside the domain of "pw", then return 0 or NaN
 * depending on whether 0 is the default value for this type of function.
 *
 * Align the parameters if needed, but "pnt" should specify a value
 * for all parameters in "pw".
 */
__isl_give isl_val *FN(PW,eval)(__isl_take PW *pw, __isl_take isl_point *pnt)
{
	int i;
	isl_bool is_void;
	isl_bool found;
	isl_ctx *ctx;
	isl_bool ok;
	isl_space *pnt_space, *pw_space;
	isl_val *v;

	FN(PW,align_params_point)(&pw, &pnt);

	pnt_space = isl_point_peek_space(pnt);
	pw_space = FN(PW,peek_space)(pw);
	ok = isl_space_is_domain_internal(pnt_space, pw_space);
	if (ok < 0)
		goto error;
	ctx = isl_point_get_ctx(pnt);
	if (!ok)
		isl_die(ctx, isl_error_invalid,
			"incompatible spaces", goto error);
	is_void = isl_point_is_void(pnt);
	if (is_void < 0)
		goto error;
	if (is_void)
		return FN(PW,eval_void)(pw, pnt);

	found = isl_bool_false;
	for (i = 0; i < pw->n; ++i) {
		found = isl_set_contains_point(pw->p[i].set, pnt);
		if (found < 0)
			goto error;
		if (found)
			break;
	}
	if (found) {
		v = FN(EL,eval)(FN(EL,copy)(pw->p[i].FIELD),
					    isl_point_copy(pnt));
	} else if (DEFAULT_IS_ZERO) {
		v = isl_val_zero(ctx);
	} else {
		v = isl_val_nan(ctx);
	}
	FN(PW,free)(pw);
	isl_point_free(pnt);
	return v;
error:
	FN(PW,free)(pw);
	isl_point_free(pnt);
	return NULL;
}
