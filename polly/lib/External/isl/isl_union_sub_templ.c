/*
 * Copyright 2010      INRIA Saclay
 *
 * Use of this software is governed by the MIT license
 *
 * Written by Sven Verdoolaege,
 * INRIA Saclay - Ile-de-France, Parc Club Orsay Universite,
 * ZAC des vignes, 4 rue Jacques Monod, 91893 Orsay, France
 */

#include "isl_union_macro.h"

/* Subtract "u2" from "u1" and return the result.
 *
 * If the base expressions have a default zero value, then
 * reuse isl_union_*_add to ensure the result
 * is computed on the union of the domains of "u1" and "u2".
 * Otherwise, compute the result directly on their shared domain.
 */
__isl_give UNION *FN(UNION,sub)(__isl_take UNION *u1, __isl_take UNION *u2)
{
#if DEFAULT_IS_ZERO
	return FN(UNION,add)(u1, FN(UNION,neg)(u2));
#else
	return FN(UNION,match_bin_op)(u1, u2, &FN(PART,sub));
#endif
}
