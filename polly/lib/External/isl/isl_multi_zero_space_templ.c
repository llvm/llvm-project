/*
 * Copyright 2020      Cerebras Systems
 *
 * Use of this software is governed by the MIT license
 *
 * Written by Sven Verdoolaege,
 * Cerebras Systems, 175 S San Antonio Rd, Los Altos, CA, USA
 */

#include <isl/space.h>

#include "isl_multi_macro.h"

/* This function performs the same operation as isl_multi_*_zero,
 * but is considered as a function on an isl_space when exported.
 */
__isl_give MULTI(BASE) *FN(isl_space_zero_multi,BASE)(
	__isl_take isl_space *space)
{
	return FN(MULTI(BASE),zero)(space);
}
