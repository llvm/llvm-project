/*
 * Copyright 2019      Cerebras Systems
 *
 * Use of this software is governed by the MIT license
 *
 * Written by Sven Verdoolaege,
 * Cerebras Systems, 175 S San Antonio Rd, Los Altos, CA, USA
 */

#include <isl_multi_macro.h>

/* Does "multi" involve any local variables?
 */
isl_bool FN(MULTI(BASE),involves_locals)(__isl_keep MULTI(BASE) *multi)
{
	return FN(MULTI(BASE),any)(multi, FN(EL,involves_locals));
}
