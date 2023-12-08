/*
 * Copyright 2014      Ecole Normale Superieure
 *
 * Use of this software is governed by the MIT license
 *
 * Written by Sven Verdoolaege,
 * Ecole Normale Superieure, 45 rue d'Ulm, 75230 Paris, France
 */

#include <isl_multi_macro.h>

/* Does "multi" involve any NaNs?
 */
isl_bool FN(MULTI(BASE),involves_nan)(__isl_keep MULTI(BASE) *multi)
{
	return FN(MULTI(BASE),any)(multi, &FN(EL,involves_nan));
}
