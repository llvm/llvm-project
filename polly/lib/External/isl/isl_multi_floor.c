/*
 * Copyright 2014      Ecole Normale Superieure
 *
 * Use of this software is governed by the MIT license
 *
 * Written by Sven Verdoolaege,
 * Ecole Normale Superieure, 45 rue d'Ulm, 75230 Paris, France
 */

#include <isl_multi_macro.h>

/* Given f, return floor(f).
 */
__isl_give MULTI(BASE) *FN(MULTI(BASE),floor)(__isl_take MULTI(BASE) *multi)
{
	return FN(MULTI(BASE),un_op)(multi, &FN(EL,floor));
}
