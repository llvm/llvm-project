/*
 * Copyright 2014      Ecole Normale Superieure
 *
 * Use of this software is governed by the MIT license
 *
 * Written by Sven Verdoolaege,
 * Ecole Normale Superieure, 45 rue d'Ulm, 75230 Paris, France
 */

#include <isl_multi_macro.h>

/* Apply "fn" to each of the base expressions of "multi".
 */
static __isl_give MULTI(BASE) *FN(MULTI(BASE),un_op)(
	__isl_take MULTI(BASE) *multi, __isl_give EL *(*fn)(__isl_take EL *el))
{
	int i;
	isl_size n;

	n = FN(MULTI(BASE),size)(multi);
	if (n < 0)
		return FN(MULTI(BASE),free)(multi);

	for (i = 0; i < n; ++i) {
		EL *el;

		el = FN(MULTI(BASE),take_at)(multi, i);
		el = fn(el);
		multi = FN(MULTI(BASE),restore_at)(multi, i, el);
	}

	return multi;
}
