/*
 * Copyright 2012,2014 Ecole Normale Superieure
 *
 * Use of this software is governed by the MIT license
 *
 * Written by Sven Verdoolaege,
 * Ecole Normale Superieure, 45 rue d'Ulm, 75230 Paris, France
 */

#include <isl/space.h>

#include <isl_multi_macro.h>

/* Create a multiple expression with a single output/set dimension
 * equal to "el".
 * For most multiple expression types, the base type has a single
 * output/set dimension and the space of the result is therefore
 * the same as the space of the input.
 * In the case of isl_multi_union_pw_aff, however, the base type
 * lives in a parameter space and we therefore need to add
 * a single set dimension.
 */
__isl_give MULTI(BASE) *FN(FN(MULTI(BASE),from),BASE)(__isl_take EL *el)
{
	isl_space *space;
	MULTI(BASE) *multi;

	space = FN(EL,get_space(el));
	if (isl_space_is_params(space)) {
		space = isl_space_set_from_params(space);
		space = isl_space_add_dims(space, isl_dim_set, 1);
	}
	multi = FN(MULTI(BASE),alloc)(space);
	multi = FN(FN(MULTI(BASE),set),BASE)(multi, 0, el);

	return multi;
}
