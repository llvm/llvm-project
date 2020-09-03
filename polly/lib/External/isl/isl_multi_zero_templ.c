/*
 * Copyright 2012      Ecole Normale Superieure
 *
 * Use of this software is governed by the MIT license
 *
 * Written by Sven Verdoolaege,
 * Ecole Normale Superieure, 45 rue d'Ulm, 75230 Paris, France
 */

#include <isl/space.h>
#include <isl/local_space.h>

#include <isl_multi_macro.h>

/* Construct a multi expression in the given space with value zero in
 * each of the output dimensions.
 */
__isl_give MULTI(BASE) *FN(MULTI(BASE),zero)(__isl_take isl_space *space)
{
	isl_size n;
	MULTI(BASE) *multi;

	n = isl_space_dim(space , isl_dim_out);
	if (n < 0)
		goto error;

	multi = FN(MULTI(BASE),alloc)(isl_space_copy(space));

	if (!n)
		isl_space_free(space);
	else {
		int i;
		isl_local_space *ls;
		EL *el;

		space = isl_space_domain(space);
		ls = isl_local_space_from_space(space);
		el = FN(EL,zero_on_domain)(ls);

		for (i = 0; i < n; ++i)
			multi = FN(FN(MULTI(BASE),set),BASE)(multi, i,
							    FN(EL,copy)(el));

		FN(EL,free)(el);
	}

	return multi;
error:
	isl_space_free(space);
	return NULL;
}
