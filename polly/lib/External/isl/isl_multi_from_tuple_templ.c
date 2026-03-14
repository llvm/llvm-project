/*
 * Copyright 2011      Sven Verdoolaege
 * Copyright 2012      Ecole Normale Superieure
 *
 * Use of this software is governed by the MIT license
 *
 * Written by Sven Verdoolaege,
 * Ecole Normale Superieure, 45 rue d'Ulm, 75230 Paris, France
 */

#include <isl_multi_macro.h>

/* Extract a multi expression with domain space "dom_space"
 * from a tuple "tuple" that was read by read_tuple.
 *
 * Check that none of the expressions depend on any other output/set dimensions.
 */
static MULTI(BASE) *FN(MULTI(BASE),from_tuple)(
	__isl_take isl_space *dom_space, __isl_take isl_multi_pw_aff *tuple)
{
	int i;
	isl_size dim, n;
	isl_space *space;
	MULTI(BASE) *multi;

	n = isl_multi_pw_aff_dim(tuple, isl_dim_out);
	dim = isl_space_dim(dom_space, isl_dim_all);
	if (n < 0 || dim < 0)
		dom_space = isl_space_free(dom_space);
	space = isl_space_range(isl_multi_pw_aff_get_space(tuple));
	space = isl_space_align_params(space, isl_space_copy(dom_space));
	if (!isl_space_is_params(dom_space))
		space = isl_space_map_from_domain_and_range(
				isl_space_copy(dom_space), space);
	isl_space_free(dom_space);
	multi = FN(MULTI(BASE),alloc)(space);

	for (i = 0; i < n; ++i) {
		isl_pw_aff *pa;
		pa = isl_multi_pw_aff_get_pw_aff(tuple, i);
		multi = FN(MULTI(BASE),set_tuple_entry)(multi, pa, i, dim, n);
	}

	isl_multi_pw_aff_free(tuple);
	return multi;
}
