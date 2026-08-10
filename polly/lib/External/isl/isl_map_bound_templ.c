/*
 * Copyright 2018      Cerebras Systems
 *
 * Use of this software is governed by the MIT license
 *
 * Written by Sven Verdoolaege,
 * Cerebras Systems, 175 S San Antonio Rd, Los Altos, CA, USA
 */

#include "isl_multi_macro.h"
#undef TYPE
#define TYPE CAT(isl_,BASE)

/* Check that "map" and "multi" live in the same space, ignoring parameters.
 */
static isl_stat FN(check_map_equal_tuples_multi,BASE)(__isl_keep isl_map *map,
	__isl_keep MULTI(BASE) *multi)
{
	isl_space *map_space, *multi_space;

	map_space = isl_map_peek_space(map);
	multi_space = FN(MULTI(BASE),peek_space)(multi);
	return isl_space_check_equal_tuples(map_space, multi_space);
}

/* Apply "map_bound" to "map" with the corresponding value in "bound"
 * for each output dimension.
 * If "bound" has an explicit domain (which implies that "bound"
 * is zero-dimensional), then intersect the domain of "map"
 * with this explicit domain instead.
 */
static __isl_give isl_map *FN(map_bound_multi,BASE)(__isl_take isl_map *map,
	__isl_take MULTI(BASE) *bound,
	__isl_give isl_map *map_bound(__isl_take isl_map *map,
		unsigned pos, __isl_take TYPE *value))
{
	int i;
	isl_size dim;

	dim = isl_map_dim(map, isl_dim_out);
	if (dim < 0 || FN(check_map_equal_tuples_multi,BASE)(map, bound) < 0)
		goto error;

	for (i = 0; i < dim; ++i) {
		TYPE *el;

		el = FN(MULTI(BASE),get_at)(bound, i);
		map = map_bound(map, i, el);
	}
	map = FN(FN(isl_map_intersect_multi,BASE),explicit_domain)(map, bound);
	FN(MULTI(BASE),free)(bound);
	return map;
error:
	isl_map_free(map);
	FN(MULTI(BASE),free)(bound);
	return NULL;
}
