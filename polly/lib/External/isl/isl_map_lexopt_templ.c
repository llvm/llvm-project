/*
 * Copyright 2010      INRIA Saclay
 * Copyright 2012      Ecole Normale Superieure
 *
 * Use of this software is governed by the MIT license
 *
 * Written by Sven Verdoolaege,
 * INRIA Saclay - Ile-de-France, Parc Club Orsay Universite,
 * ZAC des vignes, 4 rue Jacques Monod, 91893 Orsay, France
 * and Ecole Normale Superieure, 45 rue d’Ulm, 75230 Paris, France
 */

/* Function for computing the lexicographic optimum of a map
 * in the form of either an isl_map or an isl_pw_multi_aff.
 */

#define xSF(TYPE,SUFFIX) TYPE ## SUFFIX
#define SF(TYPE,SUFFIX) xSF(TYPE,SUFFIX)

/* Given a basic map "bmap", compute the lexicographically minimal
 * (or maximal) image element for each domain element in dom.
 * If empty is not NULL, then set *empty to those elements in dom
 * that do not have an image element.
 *
 * We first make sure the basic sets in dom are disjoint and then
 * simply collect the results over each of the basic sets separately.
 * We could probably improve the efficiency a bit by moving the union
 * domain down into the parametric integer programming.
 */
static __isl_give TYPE *SF(basic_map_partial_lexopt,SUFFIX)(
	__isl_take isl_basic_map *bmap, __isl_take isl_set *dom,
	__isl_give isl_set **empty, int max)
{
	int i;
	TYPE *res;
	isl_set *all_empty;

	dom = isl_set_make_disjoint(dom);
	if (!dom)
		goto error;

	if (isl_set_plain_is_empty(dom)) {
		isl_space *space = isl_basic_map_get_space(bmap);
		if (empty)
			*empty = dom;
		else
			isl_set_free(dom);
		isl_basic_map_free(bmap);
		return EMPTY(space);
	}

	res = SF(isl_basic_map_partial_lexopt,SUFFIX)(isl_basic_map_copy(bmap),
			isl_basic_set_copy(dom->p[0]), empty, max);

	if (empty)
		all_empty = *empty;
	for (i = 1; i < dom->n; ++i) {
		TYPE *res_i;

		res_i = SF(isl_basic_map_partial_lexopt,SUFFIX)(
				isl_basic_map_copy(bmap),
				isl_basic_set_copy(dom->p[i]), empty, max);

		res = ADD(res, res_i);
		if (empty)
			all_empty = isl_set_union_disjoint(all_empty, *empty);
	}

	if (empty)
		*empty = all_empty;
	isl_set_free(dom);
	isl_basic_map_free(bmap);
	return res;
error:
	if (empty)
		*empty = NULL;
	isl_set_free(dom);
	isl_basic_map_free(bmap);
	return NULL;
}

static __isl_give TYPE *SF(isl_map_partial_lexopt_aligned,SUFFIX)(
	__isl_take isl_map *map, __isl_take isl_set *dom,
	__isl_give isl_set **empty, int max);

/* Given a map "map", compute the lexicographically minimal
 * (or maximal) image element for each domain element in dom.
 * Set *empty to those elements in dom that do not have an image element.
 *
 * Align parameters if needed and then call isl_map_partial_lexopt_aligned.
 */
static __isl_give TYPE *SF(isl_map_partial_lexopt,SUFFIX)(
	__isl_take isl_map *map, __isl_take isl_set *dom,
	__isl_give isl_set **empty, int max)
{
	if (!map || !dom)
		goto error;
	if (isl_space_match(map->dim, isl_dim_param, dom->dim, isl_dim_param))
		return SF(isl_map_partial_lexopt_aligned,SUFFIX)(map, dom,
								empty, max);
	if (!isl_space_has_named_params(map->dim) ||
	    !isl_space_has_named_params(dom->dim))
		isl_die(map->ctx, isl_error_invalid,
			"unaligned unnamed parameters", goto error);
	map = isl_map_align_params(map, isl_map_get_space(dom));
	dom = isl_map_align_params(dom, isl_map_get_space(map));
	return SF(isl_map_partial_lexopt_aligned,SUFFIX)(map, dom, empty, max);
error:
	if (empty)
		*empty = NULL;
	isl_set_free(dom);
	isl_map_free(map);
	return NULL;
}

__isl_give TYPE *SF(isl_map_lexopt,SUFFIX)(__isl_take isl_map *map, int max)
{
	isl_set *dom = NULL;
	isl_space *dom_space;

	if (!map)
		goto error;
	dom_space = isl_space_domain(isl_space_copy(map->dim));
	dom = isl_set_universe(dom_space);
	return SF(isl_map_partial_lexopt,SUFFIX)(map, dom, NULL, max);
error:
	isl_map_free(map);
	return NULL;
}

__isl_give TYPE *SF(isl_map_lexmin,SUFFIX)(__isl_take isl_map *map)
{
	return SF(isl_map_lexopt,SUFFIX)(map, 0);
}

__isl_give TYPE *SF(isl_map_lexmax,SUFFIX)(__isl_take isl_map *map)
{
	return SF(isl_map_lexopt,SUFFIX)(map, 1);
}

__isl_give TYPE *SF(isl_set_lexmin,SUFFIX)(__isl_take isl_set *set)
{
	return SF(isl_map_lexmin,SUFFIX)(set);
}

__isl_give TYPE *SF(isl_set_lexmax,SUFFIX)(__isl_take isl_set *set)
{
	return SF(isl_map_lexmax,SUFFIX)(set);
}
