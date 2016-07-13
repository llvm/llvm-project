/*
 * Copyright 2010-2011 INRIA Saclay
 *
 * Use of this software is governed by the MIT license
 *
 * Written by Sven Verdoolaege, INRIA Saclay - Ile-de-France,
 * Parc Club Orsay Universite, ZAC des vignes, 4 rue Jacques Monod,
 * 91893 Orsay, France
 */

#include <assert.h>
#include <ctype.h>
#include <string.h>

#include <isl/set.h>
#include <isl/map.h>
#include <isl/constraint.h>

#include "schedule.h"

/* Construct a map from a len-dimensional domain to
 * a (len-n)-dimensional domain that projects out the n coordinates
 * starting at first.
 * "dim" prescribes the parameters.
 */
__isl_give isl_map *project_out(__isl_take isl_space *dim,
    int len, int first, int n)
{
    int i, j;
    isl_basic_map *bmap;

    dim = isl_space_add_dims(dim, isl_dim_in, len);
    dim = isl_space_add_dims(dim, isl_dim_out, len - n);
    bmap = isl_basic_map_universe(dim);

    for (i = 0, j = 0; i < len; ++i) {
        if (i >= first && i < first + n)
            continue;
	bmap = isl_basic_map_equate(bmap, isl_dim_in, i, isl_dim_out, j);
        ++j;
    }

    return isl_map_from_basic_map(bmap);
}

/* Construct a projection that maps a src_len dimensional domain
 * to its first dst_len coordinates.
 * "dim" prescribes the parameters.
 */
__isl_give isl_map *projection(__isl_take isl_space *dim,
    int src_len, int dst_len)
{
    return project_out(dim, src_len, dst_len, src_len - dst_len);
}

/* Add parameters with identifiers "ids" to "set".
 */
static __isl_give isl_set *add_params(__isl_take isl_set *set,
	__isl_keep isl_id_list *ids)
{
	int i, n;
	unsigned nparam;

	n = isl_id_list_n_id(ids);

	nparam = isl_set_dim(set, isl_dim_param);
	set = isl_set_add_dims(set, isl_dim_param, n);

	for (i = 0; i < n; ++i) {
		isl_id *id;

		id = isl_id_list_get_id(ids, i);
		set = isl_set_set_dim_id(set, isl_dim_param, nparam + i, id);
	}

	return set;
}

/* Equate the dimensions of "set" starting at "first" to
 * freshly created parameters with identifiers "ids".
 * The number of equated dimensions is equal to the number of elements in "ids".
 */
static __isl_give isl_set *parametrize(__isl_take isl_set *set,
	int first, __isl_keep isl_id_list *ids)
{
	int i, n;
	unsigned nparam;

	nparam = isl_set_dim(set, isl_dim_param);

	set = add_params(set, ids);

	n = isl_id_list_n_id(ids);
	for (i = 0; i < n; ++i)
		set = isl_set_equate(set, isl_dim_param, nparam + i,
					isl_dim_set, first + i);

	return set;
}

/* Given a parameter space "space", create a set of dimension "len"
 * of which the dimensions starting at "first" are equated to
 * freshly created parameters with identifiers "ids".
 */
__isl_give isl_set *parametrization(__isl_take isl_space *space,
	int len, int first, __isl_keep isl_id_list *ids)
{
	isl_set *set;

	space = isl_space_set_from_params(space);
	space = isl_space_add_dims(space, isl_dim_set, len);
	set = isl_set_universe(space);

	return parametrize(set, first, ids);
}

/* Extend "set" with unconstrained coordinates to a total length of "dst_len".
 */
__isl_give isl_set *extend(__isl_take isl_set *set, int dst_len)
{
    int n_set;
    isl_space *dim;
    isl_map *map;

    dim = isl_set_get_space(set);
    n_set = isl_space_dim(dim, isl_dim_set);
    dim = isl_space_drop_dims(dim, isl_dim_set, 0, n_set);
    map = projection(dim, dst_len, n_set);
    map = isl_map_reverse(map);

    return isl_set_apply(set, map);
}

/* Set max_out to the maximal number of output dimensions over
 * all maps.
 */
static isl_stat update_max_out(__isl_take isl_map *map, void *user)
{
	int *max_out = user;
	int n_out = isl_map_dim(map, isl_dim_out);

	if (n_out > *max_out)
		*max_out = n_out;

	isl_map_free(map);
	return isl_stat_ok;
}

struct align_range_data {
	int max_out;
	isl_union_map *res;
};

/* Extend the dimension of the range of the given map to data->max_out and
 * then add the result to data->res.
 */
static isl_stat map_align_range(__isl_take isl_map *map, void *user)
{
	struct align_range_data *data = user;
	int i;
	isl_space *dim;
	isl_map *proj;
	int n_out = isl_map_dim(map, isl_dim_out);

	dim = isl_union_map_get_space(data->res);
	proj = isl_map_reverse(projection(dim, data->max_out, n_out));
	for (i = n_out; i < data->max_out; ++i)
		proj = isl_map_fix_si(proj, isl_dim_out, i, 0);

	map = isl_map_apply_range(map, proj);

	data->res = isl_union_map_add_map(data->res, map);

	return isl_stat_ok;
}

/* Extend the ranges of the maps in the union map such they all have
 * the same dimension.
 */
__isl_give isl_union_map *align_range(__isl_take isl_union_map *umap)
{
	struct align_range_data data;

	data.max_out = 0;
	isl_union_map_foreach_map(umap, &update_max_out, &data.max_out);

	data.res = isl_union_map_empty(isl_union_map_get_space(umap));
	isl_union_map_foreach_map(umap, &map_align_range, &data);

	isl_union_map_free(umap);
	return data.res;
}
