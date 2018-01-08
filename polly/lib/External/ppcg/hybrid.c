/*
 * Copyright 2013      Ecole Normale Superieure
 * Copyright 2015      Sven Verdoolaege
 *
 * Use of this software is governed by the MIT license
 *
 * Written by Sven Verdoolaege,
 * Ecole Normale Superieure, 45 rue d'Ulm, 75230 Paris, France
 */

#include <string.h>

#include <isl/space.h>
#include <isl/constraint.h>
#include <isl/val.h>
#include <isl/aff.h>
#include <isl/set.h>
#include <isl/map.h>
#include <isl/union_set.h>
#include <isl/union_map.h>

#include "hybrid.h"
#include "schedule.h"

/* The hybrid tiling implemented in this file is based on
 * Grosser et al., "Hybrid Hexagonal/Classical Tiling for GPUs".
 */

/* Bounds on relative dependence distances in input to hybrid tiling.
 * upper is an upper bound on the relative dependence distances
 * in the first space dimension
 * -lower is a lower bound on the relative dependence distances
 * in all space dimensions.
 *
 * In particular,
 *
 *	d_i >= -lower_i d_0
 * and
 *	d_1 <= upper d_0
 *
 * for each dependence distance vector d, where d_1 is the component
 * corresponding to the first space dimension.
 *
 * upper and lower are always non-negative.
 * Some of the values may be NaN if no bound could be found.
 */
struct ppcg_ht_bounds {
	isl_val *upper;
	isl_multi_val *lower;
};

/* Free "bounds" along with all its fields.
 */
__isl_null ppcg_ht_bounds *ppcg_ht_bounds_free(
	__isl_take ppcg_ht_bounds *bounds)
{
	if (!bounds)
		return NULL;
	isl_val_free(bounds->upper);
	isl_multi_val_free(bounds->lower);
	free(bounds);

	return NULL;
}

/* Create a ppcg_ht_bounds object for a band living in "space".
 * The bounds are initialized to NaN.
 */
__isl_give ppcg_ht_bounds *ppcg_ht_bounds_alloc(__isl_take isl_space *space)
{
	int i, n;
	isl_ctx *ctx;
	ppcg_ht_bounds *bounds;

	if (!space)
		return NULL;

	ctx = isl_space_get_ctx(space);
	bounds = isl_alloc_type(ctx, struct ppcg_ht_bounds);
	if (!bounds)
		goto error;
	bounds->upper = isl_val_nan(ctx);
	bounds->lower = isl_multi_val_zero(space);
	n = isl_multi_val_dim(bounds->lower, isl_dim_set);
	for (i = 0; i < n; ++i) {
		isl_val *v = isl_val_copy(bounds->upper);
		bounds->lower = isl_multi_val_set_val(bounds->lower, i, v);
	}

	if (!bounds->lower || !bounds->upper)
		return ppcg_ht_bounds_free(bounds);

	return bounds;
error:
	isl_space_free(space);
	return NULL;
}

void ppcg_ht_bounds_dump(__isl_keep ppcg_ht_bounds *bounds)
{
	if (!bounds)
		return;

	fprintf(stderr, "lower: ");
	isl_multi_val_dump(bounds->lower);
	fprintf(stderr, "upper: ");
	isl_val_dump(bounds->upper);
}

/* Return the upper bound on the relative dependence distances
 * in the first space dimension.
 */
__isl_give isl_val *ppcg_ht_bounds_get_upper(__isl_keep ppcg_ht_bounds *bounds)
{
	if (!bounds)
		return NULL;
	return isl_val_copy(bounds->upper);
}

/* Replace the upper bound on the relative dependence distances
 * in the first space dimension by "upper".
 */
__isl_give ppcg_ht_bounds *ppcg_ht_bounds_set_upper(
	__isl_take ppcg_ht_bounds *bounds, __isl_take isl_val *upper)
{
	if (!bounds || !upper)
		goto error;
	isl_val_free(bounds->upper);
	bounds->upper = upper;
	return bounds;
error:
	ppcg_ht_bounds_free(bounds);
	isl_val_free(upper);
	return NULL;
}

/* Return the lower bound on the relative dependence distances
 * in space dimension "pos".
 */
__isl_give isl_val *ppcg_ht_bounds_get_lower(__isl_keep ppcg_ht_bounds *bounds,
	int pos)
{
	if (!bounds)
		return NULL;
	return isl_multi_val_get_val(bounds->lower, pos);
}

/* Replace the lower bound on the relative dependence distances
 * in space dimension "pos" by "lower".
 */
__isl_give ppcg_ht_bounds *ppcg_ht_bounds_set_lower(
	__isl_take ppcg_ht_bounds *bounds, int pos, __isl_take isl_val *lower)
{
	if (!bounds || !lower)
		goto error;
	bounds->lower = isl_multi_val_set_val(bounds->lower, pos, lower);
	if (!bounds->lower)
		return ppcg_ht_bounds_free(bounds);
	return bounds;
error:
	ppcg_ht_bounds_free(bounds);
	isl_val_free(lower);
	return NULL;
}

/* Can the bounds on relative dependence distances recorded in "bounds"
 * be used to perform hybrid tiling?
 * In particular, have appropriate lower and upper bounds been found?
 * Any NaN indicates that no corresponding bound was found.
 */
isl_bool ppcg_ht_bounds_is_valid(__isl_keep ppcg_ht_bounds *bounds)
{
	isl_bool is_nan;
	int i, n;

	if (!bounds)
		return isl_bool_error;
	is_nan = isl_val_is_nan(bounds->upper);
	if (is_nan < 0)
		return isl_bool_error;
	if (is_nan)
		return isl_bool_false;

	n = isl_multi_val_dim(bounds->lower, isl_dim_set);
	for (i = 0; i < n; ++i) {
		isl_val *v;

		v = isl_multi_val_get_val(bounds->lower, i);
		is_nan = isl_val_is_nan(v);
		if (is_nan < 0)
			return isl_bool_error;
		if (is_nan)
			return isl_bool_false;
		isl_val_free(v);
	}

	return isl_bool_true;
}

/* Structure that represents the basic hexagonal tiling,
 * along with information that is needed to perform the hybrid tiling.
 *
 * "bounds" are the bounds on the dependence distances that
 * define the hexagonal shape and the required skewing in the remaining
 * space dimensions.
 *
 * "input_node" points to the input pair of band nodes.
 * "input_schedule" is the partial schedule of this input pair of band nodes.
 * The space of this schedule is [P -> C], where P is the space
 * of the parent node and C is the space of the child node.
 *
 * "space_sizes" represent the total size of a tile for the space
 * dimensions, i.e., those corresponding to the child node.
 * The space of "space_sizes" is C.
 * If S_0 is the original tile size in the first space dimension,
 * then the first entry of "space_sizes" is equal to
 * W = 2*S_0 + floor(d_l h) + floor(d_u h).
 * The remaining entries are the same as in the original tile sizes.
 *
 * The basic hexagonal tiling "hex" is defined
 * in a "ts" (time-space) space and corresponds to the phase-1 tiles.
 * "time_tile" maps the "ts" space to outer time tile.
 * Is is equal to ts[t, s] -> floor(t/(2 * S_t)), with S_t the original tile
 * size corresponding to the parent node.
 * "local_time" maps the "ts" space to the time dimension inside each tile.
 * It is equal to ts[t, s] -> t mod (2 S_t), with S_t the original tile
 * size corresponding to the parent node.
 * "shift_space" shifts the tiles at time tile T = floor(t/(2 S_t))
 * in the space dimension such that they align to a multiple of W.
 * It is equal to ts[t, s] -> s + (-(2 * shift_s)*T) % W,
 * with shift_s = S_0 + floor(d_u h).
 * "shift_phase" is the shift taken to go from phase 0 to phase 1
 * It is equal to ts[t, s] -> ts[t + S_t, s + shift_s],
 * with shift_s = S_0 + floor(d_u h).
 *
 * "project_ts" projects the space of the input schedule to the ts-space.
 * It is equal to [P[t] -> C[s_0, ...]] -> ts[t, s_0].
 */
struct ppcg_ht_tiling {
	int ref;

	ppcg_ht_bounds *bounds;
	isl_schedule_node *input_node;
	isl_multi_union_pw_aff *input_schedule;

	isl_multi_val *space_sizes;

	isl_aff *time_tile;
	isl_aff *local_time;
	isl_aff *shift_space;
	isl_multi_aff *shift_phase;
	isl_set *hex;

	isl_multi_aff *project_ts;
};
typedef struct ppcg_ht_tiling ppcg_ht_tiling;

/* Return the space of the pair of band nodes that form the input
 * to the hybrid tiling.
 * In particular, return the space [P -> C], where P is the space
 * of the parent node and C is the space of the child node.
 */
__isl_give isl_space *ppcg_ht_tiling_get_input_space(
	__isl_keep ppcg_ht_tiling *tile)
{
	if (!tile)
		return NULL;

	return isl_multi_union_pw_aff_get_space(tile->input_schedule);
}

/* Remove a reference to "tile" and free "tile" along with all its fields
 * as soon as the reference count drops to zero.
 */
static __isl_null ppcg_ht_tiling *ppcg_ht_tiling_free(
	__isl_take ppcg_ht_tiling *tiling)
{
	if (!tiling)
		return NULL;
	if (--tiling->ref > 0)
		return NULL;

	ppcg_ht_bounds_free(tiling->bounds);
	isl_schedule_node_free(tiling->input_node);
	isl_multi_union_pw_aff_free(tiling->input_schedule);
	isl_multi_val_free(tiling->space_sizes);
	isl_aff_free(tiling->time_tile);
	isl_aff_free(tiling->local_time);
	isl_aff_free(tiling->shift_space);
	isl_multi_aff_free(tiling->shift_phase);
	isl_set_free(tiling->hex);
	isl_multi_aff_free(tiling->project_ts);
	free(tiling);

	return NULL;
}

/* Return a new reference to "tiling".
 */
__isl_give ppcg_ht_tiling *ppcg_ht_tiling_copy(
	__isl_keep ppcg_ht_tiling *tiling)
{
	if (!tiling)
		return NULL;

	tiling->ref++;
	return tiling;
}

/* Return the isl_ctx to which "tiling" belongs.
 */
isl_ctx *ppcg_ht_tiling_get_ctx(__isl_keep ppcg_ht_tiling *tiling)
{
	if (!tiling)
		return NULL;

	return isl_multi_union_pw_aff_get_ctx(tiling->input_schedule);
}

/* Representation of one of the two phases of hybrid tiling.
 *
 * "tiling" points to the shared tiling data.
 *
 * "time_tile", "local_time" and "shift_space" are equal to the corresponding
 * fields of "tiling", pulled back to the input space.
 * In case of phase 0, these expressions have also been moved
 * from phase 1 to phase 0.
 *
 * "domain" contains the hexagonal tiling of this phase.
 *
 * "space_shift" is the shift that should be added to the space band
 * in order to be able to apply rectangular tiling to the space.
 * For phase 1, it is equal to
 *
 *	[P[t] -> C[s_0, s_i]] -> C[(-(2 * shift_s)*T) % W, dl_i * u]
 *
 * with shift_s = S_0 + floor(d_u h),
 * T equal to "time_tile" and u equal to "local_time".
 * For phase 0, it is equal to
 *
 *	[P[t] -> C[s_0, s_i]] -> C[shift_s + (-(2 * shift_s)*T) % W, dl_i * u]
 *
 * "space_tile" is the space tiling.  It is equal to
 *
 *	[P[t] -> C[s]] -> C[floor((s + space_shift)/space_size]
 */
struct ppcg_ht_phase {
	ppcg_ht_tiling *tiling;

	isl_aff *time_tile;
	isl_aff *local_time;
	isl_aff *shift_space;
	isl_set *domain;

	isl_multi_aff *space_shift;
	isl_multi_aff *space_tile;
};

/* Free "phase" along with all its fields.
 */
static __isl_null ppcg_ht_phase *ppcg_ht_phase_free(
	__isl_take ppcg_ht_phase *phase)
{
	if (!phase)
		return NULL;

	ppcg_ht_tiling_free(phase->tiling);
	isl_aff_free(phase->time_tile);
	isl_aff_free(phase->local_time);
	isl_aff_free(phase->shift_space);
	isl_set_free(phase->domain);
	isl_multi_aff_free(phase->space_shift);
	isl_multi_aff_free(phase->space_tile);
	free(phase);

	return NULL;
}

/* Wrapper around ppcg_ht_phase_free for use as an argument
 * to isl_id_set_free_user.
 */
static void ppcg_ht_phase_free_wrap(void *user)
{
	ppcg_ht_phase *phase = user;

	ppcg_ht_phase_free(phase);
}

/* Return the domain of hybrid tiling phase "phase".
 */
static __isl_give isl_set *ppcg_ht_phase_get_domain(ppcg_ht_phase *phase)
{
	if (!phase)
		return NULL;

	return isl_set_copy(phase->domain);
}

/* Return the space of the pair of band nodes that form the input
 * to the hybrid tiling of which "phase" is a phase.
 * In particular, return the space [P -> C], where P is the space
 * of the parent node and C is the space of the child node.
 */
static __isl_give isl_space *ppcg_ht_phase_get_input_space(
	__isl_keep ppcg_ht_phase *phase)
{
	if (!phase)
		return NULL;

	return ppcg_ht_tiling_get_input_space(phase->tiling);
}

/* Construct the lower left constraint of the hexagonal tile, i.e.,
 *
 *	du a - b <= (2h+1) du - duh
 *	-du a + b + (2h+1) du - duh >= 0
 *
 * where duh = floor(du * h).
 *
 * This constraint corresponds to (6) in
 * "Hybrid Hexagonal/Classical Tiling for GPUs".
 */
static __isl_give isl_constraint *hex_lower_left(__isl_take isl_local_space *ls,
	__isl_keep isl_val *h, __isl_keep isl_val *du, __isl_keep isl_val *duh)
{
	isl_val *v;
	isl_aff *aff;

	v = isl_val_add_ui(isl_val_mul_ui(isl_val_copy(h), 2), 1);
	v = isl_val_mul(v, isl_val_copy(du));
	v = isl_val_sub(v, isl_val_copy(duh));
	aff = isl_aff_val_on_domain(ls, v);
	v = isl_val_neg(isl_val_copy(du));
	aff = isl_aff_set_coefficient_val(aff, isl_dim_in, 0, v);
	aff = isl_aff_set_coefficient_si(aff, isl_dim_in, 1, 1);

	return isl_inequality_from_aff(aff);
}

/* Construct the lower constraint of the hexagonal tile, i.e.,
 *
 *	a <= 2h+1
 *	-a + 2h+1 >= 0
 *
 * This constraint corresponds to (7) in
 * "Hybrid Hexagonal/Classical Tiling for GPUs".
 */
static __isl_give isl_constraint *hex_lower(__isl_take isl_local_space *ls,
	__isl_keep isl_val *h)
{
	isl_val *v;
	isl_aff *aff;

	v = isl_val_add_ui(isl_val_mul_ui(isl_val_copy(h), 2), 1);
	aff = isl_aff_val_on_domain(ls, v);
	aff = isl_aff_set_coefficient_si(aff, isl_dim_in, 0, -1);

	return isl_inequality_from_aff(aff);
}

/* Construct the lower right constraint of the hexagonal tile, i.e.,
 *
 *	dl a + b <= (2h+1) dl + duh + (s0-1)
 *	-dl a - b + (2h+1) dl + duh + (s0-1) >= 0
 *
 * where duh = floor(du * h).
 *
 * This constraint corresponds to (8) in
 * "Hybrid Hexagonal/Classical Tiling for GPUs".
 */
static __isl_give isl_constraint *hex_lower_right(
	__isl_take isl_local_space *ls, __isl_keep isl_val *h,
	__isl_keep isl_val *s0, __isl_keep isl_val *dl, __isl_keep isl_val *duh)
{
	isl_val *v;
	isl_aff *aff;

	v = isl_val_add_ui(isl_val_mul_ui(isl_val_copy(h), 2), 1);
	v = isl_val_mul(v, isl_val_copy(dl));
	v = isl_val_add(v, isl_val_copy(duh));
	v = isl_val_add(v, isl_val_copy(s0));
	v = isl_val_sub_ui(v, 1);
	aff = isl_aff_val_on_domain(ls, v);
	v = isl_val_neg(isl_val_copy(dl));
	aff = isl_aff_set_coefficient_val(aff, isl_dim_in, 0, v);
	aff = isl_aff_set_coefficient_si(aff, isl_dim_in, 1, -1);

	return isl_inequality_from_aff(aff);
}

/* Construct the upper left constraint of the hexagonal tile, i.e.,
 *
 *	dl a + b >= h dl - (d - 1)/d				with d = den(dl)
 *	dl a + b - h dl + (d - 1)/d >= 0
 *
 * This constraint corresponds to (10) in
 * "Hybrid Hexagonal/Classical Tiling for GPUs".
 */
static __isl_give isl_constraint *hex_upper_left(__isl_take isl_local_space *ls,
	__isl_keep isl_val *h, __isl_keep isl_val *dl)
{
	isl_val *v, *d;
	isl_aff *aff;

	d = isl_val_get_den_val(dl);
	v = isl_val_sub_ui(isl_val_copy(d), 1);
	v = isl_val_div(v, d);
	v = isl_val_sub(v, isl_val_mul(isl_val_copy(h), isl_val_copy(dl)));
	aff = isl_aff_val_on_domain(ls, v);
	aff = isl_aff_set_coefficient_val(aff, isl_dim_in, 0, isl_val_copy(dl));
	aff = isl_aff_set_coefficient_si(aff, isl_dim_in, 1, 1);

	return isl_inequality_from_aff(aff);
}

/* Construct the upper right constraint of the hexagonal tile, i.e.,
 *
 *	du a - b >= du h - duh - (s0-1) - dlh - (d - 1)/d	with d = den(du)
 *	du a - b - du h + duh + (s0-1) + dlh + (d - 1)/d >= 0
 *
 * where dlh = floor(dl * h) and duh = floor(du * h).
 *
 * This constraint corresponds to (12) in
 * "Hybrid Hexagonal/Classical Tiling for GPUs".
 */
static __isl_give isl_constraint *hex_upper_right(
	__isl_take isl_local_space *ls, __isl_keep isl_val *h,
	__isl_keep isl_val *s0, __isl_keep isl_val *du,
	__isl_keep isl_val *dlh, __isl_keep isl_val *duh)
{
	isl_val *v, *d;
	isl_aff *aff;

	d = isl_val_get_den_val(du);
	v = isl_val_sub_ui(isl_val_copy(d), 1);
	v = isl_val_div(v, d);
	v = isl_val_sub(v, isl_val_mul(isl_val_copy(h), isl_val_copy(du)));
	v = isl_val_add(v, isl_val_copy(duh));
	v = isl_val_add(v, isl_val_copy(dlh));
	v = isl_val_add(v, isl_val_copy(s0));
	v = isl_val_sub_ui(v, 1);
	aff = isl_aff_val_on_domain(ls, v);
	aff = isl_aff_set_coefficient_val(aff, isl_dim_in, 0, isl_val_copy(du));
	aff = isl_aff_set_coefficient_si(aff, isl_dim_in, 1, -1);

	return isl_inequality_from_aff(aff);
}

/* Construct the uppper constraint of the hexagonal tile, i.e.,
 *
 *	a >= 0
 *
 * This constraint corresponds to (13) in
 * "Hybrid Hexagonal/Classical Tiling for GPUs".
 */
static __isl_give isl_constraint *hex_upper(__isl_take isl_local_space *ls)
{
	isl_aff *aff;

	aff = isl_aff_var_on_domain(ls, isl_dim_set, 0);

	return isl_inequality_from_aff(aff);
}

/* Construct the basic hexagonal tile shape.
 * "space" is the 2D space in which the hexagon should be constructed.
 * h is st-1, with st the tile size in the time dimension
 * s0 is the tile size in the space dimension
 * dl is a bound on the negative relative dependence distances, i.e.,
 *
 *	d_s >= -dl d_t
 *
 * du is a bound on the positive relative dependence distances, i.e.,
 *
 *	d_s <= du d_t
 *
 * with (d_t,d_s) any dependence distance vector.
 * dlh = floor(dl * h)
 * duh = floor(du * h)
 *
 * The shape of the hexagon is as follows:
 *
 *		0 dlh   dlh+s0-1
 *		   ______                __
 * 0		  /      \_             /
 *		 /         \_          /
 * h		/            \ ______ /
 * h+1		\_           //      \\_
 *		  \_        //         \\_
 * 2h+1		    \______//            \\
 *		0   duh   duh+s0-1
 *		             duh+s0-1+dlh
 *		                  duh+s0-1+dlh+1+s0+1
 *
 * The next hexagon is shifted by duh + dlh + 2 * s0.
 *
 * The slope of the "/" constraints is dl.
 * The slope of the "\_" constraints is du.
 */
static __isl_give isl_set *compute_hexagon(__isl_take isl_space *space,
	__isl_keep isl_val *h, __isl_keep isl_val *s0,
	__isl_keep isl_val *dl, __isl_keep isl_val *du,
	__isl_keep isl_val *dlh, __isl_keep isl_val *duh)
{
	isl_local_space *ls;
	isl_constraint *c;
	isl_basic_set *bset;

	ls = isl_local_space_from_space(space);

	c = hex_lower_left(isl_local_space_copy(ls), h, du, duh);
	bset = isl_basic_set_from_constraint(c);

	c = hex_lower(isl_local_space_copy(ls), h);
	bset = isl_basic_set_add_constraint(bset, c);

	c = hex_lower_right(isl_local_space_copy(ls), h, s0, dl, duh);
	bset = isl_basic_set_add_constraint(bset, c);

	c = hex_upper_left(isl_local_space_copy(ls), h, dl);
	bset = isl_basic_set_add_constraint(bset, c);

	c = hex_upper_right(isl_local_space_copy(ls), h, s0, du, dlh, duh);
	bset = isl_basic_set_add_constraint(bset, c);

	c = hex_upper(ls);
	bset = isl_basic_set_add_constraint(bset, c);

	return isl_set_from_basic_set(bset);
}

/* Name of the ts-space.
 */
static const char *ts_space_name = "ts";

/* Construct and return the space ts[t, s].
 */
static __isl_give isl_space *construct_ts_space(isl_ctx *ctx)
{
	isl_space *s;

	s = isl_space_set_alloc(ctx, 0, 2);
	s = isl_space_set_tuple_name(s, isl_dim_set, ts_space_name);

	return s;
}

/* Name of the local ts-space.
 */
static const char *local_ts_space_name = "local_ts";

/* Construct and return the space local_ts[t, s].
 */
static __isl_give isl_space *construct_local_ts_space(isl_ctx *ctx)
{
	isl_space *s;

	s = isl_space_set_alloc(ctx, 0, 2);
	s = isl_space_set_tuple_name(s, isl_dim_set, local_ts_space_name);

	return s;
}

/* Compute the total size of a tile for the space dimensions,
 * i.e., those corresponding to the child node
 * of the input pattern.
 * If S_0 is the original tile size in the first space dimension,
 * then the first entry of "space_sizes" is equal to
 * W = 2*S_0 + floor(d_l h) + floor(d_u h).
 * The remaining entries are the same as in the original tile sizes.
 * "tile_sizes" contains the original tile sizes, including
 * the tile size corresponding to the parent node.
 * "dlh" is equal to floor(d_l h).
 * "duh" is equal to floor(d_u h).
 */
static __isl_give isl_multi_val *compute_space_sizes(
	__isl_keep isl_multi_val *tile_sizes,
	__isl_keep isl_val *dlh, __isl_keep isl_val *duh)
{
	isl_val *size;
	isl_multi_val *space_sizes;

	space_sizes = isl_multi_val_copy(tile_sizes);
	space_sizes = isl_multi_val_factor_range(space_sizes);
	size = isl_multi_val_get_val(space_sizes, 0);
	size = isl_val_mul_ui(size, 2);
	size = isl_val_add(size, isl_val_copy(duh));
	size = isl_val_add(size, isl_val_copy(dlh));
	space_sizes = isl_multi_val_set_val(space_sizes, 0, size);

	return space_sizes;
}

/* Compute the offset of phase 1 with respect to phase 0
 * in the ts-space ("space").
 * In particular, return
 *
 *	ts[st, s0 + duh]
 */
static __isl_give isl_multi_val *compute_phase_shift(
	__isl_keep isl_space *space, __isl_keep isl_val *st,
	__isl_keep isl_val *s0, __isl_keep isl_val *duh)
{
	isl_val *v;
	isl_multi_val *phase_shift;

	phase_shift = isl_multi_val_zero(isl_space_copy(space));
	phase_shift = isl_multi_val_set_val(phase_shift, 0, isl_val_copy(st));
	v = isl_val_add(isl_val_copy(duh), isl_val_copy(s0));
	phase_shift = isl_multi_val_set_val(phase_shift, 1, v);

	return phase_shift;
}

/* Return the function
 *
 *	ts[t, s] -> floor(t/(2 * st))
 *
 * representing the time tile.
 * "space" is the space ts[t, s].
 */
static __isl_give isl_aff *compute_time_tile(__isl_keep isl_space *space,
	__isl_keep isl_val *st)
{
	isl_val *v;
	isl_aff *t;
	isl_local_space *ls;

	ls = isl_local_space_from_space(isl_space_copy(space));
	t = isl_aff_var_on_domain(ls, isl_dim_set, 0);
	v = isl_val_mul_ui(isl_val_copy(st), 2);
	t = isl_aff_floor(isl_aff_scale_down_val(t, v));

	return t;
}

/* Compute a shift in the space dimension for tiles
 * at time tile T = floor(t/(2 * S_t))
 * such that they align to a multiple of the total space tile dimension W.
 * In particular, compute
 *
 *	ts[t, s] -> s + (-(2 * shift_s)*T) % W
 *
 * where shift_s is the shift of phase 1 with respect to phase 0
 * in the space dimension (the first element of "phase_shift").
 * W is stored in the first element of "space_sizes".
 * "time_tile" is the function
 *
 *	ts[t, s] -> floor(t/(2 * S_T))
 *
 * Since phase 1 is shifted by shift_s with respect to phase 0,
 * the next line of phase 0 (at T+1) is shifted by 2*shift_s
 * with respect to the previous line (at T).
 * A shift of -(2 * shift_s)*T therefore allows the basic pattern
 * (which starts at 0) to be applied.
 * However, this shift will be used to obtain the tile coordinate
 * in the first space dimension and if the original values
 * in the space dimension are non-negative, then the shift should
 * not make them negative.  Moreover, the shift should be as minimal
 * as possible.
 * Since the pattern repeats itself with a period of W in the space
 * dimension, the shift can be replaced by (-(2 * shift_s)*T) % W.
 */
static __isl_give isl_aff *compute_shift_space(__isl_keep isl_aff *time_tile,
	__isl_keep isl_multi_val *space_sizes,
	__isl_keep isl_multi_val *phase_shift)
{
	isl_val *v;
	isl_aff *s, *t;
	isl_local_space *ls;

	ls = isl_local_space_from_space(isl_aff_get_domain_space(time_tile));
	t = isl_aff_copy(time_tile);
	v = isl_val_mul_ui(isl_multi_val_get_val(phase_shift, 1), 2);
	v = isl_val_neg(v);
	t = isl_aff_scale_val(t, v);
	v = isl_multi_val_get_val(space_sizes, 0);
	t = isl_aff_mod_val(t, v);
	s = isl_aff_var_on_domain(ls, isl_dim_set, 1);
	s = isl_aff_add(s, t);

	return s;
}

/* Give the phase_shift ts[S_t, S_0 + floor(d_u h)],
 * compute a function that applies the shift, i.e.,
 *
 *	ts[t, s] -> ts[t + S_t, s + S_0 + floor(d_u h)],
 */
static __isl_give isl_multi_aff *compute_shift_phase(
	__isl_keep isl_multi_val *phase_shift)
{
	isl_space *space;
	isl_multi_aff *shift;

	space = isl_multi_val_get_space(phase_shift);
	shift = isl_multi_aff_multi_val_on_space(space,
					isl_multi_val_copy(phase_shift));
	space = isl_multi_aff_get_space(shift);
	shift = isl_multi_aff_add(shift, isl_multi_aff_identity(space));

	return shift;
}

/* Compute a mapping from the ts-space to the local coordinates
 * within each tile.  In particular, compute
 *
 *	ts[t, s] -> local_ts[t % (2 S_t), (s + (-(2 * shift_s)*T) % W) % W]
 *
 * "ts" is the space ts[t, s]
 * "local_ts" is the space local_ts[t, s]
 * "shift_space" is equal to ts[t, s] -> s + (-(2 * shift_s)*T) % W
 * "st" is the tile size in the time dimension S_t.
 * The first element of "space_sizes" is equal to W.
 */
static __isl_give isl_multi_aff *compute_localize(
	__isl_keep isl_space *local_ts, __isl_keep isl_aff *shift_space,
	__isl_keep isl_val *st, __isl_keep isl_multi_val *space_sizes)
{
	isl_val *v;
	isl_space *space;
	isl_aff *s, *t;
	isl_multi_aff *localize;

	space = isl_aff_get_domain_space(shift_space);
	local_ts = isl_space_copy(local_ts);
	space = isl_space_map_from_domain_and_range(space, local_ts);
	localize = isl_multi_aff_identity(space);
	t = isl_multi_aff_get_aff(localize, 0);
	v = isl_val_mul_ui(isl_val_copy(st), 2);
	t = isl_aff_mod_val(t, v);
	localize = isl_multi_aff_set_aff(localize, 0, t);
	s = isl_aff_copy(shift_space);
	v = isl_multi_val_get_val(space_sizes, 0);
	s = isl_aff_mod_val(s, v);
	localize = isl_multi_aff_set_aff(localize, 1, s);

	return localize;
}

/* Set the project_ts field of "tiling".
 *
 * This field projects the space of the input schedule to the ts-space.
 * It is equal to [P[t] -> C[s_0, ...]] -> ts[t, s_0].
 */
static __isl_give ppcg_ht_tiling *ppcg_ht_tiling_set_project_ts(
	__isl_take ppcg_ht_tiling *tiling)
{
	int n;
	isl_space *space;
	isl_multi_aff *project;

	if (!tiling)
		return NULL;

	space = ppcg_ht_tiling_get_input_space(tiling);
	n = isl_space_dim(space, isl_dim_set);
	project = isl_multi_aff_project_out_map(space, isl_dim_set, 2, n - 2);
	project = isl_multi_aff_set_tuple_name(project,
						isl_dim_out, ts_space_name);
	if (!project)
		return ppcg_ht_tiling_free(tiling);

	tiling->project_ts = project;

	return tiling;
}

/* Construct a hybrid tiling description from bounds on the dependence
 * distances "bounds".
 * "input_node" points to the original parent node.
 * "input_schedule" is the combined schedule of the parent and child
 * node in the input.
 * "tile_sizes" are the original, user specified tile sizes.
 */
static __isl_give ppcg_ht_tiling *ppcg_ht_bounds_construct_tiling(
	__isl_take ppcg_ht_bounds *bounds,
	__isl_keep isl_schedule_node *input_node,
	__isl_keep isl_multi_union_pw_aff *input_schedule,
	__isl_keep isl_multi_val *tile_sizes)
{
	isl_ctx *ctx;
	ppcg_ht_tiling *tiling;
	isl_multi_val *space_sizes, *phase_shift;
	isl_aff *time_tile, *shift_space;
	isl_multi_aff *localize;
	isl_val *h, *duh, *dlh;
	isl_val *st, *s0, *du, *dl;
	isl_space *ts, *local_ts;

	if (!bounds || !input_node || !input_schedule || !tile_sizes)
		goto error;

	ctx = isl_multi_union_pw_aff_get_ctx(input_schedule);
	tiling = isl_calloc_type(ctx, struct ppcg_ht_tiling);
	if (!tiling)
		goto error;
	tiling->ref = 1;

	st = isl_multi_val_get_val(tile_sizes, 0);
	h = isl_val_sub_ui(isl_val_copy(st), 1);
	s0 = isl_multi_val_get_val(tile_sizes, 1);
	du = ppcg_ht_bounds_get_upper(bounds);
	dl = ppcg_ht_bounds_get_lower(bounds, 0);

	duh = isl_val_floor(isl_val_mul(isl_val_copy(du), isl_val_copy(h)));
	dlh = isl_val_floor(isl_val_mul(isl_val_copy(dl), isl_val_copy(h)));

	ts = construct_ts_space(ctx);
	local_ts = construct_local_ts_space(ctx);

	space_sizes = compute_space_sizes(tile_sizes, dlh, duh);
	phase_shift = compute_phase_shift(ts, st, s0, duh);
	time_tile = compute_time_tile(ts, st);
	shift_space = compute_shift_space(time_tile, space_sizes, phase_shift);
	localize = compute_localize(local_ts, shift_space, st, space_sizes);
	isl_space_free(ts);

	tiling->input_node = isl_schedule_node_copy(input_node);
	tiling->input_schedule = isl_multi_union_pw_aff_copy(input_schedule);
	tiling->space_sizes = space_sizes;
	tiling->bounds = bounds;
	tiling->local_time = isl_multi_aff_get_aff(localize, 0);
	tiling->hex = compute_hexagon(local_ts, h, s0, dl, du, dlh, duh);
	tiling->hex = isl_set_preimage_multi_aff(tiling->hex, localize);
	tiling->time_tile = time_tile;
	tiling->shift_space = shift_space;
	tiling->shift_phase = compute_shift_phase(phase_shift);
	isl_multi_val_free(phase_shift);

	isl_val_free(duh);
	isl_val_free(dlh);
	isl_val_free(du);
	isl_val_free(dl);
	isl_val_free(s0);
	isl_val_free(st);
	isl_val_free(h);

	if (!tiling->input_schedule || !tiling->local_time || !tiling->hex ||
	    !tiling->shift_space || !tiling->shift_phase)
		return ppcg_ht_tiling_free(tiling);

	tiling = ppcg_ht_tiling_set_project_ts(tiling);

	return tiling;
error:
	ppcg_ht_bounds_free(bounds);
	return NULL;
}

/* Are all members of the band node "node" coincident?
 */
static isl_bool all_coincident(__isl_keep isl_schedule_node *node)
{
	int i, n;

	n = isl_schedule_node_band_n_member(node);
	for (i = 0; i < n; ++i) {
		isl_bool c;

		c = isl_schedule_node_band_member_get_coincident(node, i);
		if (c < 0 || !c)
			return c;
	}

	return isl_bool_true;
}

/* Does "node" satisfy the properties of the inner node in the input
 * pattern for hybrid tiling?
 * That is, is it a band node with only coincident members, of which
 * there is at least one?
 */
static isl_bool has_child_properties(__isl_keep isl_schedule_node *node)
{
	if (!node)
		return isl_bool_error;
	if (isl_schedule_node_get_type(node) != isl_schedule_node_band)
		return isl_bool_false;
	if (isl_schedule_node_band_n_member(node) < 1)
		return isl_bool_false;
	return all_coincident(node);
}

/* Does "node" satisfy the properties of the outer node in the input
 * pattern for hybrid tiling?
 * That is, is it a band node with a single member?
 */
static isl_bool has_parent_properties(__isl_keep isl_schedule_node *node)
{
	if (!node)
		return isl_bool_error;
	if (isl_schedule_node_get_type(node) != isl_schedule_node_band)
		return isl_bool_false;
	if (isl_schedule_node_band_n_member(node) != 1)
		return isl_bool_false;
	return isl_bool_true;
}

/* Does the parent of "node" satisfy the input patttern for hybrid tiling?
 * That is, does "node" satisfy the properties of the inner node and
 * does the parent of "node" satisfy the properties of the outer node?
 */
isl_bool ppcg_ht_parent_has_input_pattern(__isl_keep isl_schedule_node *node)
{
	isl_bool has_pattern;

	has_pattern = has_child_properties(node);
	if (has_pattern < 0 || !has_pattern)
		return has_pattern;

	node = isl_schedule_node_copy(node);
	node = isl_schedule_node_parent(node);
	has_pattern = has_parent_properties(node);
	isl_schedule_node_free(node);

	return has_pattern;
}

/* Does "node" satisfy the input patttern for hybrid tiling?
 * That is, does "node" satisfy the properties of the outer node and
 * does the child of "node" satisfy the properties of the inner node?
 */
isl_bool ppcg_ht_has_input_pattern(__isl_keep isl_schedule_node *node)
{
	isl_bool has_pattern;

	has_pattern = has_parent_properties(node);
	if (has_pattern < 0 || !has_pattern)
		return has_pattern;

	node = isl_schedule_node_get_child(node, 0);
	has_pattern = has_child_properties(node);
	isl_schedule_node_free(node);

	return has_pattern;
}

/* Check that "node" satisfies the input pattern for hybrid tiling.
 * Error out if it does not.
 */
static isl_stat check_input_pattern(__isl_keep isl_schedule_node *node)
{
	isl_bool has_pattern;

	has_pattern = ppcg_ht_has_input_pattern(node);
	if (has_pattern < 0)
		return isl_stat_error;
	if (!has_pattern)
		isl_die(isl_schedule_node_get_ctx(node), isl_error_invalid,
			"invalid input pattern for hybrid tiling",
			return isl_stat_error);

	return isl_stat_ok;
}

/* Extract the input schedule from "node", i.e., the product
 * of the partial schedules of the parent and child nodes
 * in the input pattern.
 */
static __isl_give isl_multi_union_pw_aff *extract_input_schedule(
	__isl_keep isl_schedule_node *node)
{
	isl_multi_union_pw_aff *partial, *partial2;

	partial = isl_schedule_node_band_get_partial_schedule(node);
	node = isl_schedule_node_get_child(node, 0);
	partial2 = isl_schedule_node_band_get_partial_schedule(node);
	isl_schedule_node_free(node);

	return isl_multi_union_pw_aff_range_product(partial, partial2);
}

/* Collect all dependences from "scop" that are relevant for performing
 * hybrid tiling on "node" and its child and map them to the schedule
 * space of this pair of nodes.
 *
 * In case live range reordering is not used,
 * the flow and the false dependences are collected.
 * In case live range reordering is used,
 * the flow and the forced dependences are collected, as well
 * as the order dependences that are adjacent to non-local
 * flow dependences.
 *
 * In all cases, only dependences that map to the same instance
 * of the outer part of the schedule are considered.
 */
static __isl_give isl_map *collect_deps(struct ppcg_scop *scop,
	__isl_keep isl_schedule_node *node)
{
	isl_space *space;
	isl_multi_union_pw_aff *prefix, *partial;
	isl_union_map *flow, *other, *dep, *umap;
	isl_map *map;

	prefix = isl_schedule_node_get_prefix_schedule_multi_union_pw_aff(node);
	partial = extract_input_schedule(node);
	space = isl_multi_union_pw_aff_get_space(partial);

	flow = isl_union_map_copy(scop->dep_flow);
	flow = isl_union_map_eq_at_multi_union_pw_aff(flow,
					isl_multi_union_pw_aff_copy(prefix));
	if (!scop->options->live_range_reordering) {
		other = isl_union_map_copy(scop->dep_false);
		other = isl_union_map_eq_at_multi_union_pw_aff(other, prefix);
	} else {
		isl_union_map *local, *non_local, *order, *adj;
		isl_union_set *domain, *range;

		other = isl_union_map_copy(scop->dep_forced);
		other = isl_union_map_eq_at_multi_union_pw_aff(other,
					isl_multi_union_pw_aff_copy(prefix));
		local = isl_union_map_copy(flow);
		local = isl_union_map_eq_at_multi_union_pw_aff(local,
					isl_multi_union_pw_aff_copy(partial));
		non_local = isl_union_map_copy(flow);
		non_local = isl_union_map_subtract(non_local, local);

		order = isl_union_map_copy(scop->dep_order);
		order = isl_union_map_eq_at_multi_union_pw_aff(order, prefix);
		adj = isl_union_map_copy(order);
		domain = isl_union_map_domain(isl_union_map_copy(non_local));
		domain = isl_union_set_coalesce(domain);
		adj = isl_union_map_intersect_range(adj, domain);
		other = isl_union_map_union(other, adj);

		adj = order;
		range = isl_union_map_range(non_local);
		range = isl_union_set_coalesce(range);
		adj = isl_union_map_intersect_domain(adj, range);
		other = isl_union_map_union(other, adj);
	}
	dep = isl_union_map_union(flow, other);

	umap = isl_union_map_from_multi_union_pw_aff(partial);
	dep = isl_union_map_apply_domain(dep, isl_union_map_copy(umap));
	dep = isl_union_map_apply_range(dep, umap);

	space = isl_space_map_from_set(space);
	map = isl_union_map_extract_map(dep, space);
	isl_union_map_free(dep);

	map = isl_map_coalesce(map);

	return map;
}

/* Given a constraint of the form
 *
 *	a i_0 + b i_1 >= 0
 * or
 *	a i_0 + b i_1 = 0
 *
 * use it to update one or both of the non-negative bounds
 * in "list" = (min, max) such that
 *
 *	i_1 >= -min i_0
 * and
 *	i_1 <= max i_0
 *
 * If b = 0, then the constraint cannot be used.
 * Otherwise, the constraint is equivalent to
 *
 *	sgn(b) i_1 >= - a/abs(b) i_0
 * i.e.,
 *	i_1 >= - a/abs(b) i_0
 * or
 *	i_1 <= a/abs(b) i_0
 *
 * Set the first or second element of "list" to max(0, a/abs(b)),
 * according to the sign of "b".  Or set both in case the constraint
 * is an equality, taking into account the sign change.
 */
static __isl_give isl_val_list *list_set_min_max(__isl_take isl_val_list *list,
	__isl_keep isl_constraint *c)
{
	isl_val *a, *b;
	int sign;
	int pos;
	isl_bool eq, is_zero, is_neg;

	eq = isl_constraint_is_equality(c);
	if (eq < 0)
		return isl_val_list_free(list);

	b = isl_constraint_get_coefficient_val(c, isl_dim_set, 1);
	is_zero = isl_val_is_zero(b);
	if (is_zero == isl_bool_true) {
		isl_val_free(b);
		return list;
	}
	a = isl_constraint_get_coefficient_val(c, isl_dim_set, 0);
	sign = isl_val_sgn(b);
	b = isl_val_abs(b);
	a = isl_val_div(a, b);

	if (eq)
		b = isl_val_copy(a);

	pos = sign > 0 ? 0 : 1;
	is_neg = isl_val_is_neg(a);
	if (is_neg == isl_bool_true)
		a = isl_val_set_si(a, 0);
	list = isl_val_list_set_val(list, pos, a);

	if (!eq)
		return is_neg < 0 ? isl_val_list_free(list) : list;

	pos = 1 - pos;
	a = isl_val_neg(b);
	is_neg = isl_val_is_neg(a);
	if (is_neg == isl_bool_true)
		a = isl_val_set_si(a, 0);
	list = isl_val_list_set_val(list, pos, a);

	return is_neg < 0 ? isl_val_list_free(list) : list;
}

/* If constraint "c" passes through the origin, then try and use it
 * to update the non-negative bounds in "list" = (min, max) such that
 *
 *	i_1 >= -min i_0
 * and
 *	i_1 <= max i_0
 */
static isl_stat set_min_max(__isl_take isl_constraint *c, void *user)
{
	isl_val *v;
	isl_val_list **list = user;
	isl_bool is_zero;

	v = isl_constraint_get_constant_val(c);
	is_zero = isl_val_is_zero(v);
	isl_val_free(v);

	if (is_zero == isl_bool_true)
		*list = list_set_min_max(*list, c);

	isl_constraint_free(c);
	return is_zero < 0 ? isl_stat_error : isl_stat_ok;
}

/* Given a set of dependence distance vectors "dist", compute
 * pair of non-negative bounds min and max such that
 *
 *	d_pos >= -min d_0
 * and
 *	d_pos <= max d_0
 *
 * and return the pair (min, max).
 * If no bound can be found in either direction, then the bound
 * is replaced by NaN.
 *
 * The dependence distances are first projected onto the (d_0, d_pos).
 * Then the zero dependence distance is added and the convex hull is computed.
 * Finally, the bounds are extracted from the constraints of the convex hull
 * that pass through the origin.
 */
static __isl_give isl_val_list *min_max_dist(__isl_keep isl_set *dist, int pos)
{
	isl_space *space;
	isl_basic_set *hull;
	int dim;
	isl_ctx *ctx;
	isl_val *nan;
	isl_val_list *list;

	ctx = isl_set_get_ctx(dist);
	nan = isl_val_nan(ctx);
	list = isl_val_list_alloc(ctx, 2);
	list = isl_val_list_add(list, isl_val_copy(nan));
	list = isl_val_list_add(list, nan);

	dist = isl_set_copy(dist);
	dim = isl_set_dim(dist, isl_dim_set);
	if (dist && pos >= dim)
		isl_die(ctx, isl_error_internal, "position out of bounds",
			dist = isl_set_free(dist));
	dist = isl_set_project_out(dist, isl_dim_set, pos + 1, dim - (pos + 1));
	dist = isl_set_project_out(dist, isl_dim_set, 1, pos - 1);

	space = isl_set_get_space(dist);
	dist = isl_set_union(dist, isl_set_from_point(isl_point_zero(space)));
	dist = isl_set_remove_divs(dist);
	hull = isl_set_convex_hull(dist);

	if (isl_basic_set_foreach_constraint(hull, &set_min_max, &list) < 0)
		list = isl_val_list_free(list);
	isl_basic_set_free(hull);

	return list;
}

/* Given a schedule node "node" that, together with its child,
 * satisfies the input pattern for hybrid tiling, compute bounds
 * on the relative dependence distances of the child node with
 * respect to the parent node.  These bounds are needed to
 * construct a hybrid tiling.
 *
 * First all relevant dependences are collected and mapped
 * to the schedule space of the pair of nodes.  Then, the
 * dependence distances are computed in this space.
 *
 * These dependence distances are then projected onto a two-dimensional
 * space consisting of the single schedule dimension of the outer node
 * and one of the schedule dimensions of the inner node.
 * The maximal and minimal relative dependence distances are extracted
 * from these projections.
 * This process is repeated for each of the schedule dimensions
 * of the inner node.  For the first dimension, both minimal and
 * maximal relative dependence distances are stored in the result.
 * For the other dimensions, only the minimal relative dependence
 * distance is stored.
 */
__isl_give ppcg_ht_bounds *ppcg_ht_compute_bounds(struct ppcg_scop *scop,
	__isl_keep isl_schedule_node *node)
{
	ppcg_ht_bounds *bnd;
	isl_space *space;
	isl_map *map;
	isl_set *dist;
	isl_val_list *pair;
	isl_schedule_node *child;
	int n;
	int i, dim;

	if (!scop || !node || check_input_pattern(node) < 0)
		return NULL;

	child = isl_schedule_node_get_child(node, 0);
	space = isl_schedule_node_band_get_space(child);
	dim = isl_schedule_node_band_n_member(child);
	isl_schedule_node_free(child);
	bnd = ppcg_ht_bounds_alloc(space);
	if (!bnd)
		return NULL;

	map = collect_deps(scop, node);

	dist = isl_map_deltas(map);
	n = isl_set_dim(dist, isl_dim_param);
	dist = isl_set_project_out(dist, isl_dim_param, 0, n);

	pair = min_max_dist(dist, 1);
	bnd = ppcg_ht_bounds_set_lower(bnd, 0, isl_val_list_get_val(pair, 0));
	bnd = ppcg_ht_bounds_set_upper(bnd, isl_val_list_get_val(pair, 1));
	isl_val_list_free(pair);

	for (i = 1; i < dim; ++i) {
		pair = min_max_dist(dist, 1 + i);
		bnd = ppcg_ht_bounds_set_lower(bnd, i,
						isl_val_list_get_val(pair, 0));
		isl_val_list_free(pair);
	}

	isl_set_free(dist);

	return bnd;
}

/* Check if all the fields of "phase" are valid, freeing "phase"
 * if they are not.
 */
static __isl_give ppcg_ht_phase *check_phase(__isl_take ppcg_ht_phase *phase)
{
	if (!phase)
		return NULL;

	if (!phase->tiling || !phase->local_time ||
	    !phase->shift_space || !phase->domain)
		return ppcg_ht_phase_free(phase);

	return phase;
}

/* Construct a ppcg_ht_phase object, that simply copies
 * information from "tiling".
 * That is, the result is defined over the "ts" space and
 * corresponds to phase 1.
 */
static __isl_give ppcg_ht_phase *construct_phase(
	__isl_keep ppcg_ht_tiling *tiling)
{
	isl_ctx *ctx;
	ppcg_ht_phase *phase;

	if (!tiling)
		return NULL;

	ctx = ppcg_ht_tiling_get_ctx(tiling);
	phase = isl_calloc_type(ctx, struct ppcg_ht_phase);
	if (!phase)
		return NULL;
	phase->tiling = ppcg_ht_tiling_copy(tiling);
	phase->time_tile = isl_aff_copy(tiling->time_tile);
	phase->local_time = isl_aff_copy(tiling->local_time);
	phase->shift_space = isl_aff_copy(tiling->shift_space);
	phase->domain = isl_set_copy(tiling->hex);

	return check_phase(phase);
}

/* Align the parameters of the elements of "phase" to those of "space".
 */
static __isl_give ppcg_ht_phase *phase_align_params(
	__isl_take ppcg_ht_phase *phase, __isl_take isl_space *space)
{
	if (!phase)
		goto error;

	phase->time_tile = isl_aff_align_params(phase->time_tile,
							isl_space_copy(space));
	phase->local_time = isl_aff_align_params(phase->local_time,
							isl_space_copy(space));
	phase->shift_space = isl_aff_align_params(phase->shift_space,
							isl_space_copy(space));
	phase->domain = isl_set_align_params(phase->domain, space);

	return check_phase(phase);
error:
	isl_space_free(space);
	return NULL;
}

/* Pull back "phase" over "ma".
 * That is, take a phase defined over the range of "ma" and
 * turn it into a phase defined over the domain of "ma".
 */
static __isl_give ppcg_ht_phase *pullback_phase(__isl_take ppcg_ht_phase *phase,
	__isl_take isl_multi_aff *ma)
{
	phase = phase_align_params(phase, isl_multi_aff_get_space(ma));
	if (!phase)
		goto error;

	phase->time_tile = isl_aff_pullback_multi_aff(phase->time_tile,
							isl_multi_aff_copy(ma));
	phase->local_time = isl_aff_pullback_multi_aff(phase->local_time,
							isl_multi_aff_copy(ma));
	phase->shift_space = isl_aff_pullback_multi_aff(phase->shift_space,
							isl_multi_aff_copy(ma));
	phase->domain = isl_set_preimage_multi_aff(phase->domain, ma);

	return check_phase(phase);
error:
	isl_multi_aff_free(ma);
	return NULL;
}

/* Pullback "phase" over phase->tiling->shift_phase, which shifts
 * phase 0 to phase 1.  The pullback therefore takes a phase 1
 * description and turns it into a phase 0 description.
 */
static __isl_give ppcg_ht_phase *shift_phase(__isl_take ppcg_ht_phase *phase)
{
	ppcg_ht_tiling *tiling;

	if (!phase)
		return NULL;

	tiling = phase->tiling;
	return pullback_phase(phase, isl_multi_aff_copy(tiling->shift_phase));
}

/* Take a "phase" defined over the ts-space and plug in the projection
 * from the input schedule space to the ts-space.
 * The result is then defined over this input schedule space.
 */
static __isl_give ppcg_ht_phase *lift_phase(__isl_take ppcg_ht_phase *phase)
{
	ppcg_ht_tiling *tiling;

	if (!phase)
		return NULL;

	tiling = phase->tiling;
	return pullback_phase(phase, isl_multi_aff_copy(tiling->project_ts));
}

/* Compute the shift that should be added to the space band
 * in order to be able to apply rectangular tiling to the space.
 * Store the shift in phase->space_shift.
 *
 * In the first dimension, it is equal to shift_space - s.
 * For phase 1, this results in
 *
 *	(-(2 * shift_s)*T) % W
 *
 * In phase 0, the "s" in shift_space has been replaced by "s + shift_s",
 * so the result is
 *
 *	shift_s + (-(2 * shift_s)*T) % W
 *
 * In the other dimensions, the shift is equal to
 *
 *	dl_i * local_time.
 */
static __isl_give ppcg_ht_phase *compute_space_shift(
	__isl_take ppcg_ht_phase *phase)
{
	int i, n;
	isl_space *space;
	isl_local_space *ls;
	isl_aff *aff, *s;
	isl_multi_aff *space_shift;

	if (!phase)
		return NULL;

	space = ppcg_ht_phase_get_input_space(phase);
	space = isl_space_unwrap(space);
	space = isl_space_range_map(space);

	space_shift = isl_multi_aff_zero(space);
	aff = isl_aff_copy(phase->shift_space);
	ls = isl_local_space_from_space(isl_aff_get_domain_space(aff));
	s = isl_aff_var_on_domain(ls, isl_dim_set, 1);
	aff = isl_aff_sub(aff, s);
	space_shift = isl_multi_aff_set_aff(space_shift, 0, aff);

	n = isl_multi_aff_dim(space_shift, isl_dim_out);
	for (i = 1; i < n; ++i) {
		isl_val *v;
		isl_aff *time;

		v = ppcg_ht_bounds_get_lower(phase->tiling->bounds, i);
		time = isl_aff_copy(phase->local_time);
		time = isl_aff_scale_val(time, v);
		space_shift = isl_multi_aff_set_aff(space_shift, i, time);
	}

	if (!space_shift)
		return ppcg_ht_phase_free(phase);
	phase->space_shift = space_shift;
	return phase;
}

/* Compute the space tiling and store the result in phase->space_tile.
 * The space tiling is of the form
 *
 *	[P[t] -> C[s]] -> C[floor((s + space_shift)/space_size]
 */
static __isl_give ppcg_ht_phase *compute_space_tile(
	__isl_take ppcg_ht_phase *phase)
{
	isl_space *space;
	isl_multi_val *space_sizes;
	isl_multi_aff *space_shift;
	isl_multi_aff *tile;

	if (!phase)
		return NULL;

	space = ppcg_ht_phase_get_input_space(phase);
	space = isl_space_unwrap(space);
	tile = isl_multi_aff_range_map(space);
	space_shift = isl_multi_aff_copy(phase->space_shift);
	tile = isl_multi_aff_add(space_shift, tile);
	space_sizes = isl_multi_val_copy(phase->tiling->space_sizes);
	tile = isl_multi_aff_scale_down_multi_val(tile, space_sizes);
	tile = isl_multi_aff_floor(tile);

	if (!tile)
		return ppcg_ht_phase_free(phase);
	phase->space_tile = tile;
	return phase;
}

/* Construct a representation for one of the two phase for hybrid tiling
 * "tiling".  If "shift" is not set, then the phase is constructed
 * directly from the hexagonal tile shape in "tiling", which represents
 * the phase-1 tiles.  If "shift" is set, then this tile shape is shifted
 * back over tiling->shift_phase to obtain the phase-0 tiles.
 *
 * First copy data from "tiling", then optionally shift the phase and
 * finally move the tiling from the "ts" space of "tiling" to
 * the space of the input pattern.
 *
 * After the basic phase has been computed, also compute
 * the corresponding space shift.
 */
static __isl_give ppcg_ht_phase *ppcg_ht_tiling_compute_phase(
	__isl_keep ppcg_ht_tiling *tiling, int shift)
{
	ppcg_ht_phase *phase;

	phase = construct_phase(tiling);
	if (shift)
		phase = shift_phase(phase);
	phase = lift_phase(phase);

	phase = compute_space_shift(phase);
	phase = compute_space_tile(phase);

	return phase;
}

/* Consruct a function that is equal to the time tile of "phase0"
 * on the domain of "phase0" and equal to the time tile of "phase1"
 * on the domain of "phase1".
 * The two domains are assumed to form a partition of the input
 * schedule space.
 */
static __isl_give isl_pw_multi_aff *combine_time_tile(
	__isl_keep ppcg_ht_phase *phase0, __isl_keep ppcg_ht_phase *phase1)
{
	isl_aff *T;
	isl_pw_aff *time, *time1;

	if (!phase0 || !phase1)
		return NULL;

	T = isl_aff_copy(phase0->time_tile);
	time = isl_pw_aff_alloc(ppcg_ht_phase_get_domain(phase0), T);

	T = isl_aff_copy(phase1->time_tile);
	time1 = isl_pw_aff_alloc(ppcg_ht_phase_get_domain(phase1), T);

	time = isl_pw_aff_union_add(time, time1);

	return isl_pw_multi_aff_from_pw_aff(time);
}

/* Name used in mark nodes that contain a pointer to a ppcg_ht_phase.
 */
static char *ppcg_phase_name = "phase";

/* Does "id" contain a pointer to a ppcg_ht_phase?
 * That is, is it called "phase"?
 */
static isl_bool is_phase_id(__isl_keep isl_id *id)
{
	const char *name;

	name = isl_id_get_name(id);
	if (!name)
		return isl_bool_error;

	return !strcmp(name, ppcg_phase_name);
}

/* Given a mark node with an identifier that points to a ppcg_ht_phase,
 * extract this ppcg_ht_phase pointer.
 */
__isl_keep ppcg_ht_phase *ppcg_ht_phase_extract_from_mark(
	__isl_keep isl_schedule_node *node)
{
	isl_bool is_phase;
	isl_id *id;
	void *p;

	if (!node)
		return NULL;
	if (isl_schedule_node_get_type(node) != isl_schedule_node_mark)
		isl_die(isl_schedule_node_get_ctx(node), isl_error_internal,
			"not a phase mark", return NULL);

	id = isl_schedule_node_mark_get_id(node);
	is_phase = is_phase_id(id);
	p = isl_id_get_user(id);
	isl_id_free(id);

	if (is_phase < 0)
		return NULL;
	if (!is_phase)
		isl_die(isl_schedule_node_get_ctx(node), isl_error_internal,
			"not a phase mark", return NULL);

	return p;
}

/* Insert a mark node at "node" holding a pointer to "phase".
 */
static __isl_give isl_schedule_node *insert_phase(
	__isl_take isl_schedule_node *node, __isl_take ppcg_ht_phase *phase)
{
	isl_ctx *ctx;
	isl_id *id;

	if (!node)
		goto error;
	ctx = isl_schedule_node_get_ctx(node);
	id = isl_id_alloc(ctx, ppcg_phase_name, phase);
	if (!id)
		goto error;
	id = isl_id_set_free_user(id, &ppcg_ht_phase_free_wrap);
	node = isl_schedule_node_insert_mark(node, id);

	return node;
error:
	ppcg_ht_phase_free(phase);
	isl_schedule_node_free(node);
	return NULL;
}

/* Construct a mapping from the elements of the original pair of bands
 * to which tiling was applied that belong to a tile of "phase"
 * to that tile, preserving the values for the outer bands.
 *
 * The mapping is of the form
 *
 *	[[outer] -> [P -> C]] -> [[outer] -> [tile]]
 *
 * where tile is defined by a concatenation of the time_tile and
 * the space_tile.
 */
static __isl_give isl_map *construct_tile_map(__isl_keep ppcg_ht_phase *phase)
{
	int depth;
	isl_space *space;
	isl_multi_aff *ma;
	isl_multi_aff *tiling;
	isl_map *el2tile;

	depth = isl_schedule_node_get_schedule_depth(
						phase->tiling->input_node);
	space = isl_aff_get_space(phase->time_tile);
	space = isl_space_params(space);
	space = isl_space_set_from_params(space);
	space = isl_space_add_dims(space, isl_dim_set, depth);
	space = isl_space_map_from_set(space);
	ma = isl_multi_aff_identity(space);

	tiling = isl_multi_aff_flat_range_product(
		isl_multi_aff_from_aff(isl_aff_copy(phase->time_tile)),
		isl_multi_aff_copy(phase->space_tile));
	el2tile = isl_map_from_multi_aff(tiling);
	el2tile = isl_map_intersect_domain(el2tile,
						isl_set_copy(phase->domain));
	el2tile = isl_map_product(isl_map_from_multi_aff(ma), el2tile);

	return el2tile;
}

/* Return a description of the full tiles of "phase" at the point
 * in the original schedule tree where the tiling was applied.
 *
 * First construct a mapping from the input schedule dimensions
 * up to an including the original pair of bands to which hybrid tiling
 * was applied to schedule dimensions in which this original pair
 * has been replaced by the tiles.
 * This mapping is of the form
 *
 *	[[outer] -> [P -> C]] -> [[outer] -> [tile]]
 *
 * Apply this mapping to the set of all values for the input
 * schedule dimensions and then apply its inverse.
 * The result is the set of values for the input schedule dimensions
 * that would map to any of the tiles.  Subtracting from this set
 * the set of values that are actually executed produces the set
 * of values that belong to a tile but that are not executed.
 * Mapping these back to the tiles produces a description of
 * the partial tiles.  Subtracting these from the set of all tiles
 * produces a description of the full tiles in the form
 *
 *	[[outer] -> [tile]]
 */
static __isl_give isl_set *compute_full_tile(__isl_keep ppcg_ht_phase *phase)
{
	isl_schedule_node *node;
	isl_union_set *domain;
	isl_union_map *prefix, *schedule;
	isl_set *all, *partial, *all_el;
	isl_map *tile2el, *el2tile;
	isl_multi_union_pw_aff *mupa;

	el2tile = construct_tile_map(phase);
	tile2el = isl_map_reverse(isl_map_copy(el2tile));

	node = phase->tiling->input_node;
	prefix = isl_schedule_node_get_prefix_schedule_union_map(node);
	domain = isl_schedule_node_get_domain(node);
	mupa = isl_multi_union_pw_aff_copy(phase->tiling->input_schedule);
	schedule = isl_union_map_from_multi_union_pw_aff(mupa);
	schedule = isl_union_map_range_product(prefix, schedule);
	all_el = isl_set_from_union_set(isl_union_set_apply(domain, schedule));
	all_el = isl_set_coalesce(all_el);

	all = isl_set_apply(isl_set_copy(all_el), isl_map_copy(el2tile));

	partial = isl_set_copy(all);
	partial = isl_set_apply(partial, tile2el);
	partial = isl_set_subtract(partial, all_el);
	partial = isl_set_apply(partial, el2tile);

	return isl_set_subtract(all, partial);
}

/* Copy the AST loop types of the non-isolated part to those
 * of the isolated part.
 */
static __isl_give isl_schedule_node *set_isolate_loop_type(
	__isl_take isl_schedule_node *node)
{
	int i, n;

	n = isl_schedule_node_band_n_member(node);
	for (i = 0; i < n; ++i) {
		enum isl_ast_loop_type type;

		type = isl_schedule_node_band_member_get_ast_loop_type(node, i);
		node = isl_schedule_node_band_member_set_isolate_ast_loop_type(
								node, i, type);
	}

	return node;
}

/* If options->isolate_full_tiles is set, then mark the full tiles
 * in "node" for isolation.  The full tiles are derived from "phase".
 * "node" may point to a part of the tiling, e.g., the space tiling.
 *
 * The full tiles are originally computed in the form
 *
 *	[[outer] -> [tile]]
 *
 * However, the band that "node" points to may only contain
 * subset of the tile dimensions.
 * The description above is therefore treated as
 *
 *	[[outer] -> [before; this; after]]
 *
 * before is of size "pos"; this is of size "dim"; and
 * after is of size "out - pos - dim".
 * The after part is first project out.  Then the range is split
 * into a before and this part and finally the before part is moved
 * to the domain, resulting in
 *
 *	[[outer; before] -> [this]]
 *
 * This description is then used as the isolate option.
 *
 * The AST loop type for the isolated part is set to be the same
 * as that of the non-isolated part.
 */
static __isl_give isl_schedule_node *ppcg_ht_phase_isolate_full_tile_node(
	__isl_keep ppcg_ht_phase *phase, __isl_take isl_schedule_node *node,
	struct ppcg_options *options)
{
	int in, out, pos, depth, dim;
	isl_space *space;
	isl_multi_aff *ma1, *ma2;
	isl_set *tile;
	isl_map *map;
	isl_set *set;
	isl_union_set *opt;

	if (!options->isolate_full_tiles)
		return node;

	depth = isl_schedule_node_get_schedule_depth(node);
	dim = isl_schedule_node_band_n_member(node);

	tile = compute_full_tile(phase);
	map = isl_set_unwrap(tile);
	in = isl_map_dim(map, isl_dim_in);
	out = isl_map_dim(map, isl_dim_out);
	pos = depth - in;
	map = isl_map_project_out(map, isl_dim_out, pos + dim,
				out - (pos + dim));
	space = isl_space_range(isl_map_get_space(map));
	ma1 = isl_multi_aff_project_out_map(isl_space_copy(space),
					   isl_dim_set, pos, dim);
	ma2 = isl_multi_aff_project_out_map(space, isl_dim_set, 0, pos);
	ma1 = isl_multi_aff_range_product(ma1, ma2);
	map = isl_map_apply_range(map, isl_map_from_multi_aff(ma1));
	map = isl_map_uncurry(map);
	map = isl_map_flatten_domain(map);
	set = isl_map_wrap(map);
	set = isl_set_set_tuple_name(set, "isolate");

	opt = isl_schedule_node_band_get_ast_build_options(node);
	opt = isl_union_set_add_set(opt, set);
	node = isl_schedule_node_band_set_ast_build_options(node, opt);
	node = set_isolate_loop_type(node);

	return node;
}

/* Insert a band node for performing the space tiling for "phase" at "node".
 * In particular, insert a band node with partial schedule
 *
 *	[P[t] -> C[s]] -> C[floor((s + space_shift)/space_size)]
 *
 * pulled back over the input schedule.
 * "options" determines whether full tiles should be separated
 * from partial tiles.
 *
 * The first tile dimension iterates over the hexagons in the same
 * phase, which are independent by construction.  The first dimension
 * is therefore marked coincident.
 * All dimensions are also marked for being generated as atomic loops
 * because separation is usually not desirable on tile loops.
 */
static __isl_give isl_schedule_node *insert_space_tiling(
	__isl_keep ppcg_ht_phase *phase, __isl_take isl_schedule_node *node,
	struct ppcg_options *options)
{
	isl_multi_aff *space_tile;
	isl_multi_union_pw_aff *mupa;

	if (!phase)
		return isl_schedule_node_free(node);

	space_tile = isl_multi_aff_copy(phase->space_tile);
	mupa = isl_multi_union_pw_aff_copy(phase->tiling->input_schedule);
	mupa = isl_multi_union_pw_aff_apply_multi_aff(mupa, space_tile);
	node = isl_schedule_node_insert_partial_schedule(node, mupa);
	node = ppcg_set_schedule_node_type(node, isl_ast_loop_atomic);
	node = ppcg_ht_phase_isolate_full_tile_node(phase, node, options);
	node = isl_schedule_node_band_member_set_coincident(node, 0, 1);

	return node;
}

/* Given a pointer "node" to (a copy of) the original child node
 * in the input pattern, adjust its partial schedule such that
 * it starts at zero within each tile.
 *
 * That is, replace "s" by (s + space_shift) % space_sizes.
 */
__isl_give isl_schedule_node *ppcg_ht_phase_shift_space_point(
	__isl_keep ppcg_ht_phase *phase, __isl_take isl_schedule_node *node)
{
	isl_multi_val *space_sizes;
	isl_multi_aff *space_shift;
	isl_multi_union_pw_aff *mupa;

	space_shift = isl_multi_aff_copy(phase->space_shift);
	mupa = isl_multi_union_pw_aff_copy(phase->tiling->input_schedule);
	mupa = isl_multi_union_pw_aff_apply_multi_aff(mupa, space_shift);
	node = isl_schedule_node_band_shift(node, mupa);
	space_sizes = isl_multi_val_copy(phase->tiling->space_sizes);
	node = isl_schedule_node_band_mod(node, space_sizes);

	return node;
}

/* Does
 *
 *	s0 > delta + 2 * {delta * h} - 1
 *
 * hold?
 */
static isl_bool wide_enough(__isl_keep isl_val *s0, __isl_keep isl_val *delta,
	__isl_keep isl_val *h)
{
	isl_val *v, *v2;
	isl_bool ok;

	v = isl_val_mul(isl_val_copy(delta), isl_val_copy(h));
	v2 = isl_val_floor(isl_val_copy(v));
	v = isl_val_sub(v, v2);
	v = isl_val_mul_ui(v, 2);
	v = isl_val_add(v, isl_val_copy(delta));
	v = isl_val_sub_ui(v, 1);
	ok = isl_val_gt(s0, v);
	isl_val_free(v);

	return ok;
}

/* Is the tile size specified by "sizes" wide enough in the first space
 * dimension, i.e., the base of the hexagon?  This ensures that,
 * after hybrid tiling using "bounds" and these sizes,
 * neighboring hexagons in the same phase are far enough apart
 * that they do not depend on each other.
 * The test is only meaningful if the bounds are valid.
 *
 * Let st be (half) the size in the time dimension and s0 the base
 * size in the first space dimension.  Let delta be the dependence
 * distance in either positive or negative direction.  In principle,
 * it should be enough to have s0 + 1 > delta, i.e., s0 >= delta.
 * However, in case of fractional delta, the tile is not extended
 * with delta * (st - 1), but instead with floor(delta * (st - 1)).
 * The condition therefore needs to be adjusted to
 *
 *	s0 + 1 > delta + 2 {delta * (st - 1)}
 *
 * (with {} the fractional part) to account for the two slanted sides.
 * The condition in the paper "Hybrid Hexagonal/Classical Tiling for GPUs"
 * translates to
 *
 *	s0 >= delta + {delta * (st - 1)}
 *
 * Since 1 > frac(delta * (st - 1)), this condition implies
 * the condition above.
 *
 * The condition is checked for both directions.
 */
isl_bool ppcg_ht_bounds_supports_sizes(__isl_keep ppcg_ht_bounds *bounds,
	__isl_keep isl_multi_val *sizes)
{
	isl_val *s0, *h;
	isl_val *delta;
	isl_bool ok;

	ok = ppcg_ht_bounds_is_valid(bounds);
	if (ok < 0 || !ok)
		return ok;

	h = isl_val_sub_ui(isl_multi_val_get_val(sizes, 0), 1);
	s0 = isl_multi_val_get_val(sizes, 1);

	delta = ppcg_ht_bounds_get_lower(bounds, 0);
	ok = wide_enough(s0, delta, h);
	isl_val_free(delta);

	delta = ppcg_ht_bounds_get_upper(bounds);
	if (ok == isl_bool_true)
		ok = wide_enough(s0, delta, h);
	isl_val_free(delta);

	isl_val_free(s0);
	isl_val_free(h);

	return ok;
}

/* Check that the tile will be wide enough in the first space
 * dimension, i.e., the base of the hexagon.  This ensures that
 * neighboring hexagons in the same phase are far enough apart
 * that they do not depend on each other.
 *
 * Error out if the condition fails to hold.
 */
static isl_stat check_width(__isl_keep ppcg_ht_bounds *bounds,
	__isl_keep isl_multi_val *sizes)
{
	isl_bool ok;

	ok = ppcg_ht_bounds_supports_sizes(bounds, sizes);

	if (ok < 0)
		return isl_stat_error;
	if (!ok)
		isl_die(isl_multi_val_get_ctx(sizes), isl_error_invalid,
			"base of hybrid tiling hexagon not sufficiently wide",
			return isl_stat_error);

	return isl_stat_ok;
}

/* Given valid bounds on the relative dependence distances for
 * the pair of nested nodes that "node" point to, as well as sufficiently
 * wide tile sizes "sizes", insert the corresponding time and space tiling
 * at "node", along with a pair of phase nodes that can be used
 * to make further changes.
 * The space of "sizes" should be the product of the spaces
 * of the schedules of the pair of parent and child nodes.
 * "options" determines whether full tiles should be separated
 * from partial tiles.
 *
 * In particular, given an input of the form
 *
 *	P - C - ...
 *
 * the output has the form
 *
 *	        /- F0 - M0 - CT0 - P - C - ...
 *	PT - seq
 *	        \- F1 - M1 - CT1 - P - C - ...
 *
 * PT is the global time tiling.  Within each of these tiles,
 * two phases are executed in order.  Within each phase, the schedule
 * space is further subdivided into tiles through CT0 and CT1.
 * The first dimension of each of these iterates over the hexagons
 * within a phase and these are independent by construction.
 * The F0 and F1 filters filter the statement instances that belong
 * to the corresponding phase.  The M0 and M1 marks contain a pointer
 * to a ppcg_ht_phase object that can be used to perform further changes.
 *
 * After checking that input satisfies the requirements,
 * a data structure is constructed that represents the tiling and
 * two additional data structures are constructed for the two phases
 * of the tiling.  These are then used to define the filters F0 and F1 and
 * combined to construct the time tiling PT.
 * Then the time tiling node PT is inserted, followed by
 * the sequence with the two filters, the CT space tiling nodes and
 * the phase markers M.
 */
__isl_give isl_schedule_node *ppcg_ht_bounds_insert_tiling(
	__isl_take ppcg_ht_bounds *bounds, __isl_take isl_multi_val *sizes,
	__isl_take isl_schedule_node *node, struct ppcg_options *options)
{
	isl_ctx *ctx;
	isl_union_set *phase0;
	isl_union_set *phase1;
	isl_multi_union_pw_aff *input, *dom_time;
	isl_union_pw_multi_aff *upma;
	isl_pw_multi_aff *time;
	isl_union_set_list *phases;
	ppcg_ht_tiling *tiling;
	ppcg_ht_phase *phase_0;
	ppcg_ht_phase *phase_1;

	if (!node || !sizes || !bounds)
		goto error;
	if (check_input_pattern(node) < 0 || check_width(bounds, sizes) < 0)
		goto error;

	ctx = isl_schedule_node_get_ctx(node);

	input = extract_input_schedule(node);

	tiling = ppcg_ht_bounds_construct_tiling(bounds, node, input, sizes);
	phase_0 = ppcg_ht_tiling_compute_phase(tiling, 1);
	phase_1 = ppcg_ht_tiling_compute_phase(tiling, 0);
	time = combine_time_tile(phase_0, phase_1);
	ppcg_ht_tiling_free(tiling);

	upma = isl_union_pw_multi_aff_from_multi_union_pw_aff(
					isl_multi_union_pw_aff_copy(input));
	phase0 = isl_union_set_from_set(ppcg_ht_phase_get_domain(phase_0));
	phase0 = isl_union_set_preimage_union_pw_multi_aff(phase0,
					isl_union_pw_multi_aff_copy(upma));
	phase1 = isl_union_set_from_set(ppcg_ht_phase_get_domain(phase_1));
	phase1 = isl_union_set_preimage_union_pw_multi_aff(phase1, upma);

	phases = isl_union_set_list_alloc(ctx, 2);
	phases = isl_union_set_list_add(phases, phase0);
	phases = isl_union_set_list_add(phases, phase1);

	dom_time = isl_multi_union_pw_aff_apply_pw_multi_aff(input, time);
	node = isl_schedule_node_insert_partial_schedule(node, dom_time);

	node = isl_schedule_node_child(node, 0);

	node = isl_schedule_node_insert_sequence(node, phases);
	node = isl_schedule_node_child(node, 0);
	node = isl_schedule_node_child(node, 0);
	node = insert_space_tiling(phase_0, node, options);
	node = insert_phase(node, phase_0);
	node = isl_schedule_node_parent(node);
	node = isl_schedule_node_next_sibling(node);
	node = isl_schedule_node_child(node, 0);
	node = insert_space_tiling(phase_1, node, options);
	node = insert_phase(node, phase_1);
	node = isl_schedule_node_parent(node);
	node = isl_schedule_node_parent(node);

	node = isl_schedule_node_parent(node);

	isl_multi_val_free(sizes);
	return node;
error:
	isl_multi_val_free(sizes);
	isl_schedule_node_free(node);
	ppcg_ht_bounds_free(bounds);
	return NULL;
}

/* Given a branch "node" that contains a sequence node with two phases
 * of hybrid tiling as input, call "fn" on each of the two phase marker
 * nodes.
 *
 * That is, the input is as follows
 *
 *	         /- F0 - M0 - ...
 *	... - seq
 *	         \- F1 - M1 - ...
 *
 * and "fn" is called on M0 and on M1.
 */
__isl_give isl_schedule_node *hybrid_tile_foreach_phase(
	__isl_take isl_schedule_node *node,
	__isl_give isl_schedule_node *(*fn)(__isl_take isl_schedule_node *node,
		void *user), void *user)
{
	int depth0, depth;

	depth0 = isl_schedule_node_get_tree_depth(node);

	while (node &&
	    isl_schedule_node_get_type(node) != isl_schedule_node_sequence)
		node = isl_schedule_node_child(node, 0);

	node = isl_schedule_node_child(node, 0);
	node = isl_schedule_node_child(node, 0);
	if (!node)
		return NULL;
	node = fn(node, user);
	node = isl_schedule_node_parent(node);
	node = isl_schedule_node_next_sibling(node);
	node = isl_schedule_node_child(node, 0);
	if (!node)
		return NULL;
	node = fn(node, user);
	node = isl_schedule_node_parent(node);
	node = isl_schedule_node_parent(node);

	depth = isl_schedule_node_get_tree_depth(node);
	node = isl_schedule_node_ancestor(node, depth - depth0);

	return node;
}

/* This function is called on each of the two phase marks
 * in a hybrid tiling tree.
 * Drop the phase mark at "node".
 */
static __isl_give isl_schedule_node *drop_phase_mark(
	__isl_take isl_schedule_node *node, void *user)
{
	isl_id *id;
	isl_bool is_phase;

	if (isl_schedule_node_get_type(node) != isl_schedule_node_mark)
		return node;

	id = isl_schedule_node_mark_get_id(node);
	is_phase = is_phase_id(id);
	isl_id_free(id);

	if (is_phase < 0)
		return isl_schedule_node_free(node);
	if (is_phase)
		node = isl_schedule_node_delete(node);

	return node;
}

/* Given a branch "node" that contains a sequence node with two phases
 * of hybrid tiling as input, remove the two phase marker nodes.
 *
 * That is, the input is as follows
 *
 *	         /- F0 - M0 - ...
 *	... - seq
 *	         \- F1 - M1 - ...
 *
 * and the output is
 *
 *	         /- F0 - ...
 *	... - seq
 *	         \- F1 - ...
 */
__isl_give isl_schedule_node *hybrid_tile_drop_phase_marks(
	__isl_take isl_schedule_node *node)
{
	return hybrid_tile_foreach_phase(node, &drop_phase_mark, NULL);
}
