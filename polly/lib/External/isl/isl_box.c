/*
 * Copyright 2010-2011 INRIA Saclay
 * Copyright 2012-2013 Ecole Normale Superieure
 *
 * Use of this software is governed by the MIT license
 *
 * Written by Sven Verdoolaege, INRIA Saclay - Ile-de-France,
 * Parc Club Orsay Universite, ZAC des vignes, 4 rue Jacques Monod,
 * 91893 Orsay, France
 * and Ecole Normale Superieure, 45 rue d'Ulm, 75230 Paris, France
 */

#include <isl/val.h>
#include <isl/space.h>
#include <isl_map_private.h>
#include <isl_aff_private.h>
#include <isl/constraint.h>
#include <isl/ilp.h>
#include <isl/fixed_box.h>

/* Representation of a box of fixed size containing the elements
 * [offset, offset + size).
 * "size" lives in the target space of "offset".
 *
 * If any of the "offsets" is NaN, then the object represents
 * the failure of finding a fixed-size box.
 */
struct isl_fixed_box {
	isl_multi_aff *offset;
	isl_multi_val *size;
};

/* Free "box" and return NULL.
 */
__isl_null isl_fixed_box *isl_fixed_box_free(__isl_take isl_fixed_box *box)
{
	if (!box)
		return NULL;
	isl_multi_aff_free(box->offset);
	isl_multi_val_free(box->size);
	free(box);
	return NULL;
}

/* Construct an isl_fixed_box with the given offset and size.
 */
static __isl_give isl_fixed_box *isl_fixed_box_alloc(
	__isl_take isl_multi_aff *offset, __isl_take isl_multi_val *size)
{
	isl_ctx *ctx;
	isl_fixed_box *box;

	if (!offset || !size)
		goto error;
	ctx = isl_multi_aff_get_ctx(offset);
	box = isl_alloc_type(ctx, struct isl_fixed_box);
	if (!box)
		goto error;
	box->offset = offset;
	box->size = size;

	return box;
error:
	isl_multi_aff_free(offset);
	isl_multi_val_free(size);
	return NULL;
}

/* Construct an initial isl_fixed_box with zero offsets
 * in the given space and zero corresponding sizes.
 */
static __isl_give isl_fixed_box *isl_fixed_box_init(
	__isl_take isl_space *space)
{
	isl_multi_aff *offset;
	isl_multi_val *size;

	offset = isl_multi_aff_zero(isl_space_copy(space));
	space = isl_space_drop_all_params(isl_space_range(space));
	size = isl_multi_val_zero(space);
	return isl_fixed_box_alloc(offset, size);
}

/* Return a copy of "box".
 */
__isl_give isl_fixed_box *isl_fixed_box_copy(__isl_keep isl_fixed_box *box)
{
	isl_multi_aff *offset;
	isl_multi_val *size;

	offset = isl_fixed_box_get_offset(box);
	size = isl_fixed_box_get_size(box);
	return isl_fixed_box_alloc(offset, size);
}

/* Replace the offset and size in direction "pos" by "offset" and "size"
 * (without checking whether "box" is a valid box).
 */
static __isl_give isl_fixed_box *isl_fixed_box_set_extent(
	__isl_take isl_fixed_box *box, int pos, __isl_keep isl_aff *offset,
	__isl_keep isl_val *size)
{
	if (!box)
		return NULL;
	box->offset = isl_multi_aff_set_aff(box->offset, pos,
							isl_aff_copy(offset));
	box->size = isl_multi_val_set_val(box->size, pos, isl_val_copy(size));
	if (!box->offset || !box->size)
		return isl_fixed_box_free(box);
	return box;
}

/* Replace the offset and size in direction "pos" by "offset" and "size",
 * if "box" is a valid box.
 */
static __isl_give isl_fixed_box *isl_fixed_box_set_valid_extent(
	__isl_take isl_fixed_box *box, int pos, __isl_keep isl_aff *offset,
	__isl_keep isl_val *size)
{
	isl_bool valid;

	valid = isl_fixed_box_is_valid(box);
	if (valid < 0 || !valid)
		return box;
	return isl_fixed_box_set_extent(box, pos, offset, size);
}

/* Replace "box" by an invalid box, by setting all offsets to NaN
 * (and all sizes to infinity).
 */
static __isl_give isl_fixed_box *isl_fixed_box_invalidate(
	__isl_take isl_fixed_box *box)
{
	int i;
	isl_size n;
	isl_space *space;
	isl_val *infty;
	isl_aff *nan;

	if (!box)
		return NULL;
	n = isl_multi_val_dim(box->size, isl_dim_set);
	if (n < 0)
		return isl_fixed_box_free(box);

	infty = isl_val_infty(isl_fixed_box_get_ctx(box));
	space = isl_space_domain(isl_fixed_box_get_space(box));
	nan = isl_aff_nan_on_domain(isl_local_space_from_space(space));
	for (i = 0; i < n; ++i)
		box = isl_fixed_box_set_extent(box, i, nan, infty);
	isl_aff_free(nan);
	isl_val_free(infty);

	if (!box->offset || !box->size)
		return isl_fixed_box_free(box);
	return box;
}

/* Project the domain of the fixed box onto its parameter space.
 * In particular, project out the domain of the offset.
 */
static __isl_give isl_fixed_box *isl_fixed_box_project_domain_on_params(
	__isl_take isl_fixed_box *box)
{
	isl_bool valid;

	valid = isl_fixed_box_is_valid(box);
	if (valid < 0)
		return isl_fixed_box_free(box);
	if (!valid)
		return box;

	box->offset = isl_multi_aff_project_domain_on_params(box->offset);
	if (!box->offset)
		return isl_fixed_box_free(box);

	return box;
}

/* Return the isl_ctx to which "box" belongs.
 */
isl_ctx *isl_fixed_box_get_ctx(__isl_keep isl_fixed_box *box)
{
	if (!box)
		return NULL;
	return isl_multi_aff_get_ctx(box->offset);
}

/* Return the space in which "box" lives.
 */
__isl_give isl_space *isl_fixed_box_get_space(__isl_keep isl_fixed_box *box)
{
	if (!box)
		return NULL;
	return isl_multi_aff_get_space(box->offset);
}

/* Does "box" contain valid information?
 */
isl_bool isl_fixed_box_is_valid(__isl_keep isl_fixed_box *box)
{
	if (!box)
		return isl_bool_error;
	return isl_bool_not(isl_multi_aff_involves_nan(box->offset));
}

/* Return the offsets of the box "box".
 */
__isl_give isl_multi_aff *isl_fixed_box_get_offset(
	__isl_keep isl_fixed_box *box)
{
	if (!box)
		return NULL;
	return isl_multi_aff_copy(box->offset);
}

/* Return the sizes of the box "box".
 */
__isl_give isl_multi_val *isl_fixed_box_get_size(__isl_keep isl_fixed_box *box)
{
	if (!box)
		return NULL;
	return isl_multi_val_copy(box->size);
}

/* Data used in set_dim_extent and compute_size_in_direction.
 *
 * "bset" is a wrapped copy of the basic map that has the selected
 * output dimension as range.
 * "pos" is the position of the variable representing the output dimension,
 * i.e., the variable for which the size should be computed.  This variable
 * is also the last variable in "bset".
 * "size" is the best size found so far
 * (infinity if no offset was found so far).
 * "offset" is the offset corresponding to the best size
 * (NULL if no offset was found so far).
 */
struct isl_size_info {
	isl_basic_set *bset;
	isl_size pos;
	isl_val *size;
	isl_aff *offset;
};

/* Is "c" a suitable bound on dimension "pos" for use as a lower bound
 * of a fixed-size range.
 * In particular, it needs to be a lower bound on "pos".
 * In order for the final offset not to be too complicated,
 * the constraint itself should also not involve any integer divisions.
 */
static isl_bool is_suitable_bound(__isl_keep isl_constraint *c, unsigned pos)
{
	isl_size n_div;
	isl_bool is_bound, any_divs;

	is_bound = isl_constraint_is_lower_bound(c, isl_dim_set, pos);
	if (is_bound < 0 || !is_bound)
		return is_bound;

	n_div = isl_constraint_dim(c, isl_dim_div);
	if (n_div < 0)
		return isl_bool_error;
	any_divs = isl_constraint_involves_dims(c, isl_dim_div, 0, n_div);
	return isl_bool_not(any_divs);
}

/* Given a constraint from the basic set describing the bounds on
 * an array index, check if it is a lower bound, say m i >= b(x), and,
 * if so, check whether the expression "i - ceil(b(x)/m) + 1" has a constant
 * upper bound.  If so, and if this bound is smaller than any bound
 * derived from earlier constraints, set the size to this bound on
 * the expression and the lower bound to ceil(b(x)/m).
 */
static isl_stat compute_size_in_direction(__isl_take isl_constraint *c,
	void *user)
{
	struct isl_size_info *info = user;
	isl_val *v;
	isl_aff *aff;
	isl_aff *lb;
	isl_bool is_bound, better;

	is_bound = is_suitable_bound(c, info->pos);
	if (is_bound < 0 || !is_bound) {
		isl_constraint_free(c);
		return is_bound < 0 ? isl_stat_error : isl_stat_ok;
	}

	aff = isl_constraint_get_bound(c, isl_dim_set, info->pos);
	aff = isl_aff_ceil(aff);

	lb = isl_aff_copy(aff);

	aff = isl_aff_neg(aff);
	aff = isl_aff_add_coefficient_si(aff, isl_dim_in, info->pos, 1);

	v = isl_basic_set_max_val(info->bset, aff);
	isl_aff_free(aff);

	v = isl_val_add_ui(v, 1);
	better = isl_val_lt(v, info->size);
	if (better >= 0 && better) {
		isl_val_free(info->size);
		info->size = isl_val_copy(v);
		lb = isl_aff_domain_factor_domain(lb);
		isl_aff_free(info->offset);
		info->offset = isl_aff_copy(lb);
	}
	isl_val_free(v);
	isl_aff_free(lb);

	isl_constraint_free(c);

	return better < 0 ? isl_stat_error : isl_stat_ok;
}

/* Look for a fixed-size range of values for the output dimension "pos"
 * of "map", by looking for a lower-bound expression in the parameters
 * and input dimensions such that the range of the output dimension
 * is a constant shifted by this expression.
 *
 * In particular, look through the explicit lower bounds on the output dimension
 * for candidate expressions and pick the one that results in the smallest size.
 * Initialize the size with infinity and if no better size is found
 * then invalidate the box.  Otherwise, set the offset and size
 * in the given direction by those that correspond to the smallest size.
 *
 * Note that while evaluating the size corresponding to a lower bound,
 * an affine expression is constructed from the lower bound.
 * This lower bound may therefore not have any unknown local variables.
 * Eliminate any unknown local variables up front.
 * No such restriction needs to be imposed on the set over which
 * the size is computed.
 */
static __isl_give isl_fixed_box *set_dim_extent(__isl_take isl_fixed_box *box,
	__isl_keep isl_map *map, int pos)
{
	struct isl_size_info info;
	isl_bool valid;
	isl_ctx *ctx;
	isl_basic_set *bset;

	if (!box || !map)
		return isl_fixed_box_free(box);

	ctx = isl_map_get_ctx(map);
	map = isl_map_copy(map);
	map = isl_map_project_onto(map, isl_dim_out, pos, 1);
	info.size = isl_val_infty(ctx);
	info.offset = NULL;
	info.pos = isl_map_dim(map, isl_dim_in);
	info.bset = isl_basic_map_wrap(isl_map_simple_hull(map));
	bset = isl_basic_set_copy(info.bset);
	bset = isl_basic_set_remove_unknown_divs(bset);
	if (info.pos < 0)
		bset = isl_basic_set_free(bset);
	if (isl_basic_set_foreach_constraint(bset,
					&compute_size_in_direction, &info) < 0)
		box = isl_fixed_box_free(box);
	isl_basic_set_free(bset);
	valid = isl_val_is_int(info.size);
	if (valid < 0)
		box = isl_fixed_box_free(box);
	else if (valid)
		box = isl_fixed_box_set_valid_extent(box, pos,
						     info.offset, info.size);
	else
		box = isl_fixed_box_invalidate(box);
	isl_val_free(info.size);
	isl_aff_free(info.offset);
	isl_basic_set_free(info.bset);

	return box;
}

/* Try and construct a fixed-size rectangular box with an offset
 * in terms of the domain of "map" that contains the range of "map".
 * If no such box can be constructed, then return an invalidated box,
 * i.e., one where isl_fixed_box_is_valid returns false.
 *
 * Iterate over the dimensions in the range
 * setting the corresponding offset and extent.
 */
__isl_give isl_fixed_box *isl_map_get_range_simple_fixed_box_hull(
	__isl_keep isl_map *map)
{
	int i;
	isl_size n;
	isl_space *space;
	isl_fixed_box *box;

	n = isl_map_dim(map, isl_dim_out);
	if (n < 0)
		return NULL;
	space = isl_map_get_space(map);
	box = isl_fixed_box_init(space);

	map = isl_map_detect_equalities(isl_map_copy(map));
	for (i = 0; i < n; ++i) {
		isl_bool valid;

		box = set_dim_extent(box, map, i);
		valid = isl_fixed_box_is_valid(box);
		if (valid < 0 || !valid)
			break;
	}
	isl_map_free(map);

	return box;
}

/* Try and construct a fixed-size rectangular box with an offset
 * in terms of the parameters of "set" that contains "set".
 * If no such box can be constructed, then return an invalidated box,
 * i.e., one where isl_fixed_box_is_valid returns false.
 *
 * Compute the box using isl_map_get_range_simple_fixed_box_hull
 * by constructing a map from the set and
 * project out the domain again from the result.
 */
__isl_give isl_fixed_box *isl_set_get_simple_fixed_box_hull(
	__isl_keep isl_set *set)
{
	isl_map *map;
	isl_fixed_box *box;

	map = isl_map_from_range(isl_set_copy(set));
	box = isl_map_get_range_simple_fixed_box_hull(map);
	isl_map_free(map);
	box = isl_fixed_box_project_domain_on_params(box);

	return box;
}

#undef BASE
#define BASE multi_val
#include "print_yaml_field_templ.c"

#undef BASE
#define BASE multi_aff
#include "print_yaml_field_templ.c"

/* Print the information contained in "box" to "p".
 * The information is printed as a YAML document.
 */
__isl_give isl_printer *isl_printer_print_fixed_box(
	__isl_take isl_printer *p, __isl_keep isl_fixed_box *box)
{
	if (!box)
		return isl_printer_free(p);

	p = isl_printer_yaml_start_mapping(p);
	p = print_yaml_field_multi_aff(p, "offset", box->offset);
	p = print_yaml_field_multi_val(p, "size", box->size);
	p = isl_printer_yaml_end_mapping(p);

	return p;
}

#undef BASE
#define BASE fixed_box
#include <print_templ_yaml.c>
