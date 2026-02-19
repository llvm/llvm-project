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
#include <isl/stream.h>

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
static __isl_keep isl_multi_aff *isl_fixed_box_peek_offset(
	__isl_keep isl_fixed_box *box)
{
	if (!box)
		return NULL;
	return box->offset;
}

/* Return a copy of the offsets of the box "box".
 */
__isl_give isl_multi_aff *isl_fixed_box_get_offset(
	__isl_keep isl_fixed_box *box)
{
	return isl_multi_aff_copy(isl_fixed_box_peek_offset(box));
}

/* Return the sizes of the box "box".
 */
static __isl_keep isl_multi_val *isl_fixed_box_peek_size(
	__isl_keep isl_fixed_box *box)
{
	if (!box)
		return NULL;
	return box->size;
}

/* Return a copy of the sizes of the box "box".
 */
__isl_give isl_multi_val *isl_fixed_box_get_size(__isl_keep isl_fixed_box *box)
{
	return isl_multi_val_copy(isl_fixed_box_peek_size(box));
}

/* Is "box1" obviously equal to "box2"?
 *
 * That is, does it have the same size and obviously the same offset?
 */
isl_bool isl_fixed_box_plain_is_equal(__isl_keep isl_fixed_box *box1,
	__isl_keep isl_fixed_box *box2)
{
	isl_multi_aff *offset1, *offset2;
	isl_multi_val *size1, *size2;
	isl_bool equal;

	size1 = isl_fixed_box_peek_size(box1);
	size2 = isl_fixed_box_peek_size(box2);
	equal = isl_multi_val_is_equal(size1, size2);
	if (equal < 0 || !equal)
		return equal;

	offset1 = isl_fixed_box_peek_offset(box1);
	offset2 = isl_fixed_box_peek_offset(box2);
	return isl_multi_aff_plain_is_equal(offset1, offset2);
}

/* Data used in set_dim_extent and compute_size_in_direction.
 *
 * "bset" is a wrapped copy of the basic map that has the selected
 * output dimension as range, without any contraction.
 * "pos" is the position of the variable representing the output dimension,
 * i.e., the variable for which the size should be computed.  This variable
 * is also the last variable in "bset".
 * "size" is the best size found so far
 * (infinity if no offset was found so far).
 * "offset" is the offset corresponding to the best size
 * (NULL if no offset was found so far).
 *
 * If "expand" is not NULL, then it maps a contracted version of "bset"
 * to the original "bset", while "domain_map" maps the space of "bset"
 * to the domain of the wrapped map.
 */
struct isl_size_info {
	isl_basic_set *bset;
	isl_size pos;
	isl_val *size;
	isl_aff *offset;

	isl_multi_aff *expand;
	isl_multi_aff *domain_map;
};

/* Detect any stride in the single output dimension of "map" and
 * set the fields of "info" used in exploiting this stride.
 * If no (non-trivial) stride can be found, then set those fields to NULL.
 *
 * If there is a non-trivial stride, then the single output dimension i
 * is of the form
 *
 *	i = offset + stride * i'
 *
 * Construct a function that maps i' to i.
 * Note that the offset may depend on the domain of the map,
 * so it needs to be of the form
 *
 *	[D -> [i']] -> [i]
 *
 * In fact, it is more convenient for the function to be of the form
 *
 *	[D -> [i']] -> [D -> [i]]
 *
 * First construct helper functions
 *
 *	[D -> [i]] -> D
 *	[D -> [i]] -> [i]
 *
 * Plug in [D -> [i]] -> D into the offset (defined on D) to obtain
 * the offset defined on [D -> [i]] and add stride times [D -> [i]] -> [i].
 * This produces the function
 *
 *	[D -> [i']] = [offset + stride i']
 *
 * Combine it with [D -> [i]] -> D again to obtain the desired result.
 */
static __isl_give isl_map *isl_size_info_detect_stride(
	struct isl_size_info *info, __isl_take isl_map *map)
{
	isl_stride_info *si;
	isl_val *stride;
	isl_aff *offset;
	isl_bool is_one;
	isl_multi_aff *domain_map, *id;
	isl_multi_aff *expand;

	info->expand = NULL;
	info->domain_map = NULL;

	si = isl_map_get_range_stride_info(map, 0);
	stride = isl_stride_info_get_stride(si);
	is_one = isl_val_is_one(stride);
	if (is_one < 0 || is_one) {
		isl_val_free(stride);
		isl_stride_info_free(si);
		return is_one < 0 ? isl_map_free(map) : map;
	}
	offset = isl_stride_info_get_offset(si);
	isl_stride_info_free(si);

	domain_map = isl_space_domain_map_multi_aff(isl_aff_get_space(offset));
	id = isl_space_range_map_multi_aff(isl_aff_get_space(offset));
	offset = isl_aff_pullback_multi_aff(offset,
					    isl_multi_aff_copy(domain_map));

	expand = isl_multi_aff_scale_val(id, stride);
	expand = isl_multi_aff_add(expand, isl_multi_aff_from_aff(offset));
	expand = isl_multi_aff_range_product(isl_multi_aff_copy(domain_map),
						expand);
	info->expand = expand;
	info->domain_map = domain_map;

	if (!expand || !domain_map)
		return isl_map_free(map);

	return map;
}

/* If any stride was detected in the single output dimension
 * in the wrapped map in "bset" (i.e., if info->expand is set),
 * then plug in the expansion to obtain a description in terms
 * of an output dimension without stride.
 * Otherwise, return the original "bset".
 */
static __isl_give isl_basic_set *isl_size_info_contract(
	struct isl_size_info *info, __isl_take isl_basic_set *bset)
{
	if (!info->expand)
		return bset;

	bset = isl_basic_set_preimage_multi_aff(bset,
					isl_multi_aff_copy(info->expand));

	return bset;
}

/* Given an affine function "aff" that maps the space of "bset"
 * to a value in the (possibly) contracted space,
 * expand it back to the original space.
 * The value of "aff" only depends on the domain of wrapped relation
 * inside "bset".
 *
 * If info->expand is not set, then no contraction was applied and
 * "aff" is returned.
 *
 * Otherwise, combine "aff" of the form [D -> [*]] -> [v'(D)]
 * with [D -> [*]] -> D to obtain [D -> [*]] -> [D -> [v'(D)]].
 * Apply the expansion [D -> [i']] = [D -> [offset + stride * i']]
 * to obtain [D -> [*]] -> [D -> [offset + stride * v'(D)]] and
 * extract out [D -> [*]] -> [offset + stride * v'(D)].
 */
static __isl_give isl_aff *isl_size_info_expand(
	struct isl_size_info *info, __isl_take isl_aff *aff)
{
	isl_multi_aff *ma;

	if (!info->expand)
		return aff;

	ma = isl_multi_aff_from_aff(aff);
	ma = isl_multi_aff_range_product(isl_multi_aff_copy(info->domain_map),
					ma);
	ma = isl_multi_aff_pullback_multi_aff(isl_multi_aff_copy(info->expand),
					ma);
	ma = isl_multi_aff_range_factor_range(ma);
	aff = isl_multi_aff_get_at(ma, 0);
	isl_multi_aff_free(ma);

	return aff;
}

/* Free all memory allocated for "info".
 */
static void isl_size_info_clear(struct isl_size_info *info)
{
	isl_val_free(info->size);
	isl_aff_free(info->offset);
	isl_basic_set_free(info->bset);

	isl_multi_aff_free(info->expand);
	isl_multi_aff_free(info->domain_map);
}

/* Is "c" a suitable bound on dimension "pos" for use as a lower bound
 * of a fixed-size range.
 * In particular, it needs to be a lower bound on "pos".
 */
static isl_bool is_suitable_bound(__isl_keep isl_constraint *c, unsigned pos)
{
	return isl_constraint_is_lower_bound(c, isl_dim_set, pos);
}

/* Given a constraint from the basic set describing the bounds on
 * an array index, check if it is a lower bound, say m i >= b(x), and,
 * if so, check whether the expression "i - ceil(b(x)/m) + 1" has a constant
 * upper bound.  If so, and if this bound is smaller than any bound
 * derived from earlier constraints, set the size to this bound on
 * the expression and the lower bound to ceil(b(x)/m).
 *
 * If any contraction was applied, then the lower bound ceil(b(x)/m)
 * is defined in the contracted space, so it needs to be expanded
 * first before applying it to the original space.
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
	aff = isl_size_info_expand(info, aff);

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
 * If the output dimension is strided, then scale it down before
 * looking for lower bounds.  The size computation is however performed
 * in the original space.
 *
 * Note that while evaluating the size corresponding to a lower bound,
 * an affine expression is constructed from the lower bound.
 * This lower bound may therefore not have any unknown local variables.
 * Eliminate any unknown local variables up front.
 * Furthermore, the lower bound can clearly not involve
 * (any local variables that involve) the output dimension itself,
 * so any such local variables are eliminated as well.
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
	map = isl_size_info_detect_stride(&info, map);
	info.size = isl_val_infty(ctx);
	info.offset = NULL;
	info.pos = isl_map_dim(map, isl_dim_in);
	info.bset = isl_basic_map_wrap(isl_map_simple_hull(map));
	bset = isl_basic_set_copy(info.bset);
	bset = isl_size_info_contract(&info, bset);
	bset = isl_basic_set_remove_unknown_divs(bset);
	if (info.pos < 0)
		bset = isl_basic_set_free(bset);
	bset = isl_basic_set_remove_divs_involving_dims(bset, isl_dim_set,
							info.pos, 1);
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
	isl_size_info_clear(&info);

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

/* Compute a fixed box from "set" using "map_box" by treating it as a map
 * with a zero-dimensional domain and
 * project out the domain again from the result.
 */
static __isl_give isl_fixed_box *fixed_box_as_map(__isl_keep isl_set *set,
	__isl_give isl_fixed_box *(*map_box)(__isl_keep isl_map *map))
{
	isl_map *map;
	isl_fixed_box *box;

	map = isl_map_from_range(isl_set_copy(set));
	box = map_box(map);
	isl_map_free(map);
	box = isl_fixed_box_project_domain_on_params(box);

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
	return fixed_box_as_map(set, &isl_map_get_range_simple_fixed_box_hull);
}

/* Check whether the output elements lie on a rectangular lattice,
 * possibly depending on the parameters and the input dimensions.
 * Return a tile in this lattice.
 * If no stride information can be found, then return a tile of size 1
 * (and offset 0).
 *
 * Obtain stride information in each output dimension separately and
 * combine the results.
 */
__isl_give isl_fixed_box *isl_map_get_range_lattice_tile(
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

	for (i = 0; i < n; ++i) {
		isl_val *stride;
		isl_aff *offset;
		isl_stride_info *si;

		si = isl_map_get_range_stride_info(map, i);
		stride = isl_stride_info_get_stride(si);
		offset = isl_stride_info_get_offset(si);
		isl_stride_info_free(si);

		box = isl_fixed_box_set_valid_extent(box, i, offset, stride);

		isl_aff_free(offset);
		isl_val_free(stride);
	}

	return box;
}

/* Check whether the elements lie on a rectangular lattice,
 * possibly depending on the parameters.
 * Return a tile in this lattice.
 * If no stride information can be found, then return a tile of size 1
 * (and offset 0).
 *
 * Consider the set as a map with a zero-dimensional domain and
 * obtain a lattice tile of that map.
 */
__isl_give isl_fixed_box *isl_set_get_lattice_tile(__isl_keep isl_set *set)
{
	return fixed_box_as_map(set, &isl_map_get_range_lattice_tile);
}

/* An enumeration of the keys that may appear in a YAML mapping
 * of an isl_fixed_box object.
 */
enum isl_fb_key {
	isl_fb_key_error = -1,
	isl_fb_key_offset,
	isl_fb_key_size,
	isl_fb_key_end,
};

/* Textual representations of the YAML keys for an isl_fixed_box object.
 */
static char *key_str[] = {
	[isl_fb_key_offset] = "offset",
	[isl_fb_key_size] = "size",
};

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
	p = print_yaml_field_multi_aff(p, key_str[isl_fb_key_offset],
					box->offset);
	p = print_yaml_field_multi_val(p, key_str[isl_fb_key_size], box->size);
	p = isl_printer_yaml_end_mapping(p);

	return p;
}

#undef BASE
#define BASE fixed_box
#include <print_templ_yaml.c>

#undef KEY
#define KEY enum isl_fb_key
#undef KEY_ERROR
#define KEY_ERROR isl_fb_key_error
#undef KEY_END
#define KEY_END isl_fb_key_end
#undef KEY_STR
#define KEY_STR key_str
#undef KEY_EXTRACT
#define KEY_EXTRACT extract_key
#undef KEY_GET
#define KEY_GET get_key
#include "extract_key.c"

#undef BASE
#define BASE multi_val
#include "read_in_string_templ.c"

#undef BASE
#define BASE multi_aff
#include "read_in_string_templ.c"

/* Read an isl_fixed_box object from "s".
 *
 * The input needs to contain both an offset and a size.
 * If either is specified multiple times, then the last specification
 * overrides all previous ones.  This is simpler than checking
 * that each is only specified once.
 */
static __isl_give isl_fixed_box *isl_stream_read_fixed_box(isl_stream *s)
{
	isl_bool more;
	isl_multi_aff *offset = NULL;
	isl_multi_val *size = NULL;

	if (isl_stream_yaml_read_start_mapping(s) < 0)
		return NULL;

	while ((more = isl_stream_yaml_next(s)) == isl_bool_true) {
		enum isl_fb_key key;

		key = get_key(s);
		if (isl_stream_yaml_next(s) < 0)
			goto error;
		switch (key) {
		case isl_fb_key_end:
		case isl_fb_key_error:
			goto error;
		case isl_fb_key_offset:
			isl_multi_aff_free(offset);
			offset = read_multi_aff(s);
			if (!offset)
				goto error;
			break;
		case isl_fb_key_size:
			isl_multi_val_free(size);
			size = read_multi_val(s);
			if (!size)
				goto error;
			break;
		}
	}
	if (more < 0)
		goto error;

	if (isl_stream_yaml_read_end_mapping(s) < 0)
		goto error;

	if (!offset) {
		isl_stream_error(s, NULL, "no offset specified");
		goto error;
	}

	if (!size) {
		isl_stream_error(s, NULL, "no size specified");
		goto error;
	}

	return isl_fixed_box_alloc(offset, size);
error:
	isl_multi_aff_free(offset);
	isl_multi_val_free(size);
	return NULL;
}

#undef TYPE_BASE
#define TYPE_BASE	fixed_box
#include "isl_read_from_str_templ.c"
