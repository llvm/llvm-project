/*
 * Copyright 2010-2011 INRIA Saclay
 * Copyright 2012-2013 Ecole Normale Superieure
 *
 * Use of this software is governed by the MIT license
 *
 * Written by Sven Verdoolaege, INRIA Saclay - Ile-de-France,
 * Parc Club Orsay Universite, ZAC des vignes, 4 rue Jacques Monod,
 * 91893 Orsay, France
 * and Ecole Normale Superieure, 45 rue d’Ulm, 75230 Paris, France
 */

#include <assert.h>
#include <stdlib.h>
#include <string.h>

#include <isl/polynomial.h>
#include <isl/union_set.h>
#include <isl/aff.h>
#include <isl/ilp.h>
#include <isl/flow.h>
#include <isl/schedule.h>
#include <isl/schedule_node.h>
#include <isl/options.h>
#include <isl/ast_build.h>

#include "cpu.h"
#include "gpu.h"
#include "gpu_array_tile.h"
#include "gpu_group.h"
#include "gpu_tree.h"
#include "schedule.h"
#include "ppcg_options.h"
#include "print.h"
#include "util.h"

struct gpu_array_info;

/* Return the name of the outer array (of structs) accessed by "access".
 */
static const char *get_outer_array_name(__isl_keep isl_map *access)
{
	isl_space *space;
	const char *name;

	space = isl_space_range(isl_map_get_space(access));
	while (space && isl_space_is_wrapping(space))
		space = isl_space_domain(isl_space_unwrap(space));
	name = isl_space_get_tuple_name(space, isl_dim_set);
	isl_space_free(space);

	return name;
}

/* Collect all references to the given array and store pointers to them
 * in array->refs.
 */
void collect_references(struct gpu_prog *prog,
	struct gpu_array_info *array)
{
	int i;
	int n;

	n = 0;
	for (i = 0; i < prog->n_stmts; ++i) {
		struct gpu_stmt *stmt = &prog->stmts[i];
		struct gpu_stmt_access *access;

		for (access = stmt->accesses; access; access = access->next) {
			const char *name;
			name = get_outer_array_name(access->access);
			if (name && !strcmp(array->name, name))
				n++;
		}
	}

	array->n_ref = n;
	array->refs = isl_alloc_array(prog->ctx, struct gpu_stmt_access *, n);
	assert(array->refs);

	n = 0;
	for (i = 0; i < prog->n_stmts; ++i) {
		struct gpu_stmt *stmt = &prog->stmts[i];
		struct gpu_stmt_access *access;

		for (access = stmt->accesses; access; access = access->next) {
			const char *name;
			name = get_outer_array_name(access->access);
			if (!name || strcmp(array->name, name))
				continue;

			array->refs[n++] = access;
		}
	}
}

/* Compute and return the extent of "array", taking into account the set of
 * accessed elements.
 *
 * In particular, the extent in the outer dimension is taken
 * from "accessed", while the extents in the remaining dimensions
 * are taken from array->extent.
 *
 * The extent in the outer dimension cannot be taken from array->extent
 * because that may be unbounded.  Furthermore, even if it is bounded,
 * it may be larger than the piece of the array that is being accessed.
 */
static __isl_give isl_set *compute_extent(struct pet_array *array,
	__isl_keep isl_set *accessed)
{
	int n_index;
	isl_id *id;
	isl_set *outer;
	isl_set *extent;

	extent = isl_set_copy(array->extent);

	n_index = isl_set_dim(accessed, isl_dim_set);
	if (n_index == 0)
		return extent;

	extent = isl_set_project_out(extent, isl_dim_set, 0, 1);
	outer = isl_set_copy(accessed);
	outer = isl_set_project_out(outer, isl_dim_set, 1, n_index - 1);
	extent = isl_set_flat_product(outer, extent);
	id = isl_set_get_tuple_id(accessed);
	extent = isl_set_set_tuple_id(extent, id);

	return extent;
}

/* Is the array "array" being extracted a read-only scalar?
 *
 * That is, is "array" a scalar that is never possibly written to.
 * An array containing structures is never considered to be a scalar.
 */
static int is_read_only_scalar(struct gpu_array_info *array,
	struct gpu_prog *prog)
{
	isl_set *space;
	isl_union_map *write;
	int empty;

	if (array->has_compound_element)
		return 0;
	if (array->n_index != 0)
		return 0;

	write = isl_union_map_copy(prog->may_write);
	space = isl_set_universe(isl_space_copy(array->space));
	write = isl_union_map_intersect_range(write,
						isl_union_set_from_set(space));
	empty = isl_union_map_is_empty(write);
	isl_union_map_free(write);

	return empty;
}

/* Compute bounds on the host array "pa" based on the corresponding
 * accessed elements in "arrays"
 * and collect all references to the array.
 * Store the results in "info".
 *
 * If the array is zero-dimensional and does not contain structures,
 * i.e., if the array is a scalar, we check whether it is read-only.
 * We also check whether the array is accessed at all.
 */
static int extract_array_info(struct gpu_prog *prog,
	struct gpu_array_info *info, struct pet_array *pa,
	__isl_keep isl_union_set *arrays)
{
	int i, empty;
	const char *name;
	int n_index;
	isl_pw_aff **bounds;
	isl_set *accessed, *extent;

	n_index = isl_set_dim(pa->extent, isl_dim_set);
	name = isl_set_get_tuple_name(pa->extent);
	bounds = isl_alloc_array(prog->ctx, isl_pw_aff *, n_index);
	if (!bounds)
		return -1;

	info->space = isl_set_get_space(pa->extent);
	info->name = strdup(name);
	info->n_index = n_index;
	info->bound = bounds;
	info->linearize = prog->scop->options->linearize_device_arrays;

	info->type = strdup(pa->element_type);
	info->size = pa->element_size;
	info->local = pa->declared && !pa->exposed;
	info->has_compound_element = pa->element_is_record;
	info->read_only_scalar = is_read_only_scalar(info, prog);

	accessed = isl_union_set_extract_set(arrays,
					    isl_space_copy(info->space));
	empty = isl_set_is_empty(accessed);
	extent = compute_extent(pa, accessed);
	isl_set_free(accessed);
	info->extent = extent;
	if (empty < 0)
		return -1;
	info->accessed = !empty;
	for (i = 0; i < n_index; ++i) {
		isl_set *dom;
		isl_local_space *ls;
		isl_aff *one;
		isl_pw_aff *bound;

		dom = isl_set_copy(extent);
		dom = isl_set_project_out(dom, isl_dim_set, i + 1,
					    n_index - (i + 1));
		dom = isl_set_project_out(dom, isl_dim_set, 0, i);
		if (!isl_set_dim_has_upper_bound(dom, isl_dim_set, 0)) {
			fprintf(stderr, "unable to determine extent of '%s' "
				"in dimension %d\n", info->name, i);
			dom = isl_set_free(dom);
		}
		bound = isl_set_dim_max(dom, 0);
		dom = isl_pw_aff_domain(isl_pw_aff_copy(bound));
		ls = isl_local_space_from_space(isl_set_get_space(dom));
		one = isl_aff_zero_on_domain(ls);
		one = isl_aff_add_constant_si(one, 1);
		bound = isl_pw_aff_add(bound, isl_pw_aff_alloc(dom, one));
		bound = isl_pw_aff_gist(bound, isl_set_copy(prog->context));

		bounds[i] = bound;
		if (!isl_pw_aff_is_cst(bound))
			info->linearize = 1;
	}

	collect_references(prog, info);

	return 0;
}

/* Remove independence from the order constraints "order" on array "array".
 * Since the pairs of iterations in the filter relation of an independence
 * are guaranteed to be completely independent by the user, there is
 * no need to ensure that live ranges are ordered along thong pairs.
 * We make an exception for local variables, though, as the independence
 * guarantee does not apply to those.
 *
 * The order constraints are used in two places.
 * Those on scalars are used in check_scalar_live_ranges to check if
 * we need to force the scalar to be private.  Any non-local scalar
 * should not be forced scalar if it only appears in independent loops.
 * Those on non-scalars are added to the coincidence constraints
 * in compute_schedule because we do not support any array expansion.
 * Accesses to non-local arrays should not prevent a loop from being
 * considered coincident so we should indeed remove those constraints
 * from the order constraints.
 */
static __isl_give isl_union_map *remove_independences(struct gpu_prog *prog,
	struct gpu_array_info *array, __isl_take isl_union_map *order)
{
	int i;

	for (i = 0; i < prog->scop->pet->n_independence; ++i) {
		struct pet_independence *pi = prog->scop->pet->independences[i];
		if (isl_union_set_contains(pi->local, array->space))
			continue;

		order = isl_union_map_subtract(order,
						isl_union_map_copy(pi->filter));
	}

	return order;
}

/* For each array in "prog", store the (untagged) order dependences
 * derived from the array in array->dep_order.
 * In particular, consider all references that access the given array
 * and take the order dependences that have one of these references
 * as source.  (Since an order dependence relates two references to
 * the same array, the target of these order dependences will also
 * be one of these references.)
 * Additionally, store the union of these array->dep_order relations
 * for all non-scalar arrays in prog->array_order.
 */
void collect_order_dependences(struct gpu_prog *prog)
{
	int i;
	isl_space *space;
	isl_union_map *accesses;

	space = isl_union_map_get_space(prog->read);
	prog->array_order = isl_union_map_empty(space);

	accesses = isl_union_map_copy(prog->scop->tagged_reads);
	accesses = isl_union_map_union(accesses,
			    isl_union_map_copy(prog->scop->tagged_may_writes));
	accesses = isl_union_map_universe(accesses);
	accesses = isl_union_map_apply_range(accesses,
					    isl_union_map_copy(prog->to_outer));

	for (i = 0; i < prog->n_array; ++i) {
		struct gpu_array_info *array = &prog->array[i];
		isl_set *set;
		isl_union_set *uset;
		isl_union_map *order;

		set = isl_set_universe(isl_space_copy(array->space));
		uset = isl_union_set_from_set(set);
		uset = isl_union_map_domain(
		    isl_union_map_intersect_range(isl_union_map_copy(accesses),
						    uset));
		order = isl_union_map_copy(prog->scop->tagged_dep_order);
		order = isl_union_map_intersect_domain(order, uset);
		order = isl_union_map_zip(order);
		order = isl_union_set_unwrap(isl_union_map_domain(order));
		order = remove_independences(prog, array, order);
		array->dep_order = order;

		if (gpu_array_is_scalar(array) && !array->has_compound_element)
			continue;

		prog->array_order = isl_union_map_union(prog->array_order,
					isl_union_map_copy(array->dep_order));
	}

	isl_union_map_free(accesses);
}

/* Construct a gpu_array_info for each array referenced by prog->scop and
 * collect them in prog->array.
 *
 * The sizes are based on the extents and the set of possibly accessed
 * elements by "prog".
 * If there are any member accesses involved, then they are first mapped
 * to the outer arrays of structs.
 *
 * If we are allowing live range reordering, then also set
 * the dep_order field.  Otherwise leave it NULL.
 */
static int collect_array_info(struct gpu_prog *prog)
{
	int i;
	int r = 0;
	isl_union_set *arrays;

	arrays = isl_union_map_range(isl_union_map_copy(prog->read));
	arrays = isl_union_set_union(arrays,
		    isl_union_map_range(isl_union_map_copy(prog->may_write)));

	arrays = isl_union_set_apply(arrays,
					isl_union_map_copy(prog->to_outer));

	arrays = isl_union_set_coalesce(arrays);

	prog->n_array = prog->scop->pet->n_array;
	prog->array = isl_calloc_array(prog->ctx,
				     struct gpu_array_info, prog->n_array);
	assert(prog->array);
	for (i = 0; i < prog->scop->pet->n_array; ++i)
		if (extract_array_info(prog, &prog->array[i],
					prog->scop->pet->arrays[i], arrays) < 0)
			r = -1;

	isl_union_set_free(arrays);

	if (prog->scop->options->live_range_reordering)
		collect_order_dependences(prog);

	return r;
}

static void free_array_info(struct gpu_prog *prog)
{
	int i, j;

	for (i = 0; i < prog->n_array; ++i) {
		int n_index = prog->array[i].n_index;
		free(prog->array[i].type);
		free(prog->array[i].name);
		for (j = 0; j < n_index; ++j)
			isl_pw_aff_free(prog->array[i].bound[j]);
		isl_space_free(prog->array[i].space);
		isl_set_free(prog->array[i].extent);
		free(prog->array[i].bound);
		free(prog->array[i].refs);
		isl_union_map_free(prog->array[i].dep_order);
	}
	free(prog->array);
}

/* Check if a gpu array is a scalar.  A scalar is a value that is not stored
 * as an array or through a pointer reference, but as a single data element.
 * At the moment, scalars are represented as zero-dimensional arrays.
 * Note that the single data element may be an entire structure.
 */
int gpu_array_is_scalar(struct gpu_array_info *array)
{
	return array->n_index == 0;
}

/* Is "array" a read-only scalar?
 */
int gpu_array_is_read_only_scalar(struct gpu_array_info *array)
{
	return array->read_only_scalar;
}

/* Does "array" need to be allocated on the device?
 * If it is a read-only scalar, then it will be passed as an argument
 * to the kernel and therefore does not require any allocation.
 * If this device memory is not accessed at all, then it does not
 * need to be allocated either.
 */
int gpu_array_requires_device_allocation(struct gpu_array_info *array)
{
	if (gpu_array_is_read_only_scalar(array))
		return 0;
	if (!array->global)
		return 0;
	return 1;
}

/* Return the set of parameter values for which the array has a positive
 * size in all dimensions.
 * If the sizes are only valid for some parameter values, then those
 * constraints are also taken into account.
 */
__isl_give isl_set *gpu_array_positive_size_guard(struct gpu_array_info *array)
{
	int i;
	isl_space *space;
	isl_set *guard;

	if (!array)
		return NULL;

	space = isl_space_params(isl_space_copy(array->space));
	guard = isl_set_universe(space);

	for (i = 0; i < array->n_index; ++i) {
		isl_pw_aff *bound;
		isl_set *guard_i, *zero;

		bound = isl_pw_aff_copy(array->bound[i]);
		guard_i = isl_pw_aff_nonneg_set(isl_pw_aff_copy(bound));
		zero = isl_pw_aff_zero_set(bound);
		guard_i = isl_set_subtract(guard_i, zero);
		guard = isl_set_intersect(guard, guard_i);
	}

	return guard;
}

/* Internal data structure for extract_size_of_type.
 * "type" specifies the name of the space that we want to extract.
 * "res" is used to store the subset of that space.
 */
struct ppcg_extract_size_data {
	const char *type;
	isl_set *res;
};

/* This function is called for each set in a union_set.
 * If the name of the set matches data->type, we store the
 * set in data->res.
 */
static isl_stat extract_size_of_type(__isl_take isl_set *size, void *user)
{
	struct ppcg_extract_size_data *data = user;
	const char *name;

	name = isl_set_get_tuple_name(size);
	if (name && !strcmp(name, data->type)) {
		data->res = size;
		return isl_stat_error;
	}

	isl_set_free(size);
	return isl_stat_ok;
}

/* Given a union map { kernel[i] -> *[...] },
 * return the range in the space called "type" for the kernel with
 * sequence number "id".
 */
static __isl_give isl_set *extract_sizes(__isl_keep isl_union_map *sizes,
	const char *type, int id)
{
	isl_space *space;
	isl_set *dom;
	isl_union_set *local_sizes;
	struct ppcg_extract_size_data data = { type, NULL };

	if (!sizes)
		return NULL;

	space = isl_union_map_get_space(sizes);
	space = isl_space_set_from_params(space);
	space = isl_space_add_dims(space, isl_dim_set, 1);
	space = isl_space_set_tuple_name(space, isl_dim_set, "kernel");
	dom = isl_set_universe(space);
	dom = isl_set_fix_si(dom, isl_dim_set, 0, id);

	local_sizes = isl_union_set_apply(isl_union_set_from_set(dom),
					isl_union_map_copy(sizes));
	isl_union_set_foreach_set(local_sizes, &extract_size_of_type, &data);
	isl_union_set_free(local_sizes);
	return data.res;
}

/* Given a singleton set, extract the first (at most *len) elements
 * of the single integer tuple into *sizes and update *len if needed.
 */
static void read_sizes_from_set(__isl_take isl_set *set, int *sizes, int *len)
{
	int i;
	int dim;

	if (!set)
		return;

	dim = isl_set_dim(set, isl_dim_set);
	if (dim < *len)
		*len = dim;

	for (i = 0; i < *len; ++i) {
		isl_val *v;

		v = isl_set_plain_get_val_if_fixed(set, isl_dim_set, i);
		assert(v);

		sizes[i] = isl_val_get_num_si(v);
		isl_val_free(v);
	}

	isl_set_free(set);
}

/* Add the map { kernel[id] -> type[sizes] } to gen->used_sizes,
 * if the option debug->dump_sizes is set.
 */
static void set_used_sizes(struct gpu_gen *gen, const char *type, int id,
	int *sizes, int len)
{
	int i;
	isl_space *space;
	isl_map *map;

	if (!gen->options->debug->dump_sizes)
		return;

	space = isl_union_map_get_space(gen->used_sizes);
	space = isl_space_set_from_params(space);
	space = isl_space_add_dims(space, isl_dim_set, 1);
	space = isl_space_set_tuple_name(space, isl_dim_set, "kernel");
	space = isl_space_from_domain(space);
	space = isl_space_add_dims(space, isl_dim_out, len);
	space = isl_space_set_tuple_name(space, isl_dim_out, type);

	map = isl_map_universe(space);
	map = isl_map_fix_si(map, isl_dim_in, 0, id);
	for (i = 0; i < len; ++i)
		map = isl_map_fix_si(map, isl_dim_out, i, sizes[i]);

	gen->used_sizes = isl_union_map_add_map(gen->used_sizes, map);
}

/* Extract user specified "tile" sizes from the "sizes" command line option,
 * defaulting to option->tile_size in each dimension.
 * *tile_len contains the maximum number of tile sizes needed.
 * Update *tile_len to the number of specified tile sizes, if any, and
 * return a pointer to the tile sizes (or NULL on error).
 * Add the effectively used sizes to gen->used_sizes.
 */
static int *read_tile_sizes(struct gpu_gen *gen, int *tile_len)
{
	int n;
	int *tile_size;
	isl_set *size;

	tile_size = isl_alloc_array(gen->ctx, int, *tile_len);
	if (!tile_size)
		return NULL;
	for (n = 0; n < *tile_len; ++n)
		tile_size[n] = gen->options->tile_size;

	size = extract_sizes(gen->sizes, "tile", gen->kernel_id);
	read_sizes_from_set(size, tile_size, tile_len);
	set_used_sizes(gen, "tile", gen->kernel_id, tile_size, *tile_len);

	return tile_size;
}

/* Extract user specified "block" sizes from the "sizes" command line option,
 * after filling in some potentially useful defaults.
 */
static void read_block_sizes(struct ppcg_kernel *kernel,
	__isl_keep isl_union_map *sizes)
{
	isl_set *size;

	if (kernel->n_block > 3)
		kernel->n_block = 3;
	switch (kernel->n_block) {
	case 1:
		kernel->block_dim[0] = 512;
		break;
	case 2:
		kernel->block_dim[0] = 32;
		kernel->block_dim[1] = 16;
		break;
	default:
		kernel->block_dim[0] = 32;
		kernel->block_dim[1] = 4;
		kernel->block_dim[2] = 4;
		break;
	}

	size = extract_sizes(sizes, "block", kernel->id);
	read_sizes_from_set(size, kernel->block_dim, &kernel->n_block);
}

/* Extract user specified "grid" sizes from the "sizes" command line option,
 * after filling in some potentially useful defaults.
 */
static void read_grid_sizes(struct ppcg_kernel *kernel,
	__isl_keep isl_union_map *sizes)
{
	isl_set *size;

	if (kernel->n_grid > 2)
		kernel->n_grid = 2;
	switch (kernel->n_grid) {
	case 1:
		kernel->grid_dim[0] = 32768;
		break;
	default:
		kernel->grid_dim[0] = 256;
		kernel->grid_dim[1] = 256;
		break;
	}

	size = extract_sizes(sizes, "grid", kernel->id);
	read_sizes_from_set(size, kernel->grid_dim, &kernel->n_grid);
}

/* Extract user specified grid and block sizes from the gen->sizes
 * command line option after filling in some potentially useful defaults.
 * Store the extracted sizes in "kernel".
 * Add the effectively used sizes to gen->used_sizes.
 */
static void read_grid_and_block_sizes(struct ppcg_kernel *kernel,
	struct gpu_gen *gen)
{
	read_block_sizes(kernel, gen->sizes);
	read_grid_sizes(kernel, gen->sizes);
	set_used_sizes(gen, "block", kernel->id,
					    kernel->block_dim, kernel->n_block);
	set_used_sizes(gen, "grid", kernel->id,
					    kernel->grid_dim, kernel->n_grid);
}

static void *free_stmts(struct gpu_stmt *stmts, int n)
{
	int i;

	if (!stmts)
		return NULL;

	for (i = 0; i < n; ++i) {
		struct gpu_stmt_access *access, *next;

		for (access = stmts[i].accesses; access; access = next) {
			next = access->next;
			isl_id_free(access->ref_id);
			isl_map_free(access->access);
			isl_map_free(access->tagged_access);
			free(access);
		}

		isl_id_free(stmts[i].id);
	}
	free(stmts);

	return NULL;
}

/* Add parameters p[i] with identifiers "ids" to "set",
 * with bounds to 0 <= p[i] < size[i].
 */
__isl_give isl_set *add_bounded_parameters(__isl_take isl_set *set,
	int *size, __isl_keep isl_id_list *ids)
{
	int i, len;
	unsigned nparam;

	len = isl_id_list_n_id(ids);
	nparam = isl_set_dim(set, isl_dim_param);
	set = isl_set_add_dims(set, isl_dim_param, len);

	for (i = 0; i < len; ++i) {
		isl_id *id;

		id = isl_id_list_get_id(ids, i);
		set = isl_set_set_dim_id(set, isl_dim_param, nparam + i, id);
		set = isl_set_lower_bound_si(set, isl_dim_param, nparam + i, 0);
		set = isl_set_upper_bound_si(set, isl_dim_param,
					    nparam + i, size[i] - 1);
	}

	return set;
}

/* Add "len" parameters p[i] with identifiers "ids" and intersect "set"
 * with
 *
 *	{ : 0 <= p[i] < size[i] }
 *
 * or an overapproximation.
 */
static __isl_give isl_set *add_bounded_parameters_dynamic(
	__isl_take isl_set *set, __isl_keep isl_multi_pw_aff *size,
	__isl_keep isl_id_list *ids)
{
	int i, len;
	unsigned nparam;
	isl_space *space;
	isl_local_space *ls;

	len = isl_multi_pw_aff_dim(size, isl_dim_out);
	nparam = isl_set_dim(set, isl_dim_param);
	set = isl_set_add_dims(set, isl_dim_param, len);

	for (i = 0; i < len; ++i) {
		isl_id *id;

		id = isl_id_list_get_id(ids, i);
		set = isl_set_set_dim_id(set, isl_dim_param, nparam + i, id);
	}

	space = isl_space_params(isl_set_get_space(set));
	ls = isl_local_space_from_space(space);
	for (i = 0; i < len; ++i) {
		isl_pw_aff *param, *size_i, *zero;
		isl_set *bound;

		param = isl_pw_aff_var_on_domain(isl_local_space_copy(ls),
						isl_dim_param, nparam + i);

		size_i = isl_multi_pw_aff_get_pw_aff(size, i);
		bound = isl_pw_aff_lt_set(isl_pw_aff_copy(param), size_i);
		bound = isl_set_from_basic_set(isl_set_simple_hull(bound));
		set = isl_set_intersect_params(set, bound);

		zero = isl_pw_aff_zero_on_domain(isl_local_space_copy(ls));
		bound = isl_pw_aff_ge_set(param, zero);
		set = isl_set_intersect_params(set, bound);
	}
	isl_local_space_free(ls);

	return set;
}

/* Return the union of all tagged access relations in the group.
 */
static __isl_give isl_union_map *group_tagged_access_relation(
	struct gpu_array_ref_group *group)
{
	int i;
	isl_union_map *access;

	access = isl_union_map_empty(isl_map_get_space(group->access));
	for (i = 0; i < group->n_ref; ++i) {
		isl_map *map_i;

		map_i = isl_map_copy(group->refs[i]->tagged_access);
		access = isl_union_map_union(access,
					    isl_union_map_from_map(map_i));
	}

	return access;
}

/* Return the extent of "array", recomputed from the bounds.
 * The recomputed extent may be simpler than the original extent.
 */
static __isl_give isl_set *array_extent(struct gpu_array_info *array)
{
	int i;
	isl_id *id;
	isl_space *space;
	isl_local_space *ls;
	isl_set *extent;

	id = isl_set_get_tuple_id(array->extent);
	space = isl_set_get_space(array->extent);
	extent = isl_set_universe(isl_space_copy(space));
	ls = isl_local_space_from_space(space);
	for (i = 0; i < array->n_index; ++i) {
		isl_pw_aff *bound;
		isl_aff *aff;
		isl_pw_aff *index;
		isl_set *lt;

		extent = isl_set_lower_bound_si(extent, isl_dim_set, i, 0);

		aff = isl_aff_var_on_domain(isl_local_space_copy(ls),
						isl_dim_set, i);
		index = isl_pw_aff_from_aff(aff);
		bound = isl_pw_aff_copy(array->bound[i]);
		bound = isl_pw_aff_from_range(bound);
		bound = isl_pw_aff_add_dims(bound, isl_dim_in, array->n_index);
		bound = isl_pw_aff_set_tuple_id(bound, isl_dim_in,
						isl_id_copy(id));
		lt = isl_pw_aff_lt_set(index, bound);
		extent = isl_set_intersect(extent, lt);
	}
	isl_local_space_free(ls);
	isl_id_free(id);

	return extent;
}

/* Return a map from the first group->depth dimensions of the computed
 * schedule to the array tile in
 * global memory that corresponds to the shared memory copy.
 *
 * In particular, return a map
 *
 *	{ D[i] -> A[a] }
 *
 * with constraints
 *
 *	tile_offset(i) <= a <= tile_offset(i) + tile_size - 1		(1)
 *
 * and
 *
 *	0 <= a <= array_size - 1					(2)
 *
 * Note that if some stride has been detected (i.e., when
 * group->shared_tile->bound[i].shift is set), then a in (1) refers
 * to the shifted and scaled down version.
 *
 * Constraints (1) are obtained by mapping the size constraints on the
 * shared/private memory tile back to the access relation.
 * Constraints (2) are obtained from the (recomputed) extent.
 */
static __isl_give isl_map *group_tile(struct gpu_array_ref_group *group)
{
	int i;
	int n_index = group->array->n_index;
	isl_map *tile;
	isl_space *space;
	isl_set *local;
	isl_set *extent;

	space = isl_multi_aff_get_space(group->shared_tile->tiling);
	space = isl_space_range(space);
	local = isl_set_universe(space);
	for (i = 0; i < n_index; ++i) {
		isl_val *bound;

		local = isl_set_lower_bound_si(local, isl_dim_set, i, 0);
		bound = isl_val_copy(group->shared_tile->bound[i].size);
		bound = isl_val_sub_ui(bound, 1);
		local = isl_set_upper_bound_val(local, isl_dim_set, i, bound);
	}
	local = isl_set_preimage_multi_aff(local,
				isl_multi_aff_copy(group->shared_tile->tiling));
	tile = isl_set_unwrap(local);
	extent = array_extent(group->array);
	tile = isl_map_intersect_range(tile, extent);

	return tile;
}

/* Given a mapping "iterator_map" from the AST schedule to a domain,
 * return the corresponding mapping from the AST schedule to
 * to the outer kernel->shared_schedule_dim dimensions of
 * the schedule computed by PPCG for this kernel.
 *
 * Note that kernel->shared_schedule_dim is at least as large as
 * the largest depth of any array reference group associated to the kernel.
 * This is needed as the returned schedule is used to extract a mapping
 * to the outer group->depth dimensions in transform_index.
 */
static __isl_give isl_pw_multi_aff *compute_sched_to_shared(
	struct ppcg_kernel *kernel, __isl_take isl_pw_multi_aff *iterator_map)
{
	isl_union_pw_multi_aff *upma;
	isl_pw_multi_aff *pma;
	isl_space *space;

	space = isl_space_range(isl_pw_multi_aff_get_space(iterator_map));
	space = isl_space_from_domain(space);
	space = isl_space_add_dims(space, isl_dim_out,
					kernel->shared_schedule_dim);

	upma = isl_union_pw_multi_aff_copy(kernel->shared_schedule);
	pma = isl_union_pw_multi_aff_extract_pw_multi_aff(upma, space);
	isl_union_pw_multi_aff_free(upma);

	return isl_pw_multi_aff_pullback_pw_multi_aff(pma, iterator_map);
}

/* If max_shared_memory is not set to infinity (-1), then make
 * sure that the total amount of shared memory required by the
 * array reference groups mapped to shared memory by "kernel"
 * is no larger than this maximum.
 *
 * We apply a greedy approach and discard (keep in global memory)
 * those groups that would result in a total memory size that
 * is larger than the maximum.
 *
 * This function should be called after any function that may
 * affect the decision on whether to place a reference group
 * in private, shared or global memory.
 */
static void check_shared_memory_bound(struct ppcg_kernel *kernel)
{
	int i, j;
	isl_val *left, *size;

	if (kernel->options->max_shared_memory < 0)
		return;

	left = isl_val_int_from_si(kernel->ctx,
				    kernel->options->max_shared_memory);

	for (i = 0; i < kernel->n_array; ++i) {
		struct gpu_local_array_info *local = &kernel->array[i];

		for (j = 0; j < local->n_group; ++j) {
			struct gpu_array_ref_group *group;

			group = local->groups[j];
			if (group->private_tile)
				continue;
			if (!group->shared_tile)
				continue;

			size = gpu_array_tile_size(group->shared_tile);
			size = isl_val_mul_ui(size, local->array->size);

			if (isl_val_le(size, left)) {
				left = isl_val_sub(left, size);
				continue;
			}
			isl_val_free(size);

			group->shared_tile =
					gpu_array_tile_free(group->shared_tile);
		}
	}

	isl_val_free(left);
}

/* Mark all arrays of "kernel" that have an array reference group
 * that is not mapped to private or shared memory as
 * accessing the corresponding global device memory.
 */
static void mark_global_arrays(struct ppcg_kernel *kernel)
{
	int i, j;

	for (i = 0; i < kernel->n_array; ++i) {
		struct gpu_local_array_info *local = &kernel->array[i];

		if (local->global)
			continue;
		for (j = 0; j < local->n_group; ++j) {
			if (gpu_array_ref_group_tile(local->groups[j]))
				continue;

			local->global = 1;
			local->array->global = 1;
			break;
		}
	}
}

/* Compute a tiling for all the array reference groups in "kernel".
 */
static void compute_group_tilings(struct ppcg_kernel *kernel)
{
	int i, j;

	for (i = 0; i < kernel->n_array; ++i) {
		struct gpu_local_array_info *array = &kernel->array[i];

		for (j = 0; j < array->n_group; ++j)
			gpu_array_ref_group_compute_tiling(array->groups[j]);
	}
}

/* Compute the size of a bounding box around the origin and "set",
 * where "set" is assumed to contain only non-negative elements.
 * In particular, compute the maximal value of "set" in each direction
 * and add one.
 */
static __isl_give isl_multi_pw_aff *extract_size(__isl_take isl_set *set,
	__isl_take isl_set *context)
{
	int i, n;
	isl_multi_pw_aff *mpa;

	context = isl_set_params(context);
	n = isl_set_dim(set, isl_dim_set);
	mpa = isl_multi_pw_aff_zero(isl_set_get_space(set));
	for (i = 0; i < n; ++i) {
		isl_space *space;
		isl_aff *one;
		isl_pw_aff *bound;

		bound = isl_set_dim_max(isl_set_copy(set), i);
		bound = isl_pw_aff_coalesce(bound);
		bound = isl_pw_aff_gist(bound, isl_set_copy(context));

		space = isl_pw_aff_get_domain_space(bound);
		one = isl_aff_zero_on_domain(isl_local_space_from_space(space));
		one = isl_aff_add_constant_si(one, 1);
		bound = isl_pw_aff_add(bound, isl_pw_aff_from_aff(one));
		mpa = isl_multi_pw_aff_set_pw_aff(mpa, i, bound);
	}
	isl_set_free(set);
	isl_set_free(context);

	return mpa;
}

/* Compute the effective grid size as a list of the sizes in each dimension.
 *
 * The grid size specified by the user or set by default
 * in read_grid_sizes() and applied by the block filter,
 * may be too large for the given code in the sense that
 * it may contain blocks that don't need to execute anything.
 * We therefore don't return this grid size, but instead the
 * smallest grid size that ensures that all blocks that actually
 * execute code are included in the grid.
 *
 * We first extract a description of the grid, i.e., the possible values
 * of the block ids, from the domain elements in "domain" and
 * kernel->block_filter.
 * The block ids are parameters in kernel->block_filter.
 * We simply need to change them into set dimensions.
 *
 * Then, for each block dimension, we compute the maximal value of the block id
 * and add one.
 */
static __isl_give isl_multi_pw_aff *extract_grid_size(
	struct ppcg_kernel *kernel, __isl_take isl_union_set *domain)
{
	int i;
	isl_set *grid;

	domain = isl_union_set_intersect(domain,
				    isl_union_set_copy(kernel->block_filter));
	grid = isl_union_set_params(domain);
	grid = isl_set_from_params(grid);
	grid = isl_set_add_dims(grid, isl_dim_set, kernel->n_grid);
	for (i = 0; i < kernel->n_grid; ++i) {
		int pos;
		isl_id *id;

		id = isl_id_list_get_id(kernel->block_ids, i);
		pos = isl_set_find_dim_by_id(grid, isl_dim_param, id);
		isl_id_free(id);
		assert(pos >= 0);
		grid = isl_set_equate(grid, isl_dim_param, pos, isl_dim_set, i);
		grid = isl_set_project_out(grid, isl_dim_param, pos, 1);
	}

	return extract_size(grid, isl_set_copy(kernel->context));
}

/* Compute the size of a fixed bounding box around the origin and "set",
 * where "set" is assumed to contain only non-negative elements,
 * and store the results in "size".
 * In particular, compute the maximal value of "set" in each direction
 * and add one.
 */
static void extract_fixed_size(__isl_take isl_set *set, int *size)
{
	int i, n;
	isl_local_space *ls;
	isl_aff *obj;

	n = isl_set_dim(set, isl_dim_set);
	ls = isl_local_space_from_space(isl_set_get_space(set));
	obj = isl_aff_zero_on_domain(ls);
	for (i = 0; i < n; ++i) {
		isl_val *max;

		obj = isl_aff_set_coefficient_si(obj, isl_dim_in, i, 1);
		max = isl_set_max_val(set, obj);
		size[i] = isl_val_get_num_si(max) + 1;
		isl_val_free(max);
		obj = isl_aff_set_coefficient_si(obj, isl_dim_in, i, 0);
	}
	isl_aff_free(obj);
	isl_set_free(set);
}

/* Compute the effective block size as a list of the sizes in each dimension
 * and store the sizes in kernel->block_dim.
 *
 * The block size specified by the user or set by default
 * in read_block_sizes() and applied by the thread filter,
 * may be too large for the given code in the sense that
 * it may contain threads that don't need to execute anything.
 * We therefore update this block size in kernel->block_dim
 * to the smallest block size that ensures that all threads
 * that actually execute code are included in the block.
 *
 * The possible values of the thread ids is obtained from
 * the domain elements "domain" and kernel->thread_filter.
 * The current implementation eliminates all parameters, ensuring
 * that the size is a fixed constant in each dimension.
 * In principle we could also compute parametric sizes.
 * We would have to make sure to project out all b%d and t%d parameters,
 * however.
 */
static void extract_block_size(struct ppcg_kernel *kernel,
	__isl_take isl_union_set *domain)
{
	int i;
	int nparam;
	isl_set *block;

	domain = isl_union_set_intersect(domain,
				    isl_union_set_copy(kernel->thread_filter));
	block = isl_union_set_params(domain);
	block = isl_set_from_params(block);
	block = isl_set_add_dims(block, isl_dim_set, kernel->n_block);
	for (i = 0; i < kernel->n_block; ++i) {
		int pos;
		isl_id *id;

		id = isl_id_list_get_id(kernel->thread_ids, i);
		pos = isl_set_find_dim_by_id(block, isl_dim_param, id);
		isl_id_free(id);
		assert(pos >= 0);
		block = isl_set_equate(block, isl_dim_param, pos,
					isl_dim_set, i);
	}
	nparam = isl_set_dim(block, isl_dim_param);
	block = isl_set_project_out(block, isl_dim_param, 0, nparam);

	extract_fixed_size(block, kernel->block_dim);
}

struct ppcg_kernel *ppcg_kernel_free(struct ppcg_kernel *kernel)
{
	int i, j;

	if (!kernel)
		return NULL;

	isl_id_list_free(kernel->block_ids);
	isl_id_list_free(kernel->thread_ids);
	isl_multi_pw_aff_free(kernel->grid_size);
	isl_set_free(kernel->context);
	isl_union_set_free(kernel->core);
	isl_union_set_free(kernel->arrays);
	isl_space_free(kernel->space);
	isl_ast_node_free(kernel->tree);
	isl_union_set_free(kernel->block_filter);
	isl_union_set_free(kernel->thread_filter);
	isl_union_pw_multi_aff_free(kernel->shared_schedule);
	isl_union_set_free(kernel->sync_writes);

	for (i = 0; i < kernel->n_array; ++i) {
		struct gpu_local_array_info *array = &kernel->array[i];

		for (j = 0; j < array->n_group; ++j)
			gpu_array_ref_group_free(array->groups[j]);
		free(array->groups);

		isl_pw_aff_list_free(array->bound);
	}
	free(kernel->array);

	for (i = 0; i < kernel->n_var; ++i) {
		free(kernel->var[i].name);
		isl_vec_free(kernel->var[i].size);
	}
	free(kernel->var);

	free(kernel);

	return NULL;
}

/* Wrapper around ppcg_kernel_free for use as a isl_id_set_free_user callback.
 */
static void ppcg_kernel_free_wrap(void *user)
{
	struct ppcg_kernel *kernel = user;

	ppcg_kernel_free(kernel);
}

static void create_kernel_var(isl_ctx *ctx, struct gpu_array_ref_group *group,
	struct ppcg_kernel_var *var)
{
	int j;
	struct gpu_array_tile *tile;
	isl_printer *p;
	char *name;

	var->array = group->array;

	tile = group->private_tile;
	var->type = ppcg_access_private;
	if (!tile) {
		tile = group->shared_tile;
		var->type = ppcg_access_shared;
	}

	p = isl_printer_to_str(ctx);
	p = gpu_array_ref_group_print_name(group, p);
	var->name = isl_printer_get_str(p);
	isl_printer_free(p);

	var->size = isl_vec_alloc(ctx, group->array->n_index);

	for (j = 0; j < group->array->n_index; ++j)
		var->size = isl_vec_set_element_val(var->size, j,
					    isl_val_copy(tile->bound[j].size));
}

static int create_kernel_vars(struct ppcg_kernel *kernel)
{
	int i, j, n;

	n = 0;
	for (i = 0; i < kernel->n_array; ++i) {
		struct gpu_local_array_info *array = &kernel->array[i];

		for (j = 0; j < array->n_group; ++j) {
			struct gpu_array_ref_group *group = array->groups[j];
			if (group->private_tile || group->shared_tile)
				++n;
		}
	}

	kernel->n_var = n;
	kernel->var = isl_calloc_array(kernel->ctx, struct ppcg_kernel_var, n);
	if (!kernel->var)
		return -1;

	n = 0;
	for (i = 0; i < kernel->n_array; ++i) {
		struct gpu_local_array_info *array = &kernel->array[i];

		for (j = 0; j < array->n_group; ++j) {
			struct gpu_array_ref_group *group = array->groups[j];
			if (!group->private_tile && !group->shared_tile)
				continue;
			create_kernel_var(kernel->ctx, group, &kernel->var[n]);
			++n;
		}
	}

	return 0;
}

/* Replace "pa" by the zero function defined over the universe domain
 * in the space of "pa".
 */
static __isl_give isl_pw_aff *set_universally_zero(__isl_take isl_pw_aff *pa)
{
	isl_space *space;
	isl_aff *zero;

	space = isl_space_domain(isl_pw_aff_get_space(pa));
	isl_pw_aff_free(pa);
	zero = isl_aff_zero_on_domain(isl_local_space_from_space(space));

	return isl_pw_aff_from_aff(zero);
}

/* The sizes of the arrays on the host that have been computed by
 * extract_array_info may depend on the parameters.  Use the extra
 * constraints on the parameters that are valid at "host_domain"
 * to simplify these expressions and store the results in kernel->array.
 *
 * We only need these localized bounds for arrays that are accessed
 * by the current kernel.  If we have found at least one reference group
 * then the array is accessed by the kernel.
 *
 * The resulting sizes may be functions that are nowhere defined
 * in case the access function cannot possibly access anything inside
 * the kernel for some reason.  If so, they are replaced by the zero
 * function.  Since the access function cannot actually access anything,
 * there is no harm in printing the array sizes as zero.
 */
static void localize_bounds(struct ppcg_kernel *kernel,
	__isl_keep isl_set *host_domain)
{
	int i, j;
	isl_set *context;

	context = isl_set_copy(host_domain);
	context = isl_set_params(context);

	for (i = 0; i < kernel->n_array; ++i) {
		struct gpu_local_array_info *local = &kernel->array[i];
		isl_pw_aff_list *bound;
		int n_index;

		if (local->n_group == 0)
			continue;

		n_index = local->array->n_index;
		bound = isl_pw_aff_list_alloc(kernel->ctx, n_index);

		for (j = 0; j < n_index; ++j) {
			isl_pw_aff *pwaff;
			int empty;

			pwaff = isl_pw_aff_copy(local->array->bound[j]);
			pwaff = isl_pw_aff_gist(pwaff, isl_set_copy(context));
			empty = isl_pw_aff_is_empty(pwaff);
			if (empty < 0)
				pwaff = isl_pw_aff_free(pwaff);
			else if (empty)
				pwaff = set_universally_zero(pwaff);
			bound = isl_pw_aff_list_add(bound, pwaff);
		}

		local->n_index = n_index;
		local->bound = bound;
	}
	isl_set_free(context);
}

/* Create the array of gpu_local_array_info structures "array"
 * inside "kernel".  The number of elements in this array is
 * the same as the number of arrays in "prog".
 * Initialize the "array" field of each local array to point
 * to the corresponding array in "prog".
 */
static struct ppcg_kernel *ppcg_kernel_create_local_arrays(
	struct ppcg_kernel *kernel, struct gpu_prog *prog)
{
	int i;
	isl_ctx *ctx;

	ctx = isl_set_get_ctx(prog->context);
	kernel->array = isl_calloc_array(ctx,
			    struct gpu_local_array_info, prog->n_array);
	if (!kernel->array)
		return ppcg_kernel_free(kernel);
	kernel->n_array = prog->n_array;

	for (i = 0; i < prog->n_array; ++i)
		kernel->array[i].array = &prog->array[i];

	return kernel;
}

/* Does "kernel" need to be passed an argument corresponding to array "i"?
 *
 * The argument is only needed if the kernel accesses this device memory.
 */
int ppcg_kernel_requires_array_argument(struct ppcg_kernel *kernel, int i)
{
	return kernel->array[i].global;
}

/* Find the element in gen->stmt that has the given "id".
 * Return NULL if no such gpu_stmt can be found.
 */
static struct gpu_stmt *find_stmt(struct gpu_prog *prog, __isl_keep isl_id *id)
{
	int i;

	for (i = 0; i < prog->n_stmts; ++i) {
		if (id == prog->stmts[i].id)
			break;
	}

	return i < prog->n_stmts ? &prog->stmts[i] : NULL;
}

void ppcg_kernel_stmt_free(void *user)
{
	int i;
	struct ppcg_kernel_stmt *stmt = user;

	if (!stmt)
		return;

	switch (stmt->type) {
	case ppcg_kernel_copy:
		isl_ast_expr_free(stmt->u.c.index);
		isl_ast_expr_free(stmt->u.c.local_index);
		break;
	case ppcg_kernel_domain:
		isl_id_to_ast_expr_free(stmt->u.d.ref2expr);
		break;
	case ppcg_kernel_sync:
		break;
	}

	free(stmt);
}

/* Return the gpu_stmt_access in the list "accesses" that corresponds
 * to "ref_id".
 */
static struct gpu_stmt_access *find_access(struct gpu_stmt_access *accesses,
	__isl_keep isl_id *ref_id)
{
	struct gpu_stmt_access *access;

	for (access = accesses; access; access = access->next)
		if (access->ref_id == ref_id)
			return access;

	return NULL;
}

/* Return the index of the array called "name" in the list of arrays.
 */
static int find_array_index(struct ppcg_kernel *kernel, const char *name)
{
	int i;

	for (i = 0; i < kernel->n_array; ++i)
		if (!strcmp(name, kernel->array[i].array->name))
			return i;

	return -1;
}

/* Internal data structure for the index and AST expression transformation
 * callbacks for pet_stmt_build_ast_exprs.
 *
 * "kernel" is the kernel for which are computing AST expressions and
 * may be NULL if we are not inside a kernel.
 * "accesses" is the list of gpu_stmt_access in the statement.
 * "iterator_map" expresses the statement iterators in terms of
 * the AST loop iterators.
 * "sched2shared" expresses the outer shared_schedule_dim dimensions of
 * the kernel schedule in terms of the AST loop iterators and
 * may be NULL if we are not inside a kernel.
 *
 * The following fields are set in transform_index and used in transform_expr.
 * "array" is the array that is being accessed.
 * "global" is set if the global array is accessed (rather than
 * shared/private memory).
 * "local_array" refers to information on the array specialized
 * to the current kernel.
 */
struct ppcg_transform_data {
        struct ppcg_options *options;
	struct ppcg_kernel *kernel;
	struct gpu_stmt_access *accesses;
	isl_pw_multi_aff *iterator_map;
	isl_pw_multi_aff *sched2shared;

	struct gpu_array_info *array;
	int global;
	struct gpu_local_array_info *local_array;
};

/* Return a pointer to the gpu_array_ref_group in "local"
 * that contains the reference "access".
 * Return NULL if no such group can be found.
 */
static struct gpu_array_ref_group *find_ref_group(
	struct gpu_local_array_info *local, struct gpu_stmt_access *access)
{
	int i, j;

	for (i = 0; i < local->n_group; ++i) {
		struct gpu_array_ref_group *group = local->groups[i];

		for (j = 0; j < group->n_ref; ++j)
			if (group->refs[j] == access)
				return group;
	}

	return NULL;
}

/* Index transformation callback for pet_stmt_build_ast_exprs.
 *
 * "index" expresses the array indices in terms of statement iterators
 *
 * We first reformulate "index" in terms of the AST loop iterators.
 * Then we check if we are accessing the global array or
 * a shared/private copy.  In particular, if we are not inside a kernel
 * then we must be accessing a global array.
 * In the former case, we simply return
 * the updated index.  If "index" is an affine expression rather
 * than an array access, then we also return the updated index here.
 *
 * If no reference groups have been computed for the array,
 * then we can only be accessing the global array.
 *
 * Otherwise, we apply the tiling to the index.
 * This tiling is of the form
 *
 *	[D -> A] -> T
 *
 * where D corresponds to the outer group->depth dimensions of
 * the kernel schedule.
 * The index is of the form
 *
 *	L -> A
 *
 * We update the tiling to refer to the AST loop iterators
 *
 *	[L -> A] -> T
 *
 * and modify index to keep track of those iterators
 *
 *	L -> [L -> A]
 *
 * Combining these two yields a tiled index expression in terms
 * of the AST loop iterators
 *
 *	L -> T
 */
static __isl_give isl_multi_pw_aff *transform_index(
	__isl_take isl_multi_pw_aff *index, __isl_keep isl_id *ref_id,
	void *user)
{
	struct ppcg_transform_data *data = user;
	struct gpu_stmt_access *access;
	struct gpu_array_ref_group *group;
	struct gpu_array_tile *tile;
	isl_pw_multi_aff *iterator_map;
	int i;
	int dim;
	const char *name;
	isl_space *space;
	isl_multi_pw_aff *tiling;
	isl_pw_multi_aff *pma;
	isl_multi_pw_aff *mpa;
	isl_pw_multi_aff *sched2depth;

	data->array = NULL;

	iterator_map = isl_pw_multi_aff_copy(data->iterator_map);
	index = isl_multi_pw_aff_pullback_pw_multi_aff(index, iterator_map);

	if (!data->kernel)
		return index;

	access = find_access(data->accesses, ref_id);
	if (!access)
		return index;
	if (!isl_map_has_tuple_name(access->access, isl_dim_out))
		return index;

	name = get_outer_array_name(access->access);
	i = find_array_index(data->kernel, name);
	if (i < 0)
		isl_die(isl_multi_pw_aff_get_ctx(index), isl_error_internal,
			"cannot find array",
			return isl_multi_pw_aff_free(index));
	data->local_array = &data->kernel->array[i];
	data->array = data->local_array->array;

	group = find_ref_group(data->local_array, access);
	if (!group) {
		data->global = 1;
		return index;
	}

	tile = group->private_tile;
	if (!tile)
		tile = group->shared_tile;
	data->global = !tile;
	if (!tile)
		return index;

	space = isl_space_range(isl_multi_pw_aff_get_space(index));
	space = isl_space_map_from_set(space);
	pma = isl_pw_multi_aff_identity(space);
	sched2depth = isl_pw_multi_aff_copy(data->sched2shared);
	dim = isl_pw_multi_aff_dim(sched2depth, isl_dim_out);
	sched2depth = isl_pw_multi_aff_drop_dims(sched2depth, isl_dim_out,
					    group->depth, dim - group->depth);
	pma = isl_pw_multi_aff_product(sched2depth, pma);
	tiling = isl_multi_pw_aff_from_multi_aff(
				    isl_multi_aff_copy(tile->tiling));
	tiling = isl_multi_pw_aff_pullback_pw_multi_aff(tiling, pma);

	space = isl_space_domain(isl_multi_pw_aff_get_space(index));
	space = isl_space_map_from_set(space);
	mpa = isl_multi_pw_aff_identity(space);
	index = isl_multi_pw_aff_range_product(mpa, index);
	index = isl_multi_pw_aff_pullback_multi_pw_aff(tiling, index);

	return index;
}

/* Dereference "expr" by adding an index [0].
 * The original "expr" is assumed not to have any indices.
 *
 * If "expr" is a member access, then the dereferencing needs
 * to be applied to the structure argument of this member access.
 */
static __isl_give isl_ast_expr *dereference(__isl_take isl_ast_expr *expr)
{
	isl_ctx *ctx;
	isl_ast_expr *arg0, *res;
	isl_ast_expr_list *list;

	arg0 = isl_ast_expr_get_op_arg(expr, 0);
	if (!arg0)
		return isl_ast_expr_free(expr);
	if (isl_ast_expr_get_type(arg0) == isl_ast_expr_op &&
	    isl_ast_expr_get_op_type(arg0) == isl_ast_op_member) {
		isl_ast_expr *arg;

		arg = isl_ast_expr_get_op_arg(arg0, 0);
		arg = dereference(arg);
		arg0 = isl_ast_expr_set_op_arg(arg0, 0, arg);
		expr = isl_ast_expr_set_op_arg(expr, 0, arg0);

		return expr;
	}
	isl_ast_expr_free(arg0);

	ctx = isl_ast_expr_get_ctx(expr);
	res = isl_ast_expr_from_val(isl_val_zero(ctx));
	list = isl_ast_expr_list_from_ast_expr(res);
	res = isl_ast_expr_get_op_arg(expr, 0);
	res = isl_ast_expr_access(res, list);
	isl_ast_expr_free(expr);

	return res;
}

/* Linearize the index expression "expr" based on the array bounds
 * of "array".
 *
 * That is, transform expression
 *
 *	A[i_0][i_1]...[i_n]
 *
 * to
 *
 *	A[(..((i_0 * b_1 + i_1) ... ) * b_n + i_n]
 *
 * where b_0, b_1, ..., b_n are the bounds on the array.
 *
 * If the base of "expr" is a member access, then the linearization needs
 * to be applied to the structure argument of this member access.
 *
 * In the base case, if "expr" has no arguments (other than the name of
 * the array), then we are passing an entire array to a function.
 * In this case, there is nothing to linearize.
 * Note that at this point an expression with no arguments can
 * only be an entire array because the scalar case and
 * the case of single struct are handled by the caller.
 *
 * If the number of specified index expressions in "expr"
 * is smaller than the dimension of the accessed array,
 * then the missing i_j also do not appear in the linearized expression.
 * Furthermore, since such an expression does not refer to a single
 * element while the default linearized expression would refer to
 * a single element, we return the expression
 *
 *	A + (..((i_0 * b_1 + i_1) ... ) * b_n]
 *
 * instead.  Note that because of the special case handling above,
 * we can assume here that here that there is at least one index expression.
 */
__isl_give isl_ast_expr *gpu_local_array_info_linearize_index(
	struct gpu_local_array_info *array, __isl_take isl_ast_expr *expr)
{
	int i, n;
	isl_ctx *ctx;
	isl_set *context;
	isl_ast_expr *arg0;
	isl_ast_expr *res;
	isl_ast_expr_list *list;
	isl_ast_build *build;

	arg0 = isl_ast_expr_get_op_arg(expr, 0);
	if (isl_ast_expr_get_type(arg0) == isl_ast_expr_op &&
	    isl_ast_expr_get_op_type(arg0) == isl_ast_op_member) {
		isl_ast_expr *arg;

		arg = isl_ast_expr_get_op_arg(arg0, 0);
		arg = gpu_local_array_info_linearize_index(array, arg);
		arg0 = isl_ast_expr_set_op_arg(arg0, 0, arg);
		expr = isl_ast_expr_set_op_arg(expr, 0, arg0);

		return expr;
	}
	isl_ast_expr_free(arg0);

	if (isl_ast_expr_get_op_n_arg(expr) == 1)
		return expr;

	ctx = isl_ast_expr_get_ctx(expr);
	context = isl_set_universe(isl_space_params_alloc(ctx, 0));
	build = isl_ast_build_from_context(context);

	n = isl_ast_expr_get_op_n_arg(expr);
	res = isl_ast_expr_get_op_arg(expr, 1);
	for (i = 1; i < array->n_index; ++i) {
		isl_pw_aff *bound_i;
		isl_ast_expr *expr_i;

		bound_i = isl_pw_aff_list_get_pw_aff(array->bound, i);
		expr_i = isl_ast_build_expr_from_pw_aff(build, bound_i);
		res = isl_ast_expr_mul(res, expr_i);

		if (i + 1 >= n)
			continue;
		expr_i = isl_ast_expr_get_op_arg(expr, i + 1);
		res = isl_ast_expr_add(res, expr_i);
	}

	isl_ast_build_free(build);

	if (1 + array->n_index > n) {
		res = isl_ast_expr_add(isl_ast_expr_get_op_arg(expr, 0), res);
	} else {
		list = isl_ast_expr_list_from_ast_expr(res);
		res = isl_ast_expr_get_op_arg(expr, 0);
		res = isl_ast_expr_access(res, list);
	}

	isl_ast_expr_free(expr);

	return res;
}

/* AST expression transformation callback for pet_stmt_build_ast_exprs.
 *
 * If the AST expression refers to an array that is not accessed
 * at all, then this means the value of the expression is not used,
 * so we might as well print zero (NULL pointer) instead.
 *
 * If the AST expression refers to a global scalar that is not
 * a read-only scalar, then its address was passed to the kernel and
 * we need to dereference it.
 *
 * If the AST expression refers to an access to a global array,
 * then we linearize the access exploiting the bounds in data->local_array.
 */
static __isl_give isl_ast_expr *transform_expr(__isl_take isl_ast_expr *expr,
	__isl_keep isl_id *id, void *user)
{
	struct ppcg_transform_data *data = user;

	if (!data->array)
		return expr;
	if (!data->array->accessed) {
		isl_ctx *ctx;

		ctx = isl_ast_expr_get_ctx(expr);
		isl_ast_expr_free(expr);
		return isl_ast_expr_from_val(isl_val_zero(ctx));
	}
	if (gpu_array_is_read_only_scalar(data->array))
		return expr;
	if (!data->global)
		return expr;
	if (data->array->n_index == 0)
		return dereference(expr);
	if (!data->array->linearize)
		return expr;

	return gpu_local_array_info_linearize_index(data->local_array, expr);
}

/* This function is called for each instance of a user statement
 * in the kernel "kernel", identified by "gpu_stmt".
 * "kernel" may be NULL if we are not inside a kernel.
 *
 * We attach a struct ppcg_kernel_stmt to the "node", containing
 * a computed AST expression for each access, through an annotation
 * with name "user".
 * These AST expressions are computed from iterator_map,
 * which expresses the domain
 * elements in terms of the generated loops, and sched2shared,
 * which expresses the outer shared_schedule_dim dimensions of
 * the kernel schedule computed by PPCG in terms of the generated loops.
 */
static __isl_give isl_ast_node *create_domain_leaf(
	struct ppcg_kernel *kernel, __isl_take isl_ast_node *node,
	__isl_keep isl_ast_build *build, struct gpu_stmt *gpu_stmt,
        struct gpu_gen *gen)
{
	struct ppcg_transform_data data;
	struct ppcg_kernel_stmt *stmt;
	isl_ctx *ctx;
	isl_id *id;
	isl_pw_multi_aff *sched2shared;
	isl_map *map;
	isl_pw_multi_aff *iterator_map;
	isl_union_map *schedule;

	if (!node)
		return NULL;
	ctx = isl_ast_node_get_ctx(node);

	stmt = isl_calloc_type(ctx, struct ppcg_kernel_stmt);
	if (!stmt)
		return isl_ast_node_free(node);

	schedule = isl_ast_build_get_schedule(build);
	map = isl_map_reverse(isl_map_from_union_map(schedule));
	iterator_map = isl_pw_multi_aff_from_map(map);
	if (kernel)
		sched2shared = compute_sched_to_shared(kernel,
					isl_pw_multi_aff_copy(iterator_map));
	else
		sched2shared = NULL;

	stmt->type = ppcg_kernel_domain;
	stmt->u.d.stmt = gpu_stmt;

	data.kernel = kernel;
	data.accesses = stmt->u.d.stmt->accesses;
	data.iterator_map = iterator_map;
	data.sched2shared = sched2shared;
	stmt->u.d.ref2expr = gen->build_ast_expr(stmt->u.d.stmt->stmt,
					    build, &transform_index, &data,
					    &transform_expr, &data);
	isl_pw_multi_aff_free(iterator_map);
	isl_pw_multi_aff_free(sched2shared);

	id = isl_id_alloc(ctx, "user", stmt);
	id = isl_id_set_free_user(id, &ppcg_kernel_stmt_free);
	return isl_ast_node_set_annotation(node, id);
}

/* This function is called for each statement node in the AST
 * for copying to or from shared/private memory.
 * Attach a pointer to a ppcg_kernel_stmt representing the copy
 * statement to the node.
 * The statement name is "read" or "write", depending on whether we are
 * reading from global memory or writing to global memory.
 *
 * The schedule is of the form
 *
 *	type[D -> A] -> L
 *
 * where D corresponds to the outer group->depth dimensions of
 * the kernel schedule, A to the global array and L to the outer
 * generated AST schedule.
 * We compute the inverse and strip off the type, resulting in
 *
 *	L -> [D -> A]
 *
 * We combine this mapping with on the one hand the projection
 *
 *	[D -> A] -> A
 *
 * and on the other hand the group tiling
 *
 *	[D -> A] -> T
 *
 * resulting in
 *
 *	L -> A		and 	L -> T
 *
 * and store the corresponding expressions in stmt->index and stmt->local_index,
 * where stmt points to the ppcg_kernel_stmt that is attached to the node.
 */
static __isl_give isl_ast_node *create_access_leaf(struct ppcg_kernel *kernel,
	struct gpu_array_ref_group *group, __isl_take isl_ast_node *node,
	__isl_keep isl_ast_build *build)
{
	struct ppcg_kernel_stmt *stmt;
	struct gpu_array_tile *tile;
	isl_id *id;
	isl_ast_expr *expr;
	isl_space *space;
	isl_map *access;
	isl_pw_multi_aff *pma, *pma2;
	const char *type;

	stmt = isl_calloc_type(kernel->ctx, struct ppcg_kernel_stmt);
	if (!stmt)
		return isl_ast_node_free(node);

	access = isl_map_from_union_map(isl_ast_build_get_schedule(build));
	type = isl_map_get_tuple_name(access, isl_dim_in);
	stmt->u.c.read = !strcmp(type, "read");
	access = isl_map_reverse(access);
	pma = isl_pw_multi_aff_from_map(access);
	pma = isl_pw_multi_aff_reset_tuple_id(pma, isl_dim_out);

	space = isl_space_range(isl_pw_multi_aff_get_space(pma));
	space = isl_space_unwrap(space);
	pma2 = isl_pw_multi_aff_range_map(space);
	pma2 = isl_pw_multi_aff_pullback_pw_multi_aff(pma2,
						    isl_pw_multi_aff_copy(pma));
	expr = isl_ast_build_access_from_pw_multi_aff(build, pma2);
	stmt->u.c.index = expr;

	tile = gpu_array_ref_group_tile(group);
	pma2 = isl_pw_multi_aff_from_multi_aff(
					    isl_multi_aff_copy(tile->tiling));
	pma2 = isl_pw_multi_aff_pullback_pw_multi_aff(pma2, pma);
	expr = isl_ast_build_access_from_pw_multi_aff(build, pma2);
	stmt->u.c.local_index = expr;

	stmt->u.c.array = group->array;
	stmt->u.c.local_array = group->local_array;
	stmt->type = ppcg_kernel_copy;

	id = isl_id_alloc(kernel->ctx, NULL, stmt);
	id = isl_id_set_free_user(id, &ppcg_kernel_stmt_free);
	return isl_ast_node_set_annotation(node, id);
}

/* Create a synchronization ppcg_kernel_stmt and
 * attach it to the node "node" representing the synchronization.
 */
static __isl_give isl_ast_node *create_sync_leaf(
	struct ppcg_kernel *kernel, __isl_take isl_ast_node *node,
	__isl_keep isl_ast_build *build)
{
	struct ppcg_kernel_stmt *stmt;
	isl_id *id;

	stmt = isl_calloc_type(kernel->ctx, struct ppcg_kernel_stmt);
	if (!stmt)
		return isl_ast_node_free(node);

	stmt->type = ppcg_kernel_sync;
	id = isl_id_alloc(kernel->ctx, NULL, stmt);
	id = isl_id_set_free_user(id, &ppcg_kernel_stmt_free);
	return isl_ast_node_set_annotation(node, id);
}

/* Internal data structure for at_domain.
 *
 * "prog" represents the entire scop.
 * "kernel" points to the kernel to which the current schedule node
 * belongs.  It is set by before_mark and reset by after_mark.
 * It may be NULL if we are outside any kernel.
 */
struct ppcg_at_domain_data {
	struct gpu_prog *prog;
	struct gpu_gen *gen;
	struct ppcg_kernel *kernel;
};

/* This function is called for each instance of a user statement
 * in the kernel.  This may be one of the original user statements
 * or a statement introduced by PPCG.
 *
 * We first check if the statement id corresponds to a gpu statement,
 * which indicates the statement is an original user statement. Any statement
 * that is not an original user statement has been introduced by PPCG and
 * requires special handling.
 *
 * If the user statement is one of the original user statements, then we call
 * create_domain_leaf.  Otherwise, we check if it is a copy or synchronization
 * statement and call the appropriate functions.  Statements that copy an array
 * to/from the device do not need any further treatment.
 */
static __isl_give isl_ast_node *at_domain(__isl_take isl_ast_node *node,
	__isl_keep isl_ast_build *build, void *user)
{
	struct ppcg_at_domain_data *data = user;
	struct gpu_stmt *gpu_stmt;
	isl_ast_expr *expr, *arg;
	isl_id *id;
	int is_sync;
	const char *name;
	void *p;

	expr = isl_ast_node_user_get_expr(node);
	arg = isl_ast_expr_get_op_arg(expr, 0);
	id = isl_ast_expr_get_id(arg);
	name = isl_id_get_name(id);
	p = isl_id_get_user(id);
	isl_ast_expr_free(expr);
	isl_ast_expr_free(arg);

	gpu_stmt = find_stmt(data->prog, id);
	is_sync = gpu_tree_id_is_sync(id, data->kernel);
	isl_id_free(id);

	if (gpu_stmt)
		return create_domain_leaf(data->kernel, node, build, gpu_stmt,
                                          data->gen);

	if (!prefixcmp(name, "to_device_") || !prefixcmp(name, "from_device_"))
		return node;
	if (is_sync < 0)
		return isl_ast_node_free(node);
	if (!strcmp(name, "read") || !strcmp(name, "write")) {
		struct gpu_array_ref_group *group = p;
		return create_access_leaf(data->kernel, group, node, build);
	}
	if (!is_sync)
		isl_die(data->prog->ctx, isl_error_internal,
			"unknown statement type",
			return isl_ast_node_free(node));
	return create_sync_leaf(data->kernel, node, build);
}

/* Given a set of wrapped references "ref", return the corresponding
 * access relations based on the tagged access relations "tagged".
 *
 * The elements of "ref" are of the form
 *
 *	[D -> R]
 *
 * with D an iteration domains and R a reference.
 * The elements of "tagged" are of the form
 *
 *	[D -> R] -> A
 *
 * with A an array.
 *
 * Extend "tagged" to include the iteration domain in the range, i.e.,
 *
 *	[D -> R] -> [D -> A]
 *
 * apply the result to "ref" and then unwrap the resulting set
 * to obtain relations of the form
 *
 *	D -> A
 */
static __isl_give isl_union_map *wrapped_reference_to_access(
	__isl_take isl_union_set *ref, __isl_take isl_union_map *tagged)
{
	isl_union_map *tag2access;

	tag2access = isl_union_map_copy(tagged);
	tag2access = isl_union_map_universe(tag2access);
	tag2access = isl_union_set_unwrap(isl_union_map_domain(tag2access));
	tag2access = isl_union_map_domain_map(tag2access);
	tag2access = isl_union_map_range_product(tag2access, tagged);

	ref = isl_union_set_coalesce(ref);
	ref = isl_union_set_apply(ref, tag2access);

	return isl_union_set_unwrap(ref);
}

/* Given an access relation "access" from one or more array reference groups,
 * remove those reads if ("read" is 1) or writes (if "read" is 0)
 * that are only needed to communicate data within
 * the same iteration of "sched".
 * "tagged" contains all tagged access relations to all
 * the array reference groups accessed by "access" from statement
 * instances scheduled by "sched".
 *
 * If the access is a read then it is either an element of
 *
 *	live_in union (range flow)
 *
 * where live_in and flow may be overapproximations, or
 * it reads an uninitialized value (that is not live-in because
 * there is an intermediate kill) or it reads a value that was
 * written within the same (compound) statement instance.
 * If the access is a write then it is either an element of
 *
 *	live_out union (domain flow)
 *
 * or it writes a value that is never read (and is not live-out
 * because of an intermediate kill) or only
 * within the same (compound) statement instance.
 * In both cases, the access relation is also a subset of
 * the group access relation.
 *
 * The cases where an uninitialized value is read or a value is written
 * that is never read or where the dataflow occurs within a statement
 * instance are also considered local and may also be removed.
 *
 * Essentially, we compute the intersection of "access" with either
 *
 *	live_in union (range non-local-flow)
 *
 * or
 *
 *	live_out union (domain non-local-flow)
 *
 * We first construct a relation "local"
 *
 *	[[D -> R] -> [D' -> R']]
 *
 * of pairs of domain iterations accessing the reference group
 * and references in the group that are coscheduled by "sched".
 *
 * If this relation does not intersect the dataflow dependences,
 * then there is nothing we can possibly remove, unless the dataflow
 * dependences themselves only relate a subset of the accesses.
 * In particular, the accesses may not be involved in any dataflow
 * dependences, either because they are uninitialized reads/dead writes
 * or because the dataflow occurs inside a statement instance.
 *
 * Since the computation below may break up the access relation
 * into smaller pieces, we only perform the intersection with
 * the non-local dependent accesses if the local pairs
 * intersect the dataflow dependences.  Otherwise, we intersect
 * with the universe of the non-local dependent accesses.
 * This should at least remove accesses from statements that
 * do not participate in any dependences.
 *
 * In particular, we remove the "local" dataflow dependences from
 * the set of all dataflow dependences, or at least those
 * that may contribute to a domain/range that intersects
 * the domain of "access".
 * Note that if the potential dataflow dependences are an overapproximation
 * of the actual dataflow dependences, then the result remains an
 * overapproximation of the non-local dataflow dependences.
 * Copying to/from global memory is only needed for the references
 * in the domain/range of the result or for accesses that are live out/in
 * for the entire scop.
 *
 * We therefore map the domain/range of the "external" relation
 * to the corresponding access relation and take the union with
 * the live out/in relation.
 */
static __isl_give isl_union_map *remove_local_accesses(
	struct gpu_prog *prog, __isl_take isl_union_map *tagged,
	__isl_take isl_union_map *access, __isl_take isl_union_map *sched,
	int read)
{
	int empty;
	isl_union_pw_multi_aff *tagger;
	isl_union_set *domain, *access_domain;
	isl_union_map *local, *external, *universe;
	isl_union_set *tag_set;

	if (isl_union_map_is_empty(access)) {
		isl_union_map_free(sched);
		isl_union_map_free(tagged);
		return access;
	}

	tagger = isl_union_pw_multi_aff_copy(prog->scop->tagger);
	domain = isl_union_map_domain(isl_union_map_copy(tagged));
	tagger = isl_union_pw_multi_aff_intersect_domain(tagger,
					isl_union_set_copy(domain));
	sched = isl_union_map_preimage_domain_union_pw_multi_aff(sched, tagger);

	local = isl_union_map_apply_range(sched,
			    isl_union_map_reverse(isl_union_map_copy(sched)));
	local = isl_union_map_intersect(local,
			isl_union_map_copy(prog->scop->tagged_dep_flow));

	empty = isl_union_map_is_empty(local);

	external = isl_union_map_copy(prog->scop->tagged_dep_flow);
	universe = isl_union_map_universe(isl_union_map_copy(access));
	access_domain = isl_union_map_domain(universe);
	domain = isl_union_set_universe(domain);
	universe = isl_union_set_unwrap(domain);
	universe = isl_union_map_intersect_domain(universe, access_domain);
	domain = isl_union_map_wrap(universe);
	if (read)
		external = isl_union_map_intersect_range(external, domain);
	else
		external = isl_union_map_intersect_domain(external, domain);
	external = isl_union_map_intersect_params(external,
				isl_set_copy(prog->scop->context));
	external = isl_union_map_subtract(external, local);

	if (read) {
		tag_set = isl_union_map_range(external);
		external = wrapped_reference_to_access(tag_set, tagged);
		external = isl_union_map_union(external,
				isl_union_map_copy(prog->scop->live_in));
	} else {
		tag_set = isl_union_map_domain(external);
		external = wrapped_reference_to_access(tag_set, tagged);
		external = isl_union_map_union(external,
				isl_union_map_copy(prog->scop->live_out));
	}

	if (empty < 0)
		external = isl_union_map_free(external);
	else if (empty)
		external = isl_union_map_universe(external);

	access = isl_union_map_intersect(access, external);

	return access;
}

/* Given an access relation "access" from "group", remove those reads
 * if ("read" is 1) or writes (if "read" is 0) that are only needed to
 * communicate data within the same iteration of the schedule at the
 * position where the copying of the group is inserted.
 * "node" points to this position, i.e., the depth at "node"
 * is equal to group->depth.
 *
 * We extract a schedule that picks out the iterations of the outer
 * group->depth dimensions and call remove_local_accesses.
 */
static __isl_give isl_union_map *remove_local_accesses_group(
	struct ppcg_kernel *kernel, struct gpu_array_ref_group *group,
	__isl_take isl_union_map *access, __isl_keep isl_schedule_node *node,
	int read)
{
	isl_union_map *sched, *tagged;

	if (isl_union_map_is_empty(access))
		return access;

	tagged = group_tagged_access_relation(group);
	sched = isl_schedule_node_get_prefix_schedule_relation(node);

	return remove_local_accesses(kernel->prog, tagged, access, sched, read);
}

/* This function is called before the AST generator starts traversing
 * the schedule subtree of a node with mark "mark".
 *
 * If the mark is called "kernel", store the kernel pointer in data->kernel
 * for use in at_domain.
 */
static int before_mark(__isl_keep isl_id *mark,
	__isl_keep isl_ast_build *build, void *user)
{
	struct ppcg_at_domain_data *data = user;

	if (!mark)
		return -1;
	if (!strcmp(isl_id_get_name(mark), "kernel"))
		data->kernel = isl_id_get_user(mark);
	return 0;
}

/* This function is called after the AST generator has finished traversing
 * the schedule subtree of a mark node.  "node" points to the corresponding
 * mark AST node.
 *
 * If the mark is called "kernel", then replace "node" by a user node
 * that "calls" the kernel, representing the launch of the kernel.
 * The original "node" is stored inside the kernel object so that
 * it can be used to print the device code.
 * Note that this assumes that a kernel is only launched once.
 * Also clear data->kernel.
 */
static __isl_give isl_ast_node *after_mark(__isl_take isl_ast_node *node,
        __isl_keep isl_ast_build *build, void *user)
{
	isl_ctx *ctx;
	isl_id *id;
	isl_ast_expr *expr;
	isl_ast_expr_list *list;
	struct ppcg_kernel *kernel;
	struct ppcg_at_domain_data *data = user;

	ctx = isl_ast_node_get_ctx(node);
	id = isl_ast_node_mark_get_id(node);
	if (!id)
		return isl_ast_node_free(node);
	if (strcmp(isl_id_get_name(id), "kernel") || !data->kernel) {
		isl_id_free(id);
		return node;
	}
	kernel = data->kernel;
	data->kernel = NULL;
	kernel->space = isl_ast_build_get_schedule_space(build);
	kernel->tree = isl_ast_node_mark_get_node(node);
	isl_ast_node_free(node);

	expr = isl_ast_expr_from_id(isl_id_copy(id));
	list = isl_ast_expr_list_alloc(ctx, 0);
	expr = isl_ast_expr_call(expr, list);
	node = isl_ast_node_alloc_user(expr);
	node = isl_ast_node_set_annotation(node, id);

	return node;
}

static isl_bool update_depth(__isl_keep isl_schedule_node *node, void *user)
{
	int *depth = user;
	int node_depth;

	if (isl_schedule_node_get_type(node) != isl_schedule_node_leaf)
		return isl_bool_true;
	node_depth = isl_schedule_node_get_schedule_depth(node);
	if (node_depth > *depth)
		*depth = node_depth;

	return isl_bool_false;
}

/* Use isl to generate code for both the host and the device
 * from "schedule".
 * The device code is marked by "kernel" mark nodes in the schedule tree,
 * containing a pointer to a ppcg_kernel object.
 * The returned AST only contains the AST for the host code.
 * The ASTs for the device code are embedded in ppcg_kernel objects
 * attached to the leaf nodes that call "kernel".
 */
__isl_give isl_ast_node *generate_code(struct gpu_gen *gen,
	__isl_take isl_schedule *schedule)
{
	struct ppcg_at_domain_data data;
	isl_ast_build *build;
	isl_ast_node *tree;
	isl_id_list *iterators;
	int depth;

	data.prog = gen->prog;
	data.gen = gen;
	data.kernel = NULL;

	depth = 0;
	if (isl_schedule_foreach_schedule_node_top_down(schedule, &update_depth,
						&depth) < 0)
		return NULL;
	build = isl_ast_build_alloc(gen->prog->ctx);
	iterators = ppcg_scop_generate_names(gen->prog->scop, depth, "c");
	build = isl_ast_build_set_iterators(build, iterators);
	build = isl_ast_build_set_at_each_domain(build, &at_domain, &data);
	build = isl_ast_build_set_before_each_mark(build, &before_mark, &data);
	build = isl_ast_build_set_after_each_mark(build, &after_mark, &data);
	if (gen->prog->scop->options->debug->dump_final_schedule)
		isl_schedule_dump(schedule);
	tree = isl_ast_build_node_from_schedule(build, schedule);
	isl_ast_build_free(build);

	return tree;
}

__isl_give isl_union_map *extract_sizes_from_str(isl_ctx *ctx, const char *str)
{
	if (!str)
		return NULL;
	return isl_union_map_read_from_str(ctx, str);
}

/* Can "node" be tiled and then mapped to block and thread identifiers?
 * That is, is it permutable with at least one coincident dimension?
 */
static int is_permutable(__isl_keep isl_schedule_node *node)
{
	if (!node)
		return -1;

	if (isl_schedule_node_get_type(node) != isl_schedule_node_band)
		return 0;
	if (!isl_schedule_node_band_get_permutable(node))
		return 0;
	if (isl_schedule_node_band_n_member(node) < 1)
		return 0;
	if (!isl_schedule_node_band_member_get_coincident(node, 0))
		return 0;

	return 1;
}

/* A isl_schedule_foreach_schedule_node_top_down callback
 * for setting *any_permutable and aborting the search
 * if "node" is a permutable band with coincident dimensions.
 * Otherwise, continue searching.
 */
static isl_bool set_permutable(__isl_keep isl_schedule_node *node, void *user)
{
	int *any_permutable = user;
	int permutable;

	permutable = is_permutable(node);
	if (permutable < 0)
		return isl_bool_error;
	if (!permutable)
		return isl_bool_true;

	*any_permutable = 1;

	return isl_bool_error;
}

/* Does "schedule" contain any permutable band with at least one coincident
 * member?
 */
int has_any_permutable_node(__isl_keep isl_schedule *schedule)
{
	int any_permutable = 0;

	if (isl_schedule_foreach_schedule_node_top_down(schedule,
				    &set_permutable, &any_permutable) < 0 &&
	    !any_permutable)
		return -1;

	return any_permutable;
}

/* Is "node" a leaf or can it be tiled and then mapped to
 * block and thread identifiers?
 */
static int is_leaf_or_tilable(__isl_keep isl_schedule_node *node)
{
	if (isl_schedule_node_get_type(node) == isl_schedule_node_leaf)
		return 1;
	return is_permutable(node);
}

/* Is "node" the outermost node in its branch that can be tiled
 * and then mapped to block and thread identifiers?
 * If there are no such nodes in the branch and if "node" is a leaf,
 * then it is accepted too.
 */
static int is_outer_tilable(__isl_keep isl_schedule_node *node)
{
	int tilable;
	isl_schedule_node *ancestor;

	tilable = is_leaf_or_tilable(node);
	if (tilable < 0)
		return -1;
	if (!tilable)
		return 0;

	tilable = 0;
	ancestor = isl_schedule_node_copy(node);
	while (isl_schedule_node_has_parent(ancestor)) {
		ancestor = isl_schedule_node_parent(ancestor);

		tilable = is_permutable(ancestor);
		if (tilable < 0 || tilable)
			break;
	}

	isl_schedule_node_free(ancestor);
	return tilable < 0 ? -1 : !tilable;
}

/* Collect the references to all writes in "group".
 * Each reference is represented by a universe set in a space
 *
 *	[S[i,j] -> R[]]
 *
 * with S[i,j] the statement instance space and R[] the array reference.
 */
static __isl_give isl_union_set *group_tagged_writes(
	struct gpu_array_ref_group *group)
{
	int i;
	isl_space *space;
	isl_union_set *writes;

	space = isl_map_get_space(group->access);
	writes = isl_union_set_empty(space);
	for (i = 0; i < group->n_ref; ++i) {
		isl_space *space;
		isl_set *writes_i;

		if (!group->refs[i]->write)
			continue;

		space = isl_map_get_space(group->refs[i]->tagged_access);
		space = isl_space_domain(space);
		writes_i = isl_set_universe(space);
		writes = isl_union_set_add_set(writes, writes_i);
	}

	return writes;
}

/* Is there any write access in "group" that requires synchronization
 * on a write to global memory?
 * We currently take into account all writes that would require
 * synchronization at the thread level depth, but if the copying
 * for this group is performed at an outer level, then we do not
 * actually need to take into account dependences at intermediate levels.
 */
static int any_sync_writes_in_group(struct ppcg_kernel *kernel,
	struct gpu_array_ref_group *group)
{
	isl_union_set *writes;
	int empty, disjoint;

	empty = isl_union_set_is_empty(kernel->sync_writes);
	if (empty < 0)
		return -1;
	if (empty)
		return 0;

	writes = group_tagged_writes(group);
	disjoint = isl_union_set_is_disjoint(kernel->sync_writes, writes);
	isl_union_set_free(writes);

	return disjoint < 0 ? -1 : !disjoint;
}

/* Collect the references to all writes in "kernel" that write directly
 * to global or shared memory, i.e., that are not mapped to private memory.
 * Each reference is represented by a universe set in a space
 *
 *	[S[i,j] -> R[]]
 *
 * with S[i,j] the statement instance space and R[] the array reference.
 */
static __isl_give isl_union_set *collect_non_private_tagged_writes(
	struct ppcg_kernel *kernel)
{
	isl_union_set *writes;
	int i, j;

	writes = isl_union_set_empty(isl_union_set_get_space(kernel->arrays));

	for (i = 0; i < kernel->n_array; ++i) {
		struct gpu_local_array_info *array = &kernel->array[i];

		for (j = 0; j < array->n_group; ++j) {
			struct gpu_array_ref_group *group = array->groups[j];
			isl_union_set *writes_ij;

			if (!group->write)
				continue;
			if (group->private_tile)
				continue;
			writes_ij = group_tagged_writes(group);
			writes = isl_union_set_union(writes, writes_ij);
		}
	}

	return writes;
}

/* Are there any direct writes to global memory that require
 * synchronization?
 */
static int any_global_or_shared_sync_writes(struct ppcg_kernel *kernel)
{
	isl_union_set *writes;
	int empty, disjoint;

	empty = isl_union_set_is_empty(kernel->sync_writes);
	if (empty < 0)
		return -1;
	if (empty)
		return 0;

	writes = collect_non_private_tagged_writes(kernel);
	disjoint = isl_union_set_is_disjoint(kernel->sync_writes, writes);
	isl_union_set_free(writes);

	return disjoint < 0 ? -1 : !disjoint;
}

/* Construct an isl_multi_val for use as tile sizes for tiling "node"
 * from the elements in "tile_size".
 */
static __isl_give isl_multi_val *construct_band_tiles_sizes(
	__isl_keep isl_schedule_node *node, int *tile_size)
{
	int i, n;
	isl_ctx *ctx;
	isl_space *space;
	isl_multi_val *mv;

	if (!node)
		return NULL;

	ctx = isl_schedule_node_get_ctx(node);
	space = isl_schedule_node_band_get_space(node);
	n = isl_schedule_node_band_n_member(node);
	mv = isl_multi_val_zero(space);
	for (i = 0; i < n; ++i) {
		isl_val *v;

		v = isl_val_int_from_si(ctx, tile_size[i]);
		mv = isl_multi_val_set_val(mv, i, v);
	}

	return mv;
}

/* Replace the partial schedule S of the band node "node" by
 *
 *	floor(S/f)
 *
 * or
 *
 *	f * floor(S/f)
 *
 * if scale_tile_loops is set, with f the integers in "factor".
 * The list that "factor" points to is assumed to contain at least
 * as many elements as the number of members in the band.
 */
static __isl_give isl_schedule_node *snap_band_to_sizes(
	__isl_take isl_schedule_node *node, int *factor,
	struct ppcg_options *options)
{
	isl_multi_val *mv;

	mv = construct_band_tiles_sizes(node, factor);
	node = isl_schedule_node_band_scale_down(node, isl_multi_val_copy(mv));
	if (options->scale_tile_loops)
		node = isl_schedule_node_band_scale(node,
							isl_multi_val_copy(mv));
	isl_multi_val_free(mv);

	return node;
}

/* Tile "band" with tile size specified by "sizes".
 *
 * Since the tile loops will be mapped to block ids, we forcibly
 * turn off tile loop scaling.  We may want to enable tile loop scaling
 * at some later point, but then we would have to support the detection
 * of strides during the mapping to block ids.
 * Similarly, since the point loops will be mapped to thread ids,
 * we forcibly shift the point loops so that they start at zero.
 */
static __isl_give isl_schedule_node *tile_band(
	__isl_take isl_schedule_node *node, __isl_take isl_multi_val *sizes)
{
	isl_ctx *ctx = isl_schedule_node_get_ctx(node);
	int scale_tile;
	int shift_point;

	scale_tile = isl_options_get_tile_scale_tile_loops(ctx);
	isl_options_set_tile_scale_tile_loops(ctx, 0);
	shift_point = isl_options_get_tile_shift_point_loops(ctx);
	isl_options_set_tile_shift_point_loops(ctx, 1);

	node = isl_schedule_node_band_tile(node, sizes);

	isl_options_set_tile_scale_tile_loops(ctx, scale_tile);
	isl_options_set_tile_shift_point_loops(ctx, shift_point);

	return node;
}

/* Extract the set of parameter values and outer schedule dimensions
 * for which any statement instance
 * in the kernel inserted at "node" needs to be executed.
 * Intersect the set of parameter values derived from the host schedule
 * relation with the context of "prog".
 */
static __isl_give isl_set *extract_context(__isl_keep isl_schedule_node *node,
	struct gpu_prog *prog)
{
	isl_union_map *schedule;
	isl_union_set *schedule_domain;
	isl_set *context;
	int empty;

	schedule = isl_schedule_node_get_prefix_schedule_relation(node);
	schedule_domain = isl_union_map_range(schedule);
	empty = isl_union_set_is_empty(schedule_domain);
	if (empty < 0) {
		isl_union_set_free(schedule_domain);
		return NULL;
	}
	if (empty) {
		int depth;
		isl_space *space;

		space = isl_union_set_get_space(schedule_domain);
		isl_union_set_free(schedule_domain);
		space = isl_space_set_from_params(space);
		depth = isl_schedule_node_get_schedule_depth(node);
		space = isl_space_add_dims(space, isl_dim_set, depth);
		context = isl_set_empty(space);
	} else {
		context = isl_set_from_union_set(schedule_domain);
	}
	context = isl_set_intersect_params(context,
					    isl_set_copy(prog->context));

	return context;
}

/* Return the set of outer array elements accessed by
 * by the statement instance in "domain" in "prog".
 */
static __isl_give isl_union_set *accessed_by_domain(
	__isl_take isl_union_set *domain, struct gpu_prog *prog)
{
	isl_union_map *access;
	isl_union_set *arrays;

	access = isl_union_map_union(isl_union_map_copy(prog->read),
				     isl_union_map_copy(prog->may_write));
	access = isl_union_map_intersect_domain(access, domain);
	arrays = isl_union_map_range(access);
	arrays = isl_union_set_apply(arrays,
				isl_union_map_copy(prog->to_outer));

	return arrays;
}

/* Return the number of outer band members of the band node "node"
 * that are marked coincident.
 */
static int n_outer_coincidence(__isl_keep isl_schedule_node *node)
{
	int i, n;

	n = isl_schedule_node_band_n_member(node);

	for (i = 0; i < n; ++i)
		if (!isl_schedule_node_band_member_get_coincident(node, i))
			break;

	return i;
}

/* If the band node "node" has more than "n" members, then split off
 * the first "n" of them.
 */
static __isl_give isl_schedule_node *split_band(
	__isl_take isl_schedule_node *node, int n)
{
	int dim;

	dim = isl_schedule_node_band_n_member(node);
	if (n < dim)
		node = isl_schedule_node_band_split(node, n);

	return node;
}

/* Scale a band node that may have been split by split_band.
 * "sizes" are the scaling factors for the original node.
 * "node" either points to the original band node, or the outer
 * of the two pieces after splitting.
 *
 * If the number of elements in "node" is smaller than the number of
 * elements in "sizes", then some splitting has occurred and we split
 * "sizes" in the same way.
 */
static __isl_give isl_schedule_node *scale_band(
	__isl_take isl_schedule_node *node, __isl_take isl_multi_val *sizes)
{
	int n, dim;

	n = isl_multi_val_dim(sizes, isl_dim_set);
	dim = isl_schedule_node_band_n_member(node);
	if (n > dim) {
		isl_multi_val *sizes2;

		sizes2 = isl_multi_val_copy(sizes);
		sizes = isl_multi_val_drop_dims(sizes,
						isl_dim_set, dim, n - dim);
		sizes2 = isl_multi_val_drop_dims(sizes2, isl_dim_set, 0, dim);
		node = isl_schedule_node_child(node, 0);
		node = isl_schedule_node_band_scale(node, sizes2);
		node = isl_schedule_node_parent(node);
	}

	return isl_schedule_node_band_scale(node, sizes);
}

/* Return an isl_multi_aff, with as elements the parameters in "space"
 * that have the names specified by the elements in "names".
 * If (some of) these parameters do not already appear in "space",
 * then they are added first.
 */
static __isl_give isl_multi_aff *parameter_vector(__isl_take isl_space *space,
	__isl_keep isl_id_list *names)
{
	int i, n;
	isl_local_space *ls;
	isl_multi_aff *ma;

	if (!names)
		space = isl_space_free(space);

	n = isl_id_list_n_id(names);
	for (i = 0; i < n; ++i) {
		int pos;
		isl_id *id;

		id = isl_id_list_get_id(names, i);
		pos = isl_space_find_dim_by_id(space, isl_dim_param, id);
		if (pos >= 0) {
			isl_id_free(id);
			continue;
		}
		pos = isl_space_dim(space, isl_dim_param);
		space = isl_space_add_dims(space, isl_dim_param, 1);
		space = isl_space_set_dim_id(space, isl_dim_param, pos, id);
	}
	ma = isl_multi_aff_zero(isl_space_copy(space));
	ls = isl_local_space_from_space(isl_space_domain(space));
	for (i = 0; i < n; ++i) {
		int pos;
		isl_id *id;
		isl_aff *aff;

		id = isl_id_list_get_id(names, i);
		pos = isl_space_find_dim_by_id(space, isl_dim_param, id);
		isl_id_free(id);
		aff = isl_aff_var_on_domain(isl_local_space_copy(ls),
					    isl_dim_param, pos);
		ma = isl_multi_aff_set_aff(ma, i, aff);
	}
	isl_local_space_free(ls);

	return ma;
}

/* Return constraints on the domain elements that equate a sequence of
 * parameters called "names", to the partial schedule
 * of "node" modulo the integers in "size".
 * The number of elements in the array "size" should be equal
 * to the number of elements in "names".
 * The number of members of the band node "node" should be smaller
 * than or equal to this number.  If it is smaller, then the first
 * elements of "names" are equated to zero.
 */
static __isl_give isl_union_set *set_schedule_modulo(
	__isl_keep isl_schedule_node *node, __isl_keep isl_id_list *names,
	int *size)
{
	int n, n_zero;
	isl_space *space;
	isl_multi_aff *ma;
	isl_multi_union_pw_aff *mupa, *mupa2;
	isl_multi_val *mv;
	isl_union_set *domain;

	if (!node)
		return NULL;
	n = isl_id_list_n_id(names);
	if (n == 0)
		return isl_schedule_node_get_universe_domain(node);
	n_zero = n - isl_schedule_node_band_n_member(node);

	mupa = isl_schedule_node_band_get_partial_schedule(node);
	mv = construct_band_tiles_sizes(node, size + n_zero);
	mupa = isl_multi_union_pw_aff_mod_multi_val(mupa, mv);

	space = isl_multi_union_pw_aff_get_space(mupa);
	space = isl_space_params(space);
	space = isl_space_set_from_params(space);
	space = isl_space_add_dims(space, isl_dim_set, n_zero);
	ma = isl_multi_aff_zero(space);

	domain = isl_schedule_node_get_universe_domain(node);
	mupa2 = isl_multi_union_pw_aff_multi_aff_on_domain(
						isl_union_set_copy(domain), ma);
	mupa = isl_multi_union_pw_aff_range_product(mupa2, mupa);

	space = isl_multi_union_pw_aff_get_space(mupa);
	ma = parameter_vector(space, names);

	mupa2 = isl_multi_union_pw_aff_multi_aff_on_domain(domain, ma);
	mupa = isl_multi_union_pw_aff_sub(mupa, mupa2);

	return isl_multi_union_pw_aff_zero_union_set(mupa);
}

/* Insert a context node at "node" introducing the block and thread
 * identifiers along with their bounds, which are stored in kernel->grid_size
 * and kernel->block_dim.
 * Note that the bounds on the block identifiers may implicitly impose
 * constraints on the parameters.  A guard needs to be inserted
 * in the schedule tree to ensure that those bounds hold at "node".
 * This guard is inserted in insert_guard.
 */
static __isl_give isl_schedule_node *insert_context(struct ppcg_kernel *kernel,
	__isl_take isl_schedule_node *node)
{
	isl_set *context;

	context = isl_set_universe(isl_set_get_space(kernel->context));

	context = add_bounded_parameters_dynamic(context,
					kernel->grid_size, kernel->block_ids);
	context = add_bounded_parameters(context,
					kernel->block_dim, kernel->thread_ids);

	node = isl_schedule_node_insert_context(node, context);

	return node;
}

/* Insert a guard that eliminates kernel launches where the kernel
 * obviously does not have any work to do.
 *
 * In particular, eliminate kernel launches where there are obviously
 * zero blocks.
 * Use the same block size constraints that are used to create the context
 * to ensure that all constraints implicit in the constructed context
 * are imposed by the guard.
 *
 * Additionally, add other constraints that are valid
 * for each executed instance ("context"), as long as this does not result
 * in a disjunction.
 */
static __isl_give isl_schedule_node *insert_guard(
	__isl_take isl_schedule_node *node, __isl_keep isl_set *context,
	__isl_keep isl_multi_pw_aff *size, struct ppcg_scop *scop)
{
	unsigned nparam, n;
	isl_set *guard;
	isl_id_list *ids;

	guard = isl_set_copy(context);
	guard = isl_set_compute_divs(guard);
	guard = isl_set_from_basic_set(isl_set_simple_hull(guard));

	nparam = isl_set_dim(guard, isl_dim_param);
	n = isl_multi_pw_aff_dim(size, isl_dim_out);
	ids = ppcg_scop_generate_names(scop, n, "__ppcg_tmp");
	guard = add_bounded_parameters_dynamic(guard, size, ids);
	isl_id_list_free(ids);
	guard = isl_set_project_out(guard, isl_dim_param, nparam, n);

	node = isl_schedule_node_insert_guard(node, guard);

	return node;
}

/* Does any array reference group mapping require the band that is mapped
 * to threads to be unrolled?
 */
static int kernel_requires_unroll(struct ppcg_kernel *kernel)
{
	int i, j;

	for (i = 0; i < kernel->n_array; ++i) {
		struct gpu_local_array_info *array = &kernel->array[i];

		for (j = 0; j < array->n_group; ++j) {
			struct gpu_array_ref_group *group = array->groups[j];
			if (gpu_array_ref_group_requires_unroll(group))
				return 1;
		}
	}

	return 0;
}

/* Mark the given band node "node" for unrolling by the AST generator and
 * then sink it to the leaves of the schedule tree.
 * All dimensions of "node" are assumed to be coincident, such that this
 * sinking is a valid operation.
 */
static __isl_give isl_schedule_node *unroll(__isl_take isl_schedule_node *node)
{
	int i, n;

	n = isl_schedule_node_band_n_member(node);
	for (i = 0; i < n; ++i)
		node = isl_schedule_node_band_member_set_ast_loop_type(node, i,
							isl_ast_loop_unroll);

	node = isl_schedule_node_band_sink(node);

	return node;
}

/* Insert a synchronization node in the schedule tree of "node"
 * after the core computation of "kernel" at the level of the band
 * that is mapped to threads, except if that level is equal to
 * that of the band that is mapped to blocks or if there are no writes
 * to global or shared memory in the core computation that require
 * synchronization.
 * If there are any writes to shared memory and the shared memory
 * copying is performed at the same level, then synchronization
 * is needed between the core and the copying anyway, so we might
 * as well add it here.  If the copying is performed at a higher
 * level, then different iterations of intermediate schedule dimensions
 * may have a different mapping from between shared memory elements and
 * threads, such that synchronization is required after the core.
 * "node" is assumed to point to the kernel node.
 */
static __isl_give isl_schedule_node *add_sync(struct ppcg_kernel *kernel,
	__isl_take isl_schedule_node *node)
{
	int kernel_depth;
	int need_sync;

	need_sync = any_global_or_shared_sync_writes(kernel);
	if (need_sync < 0)
		return isl_schedule_node_free(node);
	if (!need_sync)
		return node;

	kernel_depth = isl_schedule_node_get_schedule_depth(node);

	node = gpu_tree_move_down_to_thread(node, kernel->core);
	if (kernel_depth == isl_schedule_node_get_schedule_depth(node))
		return gpu_tree_move_up_to_kernel(node);

	node = gpu_tree_ensure_following_sync(node, kernel);

	node = gpu_tree_move_up_to_kernel(node);

	return node;
}

/* Return a read ("read" is 1) or write access relation for "group"
 * with those accesses removed that are only needed to communicate data
 * within the subtree of the schedule rooted at "node".
 * Furthermore, include the prefix schedule at "node".
 * That is, return a relation of the form
 *
 *	S -> [D -> A]
 *
 * with D the outer schedule dimensions at "node".
 */
static __isl_give isl_union_map *anchored_non_local_accesses(
	struct ppcg_kernel *kernel, struct gpu_array_ref_group *group,
	__isl_take isl_schedule_node *node, int read)
{
	isl_union_map *access;
	isl_union_map *prefix;

	access = gpu_array_ref_group_access_relation(group, read, !read);
	access = remove_local_accesses_group(kernel, group, access, node, read);
	prefix = isl_schedule_node_get_prefix_schedule_relation(node);
	access = isl_union_map_range_product(prefix, access);

	return access;
}

/* Given an array reference group "group", create a mapping
 *
 *	read[D -> A] -> [D -> A]
 *
 * if "read" is set or
 *
 *	write[D -> A] -> [D -> A]
 *
 * if "read" is not set.
 * D corresponds to the outer group->depth dimensions of
 * the kernel schedule.
 */
static __isl_give isl_multi_aff *create_from_access(isl_ctx *ctx,
	struct gpu_array_ref_group *group, int read)
{
	isl_space *space;
	isl_id *id;

	space = isl_space_copy(group->array->space);
	space = isl_space_from_range(space);
	space = isl_space_add_dims(space, isl_dim_in, group->depth);
	space = isl_space_wrap(space);
	space = isl_space_map_from_set(space);

	id = isl_id_alloc(ctx, read ? "read" : "write", group);
	space = isl_space_set_tuple_id(space, isl_dim_in, id);

	return isl_multi_aff_identity(space);
}

/* If any writes in "group" require synchronization, then make sure
 * that there is a synchronization node for "kernel" after the node
 * following "node" in a sequence.
 *
 * If "shared" is set and no synchronization is needed for
 * the writes to global memory, then add synchronization before
 * the kernel to protect shared memory from being overwritten
 * by the next iteration of the core computation.
 * No additional synchronization is needed to protect against
 * the next copy into shared memory because each element of
 * the shared memory tile is always copied by the same thread.
 */
static __isl_give isl_schedule_node *add_group_write_sync(
	__isl_take isl_schedule_node *node, struct ppcg_kernel *kernel,
	struct gpu_array_ref_group *group, int shared)
{
	int need_sync;

	need_sync = any_sync_writes_in_group(kernel, group);
	if (need_sync < 0)
		return isl_schedule_node_free(node);
	if (need_sync) {
		node = isl_schedule_node_parent(node);
		node = isl_schedule_node_next_sibling(node);
		node = isl_schedule_node_child(node, 0);
		node = gpu_tree_ensure_following_sync(node, kernel);
	} else if (shared) {
		node = isl_schedule_node_parent(node);
		node = isl_schedule_node_parent(node);
		node = gpu_tree_move_down_to_depth(node, group->depth,
							kernel->core);
		node = gpu_tree_move_left_to_sync(node, kernel);
	}

	return node;
}

/* Add copy statements to the schedule tree of "node"
 * for reading from global memory to private memory (if "read" is set) or
 * for writing back from private memory to global memory
 * (if "read" is not set) for the array reference group "group" that
 * is mapped to private memory.
 * On input, "node" points to the kernel node, and it is moved
 * back there on output.
 *
 * The copies are performed in the order of the array elements.
 * The copy statement instances include a reference to the outer
 * group->depth dimensions of the kernel schedule for ease of
 * combining them with the group tiling.
 *
 * That is, the extra schedule is of the form
 *
 *	type[D -> A] -> A
 *
 * where D corresponds to the outer group->depth dimensions of
 * the kernel schedule and A to the global array.
 * This schedule is unrolled because registers are not addressable.
 *
 * The copying is inserted in the schedule tree through an extension
 * of the form
 *
 *	D -> type[D -> A]
 *
 * where the extra domain elements type[D -> A] are those accessed
 * by the group.
 * A filter is inserted on type[D -> A] to ensure that the element
 * is read/written by the same thread that needs the element.
 * This filter is obtained by applying
 *
 *	S -> type[D -> A]
 *
 * to the thread filter for the core statements.
 *
 * The extension is inserted before the core computation in case of a read
 * and after the core computation in case of a write.
 * In the latter case, we also make sure that there is a synchronization
 * node after the write to global memory, unless this write is performed
 * at the outer level of the kernel.
 * In principle, this synchronization could be inserted higher
 * in the schedule tree depending on where the corresponding reads
 * from global memory are performed.
 */
static __isl_give isl_schedule_node *add_copies_group_private(
	struct ppcg_kernel *kernel, struct gpu_array_ref_group *group,
	__isl_take isl_schedule_node *node, int read)
{
	isl_union_map *access;
	isl_union_map *prefix;
	isl_union_set *domain;
	isl_space *space;
	isl_multi_aff *from_access;
	isl_multi_pw_aff *mpa;
	isl_multi_union_pw_aff *mupa;
	isl_schedule_node *graft;
	isl_union_set *filter;
	int kernel_depth;
	int empty;

	kernel_depth = isl_schedule_node_get_schedule_depth(node);
	node = gpu_tree_move_down_to_depth(node, group->depth, kernel->core);

	access = anchored_non_local_accesses(kernel, group, node, read);
	empty = isl_union_map_is_empty(access);
	if (empty < 0 || empty) {
		isl_union_map_free(access);
		if (empty < 0)
			return isl_schedule_node_free(node);
		return gpu_tree_move_up_to_kernel(node);
	}

	group->array->global = 1;
	group->local_array->global = 1;

	from_access = create_from_access(kernel->ctx, group, read);
	space = isl_space_domain(isl_multi_aff_get_space(from_access));
	access = isl_union_map_preimage_range_multi_aff(access, from_access);

	filter = isl_union_set_copy(kernel->thread_filter);
	filter = isl_union_set_apply(filter, isl_union_map_copy(access));
	filter = isl_union_set_detect_equalities(filter);
	filter = isl_union_set_coalesce(filter);

	domain = isl_union_map_range(access);
	access = isl_union_set_wrapped_domain_map(domain);
	access = isl_union_map_reverse(access);
	access = isl_union_map_coalesce(access);
	graft = isl_schedule_node_from_extension(access);

	space = isl_space_map_from_set(space);
	mpa = isl_multi_pw_aff_identity(space);
	mpa = isl_multi_pw_aff_range_factor_range(mpa);
	mupa = isl_multi_union_pw_aff_from_multi_pw_aff(mpa);

	graft = isl_schedule_node_child(graft, 0);
	graft = isl_schedule_node_insert_partial_schedule(graft, mupa);
	graft = unroll(graft);

	graft = isl_schedule_node_insert_filter(graft, filter);

	graft = isl_schedule_node_parent(graft);

	if (read)
		node = isl_schedule_node_graft_before(node, graft);
	else {
		node = isl_schedule_node_graft_after(node, graft);
		if (kernel_depth < group->depth)
			node = add_group_write_sync(node, kernel, group, 0);
	}

	node = gpu_tree_move_up_to_kernel(node);

	return node;
}

/* Add copy statements to the schedule tree of "node"
 * for reading from global memory to shared memory (if "read" is set) or
 * for writing back from shared memory to global memory
 * (if "read" is not set) for the array reference group "group" that
 * is mapped to shared memory.
 * On input, "node" points to the kernel node, and it is moved
 * back there on output.
 *
 * The copies are performed in the order of the corresponding shared
 * memory tile.
 * The copy statement instances include a reference to the outer
 * group->depth dimensions of the kernel schedule for ease of
 * combining them with the group tiling.
 *
 * If we are performing a read from global memory to shared memory and
 * if the array involved is not a scalar, then we copy
 * the entire tile to shared memory.  This may result in some extra
 * elements getting copied, but it should lead to simpler code
 * (which means that fewer registers may be needed) and less divergence.
 *
 * Otherwise, we only copy the elements that will be read or have been written
 * in the kernel.
 *
 * That is, the extra schedule is of the form
 *
 *	type[D -> A] -> T
 *
 * where D corresponds to the outer group->depth dimensions of
 * the kernel schedule, A to the global array and T is the corresponding
 * shared memory tile.
 *
 * The copying is inserted in the schedule tree through an extension
 * of the form
 *
 *	D -> type[D -> A]
 *
 * where the extra domain elements type[D -> A] are those accessed
 * by the group.  In the case of read from a non-scalar, this set
 * is replaced by the entire shared memory tile.
 *
 * A filter is inserted on type[D -> A] to map the copy instances
 * to the threads.  In particular, the thread identifiers are
 * equated to the position inside the shared memory tile (T)
 * modulo the block size.
 * We try to align the innermost tile dimension with the innermost
 * thread identifier (x) as a heuristic to improve coalescing.
 * In particular, if the dimension of the tile is greater than
 * the dimension of the block, then the schedule mapping to the tile
 * is broken up into two pieces and the filter is applied to the inner part.
 * If, on the other hand, the dimension of the tile is smaller than
 * the dimension of the block, then the initial thread identifiers
 * are equated to zero and the remaining thread identifiers are
 * matched to the memory tile.
 *
 * The extension is inserted before the core computation in case of a read
 * and after the core computation in case of a write.
 * In the case of a read, we first need to make sure there is some
 * synchronization before the core computation such that we can put the read
 * from global memory to shared memory before that synchronization.
 * This ensures that all threads have finished copying into shared memory
 * before the shared memory is used.
 * We also need to make sure that there is a synchronization node after
 * the core computation to ensure that the next load into shared memory
 * only happens after all data has been used.  There is no need for
 * this synchronization if we are at the outer level since then there
 * won't be a next load.
 * In the case of a write, we need to make sure there is some synchronization
 * after the core computation such taht we can put the write from shared
 * memory to global memory after that synchronization.
 * Unless we are at the outer level, we also need a synchronization node
 * after the write to ensure the data is saved to global memory
 * before the next iteration write to the same shared memory.
 * It also makes sure the data has arrived in global memory before
 * it is read in a subsequent iteration.
 */
static __isl_give isl_schedule_node *add_copies_group_shared(
	struct ppcg_kernel *kernel, struct gpu_array_ref_group *group,
	__isl_take isl_schedule_node *node, int read)
{
	struct gpu_array_tile *tile;
	isl_union_map *access;
	isl_union_set *domain;
	isl_union_set *sync;
	isl_multi_aff *ma;
	isl_multi_aff *from_access;
	isl_multi_pw_aff *mpa;
	isl_multi_union_pw_aff *mupa;
	isl_schedule_node *graft;
	isl_union_set *filter;
	int skip;
	int kernel_depth;
	int empty;

	kernel_depth = isl_schedule_node_get_schedule_depth(node);
	node = gpu_tree_move_down_to_depth(node, group->depth, kernel->core);

	access = anchored_non_local_accesses(kernel, group, node, read);
	empty = isl_union_map_is_empty(access);
	if (empty < 0 || empty) {
		isl_union_map_free(access);
		if (empty < 0)
			return isl_schedule_node_free(node);
		return gpu_tree_move_up_to_kernel(node);
	}

	group->array->global = 1;
	group->local_array->global = 1;

	from_access = create_from_access(kernel->ctx, group, read);

	tile = gpu_array_ref_group_tile(group);
	ma = isl_multi_aff_copy(tile->tiling);
	ma = isl_multi_aff_pullback_multi_aff(ma,
					    isl_multi_aff_copy(from_access));
	mpa = isl_multi_pw_aff_from_multi_aff(ma);
	mupa = isl_multi_union_pw_aff_from_multi_pw_aff(mpa);

	domain = isl_union_map_range(access);

	if (read && !gpu_array_is_scalar(group->array)) {
		isl_map *map;
		isl_union_set_free(domain);
		map = group_tile(group);
		domain = isl_union_set_from_set(isl_map_wrap(map));
	}

	domain = isl_union_set_preimage_multi_aff(domain, from_access);
	access = isl_union_set_wrapped_domain_map(domain);
	access = isl_union_map_reverse(access);
	access = isl_union_map_coalesce(access);
	graft = isl_schedule_node_from_extension(access);

	graft = isl_schedule_node_child(graft, 0);

	graft = isl_schedule_node_insert_partial_schedule(graft, mupa);

	if (tile->n > kernel->n_block && kernel->n_block > 0) {
		graft = isl_schedule_node_band_split(graft,
						tile->n - kernel->n_block);
		graft = isl_schedule_node_child(graft, 0);
	}
	if (tile->n < kernel->n_block)
		skip = kernel->n_block - tile->n;
	else
		skip = 0;
	filter = set_schedule_modulo(graft, kernel->thread_ids,
					kernel->block_dim);
	if (!kernel->options->wrap)
		graft = snap_band_to_sizes(graft, kernel->block_dim + skip,
			    kernel->options);
	if (tile->n > kernel->n_block && kernel->n_block > 0)
		graft = isl_schedule_node_parent(graft);
	graft = isl_schedule_node_insert_filter(graft, filter);

	while (graft && isl_schedule_node_has_parent(graft))
		graft = isl_schedule_node_parent(graft);

	if (read) {
		if (kernel_depth < group->depth)
			node = gpu_tree_ensure_sync_after_core(node, kernel);
		node = gpu_tree_move_left_to_sync(node, kernel);
		node = isl_schedule_node_graft_before(node, graft);
	} else {
		node = gpu_tree_move_right_to_sync(node, kernel);
		node = isl_schedule_node_graft_after(node, graft);
		if (kernel_depth < group->depth)
			node = add_group_write_sync(node, kernel, group, 1);
	}

	node = gpu_tree_move_up_to_kernel(node);

	return node;
}

/* Check whether the array reference group "group" is mapped to
 * private or shared memory and, if so,
 * add copy statements to the schedule tree of "node"
 * for reading from global memory to private or shared memory
 * (if "read" is set) or for writing back from private or shared memory
 * to global memory (if "read" is not set) for this group.
 * On input, "node" points to the kernel node, and it is moved
 * back there on output.
 */
static __isl_give isl_schedule_node *add_copies_group(
	struct ppcg_kernel *kernel, struct gpu_array_ref_group *group,
	__isl_take isl_schedule_node *node, int read)
{
	if (group->private_tile)
		return add_copies_group_private(kernel, group, node, read);
	if (group->shared_tile)
		return add_copies_group_shared(kernel, group, node, read);
	return node;
}

/* For each array reference group that is mapped to private or shared memory,
 * add copy statements to the schedule tree of "node"
 * for reading from global memory to private or shared memory
 * and for writing back.
 * On input, "node" points to the kernel node, and it is moved
 * back there on output.
 */
static __isl_give isl_schedule_node *add_copies(struct ppcg_kernel *kernel,
	__isl_take isl_schedule_node *node)
{
	int i, j;

	for (i = 0; i < kernel->n_array; ++i) {
		struct gpu_local_array_info *array = &kernel->array[i];

		for (j = 0; j < array->n_group; ++j) {
			struct gpu_array_ref_group *group = array->groups[j];

			node = add_copies_group(kernel, group, node, 1);
			if (!node)
				return NULL;
			node = add_copies_group(kernel, group, node, 0);
			if (!node)
				return NULL;
		}
	}

	return node;
}

/* Mark all dimensions in the current band node atomic.
 */
static __isl_give isl_schedule_node *atomic(__isl_take isl_schedule_node *node)
{
	int i, n;

	n = isl_schedule_node_band_n_member(node);
	for (i = 0; i < n; ++i)
		node = isl_schedule_node_band_member_set_ast_loop_type(node, i,
							isl_ast_loop_atomic);

	return node;
}

/* Mark "node" atomic, if it is a band node.
 * Do the same for all ancestors.
 * Return a pointer to "node" (in the updated schedule tree).
 */
static __isl_give isl_schedule_node *atomic_ancestors(
	__isl_take isl_schedule_node *node)
{
	int pos;

	if (!node)
		return NULL;
	if (!isl_schedule_node_has_parent(node))
		return node;

	pos = isl_schedule_node_get_child_position(node);
	node = isl_schedule_node_parent(node);
	if (isl_schedule_node_get_type(node) == isl_schedule_node_band)
		node = atomic(node);
	node = atomic_ancestors(node);
	node = isl_schedule_node_child(node, pos);

	return node;
}

/* Collect all write references that require synchronization.
 * "node" is assumed to point to the kernel node.
 * Each reference is represented by a universe set in a space
 *
 *	[S[i,j] -> R[]]
 *
 * with S[i,j] the statement instance space and R[] the array reference.
 *
 * This function should be called before block and thread filters are added.
 *
 * Synchronization is needed after a write if there is a subsequent read
 * within the same block that may not be performed by the same thread.
 * There should not be any dependences between different blocks,
 * so we start with the flow dependences within the same kernel invocation
 * and we subtract from these those dependences that are mapped
 * to the same iteration of the bands where synchronization is inserted.
 * We do not remove pairs of instances that are known to map to
 * the same thread across different iterations of the intermediate
 * bands because the read may be performed by a different thread
 * than the one that needs the value if shared memory is involved.
 *
 * We also consider all pairs of possible writes that access the same
 * memory location and that may be mapped to the same block but not
 * to the same iteration of the intermediate bands.
 * In theory, it would be possible for one thread to still be in
 * a previous iteration of a loop in these bands.
 * A write to global memory in this delayed thread could then overwrite
 * a write from another thread that has already moved on to
 * the next iteration.
 *
 * After computing the above writes paired off with reads or writes
 * that depend on them, we project onto the domain writes.
 * Sychronization is needed after writes to global memory
 * through these references.
 */
static __isl_give isl_union_set *compute_sync_writes(
	struct ppcg_kernel *kernel, __isl_keep isl_schedule_node *node)
{
	isl_union_map *local;
	isl_union_map *may_writes, *shared_access;
	isl_union_map *kernel_prefix, *thread_prefix;
	isl_union_map *equal;
	isl_union_set *wrap;
	isl_union_set *domain;

	domain = isl_schedule_node_get_universe_domain(node);
	kernel_prefix = isl_schedule_node_get_prefix_schedule_union_map(node);
	node = isl_schedule_node_copy(node);
	node = gpu_tree_move_down_to_thread(node, kernel->core);
	thread_prefix = isl_schedule_node_get_prefix_schedule_union_map(node);
	isl_schedule_node_free(node);

	may_writes = isl_union_map_copy(kernel->prog->scop->tagged_may_writes);
	may_writes = isl_union_map_curry(may_writes);
	may_writes = isl_union_map_intersect_domain(may_writes, domain);
	may_writes = isl_union_map_uncurry(may_writes);
	shared_access = isl_union_map_copy(may_writes);
	shared_access = isl_union_map_apply_range(shared_access,
					isl_union_map_reverse(may_writes));

	local = isl_union_map_copy(kernel->prog->scop->tagged_dep_flow);
	local = isl_union_map_union(local, shared_access);
	local = isl_union_map_zip(local);

	equal = isl_union_map_apply_range(kernel_prefix,
		    isl_union_map_reverse(isl_union_map_copy(kernel_prefix)));
	wrap = isl_union_map_wrap(equal);
	local = isl_union_map_intersect_domain(local, wrap);
	equal = isl_union_map_apply_range(thread_prefix,
		    isl_union_map_reverse(isl_union_map_copy(thread_prefix)));
	wrap = isl_union_map_wrap(equal);
	local = isl_union_map_subtract_domain(local, wrap);

	local = isl_union_map_zip(local);
	local = isl_union_map_universe(local);

	return isl_union_map_domain(local);
}

/* Group the domain elements into a single space, named kernelX,
 * with X the kernel sequence number "kernel_id".
 */
static __isl_give isl_schedule_node *group_statements(
	__isl_take isl_schedule_node *node, int kernel_id)
{
	char buffer[20];
	isl_id *id;

	if (!node)
		return NULL;

	snprintf(buffer, sizeof(buffer), "kernel%d", kernel_id);
	id = isl_id_alloc(isl_schedule_node_get_ctx(node), buffer, NULL);
	return isl_schedule_node_group(node, id);
}

/* Create a ppcg_kernel representing the domain instances that reach "node"
 * and insert a mark node pointing to the ppcg_kernel before "node".
 * The band that "node" points to is the band that needs to be mapped
 * to block identifiers.  The band that needs to be mapped to thread
 * identifiers should be marked by a "thread" mark by the caller.
 * This mark is removed by this function.
 * If "scale" is set, then the band that "node" points to is scaled
 * by "sizes".
 *
 * Mark all outer band nodes as atomic to ensure each kernel is only
 * scheduled once.
 * If the domain elements that reach "node" live in more than one space,
 * then group the domain elements into a single space, named kernelX,
 * with X the kernel sequence number.
 *
 * Insert a guard node governing the kernel node to ensure that
 * no kernels with zero blocks are launched.
 *
 * Insert a context node describing the block and thread
 * identifiers inside the kernel mark.
 * The context node needs to be inserted after the effective block size
 * has been determined such that the bounds on the thread identifiers
 * would reflect the effective block size.
 * Insert a filter node inside the context node mapping the statement
 * instances to block identifiers.  In particular, the block identifiers
 * are equated to the partial schedule of band that was marked for mapping
 * to blocks modulo the grid size.
 * Insert a filter node inside the "thread" mark mapping the statement
 * instances to thread identifiers.  In particular, the thread identifiers
 * are equated to the partial schedule of band that was marked for mapping
 * to threads modulo the block size.
 *
 * Compute array reference groups for all arrays, set the local
 * array bounds based on the set of domain instances that reach
 * the kernel node, check the total amount of shared memory used
 * and compute all group tilings.
 * The array reference groups are computed after the block filter
 * has been inserted because it affects the mapping to shared or
 * private memory.  This computation also requires the thread filter
 * (in the ppcg_kernel object), but this thread filter should not
 * have been added to the schedule tree yet since the computation
 * requires the schedule of the band that needs to be mapped to
 * threads before the privatization is applied.
 *
 * If any array reference group requires the band mapped to threads
 * to be unrolled, then we perform the required unrolling.
 *
 * We save a copy of the schedule that may influence the mappings
 * to shared or private memory in kernel->shared_schedule.
 *
 * Finally, we add synchronization and copy statements to the schedule tree,
 * remove the "thread" mark and create representations for the local
 * variables in the kernel.
 *
 * We keep a copy of the isl_id that points to the kernel to ensure
 * that the kernel does not get destroyed if the schedule node
 * is freed due to some error condition.
 */
static __isl_give isl_schedule_node *create_kernel(struct gpu_gen *gen,
	__isl_take isl_schedule_node *node, int scale,
	__isl_keep isl_multi_val *sizes)
{
	struct ppcg_kernel *kernel;
	isl_id *id;
	isl_schedule_node *node_thread;
	isl_union_map *host_schedule;
	isl_set *host_domain;
	isl_union_set *domain;
	int single_statement;

	kernel = isl_calloc_type(gen->ctx, struct ppcg_kernel);
	kernel = ppcg_kernel_create_local_arrays(kernel, gen->prog);
	if (!kernel)
		return isl_schedule_node_free(node);

	domain = isl_schedule_node_get_domain(node);
	single_statement = isl_union_set_n_set(domain) == 1;

	kernel->ctx = gen->ctx;
	kernel->prog = gen->prog;
	kernel->options = gen->options;
	kernel->context = extract_context(node, gen->prog);
	kernel->core = isl_union_set_universe(isl_union_set_copy(domain));
	kernel->arrays = accessed_by_domain(isl_union_set_copy(domain),
						gen->prog);
	kernel->n_grid = n_outer_coincidence(node);
	node_thread = isl_schedule_node_copy(node);
	node_thread = gpu_tree_move_down_to_thread(node_thread, kernel->core);
	node_thread = isl_schedule_node_child(node_thread, 0);
	kernel->n_block = n_outer_coincidence(node_thread);
	isl_schedule_node_free(node_thread);
	kernel->id = gen->kernel_id++;
	read_grid_and_block_sizes(kernel, gen);

	kernel->sync_writes = compute_sync_writes(kernel, node);

	host_schedule = isl_schedule_node_get_prefix_schedule_union_map(node);
	host_domain = isl_set_from_union_set(isl_union_map_range(
								host_schedule));

	node = atomic_ancestors(node);

	id = isl_id_alloc(gen->ctx, "kernel", kernel);
	id = isl_id_set_free_user(id, &ppcg_kernel_free_wrap);
	node = isl_schedule_node_insert_mark(node, isl_id_copy(id));

	if (!single_statement)
		node = group_statements(node, kernel->id);

	node = isl_schedule_node_child(node, 0);
	node = split_band(node, kernel->n_grid);
	kernel->block_ids = ppcg_scop_generate_names(gen->prog->scop,
						kernel->n_grid, "b");
	kernel->block_filter = set_schedule_modulo(node, kernel->block_ids,
						kernel->grid_dim);
	kernel->grid_size = extract_grid_size(kernel,
						isl_union_set_copy(domain));
	if (!kernel->options->wrap)
		node = snap_band_to_sizes(node, kernel->grid_dim,
						kernel->options);
	if (scale)
		node = scale_band(node, isl_multi_val_copy(sizes));
	node = isl_schedule_node_parent(node);
	if (!single_statement)
		node = isl_schedule_node_parent(node);
	node = insert_guard(node, kernel->context, kernel->grid_size,
				gen->prog->scop);
	node = gpu_tree_move_down_to_thread(node, kernel->core);
	node = isl_schedule_node_child(node, 0);
	node = split_band(node, kernel->n_block);
	kernel->thread_ids = ppcg_scop_generate_names(gen->prog->scop,
						kernel->n_block, "t");
	kernel->thread_filter = set_schedule_modulo(node, kernel->thread_ids,
						kernel->block_dim);
	extract_block_size(kernel, domain);

	node = gpu_tree_move_up_to_kernel(node);
	node = isl_schedule_node_child(node, 0);
	node = insert_context(kernel, node);
	node = isl_schedule_node_child(node, 0);
	node = isl_schedule_node_insert_filter(node,
				    isl_union_set_copy(kernel->block_filter));

	node = gpu_tree_move_up_to_kernel(node);

	if (gpu_group_references(kernel, node) < 0)
		node = isl_schedule_node_free(node);
	localize_bounds(kernel, host_domain);
	isl_set_free(host_domain);

	check_shared_memory_bound(kernel);
	mark_global_arrays(kernel);
	compute_group_tilings(kernel);

	node = gpu_tree_move_down_to_thread(node, kernel->core);
	node = isl_schedule_node_child(node, 0);
	if (!kernel->options->wrap)
		node = snap_band_to_sizes(node, kernel->block_dim,
						kernel->options);
	node = isl_schedule_node_insert_filter(node,
				    isl_union_set_copy(kernel->thread_filter));
	if (kernel_requires_unroll(kernel)) {
		node = isl_schedule_node_child(node, 0);
		node = unroll(node);
	}

	node = gpu_tree_move_up_to_thread(node);
	kernel->shared_schedule_dim =
		isl_schedule_node_get_schedule_depth(node);
	kernel->shared_schedule =
		isl_schedule_node_get_prefix_schedule_union_pw_multi_aff(node);

	node = gpu_tree_move_up_to_kernel(node);

	node = add_sync(kernel, node);
	node = add_copies(kernel, node);

	node = gpu_tree_move_down_to_thread(node, kernel->core);
	node = isl_schedule_node_delete(node);

	node = gpu_tree_move_up_to_kernel(node);

	if (create_kernel_vars(kernel) < 0)
		node = isl_schedule_node_free(node);

	if (!single_statement)
		node = isl_schedule_node_parent(node);
	node = isl_schedule_node_parent(node);

	isl_id_free(id);
	return node;
}

/* Insert a zero-dimensional permutable band at "node".
 */
static __isl_give isl_schedule_node *insert_empty_permutable_band(
	__isl_take isl_schedule_node *node)
{
	isl_space *space;
	isl_schedule *schedule;
	isl_union_set *domain;
	isl_multi_union_pw_aff *mupa;

	schedule = isl_schedule_node_get_schedule(node);
	domain = isl_schedule_get_domain(schedule);
	space = isl_union_set_get_space(domain);
	isl_union_set_free(domain);
	isl_schedule_free(schedule);

	space = isl_space_set_from_params(space);
	mupa = isl_multi_union_pw_aff_zero(space);
	node = isl_schedule_node_insert_partial_schedule(node, mupa);
	node = isl_schedule_node_band_set_permutable(node, 1);

	return node;
}

/* If "node" is the outermost permutable band that can be mapped to block and
 * thread identifiers in its branch (or a leaf with no such outer bands),
 * then mark the band as such, attaching a ppcg_kernel to the mark.
 *
 * If "node" originally points to a leaf, then insert a zero-dimensional
 * permutable band such that we can assume that "node" always
 * points to a band node.
 *
 * Tile "node" using user specified tile sizes, after splitting the band
 * if the number of specified tile sizes is smaller than the dimension
 * of the band.  Mark the point band of this tiling as the band that
 * needs to be mapped to threads.
 * Create a kernel representing the domain instances that reach "node" and
 * insert a mark node pointing to the ppcg_kernel before the band node.
 */
static __isl_give isl_schedule_node *mark_outer_permutable(
	__isl_take isl_schedule_node *node, void *user)
{
	struct gpu_gen *gen = user;
	int outer;
	int scale;
	int tile_len;
	int *tile_size;
	isl_id *id;
	isl_multi_val *sizes;

	outer = is_outer_tilable(node);
	if (outer < 0)
		return isl_schedule_node_free(node);
	if (!outer)
		return node;

	if (isl_schedule_node_get_type(node) == isl_schedule_node_leaf)
		node = insert_empty_permutable_band(node);

	tile_len = isl_schedule_node_band_n_member(node);
	tile_size = read_tile_sizes(gen, &tile_len);
	if (!tile_size)
		return isl_schedule_node_free(node);
	if (tile_len < isl_schedule_node_band_n_member(node))
		node = isl_schedule_node_band_split(node, tile_len);
	sizes = construct_band_tiles_sizes(node, tile_size);
	node = tile_band(node, isl_multi_val_copy(sizes));
	node = isl_schedule_node_child(node, 0);
	id = isl_id_alloc(gen->ctx, "thread", NULL);
	node = isl_schedule_node_insert_mark(node, id);
	node = isl_schedule_node_parent(node);

	scale = gen->options->scale_tile_loops;
	node = create_kernel(gen, node, scale, sizes);
	isl_multi_val_free(sizes);
	free(tile_size);

	return node;
}

/* Does the subtree rooted at "node" have any suitably permutable band nodes?
 * That is, does it have any nodes that are permutable and that
 * have a least one coincident dimension?
 */
static int subtree_has_permutable_bands(__isl_keep isl_schedule_node *node)
{
	int any_parallelism = 0;

	if (isl_schedule_node_foreach_descendant_top_down(node, &set_permutable,
						&any_parallelism) < 0 &&
	    !any_parallelism)
		return -1;

	return any_parallelism;
}

/* Mark all variables that are accessed by the statement instances in "domain"
 * and that are local to "prog" as requiring a declaration in the host code.
 */
static int declare_accessed_local_variables(struct gpu_prog *prog,
	__isl_keep isl_union_set *domain)
{
	isl_union_set *arrays;
	int i;

	if (!ppcg_scop_any_hidden_declarations(prog->scop))
		return 0;
	arrays = accessed_by_domain(isl_union_set_copy(domain), prog);

	for (i = 0; i < prog->n_array; ++i) {
		isl_space *space;
		isl_set *set;
		int empty;

		if (!prog->array[i].local)
			continue;
		space = isl_set_get_space(prog->array[i].extent);
		set = isl_union_set_extract_set(arrays, space);
		empty = isl_set_plain_is_empty(set);
		isl_set_free(set);
		if (empty < 0)
			goto error;
		if (!empty)
			prog->array[i].declare_local = 1;
	}

	isl_union_set_free(arrays);
	return 0;
error:
	isl_union_set_free(arrays);
	return -1;
}

/* If "node" points to a set node, then separate its children
 * into subtrees that have suitably permutable bands and
 * those that do not.
 * Adjust the schedule tree in order to execute the second group
 * after the first group and return a pointer to the first group,
 * assuming there are any such subtrees.
 * Mark all local variables in "prog" that are accessed by
 * the second group as requiring a declaration on the host.
 */
static __isl_give isl_schedule_node *isolate_permutable_subtrees(
	__isl_take isl_schedule_node *node, struct gpu_prog *prog)
{
	isl_space *space;
	isl_union_set *filter;
	int i, n;

	if (!node)
		return NULL;
	if (isl_schedule_node_get_type(node) != isl_schedule_node_set)
		return node;

	n = isl_schedule_node_n_children(node);
	if (n < 0)
		return isl_schedule_node_free(node);

	node = isl_schedule_node_child(node, 0);
	filter = isl_schedule_node_filter_get_filter(node);
	node = isl_schedule_node_parent(node);
	space = isl_union_set_get_space(filter);
	isl_union_set_free(filter);
	filter = isl_union_set_empty(space);

	for (i = 0; i < n; ++i) {
		int parallelism;

		node = isl_schedule_node_child(node, i);
		parallelism = subtree_has_permutable_bands(node);
		if (parallelism < 0) {
			node = isl_schedule_node_free(node);
		} else if (!parallelism) {
			isl_union_set *filter_i;
			filter_i = isl_schedule_node_filter_get_filter(node);
			filter = isl_union_set_union(filter, filter_i);
		}
		node = isl_schedule_node_parent(node);
	}

	if (declare_accessed_local_variables(prog, filter) < 0)
		node = isl_schedule_node_free(node);
	node = isl_schedule_node_order_after(node, filter);

	return node;
}

/* Replace any reference to an array element in the range of "copy"
 * by a reference to all array elements (defined by the extent of the array).
 */
static __isl_give isl_union_map *approximate_copy_out(
	__isl_take isl_union_map *copy, struct gpu_prog *prog)
{
	int i;
	isl_union_map *res;

	res = isl_union_map_empty(isl_union_map_get_space(copy));

	for (i = 0; i < prog->n_array; ++i) {
		isl_space *space;
		isl_set *set;
		isl_union_map *copy_i;
		isl_union_set *extent, *domain;

		space = isl_space_copy(prog->array[i].space);
		extent = isl_union_set_from_set(isl_set_universe(space));
		copy_i = isl_union_map_copy(copy);
		copy_i = isl_union_map_intersect_range(copy_i, extent);
		set = isl_set_copy(prog->array[i].extent);
		extent = isl_union_set_from_set(set);
		domain = isl_union_map_domain(copy_i);
		copy_i = isl_union_map_from_domain_and_range(domain, extent);
		res = isl_union_map_union(res, copy_i);
	}

	isl_union_map_free(copy);

	return res;
}

/* Insert "kernel" marks that point to a ppcg_kernel structure
 * in front of all outermost tilable band that (by construction)
 * have at least one parallel loop.
 */
static __isl_give isl_schedule_node *mark_kernels(struct gpu_gen *gen,
	__isl_take isl_schedule_node *node)
{
	return isl_schedule_node_map_descendant_bottom_up(node,
						&mark_outer_permutable, gen);
}

/* Save the schedule "schedule" to a file called "filename".
 * The schedule is printed in block style.
 */
static void save_schedule(__isl_keep isl_schedule *schedule,
	const char *filename)
{
	FILE *file;
	isl_ctx *ctx;
	isl_printer *p;

	if (!schedule)
		return;

	file = fopen(filename, "w");
	if (!file) {
		fprintf(stderr, "Unable to open '%s' for writing\n", filename);
		return;
	}
	ctx = isl_schedule_get_ctx(schedule);
	p = isl_printer_to_file(ctx, file);
	p = isl_printer_set_yaml_style(p, ISL_YAML_STYLE_BLOCK);
	p = isl_printer_print_schedule(p, schedule);
	isl_printer_free(p);
	fclose(file);
}

/* Load and return a schedule from a file called "filename".
 */
static __isl_give isl_schedule *load_schedule(isl_ctx *ctx,
	const char *filename)
{
	FILE *file;
	isl_schedule *schedule;

	file = fopen(filename, "r");
	if (!file) {
		fprintf(stderr, "Unable to open '%s' for reading\n", filename);
		return NULL;
	}
	schedule = isl_schedule_read_from_file(ctx, file);
	fclose(file);

	return schedule;
}

/* Construct schedule constraints from the dependences in prog->scop and
 * the array order dependences in prog->array_order.
 *
 * If live range reordering is allowed, then we need to make sure
 * that live ranges on arrays are not run in parallel since doing
 * so would require array expansion.  We therefore add the array
 * order dependences to the coincidence dependences.  Non-zero array
 * order dependences will then prevent a schedule dimension from being
 * considered parallel.
 * Live ranges derived from scalars are allowed to be run in parallel
 * since we force the scalars to be mapped to private memory in
 * check_scalar_live_ranges.
 * If live range reordering is allowed, then the false dependences
 * are not added to the validity constraints as that would prevent
 * reordering.  Instead, the external false dependences that enforce that reads
 * from potentially live-in data precede any later write and
 * that writes of potentially live-out data follow any other earlier write
 * are added to the validity and the coincidence constraints.
 * The false dependences are still added to the proximity constraints
 * for consistency with the case where live range reordering is not allowed.
 * The coincidence constraints then consist of flow dependences,
 * external false dependences and array order dependences.
 * The independences can be filtered out from the first two sets.
 * They have already been filtered out from the array order dependences
 * on a per array basis in collect_order_dependences.
 * There is no need for a per array handling of the other two sets
 * as there should be no flow or external false dependence on local
 * variables that can be filtered out.
 */
static __isl_give isl_schedule_constraints *construct_schedule_constraints(
	struct gpu_prog *prog)
{
	isl_union_set *domain;
	isl_union_map *dep_raw, *dep;
	isl_union_map *validity, *proximity, *coincidence;
	isl_schedule_constraints *sc;

	domain = isl_union_set_copy(prog->scop->domain);
	sc = isl_schedule_constraints_on_domain(domain);
	sc = isl_schedule_constraints_set_context(sc,
				isl_set_copy(prog->scop->context));
	if (prog->scop->options->live_range_reordering) {
		sc = isl_schedule_constraints_set_conditional_validity(sc,
			isl_union_map_copy(prog->scop->tagged_dep_flow),
			isl_union_map_copy(prog->scop->tagged_dep_order));
		proximity = isl_union_map_copy(prog->scop->dep_flow);
		validity = isl_union_map_copy(proximity);
		validity = isl_union_map_union(validity,
			    isl_union_map_copy(prog->scop->dep_forced));
		proximity = isl_union_map_union(proximity,
			    isl_union_map_copy(prog->scop->dep_false));
		coincidence = isl_union_map_copy(validity);
		coincidence = isl_union_map_subtract(coincidence,
			isl_union_map_copy(prog->scop->independence));
		coincidence = isl_union_map_union(coincidence,
				isl_union_map_copy(prog->array_order));
	} else {
		dep_raw = isl_union_map_copy(prog->scop->dep_flow);
		dep = isl_union_map_copy(prog->scop->dep_false);
		dep = isl_union_map_union(dep, dep_raw);
		dep = isl_union_map_coalesce(dep);
		proximity = isl_union_map_copy(dep);
		coincidence = isl_union_map_copy(dep);
		validity = dep;
	}
	sc = isl_schedule_constraints_set_validity(sc, validity);
	sc = isl_schedule_constraints_set_coincidence(sc, coincidence);
	sc = isl_schedule_constraints_set_proximity(sc, proximity);

	if (prog->scop->options->debug->dump_schedule_constraints)
		isl_schedule_constraints_dump(sc);
	return sc;
}

/* Compute an appropriate schedule based on the accesses in
 * gen->read and gen->write.
 *
 * We derive schedule constraints from the dependences in gen->prog->scop
 * and then use isl to compute a schedule that has a parallel loop
 * in each tilable band.
 */
static __isl_give isl_schedule *compute_schedule(struct gpu_gen *gen)
{
	isl_schedule_constraints *sc;
	isl_schedule *schedule;

	sc = construct_schedule_constraints(gen->prog);
	schedule = isl_schedule_constraints_compute_schedule(sc);

	return schedule;
}

/* If the band node "node" has exactly one member then mark it permutable.
 */
static __isl_give isl_schedule_node *band_set_permutable(
	__isl_take isl_schedule_node *node,
	__isl_keep isl_schedule_constraints *sc)
{
	if (isl_schedule_node_band_n_member(node) == 1)
		node = isl_schedule_node_band_set_permutable(node, 1);

	return node;
}

/* Return the coincidence constraints between pairs of instances
 * that are scheduled together by the ancestors of "node".
 * That is, select those coincidence constraints that relate
 * pairs of instances that have the same value for the prefix schedule.
 * If the schedule depth is zero, then the prefix schedule does not
 * contain any information, so we intersect domain and range
 * of the schedule constraints with the reaching domain elements instead.
 */
static __isl_give isl_union_map *get_local_coincidence(
	__isl_keep isl_schedule_node *node,
	__isl_keep isl_schedule_constraints *sc)
{
	isl_union_map *coincidence;
	isl_multi_union_pw_aff *prefix;
	isl_union_pw_multi_aff *contraction;

	coincidence = isl_schedule_constraints_get_coincidence(sc);
	contraction = isl_schedule_node_get_subtree_contraction(node);
	if (isl_schedule_node_get_schedule_depth(node) == 0) {
		isl_union_set *domain;

		domain = isl_schedule_node_get_domain(node);
		domain = isl_union_set_preimage_union_pw_multi_aff(domain,
						    contraction);
		coincidence = isl_union_map_intersect_domain(coincidence,
						    isl_union_set_copy(domain));
		coincidence = isl_union_map_intersect_range(coincidence,
						    domain);
		return coincidence;
	}

	prefix = isl_schedule_node_get_prefix_schedule_multi_union_pw_aff(node);
	prefix = isl_multi_union_pw_aff_pullback_union_pw_multi_aff(prefix,
								contraction);
	return isl_union_map_eq_at_multi_union_pw_aff(coincidence, prefix);
}

/* For each member in the band node "node", determine whether
 * it is coincident with respect to the outer nodes and mark
 * it accordingly.
 *
 * That is, for each coincidence constraint between pairs
 * of instances that are scheduled together by the outer nodes,
 * check that domain and range are assigned the same value
 * by the band member.  This test is performed by checking
 * that imposing the same value for the band member does not
 * remove any elements from the set of coincidence constraints.
 */
static __isl_give isl_schedule_node *band_set_coincident(
	__isl_take isl_schedule_node *node,
	__isl_keep isl_schedule_constraints *sc)
{
	isl_union_map *coincidence;
	isl_union_pw_multi_aff *contraction;
	isl_multi_union_pw_aff *partial;
	int i, n;

	coincidence = get_local_coincidence(node, sc);

	partial = isl_schedule_node_band_get_partial_schedule(node);
	contraction = isl_schedule_node_get_subtree_contraction(node);
	partial = isl_multi_union_pw_aff_pullback_union_pw_multi_aff(partial,
								contraction);
	n = isl_schedule_node_band_n_member(node);
	for (i = 0; i < n; ++i) {
		isl_union_map *coincidence_i;
		isl_union_pw_aff *upa;
		isl_multi_union_pw_aff *partial_i;
		int subset;

		upa = isl_multi_union_pw_aff_get_union_pw_aff(partial, i);
		partial_i = isl_multi_union_pw_aff_from_union_pw_aff(upa);
		coincidence_i = isl_union_map_copy(coincidence);
		coincidence_i = isl_union_map_eq_at_multi_union_pw_aff(
						    coincidence_i, partial_i);
		subset = isl_union_map_is_subset(coincidence, coincidence_i);
		isl_union_map_free(coincidence_i);

		if (subset < 0)
			break;
		node = isl_schedule_node_band_member_set_coincident(node, i,
								    subset);
	}
	if (i < n)
		node = isl_schedule_node_free(node);
	isl_multi_union_pw_aff_free(partial);
	isl_union_map_free(coincidence);

	return node;
}

/* If "node" is a band, then set its properties.
 *
 * In particular, if the band has exactly one member, then mark it permutable.
 * Mark the band member coincident based on the coincidence constraints
 * of "sc".
 */
static __isl_give isl_schedule_node *set_band_properties(
	__isl_take isl_schedule_node *node, void *user)
{
	isl_schedule_constraints *sc = user;

	if (isl_schedule_node_get_type(node) != isl_schedule_node_band)
		return node;
	if (isl_schedule_node_band_n_member(node) == 0)
		return node;

	node = band_set_permutable(node, sc);
	node = band_set_coincident(node, sc);

	return node;
}

/* Return the original schedule with all bands marked permutable and
 * all band members marked coincident based on the coincidence constraints.
 * The bands are explicitly marked permutable so that they will be considered
 * by mark_outer_permutable.
 */
static __isl_give isl_schedule *determine_properties_original_schedule(
	struct gpu_gen *gen)
{
	isl_schedule *schedule;
	isl_schedule_constraints *sc;

	schedule = isl_schedule_copy(gen->prog->scop->schedule);
	sc = construct_schedule_constraints(gen->prog);
	schedule = isl_schedule_map_schedule_node_bottom_up(schedule,
						    &set_band_properties, sc);
	isl_schedule_constraints_free(sc);

	return schedule;
}

/* Obtain a schedule for the scop, by reading it from
 * a file, by computing one or by determining the properties
 * of the original schedule.
 */
__isl_give isl_schedule *get_schedule(struct gpu_gen *gen)
{
	isl_schedule *schedule;

	if (gen->options->load_schedule_file) {
		schedule = load_schedule(gen->ctx,
					gen->options->load_schedule_file);
	} else {
		if (gen->options->reschedule)
			schedule = compute_schedule(gen);
		else
			schedule = determine_properties_original_schedule(gen);
		if (gen->options->save_schedule_file)
			save_schedule(schedule,
					gen->options->save_schedule_file);
	}
	if (gen->options->debug->dump_schedule)
		isl_schedule_dump(schedule);

	return schedule;
}

/* Construct the string "<a>_<b>".
 */
static char *concat(isl_ctx *ctx, const char *a, const char *b)
{
	isl_printer *p;
	char *s;

	p = isl_printer_to_str(ctx);
	p = isl_printer_print_str(p, a);
	p = isl_printer_print_str(p, "_");
	p = isl_printer_print_str(p, b);
	s = isl_printer_get_str(p);
	isl_printer_free(p);

	return s;
}

/* For each array in "prog" of which an element appears in "accessed" and
 * that is not a read only scalar, create a zero-dimensional universe set
 * of which the tuple id has name "<prefix>_<name of array>" and a user
 * pointer pointing to the array (gpu_array_info).
 *
 * If the array is local to "prog", then make sure it will be declared
 * in the host code.
 *
 * Return the list of these universe sets.
 */
static __isl_give isl_union_set_list *create_copy_filters(struct gpu_prog *prog,
	const char *prefix, __isl_take isl_union_set *accessed)
{
	int i;
	isl_ctx *ctx;
	isl_union_set_list *filters;

	ctx = prog->ctx;
	filters = isl_union_set_list_alloc(ctx, 0);
	for (i = 0; i < prog->n_array; ++i) {
		struct gpu_array_info *array = &prog->array[i];
		isl_space *space;
		isl_set *accessed_i;
		int empty;
		char *name;
		isl_id *id;
		isl_union_set *uset;

		if (gpu_array_is_read_only_scalar(array))
			continue;

		space = isl_space_copy(array->space);
		accessed_i = isl_union_set_extract_set(accessed, space);
		empty = isl_set_plain_is_empty(accessed_i);
		isl_set_free(accessed_i);
		if (empty < 0) {
			filters = isl_union_set_list_free(filters);
			break;
		}
		if (empty)
			continue;

		array->global = 1;
		if (array->local)
			array->declare_local = 1;

		name = concat(ctx, prefix, array->name);
		id = name ? isl_id_alloc(ctx, name, array) : NULL;
		free(name);
		space = isl_space_set_alloc(ctx, 0, 0);
		space = isl_space_set_tuple_id(space, isl_dim_set, id);
		uset = isl_union_set_from_set(isl_set_universe(space));

		filters = isl_union_set_list_add(filters, uset);
	}
	isl_union_set_free(accessed);

	return filters;
}

/* Make sure that code for the statements in "filters" that
 * copy arrays to or from the device is only generated when
 * the size of the corresponding array is positive.
 * That is, add a set node underneath "graft" with "filters" as children
 * and for each child add a guard that the selects the parameter
 * values for which the corresponding array has a positive size.
 * The array is available in the user pointer of the statement identifier.
 * "depth" is the schedule depth of the position where "graft"
 * will be added.
 */
static __isl_give isl_schedule_node *insert_positive_size_guards(
	__isl_take isl_schedule_node *graft,
	__isl_take isl_union_set_list *filters, int depth)
{
	int i, n;

	graft = isl_schedule_node_child(graft, 0);
	graft = isl_schedule_node_insert_set(graft, filters);
	n = isl_schedule_node_n_children(graft);
	for (i = 0; i < n; ++i) {
		isl_union_set *filter;
		isl_set *domain, *guard;
		isl_id *id;
		struct gpu_array_info *array;

		graft = isl_schedule_node_child(graft, i);
		filter = isl_schedule_node_filter_get_filter(graft);
		domain = isl_set_from_union_set(filter);
		id = isl_set_get_tuple_id(domain);
		array = isl_id_get_user(id);
		isl_id_free(id);
		isl_set_free(domain);
		guard = gpu_array_positive_size_guard(array);
		guard = isl_set_from_params(guard);
		guard = isl_set_add_dims(guard, isl_dim_set, depth);
		graft = isl_schedule_node_child(graft, 0);
		graft = isl_schedule_node_insert_guard(graft, guard);
		graft = isl_schedule_node_parent(graft);
		graft = isl_schedule_node_parent(graft);
	}
	graft = isl_schedule_node_parent(graft);

	return graft;
}

/* Create a graft for copying arrays to or from the device,
 * whenever the size of the array is strictly positive.
 * Each statement is called "<prefix>_<name of array>" and
 * the identifier has a user pointer pointing to the array.
 * The graft will be added at the position specified by "node".
 * "copy" contains the array elements that need to be copied.
 * Only arrays of which some elements need to be copied
 * will have a corresponding statement in the graph.
 * Note though that each such statement will copy the entire array.
 */
static __isl_give isl_schedule_node *create_copy_device(struct gpu_prog *prog,
	__isl_keep isl_schedule_node *node, const char *prefix,
	__isl_take isl_union_set *copy)
{
	int depth;
	isl_ctx *ctx;
	isl_space *space;
	isl_union_set *all, *domain;
	isl_union_set_list *filters;
	isl_union_map *extension;
	isl_schedule_node *graft;

	ctx = prog->ctx;
	depth = isl_schedule_node_get_schedule_depth(node);
	filters = create_copy_filters(prog, prefix, copy);
	all = isl_union_set_list_union(isl_union_set_list_copy(filters));

	space = depth < 0 ? NULL : isl_space_set_alloc(ctx, 0, depth);
	domain = isl_union_set_from_set(isl_set_universe(space));
	extension = isl_union_map_from_domain_and_range(domain, all);
	graft = isl_schedule_node_from_extension(extension);

	if (!filters)
		return isl_schedule_node_free(graft);
	if (isl_union_set_list_n_union_set(filters) == 0) {
		isl_union_set_list_free(filters);
		return graft;
	}

	return insert_positive_size_guards(graft, filters, depth);
}

/* Return (the universe spaces of) the arrays that are declared
 * inside the scop corresponding to "prog" and for which all
 * potential writes inside the scop form a subset of "domain".
 */
static __isl_give isl_union_set *extract_local_accesses(struct gpu_prog *prog,
	__isl_keep isl_union_set *domain)
{
	int i;
	isl_union_set *local;

	local = isl_union_set_empty(isl_union_set_get_space(domain));

	for (i = 0; i < prog->n_array; ++i) {
		isl_set *set;
		isl_union_map *to_outer;
		isl_union_map *may_write;
		isl_union_set *write_domain;
		isl_union_set *fields;
		int subset;

		if (!prog->array[i].local)
			continue;

		set = isl_set_universe(isl_space_copy(prog->array[i].space));
		to_outer = isl_union_map_copy(prog->to_outer);
		to_outer = isl_union_map_intersect_range(to_outer,
				    isl_union_set_from_set(isl_set_copy(set)));
		fields = isl_union_map_domain(to_outer);
		may_write = isl_union_map_copy(prog->may_write);
		may_write = isl_union_map_intersect_range(may_write, fields);
		write_domain = isl_union_map_domain(may_write);
		subset = isl_union_set_is_subset(write_domain, domain);
		isl_union_set_free(write_domain);

		if (subset < 0) {
			isl_set_free(set);
			return isl_union_set_free(local);
		} else if (subset) {
			local = isl_union_set_add_set(local, set);
		} else {
			isl_set_free(set);
		}
	}

	return local;
}

/* Internal data structure for node_may_persist.
 *
 * "tagger" maps tagged iteration domains to the corresponding untagged
 *	iteration domain.
 *
 * "may_persist_flow" is the set of all tagged dataflow dependences
 * with those dependences removed that either precede or follow
 * the kernel launch in a sequence.
 * "inner_band_flow" is the set of all tagged dataflow dependences
 * that are local to a given iteration of the outer band nodes
 * with respect to the current node.
 * "local_flow" is equal to "inner_band_flow", except that the domain
 * and the range have been intersected with intermediate filters
 * on children of sets or sequences.
 */
struct ppcg_may_persist_data {
	isl_union_pw_multi_aff *tagger;

	isl_union_map *local_flow;
	isl_union_map *inner_band_flow;
	isl_union_map *may_persist_flow;
};

/* Update the information in "data" based on the band ancestor "node".
 *
 * In particular, we restrict the dependences in data->local_flow
 * to those dependence where the source and the sink occur in
 * the same iteration of the given band node.
 * We also update data->inner_band_flow to the new value of
 * data->local_flow.
 */
static int update_may_persist_at_band(__isl_keep isl_schedule_node *node,
	struct ppcg_may_persist_data *data)
{
	isl_multi_union_pw_aff *partial;
	isl_union_pw_multi_aff *contraction;
	isl_union_map *flow;

	if (isl_schedule_node_band_n_member(node) == 0)
		return 0;

	partial = isl_schedule_node_band_get_partial_schedule(node);
	contraction = isl_schedule_node_get_subtree_contraction(node);
	partial = isl_multi_union_pw_aff_pullback_union_pw_multi_aff(partial,
								contraction);
	partial = isl_multi_union_pw_aff_pullback_union_pw_multi_aff(partial,
				isl_union_pw_multi_aff_copy(data->tagger));

	flow = data->local_flow;
	flow = isl_union_map_eq_at_multi_union_pw_aff(flow, partial);
	data->local_flow = flow;

	isl_union_map_free(data->inner_band_flow);
	data->inner_band_flow = isl_union_map_copy(data->local_flow);

	return 0;
}

/* Given a set of local reaching domain elements "domain",
 * expand them to the corresponding leaf domain elements using "contraction"
 * and insert the array references tags using data->tagger.
 */
static __isl_give isl_union_set *expand_and_tag(
	__isl_take isl_union_set *domain,
	__isl_take isl_union_pw_multi_aff *contraction,
	struct ppcg_may_persist_data *data)
{
	domain = isl_union_set_preimage_union_pw_multi_aff(domain,
			    contraction);
	domain = isl_union_set_preimage_union_pw_multi_aff(domain,
			    isl_union_pw_multi_aff_copy(data->tagger));
	return domain;
}

/* Given a filter node that is the child of a set or sequence node,
 * restrict data->local_flow to refer only to those elements
 * in the filter of the node.
 * "contraction" maps the leaf domain elements of the schedule tree
 * to the corresponding domain elements at (the parent of) "node".
 */
static int filter_flow(__isl_keep isl_schedule_node *node,
	struct ppcg_may_persist_data *data,
	__isl_take isl_union_pw_multi_aff *contraction)
{
	isl_union_set *filter;
	isl_union_map *flow;

	flow = data->local_flow;
	filter = isl_schedule_node_filter_get_filter(node);
	filter = expand_and_tag(filter, contraction, data);
	flow = isl_union_map_intersect_domain(flow, isl_union_set_copy(filter));
	flow = isl_union_map_intersect_range(flow, filter);
	data->local_flow = flow;

	return 0;
}

/* Given a filter node "node", collect the filters on all preceding siblings
 * (which are also filter nodes), add them to "filters" and return the result.
 */
static __isl_give isl_union_set *add_previous_filters(
	__isl_take isl_union_set *filters, __isl_keep isl_schedule_node *node)
{
	isl_schedule_node *sibling;

	sibling = isl_schedule_node_copy(node);
	while (sibling && isl_schedule_node_has_previous_sibling(sibling)) {
		isl_union_set *filter;

		sibling = isl_schedule_node_previous_sibling(sibling);
		filter = isl_schedule_node_filter_get_filter(sibling);
		filters = isl_union_set_union(filters, filter);
	}
	isl_schedule_node_free(sibling);
	if (!sibling)
		return isl_union_set_free(filters);

	return filters;
}

/* Given a filter node "node", collect the filters on all following siblings
 * (which are also filter nodes), add them to "filters" and return the result.
 */
static __isl_give isl_union_set *add_next_filters(
	__isl_take isl_union_set *filters, __isl_keep isl_schedule_node *node)
{
	isl_schedule_node *sibling;

	sibling = isl_schedule_node_copy(node);
	while (sibling && isl_schedule_node_has_next_sibling(sibling)) {
		isl_union_set *filter;

		sibling = isl_schedule_node_next_sibling(sibling);
		filter = isl_schedule_node_filter_get_filter(sibling);
		filters = isl_union_set_union(filters, filter);
	}
	isl_schedule_node_free(sibling);
	if (!sibling)
		return isl_union_set_free(filters);

	return filters;
}

/* Remove those flow dependences from data->may_persist_flow
 * that flow between elements of "domain" within the same iteration
 * of all outer band nodes.
 * "contraction" maps the leaf domain elements of the schedule tree
 * to the corresponding elements "domain".
 */
static void remove_external_flow(struct ppcg_may_persist_data *data,
	__isl_take isl_union_set *domain,
	__isl_keep isl_union_pw_multi_aff *contraction)
{
	isl_union_map *flow;

	contraction = isl_union_pw_multi_aff_copy(contraction);
	domain = expand_and_tag(domain, contraction, data);
	flow = isl_union_map_copy(data->local_flow);
	flow = isl_union_map_intersect_domain(flow, isl_union_set_copy(domain));
	flow = isl_union_map_intersect_range(flow, domain);

	data->may_persist_flow = isl_union_map_subtract(data->may_persist_flow,
							flow);
}

/* Update the information in "data" based on the filter ancestor "node".
 * We only need to modify anything if the filter is the child
 * of a set or sequence node.
 *
 * In the case of a sequence, we remove the dependences between
 * statement instances that are both executed either before or
 * after the subtree that will be mapped to a kernel, within
 * the same iteration of outer bands.
 *
 * In both cases, we restrict data->local_flow to the current child.
 */
static int update_may_persist_at_filter(__isl_keep isl_schedule_node *node,
	struct ppcg_may_persist_data *data)
{
	enum isl_schedule_node_type type;
	isl_schedule_node *parent;
	isl_space *space;
	isl_union_pw_multi_aff *contraction;
	isl_union_set *before, *after, *filter;
	isl_union_map *flow;

	type = isl_schedule_node_get_parent_type(node);
	if (type != isl_schedule_node_sequence && type != isl_schedule_node_set)
		return 0;

	parent = isl_schedule_node_copy(node);
	parent = isl_schedule_node_parent(parent);
	contraction = isl_schedule_node_get_subtree_contraction(parent);
	isl_schedule_node_free(parent);

	if (type == isl_schedule_node_set)
		return filter_flow(node, data, contraction);

	filter = isl_schedule_node_filter_get_filter(node);
	space = isl_union_set_get_space(filter);
	isl_union_set_free(filter);
	before = isl_union_set_empty(space);
	after = isl_union_set_copy(before);
	before = add_previous_filters(before, node);
	after = add_next_filters(after, node);

	remove_external_flow(data, before, contraction);
	remove_external_flow(data, after, contraction);

	return filter_flow(node, data, contraction);
}

/* Update the information in "data" based on the ancestor "node".
 */
static isl_stat update_may_persist_at(__isl_keep isl_schedule_node *node,
	void *user)
{
	struct ppcg_may_persist_data *data = user;

	switch (isl_schedule_node_get_type(node)) {
	case isl_schedule_node_error:
		return isl_stat_error;
	case isl_schedule_node_context:
	case isl_schedule_node_domain:
	case isl_schedule_node_expansion:
	case isl_schedule_node_extension:
	case isl_schedule_node_guard:
	case isl_schedule_node_leaf:
	case isl_schedule_node_mark:
	case isl_schedule_node_sequence:
	case isl_schedule_node_set:
		break;
	case isl_schedule_node_band:
		if (update_may_persist_at_band(node, data) < 0)
			return isl_stat_error;
		break;
	case isl_schedule_node_filter:
		if (update_may_persist_at_filter(node, data) < 0)
			return isl_stat_error;
		break;
	}

	return isl_stat_ok;
}

/* Determine the set of array elements that may need to be perserved
 * by a kernel constructed from the subtree at "node".
 * This includes the set of array elements that may need to be preserved
 * by the entire scop (prog->may_persist) and the elements for which
 * there is a potential flow dependence that may cross a kernel launch.
 *
 * To determine the second set, we start from all flow dependences.
 * From this set of dependences, we remove those that cannot possibly
 * require data to be preserved by a kernel launch.
 * In particular, we consider the following sets of dependences.
 * - dependences of which the write occurs inside the kernel.
 *   If the data is needed outside the kernel, then it will
 *   be copied out immediately after the kernel launch, so there
 *   is no need for any special care.
 * - dependences of which the read occurs inside the kernel and the
 *   corresponding write occurs inside the same iteration of the
 *   outer band nodes.  This means that the data is needed in
 *   the first kernel launch after the write, which is already
 *   taken care of by the standard copy-in.  That is, the data
 *   do not need to be preserved by any intermediate call to
 *   the same kernel.
 * - dependences of which the write and the read either both occur
 *   before the kernel launch or both occur after the kernel launch,
 *   within the same iteration of the outer band nodes with respect
 *   to the sequence that determines the ordering of the dependence
 *   and the kernel launch.  Such flow dependences cannot cross
 *   any kernel launch.
 *
 * For the remaining (tagged) dependences, we take the domain
 * (i.e., the tagged writes) and apply the tagged access relation
 * to obtain the accessed data elements.
 * These are then combined with the elements that may need to be
 * preserved by the entire scop.
 */
static __isl_give isl_union_set *node_may_persist(
	__isl_keep isl_schedule_node *node, struct gpu_prog *prog)
{
	struct ppcg_may_persist_data data;
	isl_schedule_node *root;
	isl_union_pw_multi_aff *contraction;
	isl_union_set *domain;
	isl_union_set *persist;
	isl_union_map *flow, *local_flow;

	data.tagger = prog->scop->tagger;

	flow = isl_union_map_copy(prog->scop->tagged_dep_flow);
	data.local_flow = isl_union_map_copy(flow);
	data.inner_band_flow = isl_union_map_copy(flow);
	data.may_persist_flow = flow;
	if (isl_schedule_node_foreach_ancestor_top_down(node,
					&update_may_persist_at, &data) < 0)
		data.may_persist_flow =
				    isl_union_map_free(data.may_persist_flow);
	flow = data.may_persist_flow;
	isl_union_map_free(data.local_flow);

	domain = isl_schedule_node_get_domain(node);
	contraction = isl_schedule_node_get_subtree_contraction(node);
	domain = isl_union_set_preimage_union_pw_multi_aff(domain,
				    contraction);
	domain = isl_union_set_preimage_union_pw_multi_aff(domain,
				    isl_union_pw_multi_aff_copy(data.tagger));
	flow = isl_union_map_subtract_domain(flow, isl_union_set_copy(domain));
	local_flow = data.inner_band_flow;
	local_flow = isl_union_map_intersect_range(local_flow, domain);
	flow = isl_union_map_subtract(flow, local_flow);

	persist = isl_union_map_domain(flow);
	persist = isl_union_set_apply(persist,
			isl_union_map_copy(prog->scop->tagged_may_writes));
	persist = isl_union_set_union(persist,
			isl_union_set_copy(prog->may_persist));

	return persist;
}

/* Add nodes for copying outer arrays in and out of the device
 * before and after the subtree "node", which contains one or more kernels.
 * "domain" contains the original reaching domain elements before
 * the kernels were created, i.e., before the contraction that
 * may have been performed in creating the kernels has been applied.
 * "prefix" contains the prefix schedule at that point, in terms
 * of the same original reaching domain elements.
 *
 * We first compute the sets of outer array elements that need
 * to be copied in and out and then graft in the nodes for
 * performing this copying.
 *
 * In particular, for each array that is possibly written anywhere in
 * the subtree "node" and that may be used after "node"
 * or that may be visible outside the corresponding scop,
 * we copy out its entire extent.
 *
 * Any array elements that is read without first being written inside
 * the subtree "node" needs to be copied in.
 * Furthermore, if there are any array elements that
 * are copied out, but that may not be written inside "node, then
 * they also need to be copied in to ensure that the value after execution
 * is the same as the value before execution, at least for those array
 * elements that may have their values preserved by the scop or that
 * may be written before "node" and read after "node".
 * In case the array elements are structures, we need to take into
 * account that all members of the structures need to be written
 * by "node" before we can avoid copying the data structure in.
 *
 * Note that the may_write relation is intersected with the domain,
 * which has been intersected with the context.
 * This helps in those cases where the arrays are declared with a fixed size,
 * while the accesses are parametric and the context assigns a fixed value
 * to the parameters.
 *
 * If an element from a local array is read without first being written,
 * then there is no point in copying it in since it cannot have been
 * written prior to the scop.  Warn about the uninitialized read instead.
 */
static __isl_give isl_schedule_node *add_to_from_device(
	__isl_take isl_schedule_node *node, __isl_take isl_union_set *domain,
	__isl_take isl_union_map *prefix, struct gpu_prog *prog)
{
	isl_union_set *local;
	isl_union_set *to_device, *from_device, *may_persist;
	isl_union_map *may_write, *must_write, *copy_out, *not_written;
	isl_union_map *read, *copy_in;
	isl_union_map *tagged;
	isl_union_map *local_uninitialized;
	isl_schedule_node *graft;

	tagged = isl_union_map_copy(prog->scop->tagged_reads);
	tagged = isl_union_map_union(tagged,
			    isl_union_map_copy(prog->scop->tagged_may_writes));

	may_write = isl_union_map_copy(prog->may_write);
	may_write = isl_union_map_intersect_domain(may_write,
					isl_union_set_copy(domain));
	may_write = remove_local_accesses(prog,
					isl_union_map_copy(tagged), may_write,
					isl_union_map_copy(prefix), 0);
	may_write = isl_union_map_apply_range(may_write,
					isl_union_map_copy(prog->to_outer));
	may_write = isl_union_map_apply_domain(may_write,
					isl_union_map_copy(prefix));
	may_write = approximate_copy_out(may_write, prog);
	copy_out = isl_union_map_copy(may_write);
	may_write = isl_union_map_apply_range(may_write,
					isl_union_map_copy(prog->to_inner));
	must_write = isl_union_map_copy(prog->must_write);
	must_write = isl_union_map_apply_domain(must_write,
					isl_union_map_copy(prefix));
	may_persist = node_may_persist(node, prog);
	may_write = isl_union_map_intersect_range(may_write, may_persist);
	not_written = isl_union_map_subtract(may_write, must_write);

	local = extract_local_accesses(prog, domain);
	read = isl_union_map_copy(prog->read);
	read = isl_union_map_intersect_domain(read, domain);
	read = remove_local_accesses(prog, tagged, read,
					isl_union_map_copy(prefix), 1);
	local = isl_union_set_apply(local, isl_union_map_copy(prog->to_inner));
	local_uninitialized = isl_union_map_copy(prog->scop->live_in);
	local_uninitialized = isl_union_map_intersect_range(local_uninitialized,
							    local);
	local_uninitialized = isl_union_map_intersect(local_uninitialized,
						    isl_union_map_copy(read));
	if (!isl_union_map_is_empty(local_uninitialized)) {
		fprintf(stderr,
			"possibly uninitialized reads (not copied in):\n");
		isl_union_map_dump(local_uninitialized);
	}
	read = isl_union_map_subtract(read, local_uninitialized);
	read = isl_union_map_apply_domain(read, prefix);
	copy_in = isl_union_map_union(read, not_written);
	copy_in = isl_union_map_apply_range(copy_in,
				    isl_union_map_copy(prog->to_outer));

	graft = create_copy_device(prog, node, "to_device",
						isl_union_map_range(copy_in));
	node = isl_schedule_node_graft_before(node, graft);
	graft = create_copy_device(prog, node, "from_device",
						isl_union_map_range(copy_out));
	node = isl_schedule_node_graft_after(node, graft);

	return node;
}

/* Update "schedule" for mapping to a GPU device.
 *
 * In particular, insert a context node, create kernels for
 * each outermost tilable band and introduce node for copying array
 * in and out of the device.
 * If the child of the initial root points to a set node,
 * then children of this node that do not contain any tilable bands
 * are separated from the other children and are not mapped to
 * the device.
 */
__isl_give isl_schedule *map_to_device(struct gpu_gen *gen,
	__isl_take isl_schedule *schedule)
{
	isl_schedule_node *node;
	isl_set *context;
	isl_union_set *domain;
	isl_union_map *prefix;

	context = isl_set_copy(gen->prog->context);
	context = isl_set_from_params(context);
	schedule = isl_schedule_insert_context(schedule, context);

	node = isl_schedule_get_root(schedule);
	isl_schedule_free(schedule);
	node = isl_schedule_node_child(node, 0);
	if (isl_schedule_node_get_type(node) == isl_schedule_node_context)
		node = isl_schedule_node_child(node, 0);
	node = isolate_permutable_subtrees(node, gen->prog);
	domain = isl_schedule_node_get_domain(node);
	prefix = isl_schedule_node_get_prefix_schedule_union_map(node);
	node = mark_kernels(gen, node);
	node = add_to_from_device(node, domain, prefix, gen->prog);
	schedule = isl_schedule_node_get_schedule(node);
	isl_schedule_node_free(node);

	return schedule;
}

/* Internal data structure for extract_access.
 * "next_access" points to the end of a linked list that is extended
 * by extract_access.
 * "single_expression" is set if the access expressions belong to
 * an expression statement (i.e., a statement without internal control).
 * "any_to_outer" maps all intermediate arrays to their outer arrays.
 */
struct ppcg_extract_access_data {
	struct gpu_stmt_access **next_access;
	int single_expression;
	isl_union_map *any_to_outer;
};

/* Given a tagged access relation to a single array "tagged", extract it
 * as a map, taking into account that the input may be empty.
 * If the access relation is empty, then it does not contain
 * any space information, so we try to recover it from the index
 * expression.
 * The space of the index expression is of the form I -> A,
 * with I the statement instances and A the array, or [I -> F] -> A,
 * with F the filters corresponding to arguments.
 * We first drop F, if present, obtaining I -> A.
 * Then we construct I -> R, with R the reference tag,
 * combine the two into I -> [R -> A] and uncurry to obtain
 * the final result [I -> R] -> A.
 * Note that the index expression may have a lower dimension
 * than that of the array, but this dimension is not used
 * if the access relation is empty.
 */
static __isl_give isl_map *extract_single_tagged_access(
	__isl_take isl_union_map *tagged, __isl_keep pet_expr *expr)
{
	int empty;
	isl_id *id;
	isl_space *space, *space2;
	isl_multi_pw_aff *index;

	empty = isl_union_map_is_empty(tagged);
	if (empty < 0)
		goto error;
	if (!empty)
		return isl_map_from_union_map(tagged);
	isl_union_map_free(tagged);

	index = pet_expr_access_get_index(expr);
	space = isl_multi_pw_aff_get_space(index);
	isl_multi_pw_aff_free(index);
	if (isl_space_domain_is_wrapping(space))
		space = isl_space_domain_factor_domain(space);
	space2 = isl_space_copy(space);
	space2 = isl_space_from_domain(isl_space_domain(space));
	id = pet_expr_access_get_ref_id(expr);
	space2 = isl_space_set_tuple_id(space2, isl_dim_out, id);
	space = isl_space_range_product(space2, space);
	space = isl_space_uncurry(space);

	return isl_map_empty(space);
error:
	isl_union_map_free(tagged);
	return NULL;
}

/* Extract a gpu_stmt_access from "expr", append it to the list
 * that ends in *data->next_access and update the end of the list.
 * If the access expression performs a write, then it is considered
 * exact only if it appears in a single expression statement and
 * if its may access relation is equal to its must access relation.
 *
 * The combined set of may accesses may be union if member accesses
 * are involved, but the entire set is derived from a single reference and
 * therefore from a single index expression.  These accesses therefore
 * all map to the same outer array.
 */
static int extract_access(__isl_keep pet_expr *expr, void *user)
{
	struct ppcg_extract_access_data *data = user;
	isl_union_map *tagged;
	struct gpu_stmt_access *access;
	isl_ctx *ctx = pet_expr_get_ctx(expr);
	isl_multi_pw_aff *index;

	access = isl_alloc_type(ctx, struct gpu_stmt_access);
	assert(access);
	access->next = NULL;
	access->read = pet_expr_access_is_read(expr);
	access->write = pet_expr_access_is_write(expr);
	tagged = pet_expr_access_get_tagged_may_read(expr);
	tagged = isl_union_map_union(tagged,
				pet_expr_access_get_tagged_may_write(expr));
	tagged = isl_union_map_apply_range(tagged,
					isl_union_map_copy(data->any_to_outer));
	if (!access->write) {
		access->exact_write = 1;
	} else if (!data->single_expression) {
		access->exact_write = 0;
	} else {
		isl_union_map *must, *may;
		may = isl_union_map_copy(tagged);
		may = isl_union_map_domain_factor_domain(may);
		must = pet_expr_access_get_must_write(expr);
		access->exact_write = isl_union_map_is_equal(must, may);
		isl_union_map_free(must);
		isl_union_map_free(may);
	}
	index = pet_expr_access_get_index(expr);
	access->n_index = isl_multi_pw_aff_dim(index, isl_dim_out);
	isl_multi_pw_aff_free(index);
	access->ref_id = pet_expr_access_get_ref_id(expr);
	access->tagged_access = extract_single_tagged_access(tagged, expr);
	access->access = isl_map_copy(access->tagged_access);
	access->access = isl_map_domain_factor_domain(access->access);

	*data->next_access = access;
	data->next_access = &(*data->next_access)->next;

	if (!access->access)
		return -1;

	return 0;
}

/* Construct a linked list of gpu_stmt_access objects,
 * one for each access expression in the statement body.
 * "any_to_outer" maps all intermediate arrays to their outer arrays.
 */
static int pet_stmt_extract_accesses(struct gpu_stmt *stmt,
	__isl_keep isl_union_map *any_to_outer)
{
	struct ppcg_extract_access_data data;

	stmt->accesses = NULL;
	data.next_access = &stmt->accesses;
	data.single_expression =
		pet_tree_get_type(stmt->stmt->body) == pet_tree_expr;
	data.any_to_outer = any_to_outer;
	return pet_tree_foreach_access_expr(stmt->stmt->body,
						&extract_access, &data);
}

/* Return an array of gpu_stmt representing the statements in "scop".
 */
static struct gpu_stmt *extract_stmts(isl_ctx *ctx, struct ppcg_scop *scop,
	__isl_keep isl_set *context, __isl_keep isl_union_map *any_to_outer)
{
	int i;
	struct gpu_stmt *stmts;

	stmts = isl_calloc_array(ctx, struct gpu_stmt, scop->pet->n_stmt);
	if (!stmts)
		return NULL;

	for (i = 0; i < scop->pet->n_stmt; ++i) {
		struct gpu_stmt *s = &stmts[i];

		s->id = isl_set_get_tuple_id(scop->pet->stmts[i]->domain);
		s->stmt = scop->pet->stmts[i];
		if (pet_stmt_extract_accesses(s, any_to_outer) < 0)
			return free_stmts(stmts, i + 1);
	}

	return stmts;
}

/* Callback for ppcg_print_guarded that calls the callback for generate_gpu.
 */
static __isl_give isl_printer *print_gpu(__isl_take isl_printer *p, void *user)
{
	struct gpu_gen *gen = user;

	return gen->print(p, gen->prog, gen->tree, &gen->types,
			    gen->print_user);
}

/* Generate CUDA code for "scop" and print it to "p".
 * After generating an AST for the transformed scop as explained below,
 * we call "gen->print" to print the AST in the desired output format
 * to "p".
 *
 * If it turns out that it does not make sense to generate GPU code,
 * then we generate CPU code instead.
 *
 * The GPU code is generated in a context where at least one
 * statement instance is executed.  The corresponding guard (if any) is printed
 * around the entire generated GPU code, except for the declaration
 * of the arrays that are visible outside of the scop and that therefore
 * cannot be declared inside the body of any possible guard.
 *
 * We first compute a schedule that respects the dependences
 * of the original program and select the outermost bands
 * of tilable dimensions that have at least one parallel loop.
 * If the --load-schedule is specified, then the loaded schedule
 * is used instead of a computed schedule.
 *
 * Each of these bands B is then tiled according to "tile" sizes, resulting
 * in two nested bands, with a kernel marker on top
 *
 *		K
 *		|
 *		T
 *		|
 *		P
 *
 * We then split off at most 2 parallel dimensions from the T band and
 * at most 3 parallel dimension from the P band
 *
 *		K
 *		|
 *		T
 *		T1
 *		|
 *		T2
 *		|
 *		P1
 *		|
 *		P2
 *
 * A filter is introduced in front of T1 that maps the domain instances
 * to block identifiers.  Similarly, a filter is introduced in front of P1
 * that maps the domain instances to thread identifiers.
 *
 * For each iteration of the T2 band and for each array, we compute
 * the array elements accessed by that iteration, construct a rectangular
 * box around it and shift it to the origin.  The result is used
 * as shared memory for the array.
 *
 * Copying and synchronization statements are added to this schedule tree.
 * In principle, these are added in front of the P1 band, but some of
 * them may get hoisted up to higher levels.
 *
 * The entire AST is then generated from the single resulting schedule tree.
 * During the generation the subtrees at kernel nodes (K) are saved
 * aside and replaced by kernel calls.  The result is printed as host code
 * while the saved subtrees are printed as device code.
 */
static __isl_give isl_printer *generate(__isl_take isl_printer *p,
	struct gpu_gen *gen, struct ppcg_scop *scop,
	struct ppcg_options *options)
{
	struct gpu_prog *prog;
	isl_ctx *ctx;
	isl_set *context, *guard;
	isl_schedule *schedule;
	int any_permutable;

	if (!scop)
		return isl_printer_free(p);

	ctx = isl_printer_get_ctx(p);
	prog = gpu_prog_alloc(ctx, scop);
	if (!prog)
		return isl_printer_free(p);

	context = isl_set_copy(prog->context);
	guard = isl_union_set_params(isl_union_set_copy(prog->scop->domain));
	prog->context = isl_set_intersect(prog->context, isl_set_copy(guard));

	gen->prog = prog;
	schedule = get_schedule(gen);

	any_permutable = has_any_permutable_node(schedule);
	if (any_permutable < 0 || !any_permutable) {
		isl_set_free(context);
		isl_set_free(guard);
		if (any_permutable < 0)
			p = isl_printer_free(p);
		else
			p = print_cpu(p, scop, options);
		isl_schedule_free(schedule);
	} else {
		schedule = map_to_device(gen, schedule);
		gen->tree = generate_code(gen, schedule);
		p = isl_ast_op_type_print_macro(isl_ast_op_fdiv_q, p);
		p = ppcg_print_exposed_declarations(p, prog->scop);
		p = ppcg_print_guarded(p, guard, context, &print_gpu, gen);
		isl_ast_node_free(gen->tree);
	}

	gpu_prog_free(prog);

	return p;
}

/* Wrapper around generate for use as a ppcg_transform callback.
 */
static __isl_give isl_printer *generate_wrap(__isl_take isl_printer *p,
	struct ppcg_scop *scop, void *user)
{
	struct gpu_gen *gen = user;

	return generate(p, gen, scop, gen->options);
}

/* Transform the code in the file called "input" by replacing
 * all scops by corresponding GPU code and write the results to "out".
 */
int generate_gpu(isl_ctx *ctx, const char *input, FILE *out,
	struct ppcg_options *options,
	__isl_give isl_printer *(*print)(__isl_take isl_printer *p,
		struct gpu_prog *prog, __isl_keep isl_ast_node *tree,
		struct gpu_types *types, void *user), void *user)
{
	struct gpu_gen gen;
	int r;
	int i;

	gen.ctx = ctx;
	gen.sizes = extract_sizes_from_str(ctx, options->sizes);
	gen.options = options;
	gen.kernel_id = 0;
	gen.print = print;
	gen.print_user = user;
	gen.types.n = 0;
	gen.types.name = NULL;

	if (options->debug->dump_sizes) {
		isl_space *space = isl_space_params_alloc(ctx, 0);
		gen.used_sizes = isl_union_map_empty(space);
	}

	r = ppcg_transform(ctx, input, out, options, &generate_wrap, &gen);

	if (options->debug->dump_sizes) {
		isl_union_map_dump(gen.used_sizes);
		isl_union_map_free(gen.used_sizes);
	}

	isl_union_map_free(gen.sizes);
	for (i = 0; i < gen.types.n; ++i)
		free(gen.types.name[i]);
	free(gen.types.name);

	return r;
}

/* Compute the set of inner array elements that may have their values
 * preserved by "prog".  In particular, collect the array elements of
 * arrays that are not local to "prog" and remove those elements that
 * are definitely killed or definitely written by "prog".
 */
__isl_give isl_union_set *compute_may_persist(struct gpu_prog *prog)
{
	int i;
	isl_union_set *may_persist, *killed;
	isl_union_map *must_kill;

	may_persist = isl_union_set_empty(isl_set_get_space(prog->context));
	for (i = 0; i < prog->n_array; ++i) {
		isl_set *extent;

		if (prog->array[i].local)
			continue;

		extent = isl_set_copy(prog->array[i].extent);
		may_persist = isl_union_set_add_set(may_persist, extent);
	}

	may_persist = isl_union_set_intersect_params(may_persist,
						isl_set_copy(prog->context));
	may_persist = isl_union_set_apply(may_persist,
					isl_union_map_copy(prog->to_inner));
	must_kill = isl_union_map_copy(prog->tagged_must_kill);
	killed = isl_union_map_range(must_kill);
	must_kill = isl_union_map_copy(prog->must_write);
	killed = isl_union_set_union(killed, isl_union_map_range(must_kill));

	may_persist = isl_union_set_subtract(may_persist, killed);
	return may_persist;
}

struct gpu_prog *gpu_prog_alloc(isl_ctx *ctx, struct ppcg_scop *scop)
{
	struct gpu_prog *prog;
	isl_space *space;
	isl_map *id;

	if (!scop)
		return NULL;

	prog = isl_calloc_type(ctx, struct gpu_prog);
	assert(prog);

	prog->ctx = ctx;
	prog->scop = scop;
	prog->context = isl_set_copy(scop->context);
	prog->n_stmts = scop->pet->n_stmt;
	prog->any_to_outer = pet_scop_compute_outer_to_any(scop->pet);
	prog->any_to_outer = isl_union_map_reverse(prog->any_to_outer);
	space = isl_union_map_get_space(prog->any_to_outer);
	space = isl_space_set_from_params(space);
	space = isl_space_add_dims(space, isl_dim_set, 1);
	space = isl_space_map_from_set(space);
	id = isl_map_identity(space);
	prog->any_to_outer = isl_union_map_add_map(prog->any_to_outer, id);
	prog->stmts = extract_stmts(ctx, scop,
					prog->context, prog->any_to_outer);
	prog->read = isl_union_map_copy(scop->reads);
	prog->may_write = isl_union_map_copy(scop->may_writes);
	prog->must_write = isl_union_map_copy(scop->must_writes);
	prog->tagged_must_kill = isl_union_map_copy(scop->tagged_must_kills);
	prog->to_inner = pet_scop_compute_outer_to_inner(scop->pet);
	prog->to_outer = isl_union_map_copy(prog->to_inner);
	prog->to_outer = isl_union_map_reverse(prog->to_outer);

	if (!prog->stmts)
		return gpu_prog_free(prog);

	if (collect_array_info(prog) < 0)
		return gpu_prog_free(prog);
	prog->may_persist = compute_may_persist(prog);

	return prog;
}

void *gpu_prog_free(struct gpu_prog *prog)
{
	if (!prog)
		return NULL;
	free_array_info(prog);
	free_stmts(prog->stmts, prog->n_stmts);
	isl_union_map_free(prog->any_to_outer);
	isl_union_map_free(prog->to_outer);
	isl_union_map_free(prog->to_inner);
	isl_union_map_free(prog->read);
	isl_union_map_free(prog->may_write);
	isl_union_map_free(prog->must_write);
	isl_union_map_free(prog->tagged_must_kill);
	isl_union_map_free(prog->array_order);
	isl_union_set_free(prog->may_persist);
	isl_set_free(prog->context);
	free(prog);
	return NULL;
}
