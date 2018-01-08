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

#include <isl/val.h>
#include <isl/space.h>
#include <isl/union_set.h>
#include <isl/schedule_node.h>

#include "hybrid.h"
#include "gpu_hybrid.h"
#include "gpu_tree.h"
#include "schedule.h"
#include "util.h"

/* Have all domain elements been filtered out before reaching
 * the "node" position in the schedule tree?
 */
static isl_bool has_empty_domain(__isl_keep isl_schedule_node *node)
{
	isl_union_set *domain;
	isl_bool empty;

	domain = isl_schedule_node_get_domain(node);
	empty = isl_union_set_is_empty(domain);
	isl_union_set_free(domain);

	return empty;
}

/* Given a pointer to a phase in the result of hybrid tiling,
 * map the phase to the device, provided the phase is non-empty.
 * Empty phases can occur if the input schedule domain can be
 * covered by a small number of hexagons that all belong to the same phase.
 *
 * The input has the following form:
 *
 *	M - CT - P - C - ...
 *
 * with M the phase marker, CT the space tiling, P the original
 * parent band and C the original child band.
 * The (outer dimensions of the) C band need to be mapped to threads.
 * The (outer dimension of the) CT band needs to be mapped to blocks.
 * The mapping to shared memory needs to be computed between the CT and
 * the P band.
 *
 * The C band is first shifted to start at zero.
 * Then the appropriate markers are introduced and a kernel is
 * created for the tree rooted at CT.
 * If the "unroll_gpu_tile" option is set, then the AST generator
 * is instructed to unroll the P and C bands.
 */
static __isl_give isl_schedule_node *update_phase(
	__isl_take isl_schedule_node *node, void *user)
{
	struct gpu_gen *gen = user;
	int depth0, depth;
	isl_ctx *ctx;
	isl_id *id;
	isl_bool empty_domain;
	ppcg_ht_phase *phase;

	empty_domain = has_empty_domain(node);
	if (empty_domain < 0)
		return isl_schedule_node_free(node);
	if (empty_domain)
		return node;

	if (!node)
		return NULL;
	ctx = isl_schedule_node_get_ctx(node);

	phase = ppcg_ht_phase_extract_from_mark(node);

	depth0 = isl_schedule_node_get_tree_depth(node);

	node = isl_schedule_node_child(node, 0);

	node = isl_schedule_node_child(node, 0);
	node = isl_schedule_node_child(node, 0);
	node = ppcg_ht_phase_shift_space_point(phase, node);
	if (gen->options->unroll_gpu_tile)
		node = ppcg_set_schedule_node_type(node, isl_ast_loop_unroll);
	id = isl_id_alloc(ctx, "thread", NULL);
	node = isl_schedule_node_insert_mark(node, id);
	node = isl_schedule_node_parent(node);
	if (gen->options->unroll_gpu_tile)
		node = ppcg_set_schedule_node_type(node, isl_ast_loop_unroll);
	id = isl_id_alloc(ctx, "shared", NULL);
	node = isl_schedule_node_insert_mark(node, id);
	node = isl_schedule_node_parent(node);

	node = gpu_create_kernel(gen, node, 0, NULL);

	depth = isl_schedule_node_get_tree_depth(node);
	node = isl_schedule_node_ancestor(node, depth - depth0);

	return node;
}

/* Apply hybrid tiling on "node" and its parent based on the (valid)
 * bounds on the relative dependence distances "bounds" and
 * the tile sizes in "tile_sizes".
 * The number of elements in "tile_sizes" is at least as large
 * as the sum of the dimensions of the parent and the child node.
 *
 * Convert the tile_sizes to an isl_multi_val in the right space,
 * insert the hybrid tiling and then create a kernel inside each phase.
 * Finally, remove the phase marks.
 */
__isl_give isl_schedule_node *gpu_hybrid_tile(struct gpu_gen *gen,
	__isl_take isl_schedule_node *node, __isl_take ppcg_ht_bounds *bounds,
	int *tile_sizes)
{
	isl_multi_val *mv;
	isl_space *space, *space2;

	if (!node || !bounds)
		goto error;

	space2 = isl_schedule_node_band_get_space(node);
	node = isl_schedule_node_parent(node);
	space = isl_schedule_node_band_get_space(node);
	space = isl_space_product(space, space2);
	mv = ppcg_multi_val_from_int_list(space, tile_sizes);

	node = ppcg_ht_bounds_insert_tiling(bounds, mv, node, gen->options);

	node = hybrid_tile_foreach_phase(node, &update_phase, gen);

	node = hybrid_tile_drop_phase_marks(node);

	return node;
error:
	isl_schedule_node_free(node);
	ppcg_ht_bounds_free(bounds);
	return NULL;
}
