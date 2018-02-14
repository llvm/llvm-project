#ifndef HYBRID_H
#define HYBRID_H

#include <isl/val.h>
#include <isl/schedule_node.h>

#include "ppcg.h"

struct ppcg_ht_bounds;
typedef struct ppcg_ht_bounds ppcg_ht_bounds;

struct ppcg_ht_phase;
typedef struct ppcg_ht_phase ppcg_ht_phase;

isl_bool ppcg_ht_has_input_pattern(__isl_keep isl_schedule_node *node);
isl_bool ppcg_ht_parent_has_input_pattern(__isl_keep isl_schedule_node *node);

__isl_give ppcg_ht_bounds *ppcg_ht_compute_bounds(struct ppcg_scop *scop,
	__isl_keep isl_schedule_node *node);
void ppcg_ht_bounds_dump(__isl_keep ppcg_ht_bounds *bounds);
isl_bool ppcg_ht_bounds_is_valid(__isl_keep ppcg_ht_bounds *bounds);
isl_bool ppcg_ht_bounds_supports_sizes(__isl_keep ppcg_ht_bounds *bounds,
	__isl_keep isl_multi_val *sizes);
__isl_give isl_schedule_node *ppcg_ht_bounds_insert_tiling(
	__isl_take ppcg_ht_bounds *bounds, __isl_take isl_multi_val *sizes,
	__isl_take isl_schedule_node *node, struct ppcg_options *options);
__isl_null ppcg_ht_bounds *ppcg_ht_bounds_free(
	__isl_take ppcg_ht_bounds *bounds);

__isl_keep ppcg_ht_phase *ppcg_ht_phase_extract_from_mark(
	__isl_keep isl_schedule_node *node);
__isl_give isl_schedule_node *ppcg_ht_phase_shift_space_point(
	__isl_keep ppcg_ht_phase *phase, __isl_take isl_schedule_node *node);
__isl_give isl_schedule_node *hybrid_tile_foreach_phase(
	__isl_take isl_schedule_node *node,
	__isl_give isl_schedule_node *(*fn)(__isl_take isl_schedule_node *node,
		void *user), void *user);
__isl_give isl_schedule_node *hybrid_tile_drop_phase_marks(
	__isl_take isl_schedule_node *node);

#endif
