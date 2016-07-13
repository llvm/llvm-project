#ifndef GPU_TREE_H
#define GPU_TREE_H

#include <isl/schedule_node.h>

#include "gpu.h"

int gpu_tree_node_is_kernel(__isl_keep isl_schedule_node *node);
__isl_give isl_schedule_node *gpu_tree_move_up_to_thread(
	__isl_take isl_schedule_node *node);
__isl_give isl_schedule_node *gpu_tree_move_down_to_thread(
	__isl_take isl_schedule_node *node, __isl_keep isl_union_set *core);
__isl_give isl_schedule_node *gpu_tree_move_up_to_kernel(
	__isl_take isl_schedule_node *node);
__isl_give isl_schedule_node *gpu_tree_move_down_to_depth(
	__isl_take isl_schedule_node *node, int depth,
	__isl_keep isl_union_set *core);

int gpu_tree_id_is_sync(__isl_keep isl_id *id, struct ppcg_kernel *kernel);
__isl_give isl_schedule_node *gpu_tree_ensure_sync_after_core(
	__isl_take isl_schedule_node *node, struct ppcg_kernel *kernel);
__isl_give isl_schedule_node *gpu_tree_ensure_following_sync(
	__isl_take isl_schedule_node *node, struct ppcg_kernel *kernel);
__isl_give isl_schedule_node *gpu_tree_move_left_to_sync(
	__isl_take isl_schedule_node *node, struct ppcg_kernel *kernel);
__isl_give isl_schedule_node *gpu_tree_move_right_to_sync(
	__isl_take isl_schedule_node *node, struct ppcg_kernel *kernel);

#endif
