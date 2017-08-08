#ifndef GPU_GROUP_H
#define GPU_GROUP_H

#include <isl/schedule_node.h>
#include "gpu.h"

/* A group of array references in a kernel that should be handled together.
 * If private_tile is not NULL, then it is mapped to registers.
 * Otherwise, if shared_tile is not NULL, it is mapped to shared memory.
 * Otherwise, it is accessed from global memory.
 * Note that if both private_tile and shared_tile are set, then shared_tile
 * is only used inside group_common_shared_memory_tile.
 * "depth" reflects the number of schedule dimensions that affect the tile
 * (private_tile if set; shared_tile if shared_tile is set and private_tile
 * is not).  The copying into and/or out of the tile is performed at that
 * depth.
 */
struct gpu_array_ref_group {
	/* The references in this group access this local array. */
	struct gpu_local_array_info *local_array;
	/* This is the corresponding array. */
	struct gpu_array_info *array;
	/* Position of this group in the list of reference groups of array. */
	int nr;

	/* The following fields are use during the construction of the groups.
	 * access is the combined access relation relative to the shared
	 * memory tiling.  In particular, the domain of the map corresponds
	 * to the first shared_schedule_dim dimensions of the kernel schedule.
	 * write is set if any access in the group is a write.
	 * exact_write is set if all writes are definite writes.
	 * slice is set if there is at least one access in the group
	 * that refers to more than one element
	 */
	isl_map *access;
	int write;
	int exact_write;
	int slice;

	/* The shared memory tile, NULL if none. */
	struct gpu_array_tile *shared_tile;

	/* The private memory tile, NULL if none. */
	struct gpu_array_tile *private_tile;

	int depth;

	/* References in this group; point to elements of a linked list. */
	int n_ref;
	struct gpu_stmt_access **refs;
};

int gpu_group_references(struct ppcg_kernel *kernel,
	__isl_keep isl_schedule_node *node);

__isl_give isl_printer *gpu_array_ref_group_print_name(
	struct gpu_array_ref_group *group, __isl_take isl_printer *p);
void gpu_array_ref_group_compute_tiling(struct gpu_array_ref_group *group);
__isl_give isl_union_map *gpu_array_ref_group_access_relation(
	struct gpu_array_ref_group *group, int read, int write);
int gpu_array_ref_group_requires_unroll(struct gpu_array_ref_group *group);
struct gpu_array_tile *gpu_array_ref_group_tile(
	struct gpu_array_ref_group *group);
struct gpu_array_ref_group *gpu_array_ref_group_free(
	struct gpu_array_ref_group *group);

#endif
