#ifndef GPU_ARRAY_TILE_H
#define GPU_ARRAY_TILE_H

#include <isl/aff_type.h>
#include <isl/map_type.h>
#include <isl/val.h>

/* The fields stride and shift only contain valid information
 * if shift != NULL.
 * If so, they express that current index is such that if you add shift,
 * then the result is always a multiple of stride.
 * Let D represent the initial tile->depth dimensions of the computed schedule.
 * The spaces of "lb" and "shift" are of the form
 *
 *	D -> [b]
 */
struct gpu_array_bound {
	isl_val *size;
	isl_aff *lb;

	isl_val *stride;
	isl_aff *shift;
};

/* A tile of an outer array.
 *
 * requires_unroll is set if the schedule dimensions that are mapped
 * to threads need to be unrolled for this (private) tile to be used.
 *
 * "depth" reflects the number of schedule dimensions that affect the tile.
 * The copying into and/or out of the tile is performed at that depth.
 *
 * n is the dimension of the array.
 * bound is an array of size "n" representing the lower bound
 *	and size for each index.
 *
 * tiling maps a tile in the global array to the corresponding
 * shared/private memory tile and is of the form
 *
 *	{ [D[i] -> A[a]] -> T[(a + shift(i))/stride - lb(i)] }
 *
 * where D represents the initial "depth" dimensions
 * of the computed schedule.
 */
struct gpu_array_tile {
	isl_ctx *ctx;
	int requires_unroll;
	int depth;
	int n;
	struct gpu_array_bound *bound;
	isl_multi_aff *tiling;
};

struct gpu_array_tile *gpu_array_tile_create(isl_ctx *ctx, int n_index);
struct gpu_array_tile *gpu_array_tile_free(struct gpu_array_tile *tile);

__isl_give isl_val *gpu_array_tile_size(struct gpu_array_tile *tile);

#endif
