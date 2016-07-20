#include <isl/aff.h>
#include <isl/map.h>

#include "gpu_array_tile.h"

struct gpu_array_tile *gpu_array_tile_free(struct gpu_array_tile *tile)
{
	int j;

	if (!tile)
		return NULL;

	for (j = 0; j < tile->n; ++j) {
		isl_val_free(tile->bound[j].size);
		isl_val_free(tile->bound[j].stride);
		isl_aff_free(tile->bound[j].lb);
		isl_aff_free(tile->bound[j].shift);
	}
	free(tile->bound);
	isl_multi_aff_free(tile->tiling);
	free(tile);

	return NULL;
}

/* Create a gpu_array_tile for an array of dimension "n_index".
 */
struct gpu_array_tile *gpu_array_tile_create(isl_ctx *ctx, int n_index)
{
	int i;
	struct gpu_array_tile *tile;

	tile = isl_calloc_type(ctx, struct gpu_array_tile);
	if (!tile)
		return NULL;

	tile->ctx = ctx;
	tile->bound = isl_alloc_array(ctx, struct gpu_array_bound, n_index);
	if (!tile->bound)
		return gpu_array_tile_free(tile);

	tile->n = n_index;

	for (i = 0; i < n_index; ++i) {
		tile->bound[i].size = NULL;
		tile->bound[i].lb = NULL;
		tile->bound[i].stride = NULL;
		tile->bound[i].shift = NULL;
	}

	return tile;
}

/* Compute the size of the tile specified by "tile"
 * in number of elements and return the result.
 */
__isl_give isl_val *gpu_array_tile_size(struct gpu_array_tile *tile)
{
	int i;
	isl_val *size;

	if (!tile)
		return NULL;

	size = isl_val_one(tile->ctx);

	for (i = 0; i < tile->n; ++i)
		size = isl_val_mul(size, isl_val_copy(tile->bound[i].size));

	return size;
}
