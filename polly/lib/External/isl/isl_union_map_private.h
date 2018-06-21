#define isl_union_set_list	isl_union_map_list
#define isl_union_set	isl_union_map
#include <isl/union_map.h>
#include <isl/union_set.h>

struct isl_union_map {
	int ref;
	isl_space *dim;

	struct isl_hash_table	table;
};

__isl_keep isl_space *isl_union_map_peek_space(__isl_keep isl_union_map *umap);
isl_bool isl_union_map_is_params(__isl_keep isl_union_map *umap);
isl_bool isl_union_map_space_has_equal_params(__isl_keep isl_union_map *umap,
	__isl_keep isl_space *space);
isl_bool isl_union_set_space_has_equal_params(__isl_keep isl_union_set *uset,
	__isl_keep isl_space *space);
__isl_give isl_union_map *isl_union_map_reset_range_space(
	__isl_take isl_union_map *umap, __isl_take isl_space *space);
__isl_give isl_union_map *isl_union_map_reset_equal_dim_space(
	__isl_take isl_union_map *umap, __isl_take isl_space *space);
