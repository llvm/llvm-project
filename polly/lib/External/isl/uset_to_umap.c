#include <isl/union_map_type.h>

/* Treat "uset" as a union map.
 * Internally, isl_union_set is defined to isl_union_map, so in practice,
 * this function performs a redundant cast.
 */
static __isl_give isl_union_map *uset_to_umap(__isl_take isl_union_set *uset)
{
	return (isl_union_map *) uset;
}
