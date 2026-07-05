#include <isl/union_map_type.h>

/* Return the union set that was treated as the union map "umap".
 */
static __isl_give isl_union_set *uset_from_umap(__isl_take isl_union_map *umap)
{
	return (isl_union_set *) umap;
}
