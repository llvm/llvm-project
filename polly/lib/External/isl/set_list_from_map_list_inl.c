#include <isl/map_type.h>

/* Return the set list that was treated as the map list "list".
 */
static __isl_give isl_set_list *set_list_from_map_list(
	__isl_take isl_map_list *list)
{
	return (isl_set_list *) list;
}
