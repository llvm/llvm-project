#include <isl/id_to_id.h>

#define isl_id_is_equal(id1,id2)	id1 == id2

#define KEY_BASE	id
#define KEY_EQUAL	isl_id_is_equal
#define VAL_BASE	id
#define VAL_EQUAL	isl_id_is_equal

#include <isl_hmap_templ.c>
