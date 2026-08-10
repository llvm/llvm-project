#include <isl/ctx.h>
#include <isl/id_to_id.h>
#include <isl/id.h>

#define isl_id_is_equal(id1,id2)	isl_bool_ok(id1 == id2)

#define ISL_KEY		isl_id
#define ISL_VAL		isl_id
#define ISL_HMAP_SUFFIX	id_to_id
#define ISL_HMAP	isl_id_to_id
#define ISL_HMAP_IS_EQUAL	isl_id_to_id_is_equal
#define ISL_KEY_IS_EQUAL	isl_id_is_equal
#define ISL_VAL_IS_EQUAL	isl_id_is_equal
#define ISL_KEY_PRINT		isl_printer_print_id
#define ISL_VAL_PRINT		isl_printer_print_id
#define ISL_HMAP_HAVE_READ_FROM_STR
#define ISL_KEY_READ		isl_stream_read_id
#define ISL_VAL_READ		isl_stream_read_id

#include <isl/hmap_templ.c>
