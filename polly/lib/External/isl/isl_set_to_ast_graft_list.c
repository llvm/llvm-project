#include <isl/ctx.h>
#include <isl/set_type.h>
#include "isl_ast_graft_private.h"
#include "isl_set_to_ast_graft_list.h"

#define isl_ast_graft_list_is_identical(a, b)	isl_bool_ok(a == b)

#define ISL_KEY			isl_set
#define ISL_VAL			isl_ast_graft_list
#define ISL_HMAP_SUFFIX		set_to_ast_graft_list
#define ISL_HMAP		isl_set_to_ast_graft_list
#define ISL_HMAP_IS_EQUAL	isl_set_to_ast_graft_list_plain_is_equal
#define ISL_KEY_IS_EQUAL	isl_set_plain_is_equal
#define ISL_VAL_IS_EQUAL	isl_ast_graft_list_is_identical
#define ISL_KEY_PRINT		isl_printer_print_set
#define ISL_VAL_PRINT		isl_printer_print_ast_graft_list
#define ISL_HMAP_HAVE_READ_FROM_STR
#define ISL_KEY_READ		isl_stream_read_set
#define ISL_VAL_READ		isl_stream_read_ast_graft_list

#include <isl/hmap_templ.c>
