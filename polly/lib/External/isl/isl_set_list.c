#include <isl/set.h>
#include <isl/union_set.h>

#undef EL
#define EL isl_basic_set

#include <isl_list_templ.h>

#undef EL
#define EL isl_set

#include <isl_list_templ.h>

#undef EL
#define EL isl_union_set

#include <isl_list_templ.h>

#undef EL_BASE
#define EL_BASE basic_set

#include <isl_list_templ.c>

#undef EL_BASE
#define EL_BASE set

#include <isl_list_templ.c>
#include <isl_list_read_templ.c>

#undef EL_BASE
#define EL_BASE union_set

#include <isl_list_templ.c>
#include <isl_list_read_templ.c>
