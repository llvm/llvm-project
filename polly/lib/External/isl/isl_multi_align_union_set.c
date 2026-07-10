#undef SUFFIX
#define SUFFIX	union_set
#undef ARG1
#define ARG1	MULTI(BASE)
#undef ARG2
#define ARG2	isl_union_set

#include "isl_align_params_templ.c"
