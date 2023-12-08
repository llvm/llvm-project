#undef EL_BASE
#define EL_BASE BASE
#include <isl_list_macro.h>

#define xMULTI(BASE) isl_multi_ ## BASE
#define MULTI(BASE) xMULTI(BASE)
#undef DOM
#define DOM CAT(isl_,DOMBASE)
