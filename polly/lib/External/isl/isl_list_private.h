#ifndef ISL_LIST_PRIVATE_H
#define ISL_LIST_PRIVATE_H

#include <isl/list.h>

#define ISL_DECLARE_LIST_FN_PRIVATE(EL)					\
__isl_keep isl_##EL *isl_##EL##_list_peek(				\
	__isl_keep isl_##EL##_list *list, int index);

#endif
