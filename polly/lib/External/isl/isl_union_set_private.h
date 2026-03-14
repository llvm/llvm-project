#ifndef ISL_UNION_SET_PRIVATE_H
#define ISL_UNION_SET_PRIVATE_H

#include <isl/union_set.h>

__isl_give isl_union_set *isl_union_set_combined_lineality_space(
	__isl_take isl_union_set *uset);
__isl_give isl_union_set *isl_union_set_plain_gist(
	__isl_take isl_union_set *uset, __isl_take isl_union_set *context);

#endif
