#ifndef ISL_VAL_TYPE_H
#define ISL_VAL_TYPE_H

#include <isl/list.h>

#if defined(__cplusplus)
extern "C" {
#endif

struct __isl_export isl_val;
typedef struct isl_val isl_val;

ISL_DECLARE_LIST_TYPE(val)

struct __isl_export isl_multi_val;
typedef struct isl_multi_val isl_multi_val;

#if defined(__cplusplus)
}
#endif

#endif
