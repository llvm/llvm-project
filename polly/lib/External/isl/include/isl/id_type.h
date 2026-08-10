#ifndef ISL_ID_TYPE_H
#define ISL_ID_TYPE_H

#include <isl/list.h>

#if defined(__cplusplus)
extern "C" {
#endif

struct __isl_export isl_id;
typedef struct isl_id isl_id;

ISL_DECLARE_EXPORTED_LIST_TYPE(id)

struct __isl_export isl_multi_id;
typedef struct isl_multi_id isl_multi_id;

#if defined(__cplusplus)
}
#endif

#endif
