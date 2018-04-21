/*
 * Use of this software is governed by the MIT license
 */

#ifndef ISL_STRIDE_INFO_H
#define ISL_STRIDE_INFO_H

#include <isl/val.h>
#include <isl/aff_type.h>

#if defined(__cplusplus)
extern "C" {
#endif

struct isl_stride_info;
typedef struct isl_stride_info isl_stride_info;

__isl_give isl_val *isl_stride_info_get_stride(__isl_keep isl_stride_info *si);
__isl_give isl_aff *isl_stride_info_get_offset(__isl_keep isl_stride_info *si);
__isl_null isl_stride_info *isl_stride_info_free(
	__isl_take isl_stride_info *si);

#if defined(__cplusplus)
}
#endif

#endif
