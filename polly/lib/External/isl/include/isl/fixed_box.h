/*
 * Use of this software is governed by the MIT license
 */

#ifndef ISL_FIXED_BOX_H
#define ISL_FIXED_BOX_H

#include <isl/ctx.h>
#include <isl/val_type.h>
#include <isl/space_type.h>
#include <isl/aff_type.h>

#if defined(__cplusplus)
extern "C" {
#endif

struct __isl_export isl_fixed_box;
typedef struct isl_fixed_box isl_fixed_box;

isl_ctx *isl_fixed_box_get_ctx(__isl_keep isl_fixed_box *box);
__isl_export
__isl_give isl_space *isl_fixed_box_get_space(__isl_keep isl_fixed_box *box);
__isl_export
isl_bool isl_fixed_box_is_valid(__isl_keep isl_fixed_box *box);
__isl_export
__isl_give isl_multi_aff *isl_fixed_box_get_offset(
	__isl_keep isl_fixed_box *box);
__isl_export
__isl_give isl_multi_val *isl_fixed_box_get_size(__isl_keep isl_fixed_box *box);

__isl_give isl_fixed_box *isl_fixed_box_copy(__isl_keep isl_fixed_box *box);
__isl_null isl_fixed_box *isl_fixed_box_free(__isl_take isl_fixed_box *box);

__isl_give isl_printer *isl_printer_print_fixed_box(
	__isl_take isl_printer *p, __isl_keep isl_fixed_box *box);
__isl_give char *isl_fixed_box_to_str(__isl_keep isl_fixed_box *box);
void isl_fixed_box_dump(__isl_keep isl_fixed_box *box);

#if defined(__cplusplus)
}
#endif

#endif
