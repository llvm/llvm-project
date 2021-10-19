#ifndef ISL_ID_H
#define ISL_ID_H

#include <isl/ctx.h>
#include <isl/id_type.h>
#include <isl/list.h>
#include <isl/multi.h>
#include <isl/printer_type.h>
#include <isl/stdint.h>

#if defined(__cplusplus)
extern "C" {
#endif

ISL_DECLARE_EXPORTED_LIST_FN(id)
ISL_DECLARE_EXPORTED_LIST_FN_READ(id)

ISL_DECLARE_MULTI(id)

isl_ctx *isl_id_get_ctx(__isl_keep isl_id *id);
uint32_t isl_id_get_hash(__isl_keep isl_id *id);

__isl_give isl_id *isl_id_alloc(isl_ctx *ctx,
	__isl_keep const char *name, void *user);
__isl_give isl_id *isl_id_copy(isl_id *id);
__isl_null isl_id *isl_id_free(__isl_take isl_id *id);

void *isl_id_get_user(__isl_keep isl_id *id);
__isl_export
__isl_keep const char *isl_id_get_name(__isl_keep isl_id *id);

__isl_give isl_id *isl_id_set_free_user(__isl_take isl_id *id,
	void (*free_user)(void *user));

__isl_constructor
__isl_give isl_id *isl_id_read_from_str(isl_ctx *ctx, const char *str);
__isl_give char *isl_id_to_str(__isl_keep isl_id *id);
__isl_give isl_printer *isl_printer_print_id(__isl_take isl_printer *p,
	__isl_keep isl_id *id);
void isl_id_dump(__isl_keep isl_id *id);

__isl_constructor
__isl_give isl_multi_id *isl_multi_id_read_from_str(isl_ctx *ctx,
	const char *str);
__isl_give isl_printer *isl_printer_print_multi_id(__isl_take isl_printer *p,
	__isl_keep isl_multi_id *mi);
void isl_multi_id_dump(__isl_keep isl_multi_id *mi);
__isl_give char *isl_multi_id_to_str(__isl_keep isl_multi_id *mi);

#if defined(__cplusplus)
}
#endif

#endif
