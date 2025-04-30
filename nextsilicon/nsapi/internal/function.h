/**
 * NSAPI handoff utilities.
 *
 * NOTE: DO NOT MODIFY THE FUNCTIONS IN THIS HEADER IN ANY WAY!
 * This header is a duplicate of the original header from nextutils.
 * Any changes to this header must be first made in nextutils and then copied
 * here.
 */

#ifndef _NSAPI_FUNCTION_H
#define _NSAPI_FUNCTION_H

#ifdef __cplusplus
extern "C" {
#endif

typedef const void *llns_function_handle;

#define NSAPI_INVALID_FUNCTION_HANDLE ((llns_function_handle)NULL)

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

/**
 * Check handoff status of a function, or the global handoff setting.
 * Handoff might be disabled for a certain function even when handoff is enabled
 * globally
 *
 * @func_addr: function address to check, or NULL to get the global handoff
 * setting.
 *
 * Returns true if handoff is enabled on the function (or globally), false
 * otherwise
 */
bool nsapi_function_is_handed_off(const void *func_addr);

/**
 * Retrieve a handle to a function, given its address.
 * A valid handle is only available if the function is handed off.
 *
 * @name: function address.
 *
 * Returns a handle to a function, if it exists. Otherwise, returns
 * NSAPI_INVALID_FUNCTION_HANDLE.
 */
llns_function_handle llns_get_function_handle_by_ptr(const void *func_addr);

/**
 * Retrieve a handle to a function, given its name.
 * A valid handle is only available if the function is handed off.
 *
 * @name: function name.
 *
 * Returns a handle to a function, if it exists. Otherwise, returns
 * NSAPI_INVALID_FUNCTION_HANDLE.
 */
llns_function_handle llns_get_function_handle_by_name(const char *func_name);

/**
 * Execute a function on the device. The return value of the function (if it
 * exists) is returned through an output argument.
 *
 * The function intended for handoff must only accept parameters of pointer type
 * or a primitive type equal or smaller than 64 bits. It can only return a
 * primitive type equal or smaller than 64 bits.
 *
 * @handle: function handle previously obtained using one of the handle getters.
 * @args: user allocated array of pointers to executed function's arguments.
 * @args_count: number of arguments in @args.
 * @retval [out]: user allocated memory which will hold the return value of the
 * executed function if it exists, or NULL if it doesn't.
 * @retval_size: size of the return value in bytes.
 *
 * Return true if the execution was successful, false otherwise.
 */
bool llns_execute_function(llns_function_handle handle, void *const *args,
                           size_t args_count, uint8_t *retval,
                           size_t retval_size);

#ifdef __cplusplus
}
#endif

#endif /* _NSAPI_FUNCTION_H */
