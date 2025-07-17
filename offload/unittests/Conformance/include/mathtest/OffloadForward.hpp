#pragma once

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

struct ol_error_struct_t;
typedef const ol_error_struct_t *ol_result_t;
#define OL_SUCCESS (static_cast<ol_result_t>(nullptr))

struct ol_device_impl_t;
typedef struct ol_device_impl_t *ol_device_handle_t;

struct ol_program_impl_t;
typedef struct ol_program_impl_t *ol_program_handle_t;

struct ol_symbol_impl_t;
typedef struct ol_symbol_impl_t *ol_symbol_handle_t;

#ifdef __cplusplus
}
#endif // __cplusplus
