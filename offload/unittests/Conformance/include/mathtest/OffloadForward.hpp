//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains forward declarations for the opaque types and handles
/// used by the Offload API.
///
//===----------------------------------------------------------------------===//

#ifndef MATHTEST_OFFLOADFORWARD_HPP
#define MATHTEST_OFFLOADFORWARD_HPP

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

#endif // MATHTEST_OFFLOADFORWARD_HPP
