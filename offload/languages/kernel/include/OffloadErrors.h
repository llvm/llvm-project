//===-- OffloadErrors.h - Kernel language offload errors ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_OFFLOAD_LANGUAGES_KERNEL_INCLUDE_OFFLOAD_ERRORS_H
#define LLVM_OFFLOAD_LANGUAGES_KERNEL_INCLUDE_OFFLOAD_ERRORS_H

#include "OffloadAPI.h"

namespace llvm {
namespace offload {

inline constexpr ol_error_struct_t InvalidKernelError = {
    OL_ERRC_INVALID_NULL_HANDLE, "kernel is not registered"};

inline constexpr ol_error_struct_t InvalidDeviceError = {OL_ERRC_INVALID_DEVICE,
                                                         "invalid device"};

inline constexpr ol_error_struct_t InvalidArgumentError = {
    OL_ERRC_INVALID_ARGUMENT, "invalid argument"};

inline constexpr ol_error_struct_t InvalidConfigurationError = {
    OL_ERRC_INVALID_SIZE, "invalid kernel launch configuration"};

inline constexpr ol_error_struct_t InvalidNullPointerError = {
    OL_ERRC_INVALID_NULL_POINTER, "invalid null pointer"};

inline constexpr ol_error_struct_t InvalidStreamError = {OL_ERRC_INVALID_QUEUE,
                                                         "invalid stream"};

} // namespace offload
} // namespace llvm

#endif // LLVM_OFFLOAD_LANGUAGES_KERNEL_INCLUDE_OFFLOAD_ERRORS_H
