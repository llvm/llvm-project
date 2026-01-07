//===--------- LibC.h - Simple implementation of libc functions --- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//
//===----------------------------------------------------------------------===//

#ifndef OMPTARGET_LIBC_H
#define OMPTARGET_LIBC_H

#include "DeviceTypes.h"

namespace ompx {

// SPIR-V backend does not support variadic functions except for __spirv_ocl_printf
// This is to provide a workaround to use regular printf that is used in the code.
#if defined(__SPIRV__)
template <size_t N, typename... Args>
int printf(const char (&Format)[N], Args... args) {
  return __spirv_ocl_printf(Format, args...);
}
#else    
int printf(const char *Format, ...);
#endif

} // namespace ompx

#endif
