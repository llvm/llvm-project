//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains common utilities for defining device kernel wrappers to
/// math functions.
///
//===----------------------------------------------------------------------===//

#ifndef CONFORMANCE_DEVICE_CODE_COMMON_HPP
#define CONFORMANCE_DEVICE_CODE_COMMON_HPP

#include <gpuintrin.h>
#include <stddef.h>
#include <stdint.h>

namespace common {

typedef _Float16 float16;

template <auto Func, typename OutType, typename... InTypes>
void runKernelBody(size_t NumElements, OutType *Out, const InTypes *...Ins) {
  uint32_t Index =
      __gpu_num_threads_x() * __gpu_block_id_x() + __gpu_thread_id_x();

  if (Index < NumElements) {
    Out[Index] = Func(Ins[Index]...);
  }
}
} // namespace common

#endif // CONFORMANCE_DEVICE_CODE_COMMON_HPP
