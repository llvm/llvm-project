//===--- NVVMIntrinsicFlags.h -----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// This file contains the definitions of the enumerations and flags
/// associated with NVVM Intrinsics.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_NVVMINTRINSICFLAGS_H
#define LLVM_SUPPORT_NVVMINTRINSICFLAGS_H

#include <stdint.h>

namespace llvm {
namespace nvvm {

enum class CpAsyncBulkTensorLoadMode {
  TILE = 0,
  IM2COL = 1,
};

typedef union {
  int V;
  struct {
    unsigned CacheHint : 1;
    unsigned MultiCast : 1;
    unsigned LoadMode : 3; // CpAsyncBulkTensorLoadMode
    unsigned reserved : 27;
  } U;
} CpAsyncBulkTensorFlags;

} // namespace nvvm
} // namespace llvm
#endif // LLVM_SUPPORT_NVVMINTRINSICFLAGS_H
