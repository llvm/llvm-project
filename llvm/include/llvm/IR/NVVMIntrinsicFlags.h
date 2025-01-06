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

#ifndef LLVM_IR_NVVMINTRINSICFLAGS_H
#define LLVM_IR_NVVMINTRINSICFLAGS_H

#include <stdint.h>

namespace llvm {
namespace nvvm {

// Reduction Ops supported with TMA Copy from Shared
// to Global Memory for the "cp.reduce.async.bulk.tensor.*"
// family of PTX instructions.
enum class TMAReductionOp : uint8_t {
  ADD = 0,
  MIN = 1,
  MAX = 2,
  INC = 3,
  DEC = 4,
  AND = 5,
  OR = 6,
  XOR = 7,
};

} // namespace nvvm
} // namespace llvm
#endif // LLVM_IR_NVVMINTRINSICFLAGS_H
