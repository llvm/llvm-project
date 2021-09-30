//===- ReduceOperands.h - Specialized Delta Pass --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a function to reduce operands to undef.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TOOLS_LLVM_REDUCE_DELTAS_REDUCEOPERANDS_H
#define LLVM_TOOLS_LLVM_REDUCE_DELTAS_REDUCEOPERANDS_H

#include "Delta.h"

namespace llvm {
void reduceOperandsDeltaPass(TestRunner &Test);
} // namespace llvm

#endif
