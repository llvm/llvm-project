//===- ReduceFunctions.h - Specialized Delta Pass -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a function which calls the Generic Delta pass in order
// to reduce functions (and any instruction that calls it) in the provided
// Module.
//
//===----------------------------------------------------------------------===//

#include "Delta.h"
#include "llvm/Transforms/Utils/Cloning.h"

namespace llvm {
void reduceFunctionsDeltaPass(TestRunner &Test);
} // namespace llvm
