//===- Utils.h - llvm-reduce utility functions ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains some utility functions supporting llvm-reduce.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TOOLS_LLVM_REDUCE_DELTAS_UTILS_H
#define LLVM_TOOLS_LLVM_REDUCE_DELTAS_UTILS_H

#include "llvm/IR/Value.h"

namespace llvm {

Value *getDefaultValue(Type *T);

} // namespace llvm

#endif
