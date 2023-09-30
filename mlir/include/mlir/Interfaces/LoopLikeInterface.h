//===- LoopLikeInterface.h - Loop-like operations interface ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the operation interface for loop like operations.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_INTERFACES_LOOPLIKEINTERFACE_H_
#define MLIR_INTERFACES_LOOPLIKEINTERFACE_H_

#include "mlir/IR/OpDefinition.h"

namespace mlir {
class RewriterBase;

/// A function that returns the additional yielded values during
/// `replaceWithAdditionalYields`. `newBbArgs` are the newly added region
/// iter_args. This function should return as many values as there are block
/// arguments in `newBbArgs`.
using NewYieldValuesFn = std::function<SmallVector<Value>(
    OpBuilder &b, Location loc, ArrayRef<BlockArgument> newBbArgs)>;
} // namespace mlir

/// Include the generated interface declarations.
#include "mlir/Interfaces/LoopLikeInterface.h.inc"

#endif // MLIR_INTERFACES_LOOPLIKEINTERFACE_H_
