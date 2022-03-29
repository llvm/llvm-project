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

//===----------------------------------------------------------------------===//
// LoopLike Interfaces
//===----------------------------------------------------------------------===//

/// Include the generated interface declarations.
#include "mlir/Interfaces/LoopLikeInterface.h.inc"

//===----------------------------------------------------------------------===//
// LoopLike Utilities
//===----------------------------------------------------------------------===//

namespace mlir {
/// Move loop invariant code out of a `looplike` operation.
void moveLoopInvariantCode(LoopLikeOpInterface looplike);
} // namespace mlir

#endif // MLIR_INTERFACES_LOOPLIKEINTERFACE_H_
