//===- Passes.h - Pass Entrypoints ------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header file defines a set of transforms specific for the AffineOps
// dialect.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_AFFINE_TRANSFORMS_PASSES_H
#define MLIR_DIALECT_AFFINE_TRANSFORMS_PASSES_H

#include "mlir/Support/LLVM.h"
#include <functional>
#include <limits>

namespace mlir {

class AffineForOp;
class FuncOp;
class ModuleOp;
class Pass;
template <typename T> class OpPassBase;

/// Creates a simplification pass for affine structures (maps and sets). In
/// addition, this pass also normalizes memrefs to have the trivial (identity)
/// layout map.
std::unique_ptr<OpPassBase<FuncOp>> createSimplifyAffineStructuresPass();

/// Creates a loop invariant code motion pass that hoists loop invariant
/// instructions out of affine loop.
std::unique_ptr<OpPassBase<FuncOp>> createAffineLoopInvariantCodeMotionPass();

/// Performs packing (or explicit copying) of accessed memref regions into
/// buffers in the specified faster memory space through either pointwise copies
/// or DMA operations.
std::unique_ptr<OpPassBase<FuncOp>> createAffineDataCopyGenerationPass(
    unsigned slowMemorySpace, unsigned fastMemorySpace,
    unsigned tagMemorySpace = 0, int minDmaTransferSize = 1024,
    uint64_t fastMemCapacityBytes = std::numeric_limits<uint64_t>::max());

} // end namespace mlir

#endif // MLIR_DIALECT_AFFINE_RANSFORMS_PASSES_H
