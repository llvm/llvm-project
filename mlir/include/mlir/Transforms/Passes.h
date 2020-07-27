//===- Passes.h - Pass Entrypoints ------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header file defines prototypes that expose pass constructors in the loop
// transformation library.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TRANSFORMS_PASSES_H
#define MLIR_TRANSFORMS_PASSES_H

#include "mlir/Support/LLVM.h"
#include <functional>
#include <limits>

namespace mlir {

class AffineForOp;
class FuncOp;
class ModuleOp;
class Pass;
template <typename T> class OperationPass;

/// Creates an instance of the BufferPlacement pass.
std::unique_ptr<Pass> createBufferPlacementPass();

/// Creates an instance of the Canonicalizer pass.
std::unique_ptr<Pass> createCanonicalizerPass();

/// Create a pass that removes unnecessary Copy operations.
std::unique_ptr<Pass> createCopyRemovalPass();

/// Creates a pass to perform common sub expression elimination.
std::unique_ptr<Pass> createCSEPass();

/// Creates a loop fusion pass which fuses loops. Buffers of size less than or
/// equal to `localBufSizeThreshold` are promoted to memory space
/// `fastMemorySpace'.
std::unique_ptr<OperationPass<FuncOp>>
createLoopFusionPass(unsigned fastMemorySpace = 0,
                     uint64_t localBufSizeThreshold = 0,
                     bool maximalFusion = false);

/// Creates a loop invariant code motion pass that hoists loop invariant
/// instructions out of the loop.
std::unique_ptr<Pass> createLoopInvariantCodeMotionPass();

/// Creates a pass to pipeline explicit movement of data across levels of the
/// memory hierarchy.
std::unique_ptr<OperationPass<FuncOp>> createPipelineDataTransferPass();

/// Lowers affine control flow operations (ForStmt, IfStmt and AffineApplyOp)
/// to equivalent lower-level constructs (flow of basic blocks and arithmetic
/// primitives).
std::unique_ptr<Pass> createLowerAffinePass();

/// Creates a pass that transforms perfectly nested loops with independent
/// bounds into a single loop.
std::unique_ptr<OperationPass<FuncOp>> createLoopCoalescingPass();

/// Creates a pass that transforms a single ParallelLoop over N induction
/// variables into another ParallelLoop over less than N induction variables.
std::unique_ptr<Pass> createParallelLoopCollapsingPass();

/// Creates a pass to perform optimizations relying on memref dataflow such as
/// store to load forwarding, elimination of dead stores, and dead allocs.
std::unique_ptr<OperationPass<FuncOp>> createMemRefDataFlowOptPass();

/// Creates a pass to strip debug information from a function.
std::unique_ptr<Pass> createStripDebugInfoPass();

/// Creates a pass which prints the list of ops and the number of occurrences in
/// the module.
std::unique_ptr<OperationPass<ModuleOp>> createPrintOpStatsPass();

/// Creates a pass which inlines calls and callable operations as defined by
/// the CallGraph.
std::unique_ptr<Pass> createInlinerPass();

/// Creates a pass which performs sparse conditional constant propagation over
/// nested operations.
std::unique_ptr<Pass> createSCCPPass();

/// Creates a pass which delete symbol operations that are unreachable. This
/// pass may *only* be scheduled on an operation that defines a SymbolTable.
std::unique_ptr<Pass> createSymbolDCEPass();
} // end namespace mlir

#endif // MLIR_TRANSFORMS_PASSES_H
