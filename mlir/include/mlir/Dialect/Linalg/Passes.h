//===- Passes.h - Linalg pass entry points ----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header file defines prototypes that expose pass constructors.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_LINALG_PASSES_H_
#define MLIR_DIALECT_LINALG_PASSES_H_

#include "mlir/Support/LLVM.h"
#include "llvm/ADT/ArrayRef.h"

namespace mlir {
class FuncOp;
class MLIRContext;
class ModuleOp;
template <typename T> class OperationPass;
class OwningRewritePatternList;
class Pass;

std::unique_ptr<OperationPass<FuncOp>> createLinalgFoldUnitExtentDimsPass();

std::unique_ptr<OperationPass<FuncOp>> createLinalgFusionPass();
std::unique_ptr<Pass> createLinalgFusionOfTensorOpsPass();

std::unique_ptr<OperationPass<FuncOp>>
createLinalgTilingPass(ArrayRef<int64_t> tileSizes = {});

std::unique_ptr<OperationPass<FuncOp>>
createLinalgTilingToParallelLoopsPass(ArrayRef<int64_t> tileSizes = {});

std::unique_ptr<OperationPass<FuncOp>>
createLinalgPromotionPass(bool dynamicBuffers, bool useAlloca);
std::unique_ptr<OperationPass<FuncOp>> createLinalgPromotionPass();

/// Create a pass to convert Linalg operations to scf.for loops and
/// std.load/std.store accesses.
std::unique_ptr<OperationPass<FuncOp>> createConvertLinalgToLoopsPass();

/// Create a pass to convert Linalg operations to scf.parallel loops and
/// std.load/std.store accesses.
std::unique_ptr<OperationPass<FuncOp>> createConvertLinalgToParallelLoopsPass();

/// Create a pass to convert Linalg operations to affine.for loops and
/// affine_load/affine_store accesses.
/// Placeholder for now, this is NYI.
std::unique_ptr<OperationPass<FuncOp>> createConvertLinalgToAffineLoopsPass();

/// Create a pass to convert Linalg operations which work on tensors to use
/// buffers instead.
std::unique_ptr<OperationPass<ModuleOp>>
createConvertLinalgOnTensorsToBuffersPass();

/// Patterns for fusing linalg operation on tensors.
void populateLinalgTensorOpsFusionPatterns(MLIRContext *context,
                                           OwningRewritePatternList &patterns);

/// Patterns to fold unit-extent dimensions in operands/results of linalg ops on
/// tensors.
void populateLinalgFoldUnitExtentDimsPatterns(
    MLIRContext *context, OwningRewritePatternList &patterns);

} // namespace mlir

#endif // MLIR_DIALECT_LINALG_PASSES_H_
