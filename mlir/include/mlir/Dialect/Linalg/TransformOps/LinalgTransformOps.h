//===- LinalgTransformOps.h - Linalg transform ops --------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_LINALG_TRANSFORMOPS_LINALGTRANSFORMOPS_H
#define MLIR_DIALECT_LINALG_TRANSFORMOPS_LINALGTRANSFORMOPS_H

#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Transform/IR/TransformAttrs.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Transform/IR/TransformInterfaces.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/RegionKindInterface.h"

namespace mlir {
class TilingInterface;
class RewriterBase;

namespace linalg {
class CopyOp;
struct ForallTilingResult;
class GenericOp;
class LinalgOp;
} // namespace linalg

namespace tensor {
class InsertSliceOp;
class PackOp;
class PadOp;
class UnPackOp;
} // namespace tensor

namespace transform {
class TransformHandleTypeInterface;
// Types needed for builders.
struct TileSizesSpec {};
struct NumThreadsSpec {};
} // namespace transform
} // namespace mlir

namespace mlir {
class DialectRegistry;

namespace transform {

/// Implementation of tiling operations using `scf.forall`.
DiagnosedSilenceableFailure
tileToForallOpImpl(RewriterBase &rewriter, transform::TransformState &state,
                   TransformOpInterface transformOp, Operation *target,
                   ArrayRef<OpFoldResult> mixedNumThreads,
                   ArrayRef<OpFoldResult> mixedTileSizes,
                   std::optional<ArrayAttr> mapping,
                   linalg::ForallTilingResult &tilingResult);

/// Find the first "extract" user of `producerOp` and tile it right before its
/// use. The tiled op is fused under the `containingOp`.
/// Return this fused op on success or nullptr if anything fails.
/// If tiled op has uses that are dominated by `containingOp`, return
/// a new `containingOp` with results of the fused op appended to
/// results of the `containingOp` or nullptr if there are no dominated uses.
std::tuple<SmallVector<Operation *>, Operation *>
tileAndFuseFirstExtractUse(RewriterBase &rewriter, Diagnostic &diag,
                           Operation *producerOp, Operation *containingOp);

/// First, find the first "scf::ForallOp" user of `producerOp` and ensure
/// it is exactly the `containingOp`, otherwise bail.
/// Then, find the first "extract" user of the tied block argument and tile it
/// right before its "extract" use. The tiled op is fused under the
/// `containingOp`.
/// Return this fused op on success or nullptr if anything fails.
SmallVector<Operation *>
tileAndFuseFirstExtractUseThroughContainingOpBlockArgument(
    RewriterBase &rewriter, Diagnostic &diag, Operation *producerOp,
    Operation *containingOp);

/// Find the first use of `producerOp` inside `containingOp` and fuse into
/// the containing op by cloning the producer. Return nullptr if no such
/// fusion opportunity exists.
Operation *cloneAndFuseFirstUse(RewriterBase &rewriter, Diagnostic &diag,
                                Operation *producerOp, Operation *containingOp);

} // namespace transform
} // namespace mlir

//===----------------------------------------------------------------------===//
// Linalg Transform Operations
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Linalg/TransformOps/LinalgTransformOpsEnums.h.inc"

#define GET_OP_CLASSES
#include "mlir/Dialect/Linalg/TransformOps/LinalgTransformOps.h.inc"

#endif // MLIR_DIALECT_LINALG_TRANSFORMOPS_LINALGTRANSFORMOPS_H
