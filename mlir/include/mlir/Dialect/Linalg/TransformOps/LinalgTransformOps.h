//===- LinalgTransformOps.h - Linalg transform ops --------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_LINALG_TRANSFORMOPS_LINALGTRANSFORMOPS_H
#define MLIR_DIALECT_LINALG_TRANSFORMOPS_LINALGTRANSFORMOPS_H

#include "mlir/Dialect/PDL/IR/PDLTypes.h"
#include "mlir/Dialect/Transform/IR/TransformInterfaces.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/RegionKindInterface.h"

namespace mlir {
class TilingInterface;
class RewriterBase;

namespace linalg {
class GenericOp;
class LinalgOp;
} // namespace linalg

namespace tensor {
class PackOp;
class UnPackOp;
} // namespace tensor

namespace transform {
class TransformHandleTypeInterface;
// Types needed for builders.
struct TileSizesSpec {};
struct NumThreadsSpec {};
} // namespace transform
} // namespace mlir

//===----------------------------------------------------------------------===//
// Linalg Transform Operations
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Linalg/TransformOps/LinalgTransformOpsEnums.h.inc"

#define GET_OP_CLASSES
#include "mlir/Dialect/Linalg/TransformOps/LinalgTransformOps.h.inc"

namespace mlir {
class DialectRegistry;

namespace transform {

/// Return the set of `linalgOp` iterator positions for which the indexing map
/// for `opOperand` is a permutation (i.e. an AffineDimExpr).
DenseSet<int64_t> findPermutationsIndexingOperand(linalg::LinalgOp linalgOp,
                                                  OpOperand *opOperand,
                                                  utils::IteratorType iter);

/// Possible dimension candidates that define a gemm embedded in the indexing
/// maps of a LinalgOp.
struct GemmDimsForPacking {
  DenseSet<int64_t> mPos, nPos, kPos;
};

/// Find 2 parallel (m and n) and 1 reduction (k) dimension candidates that form
/// a gemm subcomputation within `linalgOp`. These dimensions are such that:
///   1. The m dimension is involved in an outer-product along LHS
///      (i.e. it is a permutation on RES and LHS and does not appear in RHS).
///   2. The n dimension is involved in an outer-product along RHS
///      (i.e. it is a permutation on RES and RHS and does not appear in LHS).
///   3. The k dimension appears as a permutation on LHS and RHS.
///   4. m, n and k appear only once in any given indexing.
/// This allows detecting that some gemm is embedded within `linalgOp` with some
/// orthogonal heuristic.
FailureOr<GemmDimsForPacking> inferGemmDims(linalg::LinalgOp linalgOp);

/// Return true if `linalgOp` contains an embedded gemm subcomputation.
bool containsMostMinorGemm(linalg::LinalgOp linalgOp);

/// Implementation of tiling operations using `scf.forall`.
DiagnosedSilenceableFailure tileToForallOpImpl(
    RewriterBase &rewriter, transform::TransformState &state,
    TransformOpInterface transformOp, ArrayRef<Operation *> targets,
    ArrayRef<OpFoldResult> mixedNumThreads,
    ArrayRef<OpFoldResult> mixedTileSizes, std::optional<ArrayAttr> mapping,
    SmallVector<Operation *> &tileOps, SmallVector<Operation *> &tiledOps);

} // namespace transform

namespace linalg {
void registerTransformDialectExtension(DialectRegistry &registry);
} // namespace linalg
} // namespace mlir

#endif // MLIR_DIALECT_LINALG_TRANSFORMOPS_LINALGTRANSFORMOPS_H
