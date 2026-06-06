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
#include "mlir/Dialect/Transform/IR/TransformTypes.h"
#include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.h"
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

namespace scf {
struct SCFTilingResult;
} // namespace scf

namespace tensor {
class InsertSliceOp;
class PackOp;
class PadOp;
class UnPackOp;
} // namespace tensor

namespace transform {
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
                   scf::SCFTilingResult &tilingResult);

} // namespace transform
} // namespace mlir

//===----------------------------------------------------------------------===//
// Linalg Transform Operations
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Linalg/TransformOps/LinalgTransformOpsEnums.h.inc"

#define GET_OP_CLASSES
#include "mlir/Dialect/Linalg/TransformOps/LinalgTransformOps.h.inc"

#endif // MLIR_DIALECT_LINALG_TRANSFORMOPS_LINALGTRANSFORMOPS_H
