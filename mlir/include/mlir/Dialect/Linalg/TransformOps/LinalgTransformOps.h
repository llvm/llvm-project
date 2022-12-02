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
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/RegionKindInterface.h"

namespace mlir {
class TilingInterface;
class RewriterBase;
namespace linalg {
class GenericOp;
class LinalgOp;
} // namespace linalg

namespace transform {
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

/// Implementation of tiling operations using `scf.foreach_thread`.
DiagnosedSilenceableFailure tileToForeachThreadOpImpl(
    RewriterBase &rewriter, transform::TransformState &state,
    TransformOpInterface transformOp, ArrayRef<Operation *> targets,
    ArrayRef<OpFoldResult> mixedNumThreads,
    ArrayRef<OpFoldResult> mixedTileSizes, Optional<ArrayAttr> mapping,
    SmallVector<Operation *> &tileOps, SmallVector<Operation *> &tiledOps);
} // namespace transform

namespace linalg {
void registerTransformDialectExtension(DialectRegistry &registry);
} // namespace linalg
} // namespace mlir

#endif // MLIR_DIALECT_LINALG_TRANSFORMOPS_LINALGTRANSFORMOPS_H
