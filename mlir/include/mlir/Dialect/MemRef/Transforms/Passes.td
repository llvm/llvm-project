//===-- Passes.td - MemRef transformation definition file --*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_MEMREF_TRANSFORMS_PASSES
#define MLIR_DIALECT_MEMREF_TRANSFORMS_PASSES

include "mlir/Pass/PassBase.td"

def FoldSubViewOps : Pass<"fold-memref-subview-ops"> {
  let summary = "Fold memref.subview ops into consumer load/store ops";
  let description = [{
    The pass folds loading/storing from/to subview ops to loading/storing
    from/to the original memref.
  }];
  let constructor = "mlir::memref::createFoldSubViewOpsPass()";
  let dependentDialects = [
      "AffineDialect", "memref::MemRefDialect", "vector::VectorDialect"
  ];
}

def ResolveRankedShapeTypeResultDims :
    Pass<"resolve-ranked-shaped-type-result-dims"> {
  let summary = "Resolve memref.dim of result values of ranked shape type";
  let description = [{
    The pass resolves memref.dim of result of operations that
    implement the `ReifyRankedShapedTypeOpInterface` in terms of
    shapes of its operands.
  }];
  let constructor =
      "mlir::memref::createResolveRankedShapeTypeResultDimsPass()";
  let dependentDialects = [
    "memref::MemRefDialect", "tensor::TensorDialect"
  ];
}

def ResolveShapedTypeResultDims : Pass<"resolve-shaped-type-result-dims"> {
  let summary = "Resolve memref.dim of result values";
  let description = [{
    The pass resolves memref.dim of result of operations that
    implement the `InferShapedTypeOpInterface` or
    `ReifyRankedShapedTypeOpInterface` in terms of shapes of its
    operands.
  }];
  let constructor = "mlir::memref::createResolveShapedTypeResultDimsPass()";
  let dependentDialects = [
    "AffineDialect", "memref::MemRefDialect", "tensor::TensorDialect"
  ];
}

#endif // MLIR_DIALECT_MEMREF_TRANSFORMS_PASSES

