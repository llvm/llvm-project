//===-- Passes.td - Linalg pass definition file ------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_LINALG_PASSES
#define MLIR_DIALECT_LINALG_PASSES

include "mlir/Pass/PassBase.td"

def ConvertElementwiseToLinalg : FunctionPass<"convert-elementwise-to-linalg"> {
  let summary = "Convert ElementwiseMappable ops to linalg";
  let description = [{
    Convert ops with the `ElementwiseMappable` trait to linalg parallel loops.

    This pass only converts ops that operate on ranked tensors.
  }];
  let constructor = "mlir::createConvertElementwiseToLinalgPass()";
  let dependentDialects = ["linalg::LinalgDialect"];
}

def LinalgFoldUnitExtentDims : FunctionPass<"linalg-fold-unit-extent-dims"> {
  let summary = "Remove unit-extent dimension in Linalg ops on tensors";
  let constructor = "mlir::createLinalgFoldUnitExtentDimsPass()";
  let options = [
    Option<"foldOneTripLoopsOnly", "fold-one-trip-loops-only", "bool",
            /*default=*/"false",
	    "Only folds the one-trip loops from Linalg ops on tensors "
	    "(for testing purposes only)">
  ];
  let dependentDialects = ["linalg::LinalgDialect"];
}

def LinalgFusionOfTensorOps : Pass<"linalg-fusion-for-tensor-ops"> {
  let summary = "Fuse operations on RankedTensorType in linalg dialect";
  let constructor = "mlir::createLinalgFusionOfTensorOpsPass()";
  let dependentDialects = ["linalg::LinalgDialect", "AffineDialect"];
}

def LinalgFoldReshapeOpsByLinearization :
  Pass<"linalg-fold-reshape-ops-by-linearization"> {
  let summary = "Fold TensorReshapeOps with generic/indexed generic ops by "
                "linearization";
  let constructor = "mlir::createFoldReshapeOpsByLinearizationPass()";
  let dependentDialects = ["AffineDialect"];
}

def LinalgLowerToAffineLoops : FunctionPass<"convert-linalg-to-affine-loops"> {
  let summary = "Lower the operations from the linalg dialect into affine "
                "loops";
  let constructor = "mlir::createConvertLinalgToAffineLoopsPass()";
  let dependentDialects = ["linalg::LinalgDialect", "AffineDialect"];
}

def LinalgLowerToLoops : FunctionPass<"convert-linalg-to-loops"> {
  let summary = "Lower the operations from the linalg dialect into loops";
  let constructor = "mlir::createConvertLinalgToLoopsPass()";
  let dependentDialects = ["linalg::LinalgDialect", "scf::SCFDialect", "AffineDialect"];
}

def LinalgBufferize : Pass<"linalg-bufferize", "FuncOp"> {
  let summary = "Bufferize the linalg dialect";
  let constructor = "mlir::createLinalgBufferizePass()";
  let dependentDialects = ["linalg::LinalgDialect", "AffineDialect"];
}

def LinalgLowerToParallelLoops
    : FunctionPass<"convert-linalg-to-parallel-loops"> {
  let summary = "Lower the operations from the linalg dialect into parallel "
                "loops";
  let constructor = "mlir::createConvertLinalgToParallelLoopsPass()";
  let dependentDialects = ["AffineDialect", "linalg::LinalgDialect", "scf::SCFDialect"];
}

def LinalgPromotion : FunctionPass<"linalg-promote-subviews"> {
  let summary = "Promote subview ops to local buffers";
  let constructor = "mlir::createLinalgPromotionPass()";
  let options = [
    Option<"dynamicBuffers", "test-promote-dynamic", "bool",
           /*default=*/"false", "Test generation of dynamic promoted buffers">,
    Option<"useAlloca", "test-use-alloca", "bool",
           /*default=*/"false", "Test generation of alloca'ed buffers.">
  ];
  let dependentDialects = ["linalg::LinalgDialect"];
}

def LinalgTiling : FunctionPass<"linalg-tile"> {
  let summary = "Tile operations in the linalg dialect";
  let constructor = "mlir::createLinalgTilingPass()";
  let dependentDialects = [
    "AffineDialect", "linalg::LinalgDialect", "scf::SCFDialect"
  ];
  let options = [
    ListOption<"tileSizes", "linalg-tile-sizes", "int64_t",
               "Test generation of dynamic promoted buffers",
               "llvm::cl::ZeroOrMore, llvm::cl::MiscFlags::CommaSeparated">
  ];
}

def LinalgTilingToParallelLoops
    : FunctionPass<"linalg-tile-to-parallel-loops"> {
  let summary = "Tile operations in the linalg dialect to parallel loops";
  let constructor = "mlir::createLinalgTilingToParallelLoopsPass()";
  let options = [
    ListOption<"tileSizes", "linalg-tile-sizes", "int64_t",
               "Test generation of dynamic promoted buffers",
               "llvm::cl::ZeroOrMore, llvm::cl::MiscFlags::CommaSeparated">
  ];
  let dependentDialects = ["AffineDialect", "linalg::LinalgDialect", "scf::SCFDialect"];
}

def LinalgGeneralization : FunctionPass<"linalg-generalize-named-ops"> {
  let summary = "Convert named ops into generic ops";
  let constructor = "mlir::createLinalgGeneralizationPass()";
  let dependentDialects = ["linalg::LinalgDialect"];
}

#endif // MLIR_DIALECT_LINALG_PASSES
