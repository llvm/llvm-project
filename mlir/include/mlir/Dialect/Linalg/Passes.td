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

def LinalgFoldUnitExtentDims : FunctionPass<"linalg-fold-unit-extent-dims"> {
  let summary = "Remove unit-extent dimension in Linalg ops on tensors";
  let constructor = "mlir::createLinalgFoldUnitExtentDimsPass()";
  let options = [
    Option<"foldOneTripLoopsOnly", "fold-one-trip-loops-only", "bool",
            /*default=*/"false",
	    "Only folds the one-trip loops from Linalg ops on tensors "
	    "(for testing purposes only)">
  ];
}

def LinalgFusion : FunctionPass<"linalg-fusion"> {
  let summary = "Fuse operations in the linalg dialect";
  let constructor = "mlir::createLinalgFusionPass()";
}

def LinalgFusionOfTensorOps : Pass<"linalg-fusion-for-tensor-ops"> {
  let summary = "Fuse operations on RankedTensorType in linalg dialect";
  let constructor = "mlir::createLinalgFusionOfTensorOpsPass()";
}

def LinalgLowerToAffineLoops : FunctionPass<"convert-linalg-to-affine-loops"> {
  let summary = "Lower the operations from the linalg dialect into affine "
                "loops";
  let constructor = "mlir::createConvertLinalgToAffineLoopsPass()";
}

def LinalgLowerToLoops : FunctionPass<"convert-linalg-to-loops"> {
  let summary = "Lower the operations from the linalg dialect into loops";
  let constructor = "mlir::createConvertLinalgToLoopsPass()";
}

def LinalgOnTensorsToBuffers : Pass<"convert-linalg-on-tensors-to-buffers", "ModuleOp"> {
  let summary = "Convert the Linalg operations which work on tensor-type "
                "operands or results to use buffers instead";
  let constructor = "mlir::createConvertLinalgOnTensorsToBuffersPass()";
}

def LinalgLowerToParallelLoops
    : FunctionPass<"convert-linalg-to-parallel-loops"> {
  let summary = "Lower the operations from the linalg dialect into parallel "
                "loops";
  let constructor = "mlir::createConvertLinalgToParallelLoopsPass()";
}

def LinalgPromotion : FunctionPass<"linalg-promote-subviews"> {
  let summary = "Promote subview ops to local buffers";
  let constructor = "mlir::createLinalgPromotionPass()";
  let options = [
    Option<"dynamicBuffers", "test-promote-dynamic", "bool",
           /*default=*/"false", "Test generation of dynamic promoted buffers">
  ];
}

def LinalgTiling : FunctionPass<"linalg-tile"> {
  let summary = "Tile operations in the linalg dialect";
  let constructor = "mlir::createLinalgTilingPass()";
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
}

#endif // MLIR_DIALECT_LINALG_PASSES
