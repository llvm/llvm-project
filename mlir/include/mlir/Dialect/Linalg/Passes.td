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

def LinalgFusion : Pass<"linalg-fusion"> {
  let summary = "Fuse operations in the linalg dialect";
  let constructor = "mlir::createLinalgFusionPass()";
}

def LinalgFusionOfTensorOps : Pass<"linalg-fusion-for-tensor-ops"> {
  let summary = "Fuse operations on RankedTensorType in linalg dialect";
  let constructor = "mlir::createLinalgFusionOfTensorOpsPass()";
}

def LinalgLowerToAffineLoops : Pass<"convert-linalg-to-affine-loops"> {
  let summary = "Lower the operations from the linalg dialect into affine "
                "loops";
  let constructor = "mlir::createConvertLinalgToAffineLoopsPass()";
}

def LinalgLowerToLoops : Pass<"convert-linalg-to-loops"> {
  let summary = "Lower the operations from the linalg dialect into loops";
  let constructor = "mlir::createConvertLinalgToLoopsPass()";
}

def LinalgLowerToParallelLoops : Pass<"convert-linalg-to-parallel-loops"> {
  let summary = "Lower the operations from the linalg dialect into parallel "
                "loops";
  let constructor = "mlir::createConvertLinalgToParallelLoopsPass()";
}

def LinalgPromotion : Pass<"linalg-promote-subviews"> {
  let summary = "Promote subview ops to local buffers";
  let constructor = "mlir::createLinalgPromotionPass()";
}

def LinalgTiling : Pass<"linalg-tile"> {
  let summary = "Tile operations in the linalg dialect";
  let constructor = "mlir::createLinalgTilingPass()";
}

def LinalgTilingToParallelLoops : Pass<"linalg-tile-to-parallel-loops"> {
  let summary = "Tile operations in the linalg dialect to parallel loops";
  let constructor = "mlir::createLinalgTilingToParallelLoopsPass()";
}

#endif // MLIR_DIALECT_LINALG_PASSES
