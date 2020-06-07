//===-- Passes.td - ShapeOps pass definition file ----------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_SHAPE_TRANSFORMS_PASSES
#define MLIR_DIALECT_SHAPE_TRANSFORMS_PASSES

include "mlir/Pass/PassBase.td"

def ShapeToShapeLowering : FunctionPass<"shape-to-shape-lowering"> {
  let summary = "Legalize Shape dialect to be convertible to Standard";
  let constructor = "mlir::createShapeToShapeLowering()";
}

#endif // MLIR_DIALECT_SHAPE_TRANSFORMS_PASSES
