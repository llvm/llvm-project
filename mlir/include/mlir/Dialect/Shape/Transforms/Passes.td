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

def OutlineShapeComputation : Pass<"outline-shape-computation", "ModuleOp"> {
  let summary = "Using shape.func to preserve shape computation";
  let description = [{
    This pass outlines the shape computation part in high level IR by adding
    shape.func and populate corresponding mapping infoemation into
    ShapeMappingAnalysis. The shape computation part is usually introduced by
    shape reification, and each single dynamic shape is denoted by shape.with_shape.

    There're two main reasons this shape-outline pass is needed:
    1. Many passes don't take shape reification part into consideration.
       Therefore we need to "remove" the shape reification part temporarily for
       these passes.
    2. Sometimes we cannot redo shape reification after converting from dialect
       A to dialect B. Because op-level shape reification is only implemented
       on A.

    Input:

    ```mlir
    func.func @main(%arg0: tensor<?x4x?xf32>, %arg1: tensor<2x4x?xf32>) ->
      tensor<?x4x?xf32> {
      %c2 = arith.constant 2 : index
      %c0 = arith.constant 0 : index
      %c4 = arith.constant 4 : index
      %0 = shape.shape_of %arg0 : tensor<?x4x?xf32> -> tensor<3xindex>
      %1 = shape.get_extent %0, %c2 : tensor<3xindex>, index -> index
      %2 = "test.abs"(%arg0) : (tensor<?x4x?xf32>) -> tensor<?x4x?xf32>
      %3 = shape.with_shape %2, %0 : tensor<?x4x?xf32>, tensor<3xindex>
      %4 = shape.value_of %3 : tensor<?x4x?xf32>
      %5 = "test.concat"(%4, %arg1) {axis = 0 : i64} : (tensor<?x4x?xf32>,
            tensor<2x4x?xf32>) -> tensor<?x4x?xf32>
      %6 = shape.get_extent %0, %c0 : tensor<3xindex>, index -> index
      %7 = arith.addi %6, %c2 : index
      %8 = shape.from_extents %7, %c4, %1 : index, index, index
      %9 = shape.with_shape %5, %8 : tensor<?x4x?xf32>, !shape.shape
      %10 = shape.value_of %9 : tensor<?x4x?xf32>
      return %10 : tensor<?x4x?xf32>
    }
    ```

    Output
    ```mlir
    func.func @main(%arg0: tensor<?x4x?xf32>, %arg1: tensor<2x4x?xf32>) ->
      tensor<?x4x?xf32> {
      %0 = "test.abs"(%arg0) : (tensor<?x4x?xf32>) -> tensor<?x4x?xf32>
      %1 = "test.concat"(%0, %arg1) {axis = 0 : i64} : (tensor<?x4x?xf32>,
            tensor<2x4x?xf32>) -> tensor<?x4x?xf32>
      return %1 : tensor<?x4x?xf32>
    }
    shape.func private @shape_cal_1(%arg0: tensor<?x4x?xf32>) -> !shape.shape {
      %c2 = arith.constant 2 : index
      %c0 = arith.constant 0 : index
      %c4 = arith.constant 4 : index
      %0 = shape_of %arg0 : tensor<?x4x?xf32> -> tensor<3xindex>
      %1 = get_extent %0, %c2 : tensor<3xindex>, index -> index
      %2 = get_extent %0, %c0 : tensor<3xindex>, index -> index
      %3 = arith.addi %2, %c2 : index
      %4 = from_extents %3, %c4, %1 : index, index, index
      return %4 : !shape.shape
    }
    shape.func private @shape_cal_0(%arg0: tensor<?x4x?xf32>) -> tensor<3xindex> {
      %0 = shape_of %arg0 : tensor<?x4x?xf32> -> tensor<3xindex>
      return %0 : tensor<3xindex>
    }
    ```

    For the above example, the shape computation is inlined in the input IR,
    which is used for two values' (test.abs and test.concat) shape. And the shape
    compuatation part is outlined in the output IR.

    And the shape mapping infomation will be:

    ```
    // ---- Shape Mapping Infomation -----
    // - Shape for: %0 = "test.abs"(%arg0) : (tensor<?x4x?xf32>) -> tensor<?x4x?xf32> :: @shape_cal_0(<block argument> of type 'tensor<?x4x?xf32>' at index: 0)
    // - Shape for: %1 = "test.concat"(%0, %arg1) {axis = 0 : i64} : (tensor<?x4x?xf32>, tensor<2x4x?xf32>) -> tensor<?x4x?xf32> :: @shape_cal_1(<block argument> of type 'tensor<?x4x?xf32>' at index: 0)
    ```
  }];
  let constructor = "mlir::createOutlineShapeComputationPass()";
  let dependentDialects = ["shape::ShapeDialect"];
}

def RemoveShapeConstraints : Pass<"remove-shape-constraints", "func::FuncOp"> {
  let summary = "Replace all cstr_ ops with a true witness";
  let constructor = "mlir::createRemoveShapeConstraintsPass()";
}

def ShapeToShapeLowering : Pass<"shape-to-shape-lowering", "func::FuncOp"> {
  let summary = "Legalize Shape dialect to be convertible to Arith";
  let constructor = "mlir::createShapeToShapeLowering()";
}

// TODO: Generalize this to allow any type conversions desired.
def ShapeBufferize : Pass<"shape-bufferize", "func::FuncOp"> {
  let summary = "Bufferize the shape dialect.";
  let constructor = "mlir::createShapeBufferizePass()";
  let dependentDialects = ["bufferization::BufferizationDialect",
                           "memref::MemRefDialect"];
}
#endif // MLIR_DIALECT_SHAPE_TRANSFORMS_PASSES
