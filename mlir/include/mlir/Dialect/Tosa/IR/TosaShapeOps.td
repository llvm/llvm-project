//===-- TosaShapeOps.td - TOSA dialect utility operations --*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines shape operators for the TOSA dialect.
//
//===----------------------------------------------------------------------===//

#ifndef TOSA_SHAPE_OPS
#define TOSA_SHAPE_OPS

include "mlir/IR/OpBase.td"

include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/LoopLikeInterface.td"
include "mlir/Dialect/Tosa/IR/TosaInterfaces.td"

include "mlir/Dialect/Tosa/IR/TosaTypesBase.td"
include "mlir/Dialect/Tosa/IR/TosaOpBase.td"

// Op trait: operator has operands and results with TOSA shape type
def TosaShapeOperator : NativeOpTrait<"TosaShapeOperator"> {
  let cppNamespace = "mlir::OpTrait::tosa";
}

class Tosa_ShapeOp<string mnemonic, list<Trait> traits = []>
    : Tosa_Op<mnemonic, !listconcat(traits, [TosaShapeOperator, Pure])> {
  list<Availability> availability = [
    Profile<[Tosa_PRO_INT, Tosa_PRO_FP]>,
    Extension<[]>,
  ];

  let assemblyFormat =
      "operands attr-dict `:` functional-type(operands, results)";

  let hasFolder = 1;
}

// op trait: shape operator has same ranks for operands and results
def TosaShapeOperatorWithSameRanks
    : NativeOpTrait<"TosaShapeOperatorWithSameRanks"> {
  let cppNamespace = "mlir::OpTrait::tosa";
}

class Tosa_ElementwiseShapeOp<string mnemonic, list<Trait> traits = []>
    : Tosa_ShapeOp<mnemonic,
                   !listconcat(traits, [TosaShapeOperatorWithSameRanks])> {
}


//===----------------------------------------------------------------------===//
// Operator: ConstShape
//===----------------------------------------------------------------------===//
def Tosa_ConstShapeOp : Tosa_ShapeOp<"const_shape", [ConstantLike, Pure]> {
  let summary = "Constant Shape op.";

  let description = [{
    A node containing constant data for use as the input to an shape operation. May
    hold data only in index data type.

    Example:

    ```mlir
    // Generic form
    %out = "tosa.const_shape"() {value = dense<0> : tensor<4xindex>} : () -> !tosa.shape<4>
    ```
  }];

  let arguments = (ins IndexElementsAttr : $value);

  let results = (outs Tosa_Shape : $output);

  let hasVerifier = 1;
}

#endif // TOSA_SHAPE_OPS
