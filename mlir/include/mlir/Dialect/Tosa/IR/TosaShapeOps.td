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
    Extension<[Tosa_EXT_SHAPE]>,
  ];

  let assemblyFormat =
      "operands attr-dict `:` functional-type(operands, results)";
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
// Operator: AddShape
//===----------------------------------------------------------------------===//
def Tosa_AddShapeOp : Tosa_ElementwiseShapeOp<"add_shape", [Pure]> {
  let summary = "Elementwise addition of shapes.";

  let description = [{
      Elementwise addition of input1 and input2. Size of shapes must match.
  }];

  let arguments = (ins
    Tosa_Shape:$input1,
    Tosa_Shape:$input2
  );

  let results = (outs Tosa_Shape:$output);
}

//===----------------------------------------------------------------------===//
// Operator: ConcatShape
//===----------------------------------------------------------------------===//
def Tosa_ConcatShapeOp : Tosa_ShapeOp<"concat_shape", [Pure]> {
  let summary = "Concatenates a list of shapes.";

  let description = [{
    Concatenates a list of shapes into a new shape with length equal to the sum
    of the lengths of the inputs.
  }];

  let arguments = (ins
    Variadic<Tosa_Shape>:$input
  );

  let results = (outs Tosa_Shape:$output);

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// Operator: ConstShape
//===----------------------------------------------------------------------===//
def Tosa_ConstShapeOp : Tosa_ShapeOp<"const_shape", [ConstantLike, Pure]> {
  let summary = "Constant Shape operator.";

  let description = [{
    A node containing a constant shape.

    Example:

    ```mlir
    // Generic form
    %out = "tosa.const_shape"() {values = dense<0> : tensor<4xindex>} : () -> !tosa.shape<4>
    ```
  }];

  let arguments = (ins IndexElementsAttr:$values);

  let results = (outs Tosa_Shape:$output);

  list<Availability> availability = [
    Profile<[Tosa_PRO_INT, Tosa_PRO_FP]>,
    Extension<[]>,
  ];

  let hasVerifier = 1;
  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// Operator: Dim
//===----------------------------------------------------------------------===//
def Tosa_DimOp : Tosa_ShapeOp<"dim", [Pure]> {
  let summary = "Extract size of dimension from input tensor.";

  let description = [{
    Returns a length 1 shape_t of the size of the input tensor for the given axis.
  }];

  let arguments = (ins
    Tosa_TensorAtLeast1D:$input1,
    I32Attr:$axis
  );

  let results = (outs Tosa_Shape:$output);

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// Operator: DivCeilShape
//===----------------------------------------------------------------------===//
def Tosa_DivCeilShapeOp : Tosa_ElementwiseShapeOp<"div_ceil_shape", [Pure]> {
  let summary = "Elementwise ceiling divide of shapes.";

  let description = [{
      Elementwise divide of input1 by input2. The result of the divide is rounded up.
  }];

  let arguments = (ins
    Tosa_Shape:$input1,
    Tosa_Shape:$input2
  );

  let results = (outs Tosa_Shape:$output);
}

//===----------------------------------------------------------------------===//
// Operator: DivFloorShape
//===----------------------------------------------------------------------===//
def Tosa_DivFloorShapeOp : Tosa_ElementwiseShapeOp<"div_floor_shape", [Pure]> {
  let summary = "Elementwise floor divide of shapes.";

  let description = [{
      Elementwise integer divide of input1 by input2. The result of the divide is rounded down.
  }];

  let arguments = (ins
    Tosa_Shape:$input1,
    Tosa_Shape:$input2
  );

  let results = (outs Tosa_Shape:$output);
}

//===----------------------------------------------------------------------===//
// Operator: MulShape
//===----------------------------------------------------------------------===//
def Tosa_MulShapeOp : Tosa_ElementwiseShapeOp<"mul_shape", [Pure]> {
  let summary = "Elementwise multiplication of shapes.";

  let description = [{
      Elementwise multiplication of input1 and input2.
  }];

  let arguments = (ins
    Tosa_Shape:$input1,
    Tosa_Shape:$input2
  );

  let results = (outs Tosa_Shape:$output);
}

//===----------------------------------------------------------------------===//
// Operator: SliceShape
//===----------------------------------------------------------------------===//
def Tosa_SliceShapeOp : Tosa_ShapeOp<"slice_shape", [Pure]> {
  let summary = "Extract slice of a shape.";

  let description = [{
    Extract a shape of size from input.
  }];

  let arguments = (ins
    Tosa_Shape:$input,
    Tosa_ScalarInt32Tensor:$start,
    Tosa_ScalarInt32Tensor:$size
  );

  let results = (outs Tosa_Shape:$output);

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// Operator: SubShape
//===----------------------------------------------------------------------===//
def Tosa_SubShapeOp : Tosa_ElementwiseShapeOp<"sub_shape", [Pure]> {
  let summary = "Elementwise subtraction of shapes.";

  let description = [{
      Elementwise subtraction of input1 and input2. Size of shapes must match.
  }];

  let arguments = (ins
    Tosa_Shape:$input1,
    Tosa_Shape:$input2
  );

  let results = (outs Tosa_Shape:$output);
}

#endif // TOSA_SHAPE_OPS
