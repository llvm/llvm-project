//===-- TosaOps.td - TOSA dialect operation definitions ----*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the operation set for the TOSA dialect as defined in
// the TOSA specfication (https://developer.mlplatform.org/w/tosa/).
//
//===----------------------------------------------------------------------===//

#ifndef TOSA_OPS
#define TOSA_OPS

include "mlir/IR/OpBase.td"

include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/LoopLikeInterface.td"
include "mlir/Dialect/Tosa/IR/TosaInterfaces.td"

include "mlir/Dialect/Tosa/IR/TosaTypesBase.td"
include "mlir/Dialect/Tosa/IR/TosaOpBase.td"

//===----------------------------------------------------------------------===//
// TOSA Spec Section 2.2
// Operator Class: Tensor Data Engine Operators.
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Operator: argmax
//===----------------------------------------------------------------------===//
def Tosa_ArgMaxOp : Tosa_Op<"argmax", [
    DeclareOpInterfaceMethods<InferShapedTypeOpInterface,
                              ["inferReturnTypeComponents"]>,
    Pure]> {
  let summary = "Perform argmax on the input.";

  let description = [{
    This returns the index with the largest value across the given axis of the
    input tensor.
  }];

  let arguments = (ins
    Tosa_Tensor1Dto4D: $input,
    I64Attr: $axis
  );

  let results = (outs
    Tosa_TensorUpto4D: $output
  );
}

//===----------------------------------------------------------------------===//
// Operator: avg_pool2d
//===----------------------------------------------------------------------===//
def Tosa_AvgPool2dOp : Tosa_Op<"avg_pool2d", [
    DeclareOpInterfaceMethods<InferShapedTypeOpInterface,
                              ["inferReturnTypeComponents"]>,
    Pure]> {
  let summary = "Performs max pooling on the input.";

  let description = [{
    This performs an average pooling over the given input tensor. A sliding
    window of size given by <kernel size> is passed over the input tensor, with
    the mean value being placed in the output tensor.
  }];

  let arguments = (ins
    Tosa_Tensor4D:$input,

    Tosa_IntArrayAttr2:$kernel,
    Tosa_IntArrayAttr2:$stride,
    Tosa_IntArrayAttr4:$pad,
    OptionalAttr<Tosa_UnaryOpQuantizationAttr>:$quantization_info
  );

  let results = (outs
    Tosa_Tensor4D:$output
  );

  let builders = [Tosa_AvgPool2dOpQuantInfoBuilder];
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// Operator: conv2d
//===----------------------------------------------------------------------===//
def Tosa_Conv2DOp : Tosa_Op<"conv2d", [
    DeclareOpInterfaceMethods<InferShapedTypeOpInterface,
                              ["inferReturnTypeComponents"]>,
    Pure]> {
  let summary = "2D Convolution Operator";

  let description = [{
    Performs a 2D convolution over the given tensor input, using the weight
    tensor.
  }];

  let arguments = (ins
    Tosa_Tensor4D:$input,
    Tosa_Tensor4D:$weight,
    Tosa_Tensor1D:$bias,

    Tosa_IntArrayAttr4:$pad,
    Tosa_IntArrayAttr2:$stride,
    Tosa_IntArrayAttr2:$dilation,
    OptionalAttr<Tosa_ConvOpQuantizationAttr>:$quantization_info
  );

  let results = (outs
    Tosa_Tensor4D:$output
  );

  let builders = [Tosa_ConvOpQuantInfoBuilder];
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// Operator: conv3d
//===----------------------------------------------------------------------===//
def Tosa_Conv3DOp : Tosa_Op<"conv3d", [
    DeclareOpInterfaceMethods<InferShapedTypeOpInterface,
                              ["inferReturnTypeComponents"]>,
    Pure]> {
  let summary = "3D Convolution operator";

  let description = [{
    Performs a 3D convolution over the given input tensor.
  }];

  let arguments = (ins
    Tosa_Tensor5D:$input,
    Tosa_Tensor5D:$weight,
    Tosa_Tensor1D:$bias,

    Tosa_IntArrayAttr6:$pad,
    Tosa_IntArrayAttr3:$stride,
    Tosa_IntArrayAttr3:$dilation,
    OptionalAttr<Tosa_ConvOpQuantizationAttr>:$quantization_info
  );

  let results = (outs
    Tosa_Tensor5D:$output
  );

  let builders = [Tosa_ConvOpQuantInfoBuilder];
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// Operator: depthwise_conv2d
//===----------------------------------------------------------------------===//
def Tosa_DepthwiseConv2DOp : Tosa_Op<"depthwise_conv2d", [
    DeclareOpInterfaceMethods<InferShapedTypeOpInterface,
                              ["inferReturnTypeComponents"]>,
    Pure]> {
  let summary = "Depthwise 2D Convolution operator";

  let description = [{
    Performs 2D convolutions separately over each channel of the given tensor
    input, using the weight tensor.
  }];

  let arguments = (ins
    Tosa_Tensor4D:$input,
    Tosa_Tensor4D:$weight,
    Tosa_Tensor1D:$bias,

    Tosa_IntArrayAttr4:$pad,
    Tosa_IntArrayAttr2:$stride,
    Tosa_IntArrayAttr2:$dilation,
    OptionalAttr<Tosa_ConvOpQuantizationAttr>:$quantization_info
  );

  let results = (outs
    Tosa_Tensor4D:$output
  );

  let builders = [Tosa_ConvOpQuantInfoBuilder];
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// Operator: fft2d
//===----------------------------------------------------------------------===//
def Tosa_FFT2dOp : Tosa_Op<"fft2d", [
    DeclareOpInterfaceMethods<InferShapedTypeOpInterface,
                              ["inferReturnTypeComponents"]>,
    Pure]> {
  let summary = "Performs FFT2D operation on the input.";

  let description = [{
    Performs a batched complex 2D Fast Fourier Transform over the input. The
    complex input values are constructed from the corresponding values in the
    input_real and input_imag tensors. The resulting values in the output are
    split into the output_real and output_imag tensors. No normalization is
    applied on either the forward or inverse versions of the operation.
  }];

  let arguments = (ins
    Tosa_Tensor3D:$input_real,
    Tosa_Tensor3D:$input_imag,

    BoolAttr:$inverse
  );

  let results = (outs
    Tosa_Tensor3D:$output_real,
    Tosa_Tensor3D:$output_imag
  );
}

//===----------------------------------------------------------------------===//
// Operator: fully_connected
//===----------------------------------------------------------------------===//
def Tosa_FullyConnectedOp : Tosa_Op<"fully_connected", [
    DeclareOpInterfaceMethods<InferShapedTypeOpInterface,
                              ["inferReturnTypeComponents"]>,
    Pure]> {
  let summary = "Fully Connected operator";

  let description = [{
    Performs a fully connected network.
  }];

  let arguments = (ins
    Tosa_Tensor2D:$input,
    Tosa_Tensor2D:$weight,
    Tosa_Tensor1D:$bias,
    OptionalAttr<Tosa_ConvOpQuantizationAttr>:$quantization_info
  );

  let results = (outs
    Tosa_Tensor2D:$output
  );

  let builders = [Tosa_FCOpQuantInfoBuilder];
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// Operator: matmul
//===----------------------------------------------------------------------===//
def Tosa_MatMulOp : Tosa_Op<"matmul", [
    DeclareOpInterfaceMethods<InferShapedTypeOpInterface,
                              ["inferReturnTypeComponents"]>,
    Pure]> {
  let summary = "Matrix multiplication with bias";

  let description = [{
    Performs a two dimensional matrix multiplication. This allows both inputs to
    be activations, rather than reserving weights as an attribute in the
    FULLY_CONNECTED operator.
  }];

  let arguments = (ins
    Tosa_Tensor3D:$a,
    Tosa_Tensor3D:$b,
    OptionalAttr<Tosa_MatMulOpQuantizationAttr>:$quantization_info
  );

  let results = (outs
    Tosa_Tensor3D:$c
  );

  let builders = [Tosa_MatMulOpQuantInfoBuilder];
}

//===----------------------------------------------------------------------===//
// Operator: max_pool2d
//===----------------------------------------------------------------------===//
def Tosa_MaxPool2dOp : Tosa_Op<"max_pool2d", [
    DeclareOpInterfaceMethods<InferShapedTypeOpInterface,
                              ["inferReturnTypeComponents"]>,
    Pure]> {
  let summary = "Performs max pooling on the input.";

  let description = [{
    This performs a max pooling over the given input tensor. A sliding window of
    size given by <kernel size> is passed over the input tensor, with the
    maximum value being placed in the
    output tensor.
  }];

  let arguments = (ins
    Tosa_Tensor4D:$input,

    Tosa_IntArrayAttr2:$kernel,
    Tosa_IntArrayAttr2:$stride,
    Tosa_IntArrayAttr4:$pad
  );

  let results = (outs
    Tosa_Tensor4D:$output
  );

  let hasCanonicalizer = 1;
}

//===----------------------------------------------------------------------===//
// Operator: rfft2d
//===----------------------------------------------------------------------===//
def Tosa_RFFT2dOp : Tosa_Op<"rfft2d", [
    DeclareOpInterfaceMethods<InferShapedTypeOpInterface,
                              ["inferReturnTypeComponents"]>,
    Pure]> {
  let summary = "Performs RFFT2D operation on the input.";

  let description = [{
    Performs a batched 2D real-valued Fast Fourier Transform over the input where
    the input tensor consists of real values producing complex valued output. The
    complex output values will be split into the output_real and output_imag
    tensor arguments. RFFT2D takes advantage of Hermitian symmetry to only
    calculate the first half of the final output axis. Imaginary values with
    locations (0,0), (0,W/2), (H/2,0) and (H/2,W/2) are zero.
  }];

  let arguments = (ins
    Tosa_Tensor3D:$input
  );

  let results = (outs
    Tosa_Tensor3D:$output_real,
    Tosa_Tensor3D:$output_imag
  );
}

//===----------------------------------------------------------------------===//
// Operator: transpose_conv2d
//===----------------------------------------------------------------------===//
def Tosa_TransposeConv2DOp : Tosa_Op<"transpose_conv2d", [
    DeclareOpInterfaceMethods<InferShapedTypeOpInterface,
                              ["inferReturnTypeComponents"]>,
    Pure]> {
  let summary = "Transpose 2D Convolution operator.";

  let description = [{
    Performs a 2D transposed convolution over the given tensor input, using the
    weights tensor.
  }];

  let arguments = (ins
    Tosa_Tensor4D:$input,
    Tosa_Tensor4D:$filter,
    Tosa_Tensor1D:$bias,

    Tosa_IntArrayAttr4:$out_pad,
    Tosa_IntArrayAttr2:$stride,
    Tosa_IntArrayAttrUpto4:$out_shape,
    OptionalAttr<Tosa_ConvOpQuantizationAttr>:$quantization_info
  );

  let results = (outs
    Tosa_Tensor4D:$output
  );

  let builders = [Tosa_TransConvOpQuantInfoBuilder];
}

//===----------------------------------------------------------------------===//
// TOSA Spec Section 2.3
// Operator Class: Activation Functions.
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Operator: clamp
//===----------------------------------------------------------------------===//
def Tosa_ClampOp : Tosa_Op<"clamp", [
    DeclareOpInterfaceMethods<InferShapedTypeOpInterface,
                              ["inferReturnTypeComponents"]>,
    Pure]> {
  let summary = "Computes clamp(features, min, max).";

  let description = [{
    Clamp to an arbitrary minimum and maximum value.
    Maximum and minimum values are specified as values in the range of the
    input type.
    No zero point subtraction is done to the values, thus to clamp to the zero
    point value, the zero point itself should be supplied as the minimum value.
  }];

  let arguments = (ins
    Tosa_Tensor:$input,
    I64Attr:$min_int,
    I64Attr:$max_int,
    F32Attr:$min_fp,
    F32Attr:$max_fp
  );

  let results = (outs
    Tosa_Tensor:$output
  );

  let hasCanonicalizer = 1;
}

//===----------------------------------------------------------------------===//
// Operator: sigmoid
//===----------------------------------------------------------------------===//
def Tosa_SigmoidOp : Tosa_Op<"sigmoid", [
    DeclareOpInterfaceMethods<InferShapedTypeOpInterface,
                              ["inferReturnTypeComponents"]>,
    Pure]> {
  let summary = "Computes elementwise sigmoid of input.";

  let description = [{
    Sigmoid function: output = 1 / (1 + exp(-input))
    For quantized integer data types, the TABLE operator should be used instead
    with the following definition.  The sigmoid table has 513 entries each of
    16-bit precision and covering the input range -16.0 to +16.0
    in steps of 1/16.
  }];

  let arguments = (ins
    Tosa_Tensor:$input
  );

  let results = (outs
    Tosa_Tensor:$output
  );
}

//===----------------------------------------------------------------------===//
// Operator: tanh
//===----------------------------------------------------------------------===//
def Tosa_TanhOp : Tosa_Op<"tanh", [
    DeclareOpInterfaceMethods<InferShapedTypeOpInterface,
                              ["inferReturnTypeComponents"]>,
    Pure]> {
  let summary = "Computes elementwise hyperbolic tangent of input";

  let description = [{
    Parameterized hyperbolic tangent.
    For quantized integer data types, the TABLE operator should be used instead
    with the following definition.  The tanh_table has 513 entries each of
    16-bit precision and covering the input range -8.0 to +8.0 in steps of 1/32.
  }];

  let arguments = (ins
    Tosa_Tensor:$input
  );

  let results = (outs
    Tosa_Tensor:$output
  );
}

//===----------------------------------------------------------------------===//
// TOSA Spec Section 2.4
// Operator Class: Elementwise unary/binary/ternary operators.
// Operator Subclass: Elementwise binary ops.
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Operator: add
//===----------------------------------------------------------------------===//
def Tosa_AddOp : Tosa_Op<"add", [
    DeclareOpInterfaceMethods<InferShapedTypeOpInterface,
                              ["inferReturnTypeComponents"]>,
    ResultsBroadcastableShape, Pure, Commutative]> {
  let summary = "Elementwise addition operator";

  let description = [{
    Elementwise addition of input1 and input2. Axis of size 1 will be broadcast,
    as necessary.
  }];

  let arguments = (ins
    Tosa_Tensor:$input1,
    Tosa_Tensor:$input2
  );

  let results = (outs
    Tosa_Tensor:$output
  );

  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// Operator: arithmetic_right_shift
//===----------------------------------------------------------------------===//
def Tosa_ArithmeticRightShiftOp : Tosa_Op<"arithmetic_right_shift", [
    DeclareOpInterfaceMethods<InferShapedTypeOpInterface,
                              ["inferReturnTypeComponents"]>,
    ResultsBroadcastableShape, Pure]> {
  let summary = "Elementwise Arithmetic Right Shift";

  let description = [{
    Elementwise arithmetic right shift of input1 by the amount specified in
    input2. Axis of size 1 will be broadcast, as necessary.
  }];

  let arguments = (ins
    Tosa_Tensor:$input1,
    Tosa_Tensor:$input2,
    BoolAttr:$round
  );

  let results = (outs
    Tosa_Tensor:$output
  );
}

//===----------------------------------------------------------------------===//
// Operator: bitwise_and
//===----------------------------------------------------------------------===//
def Tosa_BitwiseAndOp : Tosa_Op<"bitwise_and", [
    DeclareOpInterfaceMethods<InferShapedTypeOpInterface,
                              ["inferReturnTypeComponents"]>,
    ResultsBroadcastableShape, Pure, Commutative]> {
  let summary = "Bitwise AND operator";

  let description = [{
    Elementwise bitwise AND of input1 and input2. Axis of size 1
    will be broadcast as necessary.
  }];

  let arguments = (ins
    Tosa_Tensor:$input1,
    Tosa_Tensor:$input2
  );

  let results = (outs
    Tosa_Tensor:$output
  );
}

//===----------------------------------------------------------------------===//
// Operator: bitwise_or
//===----------------------------------------------------------------------===//
def Tosa_BitwiseOrOp : Tosa_Op<"bitwise_or", [
    DeclareOpInterfaceMethods<InferShapedTypeOpInterface,
                              ["inferReturnTypeComponents"]>,
    ResultsBroadcastableShape, Pure, Commutative]> {
  let summary = "Bitwise OR operator";

  let description = [{
    Elementwise bitwise OR of input1 and input2. Axis of size 1 will be
    broadcast as necessary.
  }];

  let arguments = (ins
    Tosa_Tensor:$input1,
    Tosa_Tensor:$input2
  );

  let results = (outs
    Tosa_Tensor:$output
  );
}

//===----------------------------------------------------------------------===//
// Operator: bitwise_xor
//===----------------------------------------------------------------------===//
def Tosa_BitwiseXorOp : Tosa_Op<"bitwise_xor", [
    DeclareOpInterfaceMethods<InferShapedTypeOpInterface,
                              ["inferReturnTypeComponents"]>,
    ResultsBroadcastableShape, Pure, Commutative]> {
  let summary = "Bitwise XOR operator";

  let description = [{
    Elementwise bitwise XOR of input1 and input2. Axis of size 1 will be
    broadcast as necessary.
  }];

  let arguments = (ins
    Tosa_Tensor:$input1,
    Tosa_Tensor:$input2
  );

  let results = (outs
    Tosa_Tensor:$output
  );
}

//===----------------------------------------------------------------------===//
// Operator: div
//===----------------------------------------------------------------------===//
def Tosa_DivOp : Tosa_Op<"div", [
    DeclareOpInterfaceMethods<InferShapedTypeOpInterface,
                              ["inferReturnTypeComponents"]>,
    ResultsBroadcastableShape, Pure]> {
  let summary = "Integer divide operator";

  let description = [{
    Elementwise integer divide operator of input1 by input2. Axis of size 1
    will be broadcast, as necessary.
  }];

  let arguments = (ins
    Tosa_Int32Tensor:$input1,
    Tosa_Int32Tensor:$input2
  );

  let results = (outs
    Tosa_Int32Tensor:$output
  );

  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// Operator: logical_and
//===----------------------------------------------------------------------===//
def Tosa_LogicalAndOp : Tosa_Op<"logical_and", [
    DeclareOpInterfaceMethods<InferShapedTypeOpInterface,
                              ["inferReturnTypeComponents"]>,
    ResultsBroadcastableShape, Commutative, Pure]> {
  let summary = "Returns the truth value of x AND y element-wise.";

  let description = [{
    Elementwise logical AND of input1 and input2. Axis of size 1 will be
    broadcast, as necessary.
  }];

  let arguments = (ins
    I1Tensor:$input1,
    I1Tensor:$input2
  );

  let results = (outs
    I1Tensor:$z
  );
}

//===----------------------------------------------------------------------===//
// Operator: logical_left_shift
//===----------------------------------------------------------------------===//
def Tosa_LogicalLeftShiftOp : Tosa_Op<"logical_left_shift", [
    DeclareOpInterfaceMethods<InferShapedTypeOpInterface,
                              ["inferReturnTypeComponents"]>,
    ResultsBroadcastableShape, Pure]> {
  let summary = "Elementwise Logical Left Shift";

  let description = [{
    Elementwise left shift of input1 and input2. Axis of size 1 will be
    broadcast, as necessary.
  }];

  let arguments = (ins
    Tosa_Tensor:$input1,
    Tosa_Tensor:$input2
  );

  let results = (outs
    Tosa_Tensor:$output
  );
}

//===----------------------------------------------------------------------===//
// Operator: logical_right_shift
//===----------------------------------------------------------------------===//
def Tosa_LogicalRightShiftOp : Tosa_Op<"logical_right_shift", [
    DeclareOpInterfaceMethods<InferShapedTypeOpInterface,
                              ["inferReturnTypeComponents"]>,
    ResultsBroadcastableShape, Pure]> {
  let summary = "Elementwise Logical Right Shift";

  let description = [{
    Elementwise logical right shift of input1 by the amount specified in input2.
    Axis of size 1 will be broadcast, as necessary.
  }];

  let arguments = (ins
    Tosa_Tensor:$input1,
    Tosa_Tensor:$input2
  );

  let results = (outs
    Tosa_Tensor:$output
  );
}

//===----------------------------------------------------------------------===//
// Operator: logical_or
//===----------------------------------------------------------------------===//
def Tosa_LogicalOrOp : Tosa_Op<"logical_or", [
    DeclareOpInterfaceMethods<InferShapedTypeOpInterface,
                              ["inferReturnTypeComponents"]>,
    ResultsBroadcastableShape, Commutative, Pure]> {
  let summary = "Returns the truth value of x OR y element-wise.";

  let description = [{
    Elementwise logical OR of input1 and input2. Axis of size 1 will be
    broadcast as necessary.
  }];

  let arguments = (ins
    I1Tensor:$input1,
    I1Tensor:$input2
  );

  let results = (outs
    I1Tensor:$z
  );
}

//===----------------------------------------------------------------------===//
// Operator: logical_xor
//===----------------------------------------------------------------------===//
def Tosa_LogicalXorOp : Tosa_Op<"logical_xor", [
    DeclareOpInterfaceMethods<InferShapedTypeOpInterface,
                              ["inferReturnTypeComponents"]>,
    ResultsBroadcastableShape, Commutative, Pure]> {
  let summary = "Returns the truth value of x XOR y element-wise.";

  let description = [{
    Elementwise logical XOR of input1 and input2.  Axis of size 1 will be
    broadcast as necessary.
  }];

  let arguments = (ins
    I1Tensor:$input1,
    I1Tensor:$input2
  );

  let results = (outs
    I1Tensor:$z
  );
}

//===----------------------------------------------------------------------===//
// Operator: maximum
//===----------------------------------------------------------------------===//
def Tosa_MaximumOp : Tosa_Op<"maximum", [
    DeclareOpInterfaceMethods<InferShapedTypeOpInterface,
                              ["inferReturnTypeComponents"]>,
    ResultsBroadcastableShape, Pure, Commutative]> {
  let summary = "Elementwise Maximum";

  let description = [{
    Elementwise max of input1 and input2. Axis of size 1 will be broadcast, as
    necessary.
  }];

  let arguments = (ins
    Tosa_Tensor:$input1,
    Tosa_Tensor:$input2
  );

  let results = (outs
    Tosa_Tensor:$output
  );
}

//===----------------------------------------------------------------------===//
// Operator: minimum
//===----------------------------------------------------------------------===//
def Tosa_MinimumOp : Tosa_Op<"minimum", [
    DeclareOpInterfaceMethods<InferShapedTypeOpInterface,
                              ["inferReturnTypeComponents"]>,
    ResultsBroadcastableShape, Pure, Commutative]> {
  let summary = "Elementwise Minimum";

  let description = [{
    Elementwise minimum of input1 and input2. Axis of size 1
    will be broadcast, as necessary.
  }];

  let arguments = (ins
    Tosa_Tensor:$input1,
    Tosa_Tensor:$input2
  );

  let results = (outs
    Tosa_Tensor:$output
  );
}

//===----------------------------------------------------------------------===//
// Operator: mul
//===----------------------------------------------------------------------===//
def Tosa_MulOp : Tosa_Op<"mul", [
    DeclareOpInterfaceMethods<InferShapedTypeOpInterface,
                              ["inferReturnTypeComponents"]>,
    ResultsBroadcastableShape, Pure, Commutative]> {
  let summary = "Multiplication operator";

  let description = [{
    Elementwise multiplication (Hadamard product) of input1 and input2.
    Axis of size 1 will be broadcast, as necessary.
  }];

  let arguments = (ins
    Tosa_Tensor:$input1,
    Tosa_Tensor:$input2,
    I32Attr:$shift
  );

  let results = (outs
    Tosa_Tensor:$output
  );

  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// Operator: pow
//===----------------------------------------------------------------------===//
def Tosa_PowOp : Tosa_Op<"pow", [
    DeclareOpInterfaceMethods<InferShapedTypeOpInterface,
                              ["inferReturnTypeComponents"]>,
    ResultsBroadcastableShape, Pure]> {
  let summary = "Computes the power of one value to another.";

  let description = [{
    Elementwise input1 raised to the power of input2.
    Axis of size 1 will be broadcast, as necessary.
  }];

  let arguments = (ins
    Tosa_Tensor:$input1,
    Tosa_Tensor:$input2
  );

  let results = (outs
    Tosa_Tensor:$z
  );
}

//===----------------------------------------------------------------------===//
// Operator: sub
//===----------------------------------------------------------------------===//
def Tosa_SubOp : Tosa_Op<"sub", [
    DeclareOpInterfaceMethods<InferShapedTypeOpInterface,
                              ["inferReturnTypeComponents"]>,
    ResultsBroadcastableShape, Pure]> {
  let summary = "Elementwise subtraction operator";

  let description = [{
    Elementwise subtraction of input1 and input2. Axis of size 1 will be
    broadcast as necessary.
  }];

  let arguments = (ins
    Tosa_Tensor:$input1,
    Tosa_Tensor:$input2
  );

  let results = (outs
    Tosa_Tensor:$output
  );

  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// Operator: table
//===----------------------------------------------------------------------===//
def Tosa_TableOp : Tosa_Op<"table", [
    DeclareOpInterfaceMethods<InferShapedTypeOpInterface,
                              ["inferReturnTypeComponents"]>,
    Pure]> {
  let summary = "Table lookup op";

  let description = [{
    Interpolated table lookup operation. Input values are scaled to create a
    fixed-point 9.7 value.    The high 9 bits are used to index into the table.
    The fractional bits are used to interpolate based on the looked up value and
    the index+1 value in the table. The TABLE operator then returns a 16.7
    interpolated value. Note that there must be 513 values to handle the full
    range of inputs.

    The TABLE operator is expected to be used as follows:
    * A RESCALE node is expected before the TABLE operator to scale the input
      to a full int16_t range for the table lookup
    * If an int16_t result is required then follow the TABLE operator with a
      RESCALE with a right shift of 7
    * If an int8_t result is required then follow the TABLE operator with a
      RESCALE with a right shift of 15
  }];

  let arguments = (ins
    Tosa_Tensor: $input,
    Tosa_Tensor1D: $table
  );

  let results = (outs
    Tosa_Tensor:$output
  );

}

//===----------------------------------------------------------------------===//
// TOSA Spec Section 2.5
// Operator Class: Elementwise unary/binary/ternary operators.
// Operator Subclass: Elementwise unary ops.
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Operator: abs
//===----------------------------------------------------------------------===//
def Tosa_AbsOp : Tosa_Op<"abs", [
    DeclareOpInterfaceMethods<InferShapedTypeOpInterface,
                              ["inferReturnTypeComponents"]>,
    Pure]> {
  let summary = "Elementwise abs op";

  let description = [{
    Elementwise absolute value operation
  }];

  let arguments = (ins
    Tosa_Tensor:$input1
  );

  let results = (outs
    Tosa_Tensor:$output
  );
}

//===----------------------------------------------------------------------===//
// Operator: bitwise_not
//===----------------------------------------------------------------------===//
def Tosa_BitwiseNotOp : Tosa_Op<"bitwise_not", [
    DeclareOpInterfaceMethods<InferShapedTypeOpInterface,
                              ["inferReturnTypeComponents"]>,
    ResultsBroadcastableShape, Pure]> {
  let summary = "Bitwise NOT operator";

  let description = [{
    Elementwise bitwise NOT of input tensor.
  }];

  let arguments = (ins
    Tosa_Tensor:$input1
  );

  let results = (outs
    Tosa_Tensor:$output
  );
}

//===----------------------------------------------------------------------===//
// Operator: ceil
//===----------------------------------------------------------------------===//
def Tosa_CeilOp : Tosa_Op<"ceil", [
    DeclareOpInterfaceMethods<InferShapedTypeOpInterface,
                              ["inferReturnTypeComponents"]>,
    Pure]> {
  let summary = "Elementwise ceil op";

  let description = [{
    Elementwise ceiling operation
  }];

  let arguments = (ins
    Tosa_Tensor:$input1
  );

  let results = (outs
    Tosa_Tensor:$output
  );
}

//===----------------------------------------------------------------------===//
// Operator: clz
//===----------------------------------------------------------------------===//
def Tosa_ClzOp : Tosa_Op<"clz", [
    DeclareOpInterfaceMethods<InferShapedTypeOpInterface,
                              ["inferReturnTypeComponents"]>,
    Pure]> {
  let summary = "Elementwise count leading zero op";

  let description = [{
    Elementwise count leading zeros operation
  }];

  let arguments = (ins
    Tosa_Tensor:$input1
  );

  let results = (outs
    Tosa_Tensor:$output
  );
}

//===----------------------------------------------------------------------===//
// Operator: exp
//===----------------------------------------------------------------------===//
def Tosa_ExpOp : Tosa_Op<"exp", [
    DeclareOpInterfaceMethods<InferShapedTypeOpInterface,
                              ["inferReturnTypeComponents"]>,
    Pure]> {
  let summary = "Elementwise exp op";

  let description = [{
    Elementwise e to the x operation
  }];

  let arguments = (ins
    Tosa_Tensor:$input1
  );

  let results = (outs
    Tosa_Tensor:$output
  );

  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// Operator: floor
//===----------------------------------------------------------------------===//
def Tosa_FloorOp : Tosa_Op<"floor", [
    DeclareOpInterfaceMethods<InferShapedTypeOpInterface,
                              ["inferReturnTypeComponents"]>,
    Pure]> {
  let summary = "Elementwise floor op";

  let description = [{
    Elementwise floor operation
  }];

  let arguments = (ins
    Tosa_Tensor:$input1
  );

  let results = (outs
    Tosa_Tensor:$output
  );
}

//===----------------------------------------------------------------------===//
// Operator: log
//===----------------------------------------------------------------------===//
def Tosa_LogOp : Tosa_Op<"log", [
    DeclareOpInterfaceMethods<InferShapedTypeOpInterface,
                              ["inferReturnTypeComponents"]>,
    Pure]> {
  let summary = "Elementwise log op";

  let description = [{
    Elementwise natural logarithm operation
  }];

  let arguments = (ins
    Tosa_Tensor:$input1
  );

  let results = (outs
    Tosa_Tensor:$output
  );

  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// Operator: logical_not
//===----------------------------------------------------------------------===//
def Tosa_LogicalNotOp : Tosa_Op<"logical_not", [
    DeclareOpInterfaceMethods<InferShapedTypeOpInterface,
                              ["inferReturnTypeComponents"]>,
    Pure, SameOperandsAndResultType]> {
  let summary = "Returns the truth value of NOT x element-wise.";

  let description = [{
    Elementwise logical NOT of input.
  }];

  let arguments = (ins
    I1Tensor:$input1
  );

  let results = (outs
    I1Tensor:$output
  );
}

//===----------------------------------------------------------------------===//
// Operator: negate
//===----------------------------------------------------------------------===//
def Tosa_NegateOp : Tosa_Op<"negate", [
    DeclareOpInterfaceMethods<InferShapedTypeOpInterface,
                              ["inferReturnTypeComponents"]>,
    Pure]> {
  let summary = "Elementwise negate op";

  let description = [{
    Elementwise negation operation
  }];

  let arguments = (ins
      Tosa_Tensor:$input1,
      OptionalAttr<Tosa_UnaryOpQuantizationAttr>:$quantization_info
  );

  let results = (outs
    Tosa_Tensor:$output
  );

  let builders = [Tosa_UnaryOpQuantInfoBuilder];
}

//===----------------------------------------------------------------------===//
// Operator: reciprocal
//===----------------------------------------------------------------------===//
def Tosa_ReciprocalOp : Tosa_Op<"reciprocal", [
    DeclareOpInterfaceMethods<InferShapedTypeOpInterface,
                              ["inferReturnTypeComponents"]>,
    Pure]> {
  let summary = "Elementwise reciprocal op";

  let description = [{
    Elementwise reciprocal operation. For integer operation, a TABLE should be
    used with the appropriate ranges.
  }];

  let arguments = (ins
    Tosa_Tensor:$input1
  );

  let results = (outs
    Tosa_Tensor:$output
  );
}

//===----------------------------------------------------------------------===//
// Operator: rsqrt
//===----------------------------------------------------------------------===//
def Tosa_RsqrtOp : Tosa_Op<"rsqrt", [
    DeclareOpInterfaceMethods<InferShapedTypeOpInterface,
                              ["inferReturnTypeComponents"]>,
    Pure]> {
  let summary = "Elementwise 1/sqrt op";

  let description = [{
    Elementwise reciprocal square root operation. For integer operation, a TABLE
    should be used with the appropriate ranges.
  }];

  let arguments = (ins
    Tosa_Tensor:$input1
  );

  let results = (outs
    Tosa_Tensor:$output
  );
}

//===----------------------------------------------------------------------===//
// TOSA Spec Section 2.6
// Operator Class: Elementwise unary/binary/ternary operators.
// Operator Subclass: Elementwise ternary ops.
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Operator: select
//===----------------------------------------------------------------------===//
def Tosa_SelectOp : Tosa_Op<"select", [
    DeclareOpInterfaceMethods<InferShapedTypeOpInterface,
                              ["inferReturnTypeComponents"]>, Pure]> {
  let summary = "Elementwise select operator";

  let description = [{
    Elementwise select of the output based on a condition.
  }];

  let arguments = (ins
    I1Tensor:$pred,
    Tosa_Tensor:$on_true,
    Tosa_Tensor:$on_false
  );

  let results = (outs
    Tosa_Tensor:$output
  );
  let hasCanonicalizeMethod = 1;
  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// TOSA Spec Section 2.7
// Operator Class: Logical Operations.
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Operator: equal
//===----------------------------------------------------------------------===//
def Tosa_EqualOp : Tosa_Op<"equal", [InferTensorType, ResultsBroadcastableShape,
    Commutative, Pure]> {
  let summary = "Returns the truth value of (x == y) element-wise.";

  let description = [{
     Elementwise comparison operation
  }];

  let arguments = (ins
    Tosa_Tensor:$input1,
    Tosa_Tensor:$input2
  );

  let results = (outs
    I1Tensor:$output
  );

  let extraClassDeclaration = [{
    /// Returns when two result types are compatible for this op; method used by
    /// InferTypeOpInterface.
    static bool isCompatibleReturnTypes(TypeRange l, TypeRange r);
  }];

  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// Operator: greater
//===----------------------------------------------------------------------===//
def Tosa_GreaterOp : Tosa_Op<"greater", [
    DeclareOpInterfaceMethods<InferShapedTypeOpInterface,
                              ["inferReturnTypeComponents"]>,
    ResultsBroadcastableShape, Pure]> {
  let summary = "Returns the truth value of (x > y) element-wise.";

  let description = [{
    Elementwise greater than comparison operation
  }];

  let arguments = (ins
    Tosa_Tensor:$input1,
    Tosa_Tensor:$input2
  );

  let results = (outs
    I1Tensor:$output
  );

  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// Operator: greater_equal
//===----------------------------------------------------------------------===//
def Tosa_GreaterEqualOp : Tosa_Op<"greater_equal", [
    DeclareOpInterfaceMethods<InferShapedTypeOpInterface,
                              ["inferReturnTypeComponents"]>,
    ResultsBroadcastableShape, Pure]> {
  let summary = "Returns the truth value of (x >= y) element-wise.";

  let description = [{
    Elementwise comparison operation
  }];

  let arguments = (ins
    Tosa_Tensor:$input1,
    Tosa_Tensor:$input2
  );

  let results = (outs
    I1Tensor:$output
  );

  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// TOSA Spec Section 2.8
// Operator Class: Reduction Ops.
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Operator: reduce_all
//===----------------------------------------------------------------------===//
def Tosa_ReduceAllOp : Tosa_Op<"reduce_all", [
    InferTensorType, Pure]> {
  let summary = "Reduce All operator";

  let description = [{
    Reduce a tensor along the given axis with a logical AND operation
  }];

  let arguments = (ins
    Tosa_Tensor1Dto4D:$input,
    I64Attr:$axis
  );

  let results = (outs
    Tosa_Tensor1Dto4D:$output
  );

  let hasFolder = 1;

  let extraClassDeclaration = [{
    /// Returns true when two result types are compatible for this op;
    /// Method used by InferTypeOpInterface.
    static bool isCompatibleReturnTypes(TypeRange l, TypeRange r);
  }];
}

//===----------------------------------------------------------------------===//
// Operator: reduce_any
//===----------------------------------------------------------------------===//
def Tosa_ReduceAnyOp : Tosa_Op<"reduce_any", [
    InferTensorType, Pure]> {
  let summary = "Reduce Any operator";

  let description = [{
    Reduce a tensor along the given axis with a logical OR operation
  }];

  let arguments = (ins
    Tosa_Tensor1Dto4D:$input,
    I64Attr:$axis
  );

  let results = (outs
    Tosa_Tensor1Dto4D:$output
  );

  let hasFolder = 1;

  let extraClassDeclaration = [{
    /// Returns true when two result types are compatible for this op;
    /// Method used by InferTypeOpInterface.
    static bool isCompatibleReturnTypes(TypeRange l, TypeRange r);
  }];
}

//===----------------------------------------------------------------------===//
// Operator: reduce_max
//===----------------------------------------------------------------------===//
def Tosa_ReduceMaxOp : Tosa_Op<"reduce_max", [
    InferTensorType, Pure]> {
  let summary = "Reduce Max operator";

  let description = [{
    Reduce a tensor along the given axis with a maximum operation
  }];

  let arguments = (ins
    Tosa_Tensor1Dto4D:$input,
    I64Attr:$axis
  );

  let results = (outs
    Tosa_Tensor1Dto4D:$output
  );

  let hasFolder = 1;

  let extraClassDeclaration = [{
    /// Returns true when two result types are compatible for this op;
    /// Method used by InferTypeOpInterface.
    static bool isCompatibleReturnTypes(TypeRange l, TypeRange r);
  }];
}

//===----------------------------------------------------------------------===//
// Operator: reduce_min
//===----------------------------------------------------------------------===//
def Tosa_ReduceMinOp : Tosa_Op<"reduce_min", [
    InferTensorType, Pure]> {
  let summary = "Reduce Min operator";

  let description = [{
    Reduce a tensor along the given axis with a minimum operation
  }];

  let arguments = (ins
    Tosa_Tensor1Dto4D:$input,
    I64Attr:$axis
  );

  let results = (outs
    Tosa_Tensor1Dto4D:$output
  );

  let hasFolder = 1;

  let extraClassDeclaration = [{
    /// Returns true when two result types are compatible for this op;
    /// Method used by InferTypeOpInterface.
    static bool isCompatibleReturnTypes(TypeRange l, TypeRange r);
  }];
}

//===----------------------------------------------------------------------===//
// Operator: reduce_prod
//===----------------------------------------------------------------------===//
def Tosa_ReduceProdOp : Tosa_Op<"reduce_prod", [
    InferTensorType, Pure]> {
  let summary = "Reduce Prod operator";

  let description = [{
    Reduce a tensor along the given axis by computing the product of the axis.
  }];

  let arguments = (ins
    Tosa_Tensor1Dto4D:$input,
    I64Attr:$axis
  );

  let results = (outs
    Tosa_Tensor1Dto4D:$output
  );

  let hasFolder = 1;

  let extraClassDeclaration = [{
    /// Returns true when two result types are compatible for this op;
    /// Method used by InferTypeOpInterface.
    static bool isCompatibleReturnTypes(TypeRange l, TypeRange r);
  }];
}

//===----------------------------------------------------------------------===//
// Operator: reduce_sum
//===----------------------------------------------------------------------===//
def Tosa_ReduceSumOp : Tosa_Op<"reduce_sum", [
    InferTensorType, Pure]> {
  let summary = "Reduce Sum operator";

  let description = [{
    Reduce a tensor along the given axis by computing the sum of the axis.
  }];

  let arguments = (ins
    Tosa_Tensor1Dto4D:$input,
    I64Attr:$axis
  );

  let results = (outs
    Tosa_Tensor1Dto4D:$output
  );

  let hasFolder = 1;

  let extraClassDeclaration = [{
    /// Returns true when two result types are compatible for this op;
    /// Method used by InferTypeOpInterface.
    static bool isCompatibleReturnTypes(TypeRange l, TypeRange r);
  }];
}

//===----------------------------------------------------------------------===//
// TOSA Spec Section 2.9
// Operator Class: Data Layout / Memory Reinterpretation.
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Operator: concat
//===----------------------------------------------------------------------===//
def Tosa_ConcatOp : Tosa_Op<"concat", [
    InferTensorType, Pure]> {
  let summary = "Concatenates tensors along one dimension.";

  let description = [{
    Concatenate a variadic amount of tensors along a given axis. No data
    conversion happens during a concat operation.
  }];

  let arguments = (ins
    Variadic<Tosa_Tensor>:$input1,
    I64Attr:$axis
  );

  let results = (outs
    Tosa_Tensor:$output
  );

  let hasCanonicalizer = 1;

  let extraClassDeclaration = [{
    /// Returns true when two result types are compatible for this op;
    /// Method used by InferTypeOpInterface.
    static bool isCompatibleReturnTypes(TypeRange l, TypeRange r);
  }];
}

//===----------------------------------------------------------------------===//
// Operator: pad
//===----------------------------------------------------------------------===//
def Tosa_PadOp : Tosa_Op<"pad", [
    DeclareOpInterfaceMethods<InferShapedTypeOpInterface,
                              ["inferReturnTypeComponents"]>,
    Pure]> {
  let summary = "Pads a tensor with value specified.";

  let description = [{
    Pads a tensor along borders of each dimension with pad_value.
  }];

  let arguments = (ins
    Tosa_RankedTensor:$input1,
    Tosa_Int32Or64Tensor:$padding,
    Optional<Tosa_ScalarTensor>:$pad_const,
    OptionalAttr<Tosa_PadOpQuantizationAttr>:$quantization_info
  );

  let results = (outs
    Tosa_RankedTensor:$output
  );

  let builders = [Tosa_PadOpQuantInfoBuilder,
                  Tosa_ExplicitValuePadOpQuantInfoBuilder];

  let hasCanonicalizer = 1;
  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// Operator: reshape
//===----------------------------------------------------------------------===//
def Tosa_ReshapeOp: Tosa_Op<"reshape", [
  InferTensorType, Pure]> {
  let summary = "Reshape operator";

  let description = [{
    Returns a tensor with the same type/values as the input, with a new shape
    specified by the shape argument. Reshape may operate on tensors of any rank.
    No data conversion happens during a reshape operation.
  }];

  let hasCanonicalizer = 1;
  let hasFolder = 1;
  let hasVerifier = 1;

  let arguments = (ins
    Tosa_Tensor:$input1,
    DenseI64ArrayAttr:$new_shape
  );

  let results = (outs
    Tosa_RankedTensor:$output
  );

  let extraClassDeclaration = [{
    /// Returns true when two result types are compatible for this op;
    /// Method used by InferTypeOpInterface.
    static bool isCompatibleReturnTypes(TypeRange l, TypeRange r);
  }];
}

//===----------------------------------------------------------------------===//
// Operator: reverse
//===----------------------------------------------------------------------===//
def Tosa_ReverseOp: Tosa_Op<"reverse", [
    DeclareOpInterfaceMethods<InferShapedTypeOpInterface,
                              ["inferReturnTypeComponents"]>, Pure]> {
  let summary = "Reverse operator";

  let description = [{
    Returns a tensor with the same type/values as the input, with the data
    reversed along the given axis. No data conversion happens during a reverse
    operation.
  }];

  let arguments = (ins
    Tosa_Tensor1Dto4D:$input,
    I64Attr:$axis
  );

  let results = (outs
    Tosa_Tensor1Dto4D:$output
  );

  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// Operator: slice
//===----------------------------------------------------------------------===//
def Tosa_SliceOp: Tosa_Op<"slice", [
      DeclareOpInterfaceMethods<InferShapedTypeOpInterface,
                              ["inferReturnTypeComponents"]>, Pure]> {
  let summary = "Slice operator";

  let description = [{
    Extracts a slice of the input1 on the given axis, beginning at the
    start coordinates, and extending for size elements in each direction.  No
    data conversion happens during a slice operation.
  }];

  let arguments = (ins
    Tosa_Tensor1Dto6D:$input,
    DenseI64ArrayAttr:$start,
    DenseI64ArrayAttr:$size
  );

  let results = (outs
    Tosa_Tensor1Dto6D:$output
  );

  let hasCanonicalizer = 1;
  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// Operator: tile
//===----------------------------------------------------------------------===//
def Tosa_TileOp: Tosa_Op<"tile", [
      DeclareOpInterfaceMethods<InferShapedTypeOpInterface,
                              ["inferReturnTypeComponents"]>,
      Pure]> {
  let summary = "Tile operator";

  let description = [{
    Replicates input 0 multiplies times along each dimension.
  }];

  let arguments = (ins
    Tosa_Tensor1Dto4D:$input1,
    DenseI64ArrayAttr:$multiples);

  let results = (outs
    Tosa_Tensor1Dto4D:$output
  );

  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// Operator: transpose
//===----------------------------------------------------------------------===//
def Tosa_TransposeOp : Tosa_Op<"transpose", [
      DeclareOpInterfaceMethods<InferShapedTypeOpInterface,
                              ["inferReturnTypeComponents"]>,
      Pure]> {
  let summary = "Transpose operator";

  let description = [{
    Permutes the dimensions based on perm.
  }];

  let arguments = (ins
    Tosa_Tensor1Dto6D:$input1,
    Tosa_Int32Or64Tensor:$perms
  );

  let results = (
    outs Tosa_Tensor1Dto6D:$output
  );

  let extraClassDeclaration = [{
    LogicalResult getConstantPerms(llvm::SmallVector<int64_t> &perms);
  }];

  let hasCanonicalizer = 1;
  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// TOSA Spec Section 2.10
// Operator Class: Scatter/gather Operations.
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Operator: gather
//===----------------------------------------------------------------------===//
def Tosa_GatherOp : Tosa_Op<"gather", [
      DeclareOpInterfaceMethods<InferShapedTypeOpInterface,
                              ["inferReturnTypeComponents"]>,
      Pure]> {
  let summary = "Gather operation,";

  let description = [{
    Generate a tensor for which each element in the output is a slice of the
    values tensor based on the value of indices.
  }];

  let arguments = (ins
    Tosa_Tensor3D:$values,
    2DTensorOf<[Tosa_Int32]>:$indices
  );

  let results = (outs
    Tosa_Tensor3D:$output
  );
}

//===----------------------------------------------------------------------===//
// Operator: scatter
//===----------------------------------------------------------------------===//
def Tosa_ScatterOp : Tosa_Op<"scatter", [
      DeclareOpInterfaceMethods<InferShapedTypeOpInterface,
                              ["inferReturnTypeComponents"]>,
      Pure]> {
  let summary = "Scatter operation,";

  let description = [{
    The values_out tensor is set to the values_in tensor with data modified as follows:
    data from the input tensor is inserted at the positions specified by the indices tensor.
  }];

  let arguments = (ins
    Tosa_Tensor3D:$values_in,
    2DTensorOf<[Tosa_Int32]>:$indices,
    Tosa_Tensor3D:$input
  );

  let results = (outs
    Tosa_Tensor3D:$values_out
  );
}

//===----------------------------------------------------------------------===//
// TOSA Spec Section 2.11
// Operator Class: Image Frontend Functions.
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Operator: resize
//===----------------------------------------------------------------------===//
def Tosa_ResizeOp : Tosa_Op<"resize", [
      DeclareOpInterfaceMethods<InferShapedTypeOpInterface,
                              ["inferReturnTypeComponents"]>,
      Pure]> {

  let summary = "Resize operation, supports various resize/upsample modes";

  let description = [{
    Resizes a tensor. Resize is only allowed in the H and W dimensions. In
    expected use, The height dimension is scaled by factor (scale_y_n/scale_y_d).
    And the width dimension is scaled by factor (scale_x_n/scale_x_d). Thus the
    output dimensions can be derived from the input dimensions by inverting the
    scale. And the [order_y, border_x] values adjust the output size to allow
    fractional sampling beyond integer input position (IH-1,IW-1).
  }];

  let arguments = (ins
    Tosa_Tensor4D:$input,
    Tosa_IntArrayAttr4:$scale,
    Tosa_IntArrayAttr2:$offset,
    Tosa_IntArrayAttr2:$border,
    Tosa_ResizeTypeAttr:$mode
  );

  let results = (outs
    Tosa_Tensor4D:$output
  );

  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// TOSA Spec Section 2.12
// Operator Class: Type Conversion.
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Operator: cast
//===----------------------------------------------------------------------===//
def Tosa_CastOp: Tosa_Op<"cast", [Pure,
      DeclareOpInterfaceMethods<InferShapedTypeOpInterface,
                              ["inferReturnTypeComponents"]>]> {

  let summary = "Cast operation";

  let description = [{
    Performs a set of permissible cast operations
        Mode                    Input   Output
        ---------------------------------------
        signed 8 to bool        int8    Boolean
        signed 16 to bool       int16   Boolean
        signed 32 to bool       int32   Boolean
        bool to 8               Boolean int8
        bool to 16              Boolean int16
        bool to 32              Boolean int32
        signed 8 to signed 16   int8    int16
        signed 8 to signed 32   int8    int32
        signed 16 to signed 8   int16   int8
        signed 16 to signed 32  int16   int32
        signed 32 to signed 8   int32   int8
        signed 32 to signed 16  int32   int16
        float to signed 8       float   int8
        float to signed 16      float   int16
        signed 8 to float       int8    float
        signed 16 to float      int16   float
        float 32 to float 64    float32 float64
        float 64 to float 32    float64 float32
  }];

  let arguments = (ins
    Tosa_Tensor_Plus_F64:$input
  );

  let results = (outs
    Tosa_Tensor_Plus_F64:$output
  );

  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// Operator: rescale
//===----------------------------------------------------------------------===//
def Tosa_RescaleOp: Tosa_Op<"rescale", [Pure,
      DeclareOpInterfaceMethods<InferShapedTypeOpInterface,
                              ["inferReturnTypeComponents"]>]> {
  let summary = "Tosa rescale operator";

  let description = [{
    Rescale quantized values into a new domain. Supported rescalings are:
    Mode                    Input   Output
    signed 8 to 8           int8    int8
    signed 8 to 16          int8    int16
    signed 8 to 32          int8    int32
    signed 16 to 8          int16   int8
    signed 16 to 16         int16   int16
    signed 16 to 32         int16   int32
    signed 32 to 8          int32   int8
    signed 32 to 16         int32   int16
    signed 32 to 32         int32   int32
    signed 48 to 8          int48   int8
    signed 48 to 16         int48   int16
    signed 48 to 32         int48   int32
    unsigned 8 to signed 8  uint8   int8
    signed 8 to unsigned 8  int8    uint8
  }];

  let arguments = (ins
    Tosa_Tensor:$input,
    I32Attr:$input_zp,
    I32Attr:$output_zp,
    DenseI32ArrayAttr:$multiplier,
    DenseI32ArrayAttr:$shift,
    BoolAttr:$scale32,
    BoolAttr:$double_round,
    BoolAttr:$per_channel
  );

  let results = (outs
    Tosa_Tensor:$output
  );
}

//===----------------------------------------------------------------------===//
// TOSA Spec Section 2.13
// Operator Class: Data Node Ops.
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Operator: const
//===----------------------------------------------------------------------===//
def Tosa_ConstOp : Tosa_Op<"const", [ConstantLike, Pure,
                                     FirstAttrDerivedResultType]> {
  let summary = "Constant op.";

  let description = [{
    A node containing constant data for use as the input to an operation. May
    hold data in any of the supported data formats.
  }];

  let arguments = (ins
    ElementsAttr:$value
  );

  let results = (outs
    Tosa_Tensor_Plus_F64:$output
  );
  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// Operator: identity
//===----------------------------------------------------------------------===//
def Tosa_IdentityOp: Tosa_Op<"identity", [Pure,
      DeclareOpInterfaceMethods<InferShapedTypeOpInterface,
                              ["inferReturnTypeComponents"]>]> {
  let summary = "Identity operator";
  let description = [{
    Returns a tensor with the same shape, size, type
    and content as the input.
  }];

  let arguments = (ins
    Tosa_Tensor:$input1
  );

  let results = (outs
    Tosa_Tensor:$output
  );
}

//===----------------------------------------------------------------------===//
// TOSA Spec Section 2.14
// Operator Class: Custom Operators.
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Operator: custom
//===----------------------------------------------------------------------===//
def Tosa_CustomOp : Tosa_Op<"custom"> {

  let summary = "Custom operator wrapper for Tosa";

  let description = [{
    Hardware implementing TOSA may choose to add additional custom operators
    that are not expressed in the existing TOSA operations. These operators are
    not expected to be portable across TOSA implementations. The input and
    output signatures must be expressed in the corresponding TOSA node.

    `identifier` is a string that tells the backend which custom operator is being
    called.

    `config` is a string identifier which can help avoid name collisions on the
    identifier field.

    `implementation_attrs` is a string which is a backend and identifier specific
    set of attributes to the custom operator.

    `inputs` is the set of tensor inputs to the custom operator.

    `outputs is the list of tensors returned by the operator. The number of operators
    is backend specific.
  }];

  let arguments = (ins
    StrAttr:$identifier,
    StrAttr:$config,
    StrAttr:$implementation_attrs,
    Variadic<Tosa_Tensor>:$inputs
  );

  let results = (outs
    Variadic<Tosa_Tensor>:$outputs
  );
}

//===----------------------------------------------------------------------===//
// TOSA Spec Section 2.15
// Operator Class: Control Flow Operators.
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Operator: cond_if
//===----------------------------------------------------------------------===//
//===----------------------------------------------------------------------===//
// Further described in docs/Rationale/RationaleTOSADialect.md .
//===----------------------------------------------------------------------===//
def Tosa_IfOp : Tosa_Op<"cond_if", [
      DeclareOpInterfaceMethods<InferShapedTypeOpInterface,
                                ["inferReturnTypeComponents"]>,
       SingleBlockImplicitTerminator<"YieldOp">,
       RecursiveMemoryEffects]> {
  let summary = "Conditional if operator";

  let description = [{
    Evaluates a Boolean condition and then takes one of two distinct execution
    paths. This implements the semantic If-then-else structure.
  }];

  let arguments = (ins
    I1Tensor:$cond,
    Variadic<Tosa_Tensor>:$inputs
  );

  let results = (outs
    Variadic<Tosa_Tensor>:$output
  );

  let regions = (region
    SizedRegion<1>:$then_branch,
    SizedRegion<1>:$else_branch
  );
}

//===----------------------------------------------------------------------===//
// Operator: while_loop
//===----------------------------------------------------------------------===//
//===----------------------------------------------------------------------===//
// Further described in docs/Rationale/RationaleTOSADialect.md .
//===----------------------------------------------------------------------===//
def Tosa_WhileOp : Tosa_Op<"while_loop", [
       DeclareOpInterfaceMethods<LoopLikeOpInterface>,
       DeclareOpInterfaceMethods<InferShapedTypeOpInterface,
                                 ["inferReturnTypeComponents"]>,
       SingleBlockImplicitTerminator<"YieldOp">,
       RecursiveMemoryEffects]> {
  let summary = "output = input; While (Cond(output)) {output = Body(output)}";

  let description = [{
    Generates and evaluates a Bool condition and either executes a loop body or
    exits to another control point. This action is performed repeatedly after
    updating and re-evaluating the Boolean condition every iteration. This
    implements the semantic foreach or while iterative loop structure.
  }];

  let arguments = (ins
    Variadic<Tosa_Tensor>:$inputs
  );

  let results = (outs
    Variadic<Tosa_Tensor>:$output
  );

  let regions = (region
    SizedRegion<1>:$cond,
    SizedRegion<1>:$body
  );
}

include "mlir/Dialect/Tosa/IR/TosaUtilOps.td"

#endif // TOSA_OPS
