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
// Operator Class: Tensor Data Engine Operators.
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Operator: argmax
//===----------------------------------------------------------------------===//
def Tosa_ArgMaxOp : Tosa_InferShapedTypeOp<"argmax"> {
  let summary = "Perform argmax on the input.";

  let description = [{
    This returns the index with the largest value across the given axis of the
    input tensor. If multiple locations have equal values, returns the first
    match along the search axis.
  }];

  let arguments = (ins
    Tosa_Tensor: $input,
    I32Attr: $axis,
    DefaultValuedAttr<Tosa_NanPropagationAttr, "\"PROPAGATE\"">:$nan_mode
  );

  let results = (outs
    Tosa_Tensor: $output
  );

  list<Availability> availability = [
    Profile<[Tosa_PRO_INT, Tosa_PRO_FP]>,
    Extension<[Tosa_EXT_INT16, Tosa_EXT_FP8E4M3, Tosa_EXT_FP8E5M2, Tosa_EXT_BF16]>,
  ];

  let hasFolder = 1;
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// Accumulator types.
//===----------------------------------------------------------------------===//

def Tosa_AccType : AnyTypeOf<[I<32>, I<48>, F16, F32]>;

//===----------------------------------------------------------------------===//
// Operator: avg_pool2d
//===----------------------------------------------------------------------===//
def Tosa_AvgPool2dOp : Tosa_InferShapedTypeOp<"avg_pool2d"> {
  let summary = "Performs average pooling on the input.";

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
    TypeAttrOf<Tosa_AccType>:$acc_type,
    OptionalAttr<I32Attr>:$input_zp,
    OptionalAttr<I32Attr>:$output_zp
  );

  let results = (outs
    Tosa_Tensor4D:$output
  );

  list<Availability> availability = [
    Profile<[Tosa_PRO_INT, Tosa_PRO_FP]>,
    Extension<[Tosa_EXT_INT16, Tosa_EXT_FP8E4M3, Tosa_EXT_FP8E5M2, Tosa_EXT_BF16]>,
  ];

  let builders = [Tosa_AvgPool2dOpQuantInfoBuilder];
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// Operator: conv2d
//===----------------------------------------------------------------------===//
def Tosa_Conv2DOp : Tosa_ConvOp<"conv2d"> {
  let summary = "2D Convolution Operator";

  let description = [{
    Performs a 2D convolution over the given tensor input, using the weight
    tensor. Implementations may choose to skip calculation of multiplies in
    the padding area.
  }];

  let arguments = (ins
    Tosa_Tensor4D:$input,
    TosaTensorRankOf<[Tosa_Weight], [4]>:$weight,
    Tosa_Tensor1D:$bias,
    Tosa_ScalarTensor:$input_zp,
    Tosa_ScalarTensor:$weight_zp,

    Tosa_IntArrayAttr4:$pad,
    Tosa_IntArrayAttr2:$stride,
    Tosa_IntArrayAttr2:$dilation,
    TypeAttrOf<Tosa_AccType>:$acc_type,
    DefaultValuedOptionalAttr<BoolAttr, "false">:$local_bound
  );

  let results = (outs
    Tosa_Tensor4D:$output
  );

  list<Availability> availability = [
    Profile<[Tosa_PRO_INT, Tosa_PRO_FP]>,
    Extension<[Tosa_EXT_INT4, Tosa_EXT_INT16, Tosa_EXT_FP8E4M3, Tosa_EXT_FP8E5M2, Tosa_EXT_BF16]>,
  ];

  let extraClassDeclaration = [{
    LogicalResult getInputZeroPoint(int64_t &zp);
    LogicalResult getWeightZeroPoint(int64_t &zp);
    LogicalResult verifyInputZeroPoint(int64_t zp);
    LogicalResult verifyWeightZeroPoint(int64_t zp);
  }];

  let builders = [Tosa_ConvOpQuantInfoBuilder];
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// Operator: conv3d
//===----------------------------------------------------------------------===//
def Tosa_Conv3DOp : Tosa_ConvOp<"conv3d"> {
  let summary = "3D Convolution operator";

  let description = [{
    Performs a 3D convolution over the given input tensor. Implementations
    may choose to skip calculation of multiplies in the padding area.
  }];

  let arguments = (ins
    Tosa_Tensor5D:$input,
    TosaTensorRankOf<[Tosa_Weight], [5]>:$weight,
    Tosa_Tensor1D:$bias,
    Tosa_ScalarTensor:$input_zp,
    Tosa_ScalarTensor:$weight_zp,

    Tosa_IntArrayAttr6:$pad,
    Tosa_IntArrayAttr3:$stride,
    Tosa_IntArrayAttr3:$dilation,
    TypeAttrOf<Tosa_AccType>:$acc_type,
    DefaultValuedOptionalAttr<BoolAttr, "false">:$local_bound
  );

  let results = (outs
    Tosa_Tensor5D:$output
  );

  list<Availability> availability = [
    Profile<[Tosa_PRO_INT, Tosa_PRO_FP]>,
    Extension<[Tosa_EXT_INT4, Tosa_EXT_INT16, Tosa_EXT_FP8E4M3, Tosa_EXT_FP8E5M2, Tosa_EXT_BF16]>,
  ];

  let extraClassDeclaration = [{
    LogicalResult getInputZeroPoint(int64_t &zp);
    LogicalResult getWeightZeroPoint(int64_t &zp);
    LogicalResult verifyInputZeroPoint(int64_t zp);
    LogicalResult verifyWeightZeroPoint(int64_t zp);
  }];

  let builders = [Tosa_ConvOpQuantInfoBuilder];
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// Operator: depthwise_conv2d
//===----------------------------------------------------------------------===//
def Tosa_DepthwiseConv2DOp : Tosa_ConvOp<"depthwise_conv2d"> {
  let summary = "Depthwise 2D Convolution operator";

  let description = [{
    Performs 2D convolutions separately over each channel of the given tensor
    input, using the weight tensor. Implementations may choose to skip
    calculation of multiplies in the padding area.
  }];

  let arguments = (ins
    Tosa_Tensor4D:$input,
    TosaTensorRankOf<[Tosa_Weight], [4]>:$weight,
    Tosa_Tensor1D:$bias,
    Tosa_ScalarTensor:$input_zp,
    Tosa_ScalarTensor:$weight_zp,

    Tosa_IntArrayAttr4:$pad,
    Tosa_IntArrayAttr2:$stride,
    Tosa_IntArrayAttr2:$dilation,
    TypeAttrOf<Tosa_AccType>:$acc_type,
    DefaultValuedOptionalAttr<BoolAttr, "false">:$local_bound
  );

  let results = (outs
    Tosa_Tensor4D:$output
  );

  list<Availability> availability = [
    Profile<[Tosa_PRO_INT, Tosa_PRO_FP]>,
    Extension<[Tosa_EXT_INT4, Tosa_EXT_INT16, Tosa_EXT_FP8E4M3, Tosa_EXT_FP8E5M2, Tosa_EXT_BF16]>,
  ];

  let extraClassDeclaration = [{
    LogicalResult getInputZeroPoint(int64_t &zp);
    LogicalResult getWeightZeroPoint(int64_t &zp);
    LogicalResult verifyInputZeroPoint(int64_t zp);
    LogicalResult verifyWeightZeroPoint(int64_t zp);
  }];

  let builders = [Tosa_ConvOpQuantInfoBuilder];
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// Operator: fft2d
//===----------------------------------------------------------------------===//
def Tosa_FFT2dOp : Tosa_InferShapedTypeOp<"fft2d"> {
  let summary = "Performs FFT2D operation on the input.";

  let description = [{
    Performs a batched complex 2D Fast Fourier Transform over the input. The
    complex input values are constructed from the corresponding values in the
    input_real and input_imag tensors. The resulting values in the output are
    split into the output_real and output_imag tensors. No normalization is
    applied on either the forward or inverse versions of the operation.

    Example:

    ```mlir
     %out_real, %out_imag = tosa.fft2d %in_real, %in_imag : (tensor<8x9xf32>, tensor<8x9xf32>) -> (tensor<8x9xf32>, tensor<8x9xf32>)
    ```
  }];

  let arguments = (ins
    Tosa_Tensor3D:$input_real,
    Tosa_Tensor3D:$input_imag,

    BoolAttr:$inverse,
    DefaultValuedOptionalAttr<BoolAttr, "false">:$local_bound
  );

  let results = (outs
    Tosa_Tensor3D:$output_real,
    Tosa_Tensor3D:$output_imag
  );

  list<Availability> availability = [
    Profile<[]>,
    Extension<[Tosa_EXT_FFT]>,
  ];

  let assemblyFormat = [{
    $input_real `,` $input_imag attr-dict `:` `(` type($input_real) `,`
    type($input_imag) `)` `->` `(` type($output_real) `,` type($output_imag) `)`
  }];
}

//===----------------------------------------------------------------------===//
// Operator: matmul
//===----------------------------------------------------------------------===//
def Tosa_MatMulOp : Tosa_InferShapedTypeOp<"matmul"> {
  let summary = "Matrix multiplication with bias";

  let description = [{
    Performs a two dimensional matrix multiplication. This allows both inputs to
    be activations, rather than reserving weights as an attribute in the
    FULLY_CONNECTED operator.
  }];

  let arguments = (ins
    Tosa_Tensor3D:$a,
    Tosa_Tensor3D:$b,
    OptionalAttr<I32Attr>:$a_zp,
    OptionalAttr<I32Attr>:$b_zp
  );

  let results = (outs
    Tosa_Tensor3D:$output
  );

  list<Availability> availability = [
    Profile<[Tosa_PRO_INT, Tosa_PRO_FP]>,
    Extension<[Tosa_EXT_INT16, Tosa_EXT_FP8E4M3, Tosa_EXT_FP8E5M2, Tosa_EXT_BF16]>,
  ];

  let builders = [Tosa_MatMulOpQuantInfoBuilder];
}

//===----------------------------------------------------------------------===//
// Operator: max_pool2d
//===----------------------------------------------------------------------===//
def Tosa_MaxPool2dOp : Tosa_InferShapedTypeOp<"max_pool2d"> {
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
    Tosa_IntArrayAttr4:$pad,
    DefaultValuedAttr<Tosa_NanPropagationAttr, "\"PROPAGATE\"">:$nan_mode
  );

  let results = (outs
    Tosa_Tensor4D:$output
  );

  list<Availability> availability = [
    Profile<[Tosa_PRO_INT, Tosa_PRO_FP]>,
    Extension<[Tosa_EXT_INT16, Tosa_EXT_FP8E4M3, Tosa_EXT_FP8E5M2, Tosa_EXT_BF16]>,
  ];

  let hasCanonicalizer = 1;
}

//===----------------------------------------------------------------------===//
// Operator: rfft2d
//===----------------------------------------------------------------------===//
def Tosa_RFFT2dOp : Tosa_InferShapedTypeOp<"rfft2d"> {
  let summary = "Performs RFFT2D operation on the input.";

  let description = [{
    Performs a batched 2D real-valued Fast Fourier Transform over the input where
    the input tensor consists of real values producing complex valued output. The
    complex output values will be split into the output_real and output_imag
    tensor arguments. RFFT2D takes advantage of Hermitian symmetry to only
    calculate the first half of the final output axis. Imaginary values with
    locations (0,0), (0,W/2), (H/2,0) and (H/2,W/2) are zero.

    Example:

    ```mlir
     %real, %imag = tosa.rfft2d %in : (tensor<8x16xf32>) -> (tensor<8x9xf32>, tensor<8x9xf32>)
    ```
  }];

  let arguments = (ins
    Tosa_Tensor3D:$input,
    DefaultValuedOptionalAttr<BoolAttr, "false">:$local_bound
  );

  let results = (outs
    Tosa_Tensor3D:$output_real,
    Tosa_Tensor3D:$output_imag
  );

  list<Availability> availability = [
    Profile<[]>,
    Extension<[Tosa_EXT_FFT]>,
  ];

  let assemblyFormat = [{
    $input attr-dict `:` `(` type($input) `)` `->` `(` type($output_real) `,` type($output_imag) `)`
  }];
}

//===----------------------------------------------------------------------===//
// Operator: transpose_conv2d
//===----------------------------------------------------------------------===//
def Tosa_TransposeConv2DOp : Tosa_ConvOp<"transpose_conv2d"> {
  let summary = "Transpose 2D Convolution operator.";

  let description = [{
    Performs a 2D transposed convolution over the given tensor input, using the
    weights tensor. Implementations may choose to skip calculation of multiplies
    by zero at fractional input positions.
  }];

  let arguments = (ins
    Tosa_Tensor4D:$input,
    TosaTensorRankOf<[Tosa_Weight], [4]>:$weight,
    Tosa_Tensor1D:$bias,
    Tosa_ScalarTensor:$input_zp,
    Tosa_ScalarTensor:$weight_zp,

    Tosa_IntArrayAttr4:$out_pad,
    Tosa_IntArrayAttr2:$stride,
    TypeAttrOf<Tosa_AccType>:$acc_type,
    DefaultValuedOptionalAttr<BoolAttr, "false">:$local_bound
  );

  let results = (outs
    Tosa_Tensor4D:$output
  );

  list<Availability> availability = [
    Profile<[Tosa_PRO_INT, Tosa_PRO_FP]>,
    Extension<[Tosa_EXT_INT4, Tosa_EXT_INT16, Tosa_EXT_FP8E4M3, Tosa_EXT_FP8E5M2, Tosa_EXT_BF16]>,
  ];

  let extraClassDeclaration = [{
    LogicalResult getInputZeroPoint(int64_t &zp);
    LogicalResult getWeightZeroPoint(int64_t &zp);
    LogicalResult verifyInputZeroPoint(int64_t zp);
    LogicalResult verifyWeightZeroPoint(int64_t zp);
  }];

  let builders = [Tosa_TransConvOpQuantInfoBuilder];
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// Operator Class: Activation Functions.
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Operator: clamp
//===----------------------------------------------------------------------===//
def Tosa_ClampOp : Tosa_ElementwiseUnaryOp<"clamp"> {
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
    Tosa_IntOrFloatAttr:$min_val,
    Tosa_IntOrFloatAttr:$max_val,
    DefaultValuedAttr<Tosa_NanPropagationAttr, "\"PROPAGATE\"">:$nan_mode
  );

  let results = (outs
    Tosa_Tensor:$output
  );

  list<Availability> availability = [
    Profile<[Tosa_PRO_INT, Tosa_PRO_FP]>,
    Extension<[Tosa_EXT_INT16, Tosa_EXT_BF16]>,
  ];

  let hasCanonicalizer = 1;
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// Operator: sigmoid
//===----------------------------------------------------------------------===//
def Tosa_SigmoidOp : Tosa_ElementwiseUnaryOp<"sigmoid"> {
  let summary = "Computes elementwise sigmoid of input.";

  let description = [{
    Applies the sigmoid logistic function to each element of the input tensor:
    $ sigmoid(x) = \frac{1}{1 + e^{-x}} $.

    For quantized integer data types, the TABLE operator should be used instead.
    Each implementation may choose an appropriate TABLE given the scale and zero
    point of the input data. Eight or sixteen bit precision tables may be used
    based on the input tensor to the sigmoid function. The sigmoid table has 513
    entries each of 16-bit precision and covering the input range -16.0 to +16.0
    in steps of 1/16.
  }];

  let arguments = (ins
    Tosa_Tensor:$input
  );

  let results = (outs
    Tosa_Tensor:$output
  );

  list<Availability> availability = [
    Profile<[Tosa_PRO_FP]>,
    Extension<[Tosa_EXT_BF16]>,
  ];
}

//===----------------------------------------------------------------------===//
// Operator: tanh
//===----------------------------------------------------------------------===//
def Tosa_TanhOp : Tosa_ElementwiseUnaryOp<"tanh"> {
  let summary = "Computes elementwise hyperbolic tangent of input";

  let description = [{
    Parameterized hyperbolic tangent: $ tanh(x) = \frac{1 - e^{-2x}}{1 + e^{-2x}} $.

    For quantized integer data types, the TABLE operator should be used instead.
    Each implementation may choose an appropriate TABLE given the scale and zero
    point of the input data. Eight or sixteen bit precision tables may be used
    based on the input tensor to the tanh function. The tanh_table has 513
    entries each of 16-bit precision and covering the input range -8.0 to +8.0
    in steps of 1/32.
  }];

  let arguments = (ins
    Tosa_Tensor:$input
  );

  let results = (outs
    Tosa_Tensor:$output
  );

  list<Availability> availability = [
    Profile<[Tosa_PRO_FP]>,
    Extension<[Tosa_EXT_BF16]>,
  ];
}

//===----------------------------------------------------------------------===//
// Operator: erf
//===----------------------------------------------------------------------===//
def Tosa_ErfOp : Tosa_ElementwiseUnaryOp<"erf"> {
  let summary = "Computes gauss error function of input";

  let description = [{
    Gauss error function: $ erf(x) = \frac{2}{\sqrt{\pi}} \int_{0}^{x} e^{-t^2} dt $
    For quantized integer data types, the TABLE operator should be used instead
    with the following definition. The ERF table has 513 entries each of
    16-bit precision and covering the input range -4.0 to +4.0 in steps of 1/64.
  }];

  let arguments = (ins
    Tosa_Tensor:$input
  );

  let results = (outs
    Tosa_Tensor:$output
  );

  list<Availability> availability = [
    Profile<[Tosa_PRO_FP]>,
    Extension<[Tosa_EXT_BF16]>,
  ];

  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Operator Class: Elementwise unary/binary/ternary operators.
// Operator Subclass: Elementwise binary ops.
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Operator: add
//===----------------------------------------------------------------------===//
def Tosa_AddOp : Tosa_ElementwiseOp<"add", [
    Commutative,
    SameOperandsAndResultElementType]> {
  let summary = "Elementwise addition operator";

  let description = [{
    Elementwise addition of input1 and input2. Axis of size 1 will be broadcast,
    as necessary. Rank of input tensors must match.

    Example:

    ```mlir
    // Elementwise addition.
    %out = tosa.add %in1, %in2 : tensor<12x6xf32>, tensor<12x6xf32> -> tensor<12x6xf32>

    // Elementwise addition with broadcasting.
    %out = tosa.add %in1, %in2 : tensor<12x6xsi32>, tensor<1x1xsi32> -> tensor<12x6xsi32>
    ```
  }];

  let arguments = (ins
    Tosa_Tensor:$input1,
    Tosa_Tensor:$input2
  );

  let results = (outs
    Tosa_Tensor:$output
  );

  list<Availability> availability = [
    Profile<[Tosa_PRO_INT, Tosa_PRO_FP]>,
    Extension<[Tosa_EXT_BF16]>,
  ];

  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// Operator: arithmetic_right_shift
//===----------------------------------------------------------------------===//
def Tosa_ArithmeticRightShiftOp : Tosa_ElementwiseOp<"arithmetic_right_shift",
    [SameOperandsAndResultElementType]> {
  let summary = "Elementwise Arithmetic Right Shift";

  let description = [{
    Elementwise arithmetic right shift of input1 by the amount specified in
    input2. Axis of size 1 will be broadcast, as necessary. Rank of input tensors
    must match.
  }];

  let arguments = (ins
    Tosa_Tensor:$input1,
    Tosa_Tensor:$input2,
    BoolAttr:$round
  );

  let results = (outs
    Tosa_Tensor:$output
  );

  list<Availability> availability = [
    Profile<[Tosa_PRO_INT]>,
    Extension<[]>,
  ];
}

//===----------------------------------------------------------------------===//
// Operator: bitwise_and
//===----------------------------------------------------------------------===//
def Tosa_BitwiseAndOp : Tosa_ElementwiseOp<"bitwise_and", [
    Commutative,
    SameOperandsAndResultElementType]> {
  let summary = "Bitwise AND operator";

  let description = [{
    Elementwise bitwise AND of input1 and input2. Axis of size 1
    will be broadcast as necessary. Rank of input tensors must match.
  }];

  let arguments = (ins
    Tosa_Tensor:$input1,
    Tosa_Tensor:$input2
  );

  let results = (outs
    Tosa_Tensor:$output
  );

  list<Availability> availability = [
    Profile<[Tosa_PRO_INT]>,
    Extension<[]>,
  ];
}

//===----------------------------------------------------------------------===//
// Operator: bitwise_or
//===----------------------------------------------------------------------===//
def Tosa_BitwiseOrOp : Tosa_ElementwiseOp<"bitwise_or", [
    Commutative,
    SameOperandsAndResultElementType]> {
  let summary = "Bitwise OR operator";

  let description = [{
    Elementwise bitwise OR of input1 and input2. Axis of size 1 will be
    broadcast as necessary. Rank of input tensors must match.
  }];

  let arguments = (ins
    Tosa_Tensor:$input1,
    Tosa_Tensor:$input2
  );

  let results = (outs
    Tosa_Tensor:$output
  );

  list<Availability> availability = [
    Profile<[Tosa_PRO_INT]>,
    Extension<[]>,
  ];
}

//===----------------------------------------------------------------------===//
// Operator: bitwise_xor
//===----------------------------------------------------------------------===//
def Tosa_BitwiseXorOp : Tosa_ElementwiseOp<"bitwise_xor", [
    Commutative,
    SameOperandsAndResultElementType]> {
  let summary = "Bitwise XOR operator";

  let description = [{
    Elementwise bitwise XOR of input1 and input2. Axis of size 1 will be
    broadcast as necessary. Rank of input tensors must match.
  }];

  let arguments = (ins
    Tosa_Tensor:$input1,
    Tosa_Tensor:$input2
  );

  let results = (outs
    Tosa_Tensor:$output
  );

  list<Availability> availability = [
    Profile<[Tosa_PRO_INT]>,
    Extension<[]>,
  ];
}

//===----------------------------------------------------------------------===//
// Operator: int_div
//===----------------------------------------------------------------------===//
def Tosa_IntDivOp : Tosa_ElementwiseOp<"int_div", [SameOperandsAndResultElementType]> {
  let summary = "Integer divide operator";

  let description = [{
    Elementwise integer divide operator of input1 by input2. The result of the divide
    is truncated towards zero. Expected use is for operations on non-scaled integers.
    Floating point divide should use RECIPROCAL and MUL. Quantized integer divide
    should use TABLE (for 1/x) and MUL.
  }];

  let arguments = (ins
    Tosa_Int32Tensor:$input1,
    Tosa_Int32Tensor:$input2
  );

  let results = (outs
    Tosa_Int32Tensor:$output
  );

  list<Availability> availability = [
    Profile<[Tosa_PRO_INT, Tosa_PRO_FP]>,
    Extension<[]>,
  ];

  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// Operator: logical_and
//===----------------------------------------------------------------------===//
def Tosa_LogicalAndOp : Tosa_ElementwiseOp<"logical_and", [
    Commutative,
    SameOperandsAndResultElementType]> {
  let summary = "Returns the truth value of x AND y element-wise.";

  let description = [{
    Elementwise logical AND of input1 and input2. Axis of size 1 will be
    broadcast, as necessary. Rank of input tensors must match.
  }];

  let arguments = (ins
    Tosa_I1Tensor:$input1,
    Tosa_I1Tensor:$input2
  );

  let results = (outs
    Tosa_I1Tensor:$z
  );

  list<Availability> availability = [
    Profile<[Tosa_PRO_INT]>,
    Extension<[]>,
  ];
}

//===----------------------------------------------------------------------===//
// Operator: logical_left_shift
//===----------------------------------------------------------------------===//
def Tosa_LogicalLeftShiftOp : Tosa_ElementwiseOp<"logical_left_shift",
    [SameOperandsAndResultElementType]> {
  let summary = "Elementwise Logical Left Shift";

  let description = [{
    Elementwise left shift of input1 and input2. Axis of size 1 will be
    broadcast, as necessary. Rank of input tensors must match.
  }];

  let arguments = (ins
    Tosa_Tensor:$input1,
    Tosa_Tensor:$input2
  );

  let results = (outs
    Tosa_Tensor:$output
  );

  list<Availability> availability = [
    Profile<[Tosa_PRO_INT]>,
    Extension<[]>,
  ];
}

//===----------------------------------------------------------------------===//
// Operator: logical_right_shift
//===----------------------------------------------------------------------===//
def Tosa_LogicalRightShiftOp : Tosa_ElementwiseOp<"logical_right_shift",
    [SameOperandsAndResultElementType]> {
  let summary = "Elementwise Logical Right Shift";

  let description = [{
    Elementwise logical right shift of input1 by the amount specified in input2.
    Axis of size 1 will be broadcast, as necessary. Rank of input tensors must
    match.
  }];

  let arguments = (ins
    Tosa_Tensor:$input1,
    Tosa_Tensor:$input2
  );

  let results = (outs
    Tosa_Tensor:$output
  );

  list<Availability> availability = [
    Profile<[Tosa_PRO_INT]>,
    Extension<[]>,
  ];
}

//===----------------------------------------------------------------------===//
// Operator: logical_or
//===----------------------------------------------------------------------===//
def Tosa_LogicalOrOp : Tosa_ElementwiseOp<"logical_or", [
    Commutative,
    SameOperandsAndResultElementType]> {
  let summary = "Returns the truth value of x OR y element-wise.";

  let description = [{
    Elementwise logical OR of input1 and input2. Axis of size 1 will be
    broadcast as necessary. Rank of input tensors must match.
  }];

  let arguments = (ins
    Tosa_I1Tensor:$input1,
    Tosa_I1Tensor:$input2
  );

  let results = (outs
    Tosa_I1Tensor:$z
  );

  list<Availability> availability = [
    Profile<[Tosa_PRO_INT]>,
    Extension<[]>,
  ];
}

//===----------------------------------------------------------------------===//
// Operator: logical_xor
//===----------------------------------------------------------------------===//
def Tosa_LogicalXorOp : Tosa_ElementwiseOp<"logical_xor", [
    Commutative,
    SameOperandsAndResultElementType]> {
  let summary = "Returns the truth value of x XOR y element-wise.";

  let description = [{
    Elementwise logical XOR of input1 and input2. Axis of size 1 will be
    broadcast as necessary. Rank of input tensors must match.
  }];

  let arguments = (ins
    Tosa_I1Tensor:$input1,
    Tosa_I1Tensor:$input2
  );

  let results = (outs
    Tosa_I1Tensor:$z
  );

  list<Availability> availability = [
    Profile<[Tosa_PRO_INT]>,
    Extension<[]>,
  ];
}

//===----------------------------------------------------------------------===//
// Operator: maximum
//===----------------------------------------------------------------------===//
def Tosa_MaximumOp : Tosa_ElementwiseOp<"maximum", [
    Commutative,
    SameOperandsAndResultElementType]> {
  let summary = "Elementwise Maximum";

  let description = [{
    Elementwise max of input1 and input2. Axis of size 1 will be broadcast, as
    necessary. Rank of input tensors must match.
  }];

  let arguments = (ins
    Tosa_Tensor:$input1,
    Tosa_Tensor:$input2,
    DefaultValuedAttr<Tosa_NanPropagationAttr, "\"PROPAGATE\"">:$nan_mode
  );

  let results = (outs
    Tosa_Tensor:$output
  );

  list<Availability> availability = [
    Profile<[Tosa_PRO_INT, Tosa_PRO_FP]>,
    Extension<[Tosa_EXT_BF16]>,
  ];
}

//===----------------------------------------------------------------------===//
// Operator: minimum
//===----------------------------------------------------------------------===//
def Tosa_MinimumOp : Tosa_ElementwiseOp<"minimum", [
    Commutative,
    SameOperandsAndResultElementType]> {
  let summary = "Elementwise Minimum";

  let description = [{
    Elementwise minimum of input1 and input2. Axis of size 1
    will be broadcast, as necessary. Rank of input tensors must match.
  }];

  let arguments = (ins
    Tosa_Tensor:$input1,
    Tosa_Tensor:$input2,
    DefaultValuedAttr<Tosa_NanPropagationAttr, "\"PROPAGATE\"">:$nan_mode
  );

  let results = (outs
    Tosa_Tensor:$output
  );

  list<Availability> availability = [
    Profile<[Tosa_PRO_INT, Tosa_PRO_FP]>,
    Extension<[Tosa_EXT_BF16]>,
  ];
}

def MulOperandsAndResultElementType :
  NativeOpTrait<"MulOperandsAndResultElementType"> {
  let cppNamespace = "mlir::OpTrait::tosa";
}

//===----------------------------------------------------------------------===//
// Operator: mul
//===----------------------------------------------------------------------===//
def Tosa_MulOp : Tosa_Op<"mul", [
    DeclareOpInterfaceMethods<InferShapedTypeOpInterface,
                              ["inferReturnTypeComponents"]>,
    Commutative,
    Pure]> {
  let summary = "Multiplication operator";

  let description = [{
    Elementwise multiplication (Hadamard product) of input1 and input2.
    Axis of size 1 will be broadcast, as necessary. Rank of input tensors must
    match.
  }];

  let arguments = (ins
    Tosa_Tensor:$input1,
    Tosa_Tensor:$input2,
    // Apply right shift on i32_t input data only
    Tosa_ScalarInt8Tensor:$shift
  );

  let results = (outs
    Tosa_Tensor:$output
  );

  list<Availability> availability = [
    Profile<[Tosa_PRO_INT, Tosa_PRO_FP]>,
    Extension<[Tosa_EXT_BF16]>,
  ];

  let hasFolder = 1;
  let hasVerifier = 1;

  let assemblyFormat =
      "operands attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Operator: pow
//===----------------------------------------------------------------------===//
def Tosa_PowOp : Tosa_ElementwiseOp<"pow", [SameOperandsAndResultElementType]> {
  let summary = "Computes the power of one value to another.";

  let description = [{
    Elementwise input1 raised to the power of input2.
    Axis of size 1 will be broadcast, as necessary. Rank of input tensors must
    match.
  }];

  let arguments = (ins
    Tosa_Tensor:$input1,
    Tosa_Tensor:$input2
  );

  let results = (outs
    Tosa_Tensor:$output
  );

  list<Availability> availability = [
    Profile<[Tosa_PRO_FP]>,
    Extension<[Tosa_EXT_BF16]>,
  ];
}

//===----------------------------------------------------------------------===//
// Operator: sub
//===----------------------------------------------------------------------===//
def Tosa_SubOp : Tosa_ElementwiseOp<"sub", [SameOperandsAndResultElementType]> {
  let summary = "Elementwise subtraction operator";

  let description = [{
    Elementwise subtraction of input1 and input2. Axis of size 1 will be
    broadcast as necessary. Rank of input tensors must match.
  }];

  let arguments = (ins
    Tosa_Tensor:$input1,
    Tosa_Tensor:$input2
  );

  let results = (outs
    Tosa_Tensor:$output
  );

  list<Availability> availability = [
    Profile<[Tosa_PRO_INT, Tosa_PRO_FP]>,
    Extension<[Tosa_EXT_BF16]>,
  ];

  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// Operator: table
//===----------------------------------------------------------------------===//
def Tosa_TableOp : Tosa_InferShapedTypeOp<"table"> {
  let summary = "Table lookup op";

  let description = [{
    Table lookup operation. For int8_t TABLE operation, perform a 256 entry
    table lookup returning an int8_t value. For int16_t tables, the int16_t
    input is treated as a fixed-point 9.7 value. The most significant 9 bits
    are used to index into the table. The fractional 7 bits are used to
    interpolate based on table[index] and table[index+1]. For int16_t inputs,
    the TABLE operator returns a 16.7 interpolated value in an int32_t. This
    value can then be input to the RESCALE operator to scale to the required
    output data type. Note that int16_t table has 513 values to handle
    table[index+1] when index=511.

    An int16_t to int16_t table lookup can be constructed in TOSA as follows:
    * Use the TABLE operator to produce a fixed point 16.7 interpolated result
    * Use RESCALE (in_t=int32_t, out_t=int16_t, scale=1<<14, shift=21) to
      scale the output to int16_t range (or alternate scale as required)
  }];

  let arguments = (ins
    Tosa_Tensor: $input1,
    Tosa_Tensor1D: $table
  );

  let results = (outs
    Tosa_Tensor:$output
  );

  list<Availability> availability = [
    Profile<[Tosa_PRO_INT]>,
    Extension<[Tosa_EXT_BF16]>,
  ];

  let assemblyFormat = [{
    $input1 `,` $table attr-dict `:` `(` type($input1) `,` type($table) `)` `->` type($output)
  }];

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// Operator Class: Elementwise unary/binary/ternary operators.
// Operator Subclass: Elementwise unary ops.
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Operator: abs
//===----------------------------------------------------------------------===//
def Tosa_AbsOp : Tosa_ElementwiseUnaryOp<"abs"> {
  let summary = "Elementwise abs op";

  let description = [{
    Elementwise absolute value operation.

    Example:

    ```mlir
    %out = tosa.abs(%in) : (tensor<21x3xf32>) -> tensor<21x3xf32>
    ```
  }];

  let arguments = (ins
    Tosa_Tensor:$input1
  );

  let results = (outs
    Tosa_Tensor:$output
  );

  list<Availability> availability = [
    Profile<[Tosa_PRO_INT, Tosa_PRO_FP]>,
    Extension<[Tosa_EXT_BF16]>,
  ];

  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// Operator: bitwise_not
//===----------------------------------------------------------------------===//
def Tosa_BitwiseNotOp : Tosa_ElementwiseUnaryOp<"bitwise_not"> {
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

  list<Availability> availability = [
    Profile<[Tosa_PRO_INT]>,
    Extension<[]>,
  ];
}

//===----------------------------------------------------------------------===//
// Operator: ceil
//===----------------------------------------------------------------------===//
def Tosa_CeilOp : Tosa_ElementwiseUnaryOp<"ceil"> {
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

  list<Availability> availability = [
    Profile<[Tosa_PRO_FP]>,
    Extension<[Tosa_EXT_BF16]>,
  ];
}

//===----------------------------------------------------------------------===//
// Operator: clz
//===----------------------------------------------------------------------===//
def Tosa_ClzOp : Tosa_ElementwiseUnaryOp<"clz"> {
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

  list<Availability> availability = [
    Profile<[Tosa_PRO_INT]>,
    Extension<[]>,
  ];
}

//===----------------------------------------------------------------------===//
// Operator: cos
//===----------------------------------------------------------------------===//
def Tosa_CosOp : Tosa_ElementwiseUnaryOp<"cos"> {
  let summary = "Elementwise cos op";

  let description = [{
    Elementwise cosine operation for values given in radians.
  }];

  let arguments = (ins
    Tosa_FloatTensor:$input1
  );

  let results = (outs
    Tosa_FloatTensor:$output
  );

  list<Availability> availability = [
    Profile<[Tosa_PRO_FP]>,
    Extension<[Tosa_EXT_BF16]>,
  ];
}

//===----------------------------------------------------------------------===//
// Operator: exp
//===----------------------------------------------------------------------===//
def Tosa_ExpOp : Tosa_ElementwiseUnaryOp<"exp"> {
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

  list<Availability> availability = [
    Profile<[Tosa_PRO_FP]>,
    Extension<[Tosa_EXT_BF16]>,
  ];

  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// Operator: floor
//===----------------------------------------------------------------------===//
def Tosa_FloorOp : Tosa_ElementwiseUnaryOp<"floor"> {
  let summary = "Elementwise floor op";

  let description = [{
    Elementwise floor operation.
  }];

  let arguments = (ins
    Tosa_Tensor:$input1
  );

  let results = (outs
    Tosa_Tensor:$output
  );

  list<Availability> availability = [
    Profile<[Tosa_PRO_FP]>,
    Extension<[Tosa_EXT_BF16]>,
  ];
}

//===----------------------------------------------------------------------===//
// Operator: log
//===----------------------------------------------------------------------===//
def Tosa_LogOp : Tosa_ElementwiseUnaryOp<"log"> {
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

  list<Availability> availability = [
    Profile<[Tosa_PRO_FP]>,
    Extension<[Tosa_EXT_BF16]>,
  ];

  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// Operator: logical_not
//===----------------------------------------------------------------------===//
def Tosa_LogicalNotOp : Tosa_ElementwiseUnaryOp<"logical_not"> {
  let summary = "Returns the truth value of NOT x element-wise.";

  let description = [{
    Elementwise logical NOT of input.
  }];

  let arguments = (ins
    Tosa_I1Tensor:$input1
  );

  let results = (outs
    Tosa_I1Tensor:$output
  );

  list<Availability> availability = [
    Profile<[Tosa_PRO_INT, Tosa_PRO_FP]>,
    Extension<[]>,
  ];
}

//===----------------------------------------------------------------------===//
// Operator: negate
//===----------------------------------------------------------------------===//
def Tosa_NegateOp : Tosa_ElementwiseUnaryOp<"negate"> {
  let summary = "Elementwise negate op";

  let description = [{
    Elementwise negation operation.
  }];

  let arguments = (ins
      Tosa_Tensor:$input1,
      OptionalAttr<I32Attr>:$input1_zp,
      OptionalAttr<I32Attr>:$output_zp
  );

  let results = (outs
    Tosa_Tensor:$output
  );

  list<Availability> availability = [
    Profile<[Tosa_PRO_INT, Tosa_PRO_FP]>,
    Extension<[Tosa_EXT_BF16]>,
  ];

  let builders = [Tosa_UnaryOpQuantInfoBuilder];

  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// Operator: reciprocal
//===----------------------------------------------------------------------===//
def Tosa_ReciprocalOp : Tosa_ElementwiseUnaryOp<"reciprocal"> {
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

  list<Availability> availability = [
    Profile<[Tosa_PRO_FP]>,
    Extension<[Tosa_EXT_BF16]>,
  ];

  let extraClassDeclaration = [{
    /// Return the reciprocal result on the operand.
    static inline APFloat calcOneElement(const APFloat &operand) {
      APFloat recip = APFloat(operand.getSemantics(), 1);
      recip.divide(operand, APFloat::rmNearestTiesToEven);
      return recip;
    }
  }];

  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// Operator: rsqrt
//===----------------------------------------------------------------------===//
def Tosa_RsqrtOp : Tosa_ElementwiseUnaryOp<"rsqrt"> {
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

  list<Availability> availability = [
    Profile<[Tosa_PRO_FP]>,
    Extension<[Tosa_EXT_BF16]>,
  ];
}

//===----------------------------------------------------------------------===//
// Operator: sin
//===----------------------------------------------------------------------===//
def Tosa_SinOp : Tosa_ElementwiseUnaryOp<"sin"> {
  let summary = "Elementwise sin op";

  let description = [{
    Elementwise sine operation for values given in radians.
  }];

  let arguments = (ins
    Tosa_FloatTensor:$input1
  );

  let results = (outs
    Tosa_FloatTensor:$output
  );

  list<Availability> availability = [
    Profile<[Tosa_PRO_FP]>,
    Extension<[Tosa_EXT_BF16]>,
  ];
}

//===----------------------------------------------------------------------===//
// Operator Class: Elementwise unary/binary/ternary operators.
// Operator Subclass: Elementwise ternary ops.
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Operator: select
//===----------------------------------------------------------------------===//
def Tosa_SelectOp : Tosa_ElementwiseOp<"select"> {
  let summary = "Elementwise select operator";

  let description = [{
    Elementwise select of the output based on a condition.
  }];

  let arguments = (ins
    Tosa_I1Tensor:$input1,
    Tosa_Tensor:$input2,
    Tosa_Tensor:$input3
  );

  let results = (outs
    Tosa_Tensor:$output
  );

  list<Availability> availability = [
    Profile<[Tosa_PRO_INT, Tosa_PRO_FP]>,
    Extension<[Tosa_EXT_BF16]>,
  ];

  let hasCanonicalizeMethod = 1;
  let hasFolder = 1;

  let assemblyFormat = [{
    operands attr-dict `:` `(` type($input1) `,` type($input2) `,` type($input3)
    `)` `->` type($output)
  }];
}

//===----------------------------------------------------------------------===//
// Operator Class: Logical Operations.
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Operator: equal
//===----------------------------------------------------------------------===//
def Tosa_EqualOp : Tosa_ElementwiseOp<"equal", [
    InferTensorType,
    Commutative,
    SameOperandsElementType]> {
  let summary = "Returns the truth value of (x == y) element-wise.";

  let description = [{
     Elementwise comparison operation
  }];

  let arguments = (ins
    Tosa_Tensor:$input1,
    Tosa_Tensor:$input2
  );

  let results = (outs
    Tosa_I1Tensor:$output
  );

  list<Availability> availability = [
    Profile<[Tosa_PRO_INT, Tosa_PRO_FP]>,
    Extension<[Tosa_EXT_BF16]>,
  ];

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
def Tosa_GreaterOp : Tosa_ElementwiseOp<"greater", [SameOperandsElementType]> {
  let summary = "Returns the truth value of (x > y) element-wise.";

  let description = [{
    Elementwise greater than comparison operation
  }];

  let arguments = (ins
    Tosa_Tensor:$input1,
    Tosa_Tensor:$input2
  );

  let results = (outs
    Tosa_I1Tensor:$output
  );

  list<Availability> availability = [
    Profile<[Tosa_PRO_INT, Tosa_PRO_FP]>,
    Extension<[Tosa_EXT_BF16]>,
  ];

  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// Operator: greater_equal
//===----------------------------------------------------------------------===//
def Tosa_GreaterEqualOp : Tosa_ElementwiseOp<"greater_equal",
    [SameOperandsElementType]> {
  let summary = "Returns the truth value of (x >= y) element-wise.";

  let description = [{
    Elementwise comparison operation
  }];

  let arguments = (ins
    Tosa_Tensor:$input1,
    Tosa_Tensor:$input2
  );

  let results = (outs
    Tosa_I1Tensor:$output
  );

  list<Availability> availability = [
    Profile<[Tosa_PRO_INT, Tosa_PRO_FP]>,
    Extension<[Tosa_EXT_BF16]>,
  ];

  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// Operator Class: Reduction Ops.
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Operator: reduce_all
//===----------------------------------------------------------------------===//
def Tosa_ReduceAllOp : Tosa_InferTensorTypeOp<"reduce_all"> {
  let summary = "Reduce All operator";

  let description = [{
    Reduce a tensor along the given axis with a logical AND operation
  }];

  let arguments = (ins
    Tosa_Tensor:$input,
    I32Attr:$axis
  );

  let results = (outs
    Tosa_Tensor:$output
  );

  list<Availability> availability = [
    Profile<[Tosa_PRO_INT, Tosa_PRO_FP]>,
    Extension<[]>,
  ];

  let hasFolder = 1;
  let hasVerifier = 1;

  let extraClassDeclaration = [{
    /// Returns true when two result types are compatible for this op;
    /// Method used by InferTypeOpInterface.
    static bool isCompatibleReturnTypes(TypeRange l, TypeRange r);

    /// Return the AND result between two integer operands
    static inline APInt calcOneElement(APInt leftOperand, APInt rightOperand) {
      return leftOperand & rightOperand;
    }
  }];
}

//===----------------------------------------------------------------------===//
// Operator: reduce_any
//===----------------------------------------------------------------------===//
def Tosa_ReduceAnyOp : Tosa_InferTensorTypeOp<"reduce_any"> {
  let summary = "Reduce Any operator";

  let description = [{
    Reduce a tensor along the given axis with a logical OR operation
  }];

  let arguments = (ins
    Tosa_Tensor:$input,
    I32Attr:$axis
  );

  let results = (outs
    Tosa_Tensor:$output
  );

  list<Availability> availability = [
    Profile<[Tosa_PRO_INT, Tosa_PRO_FP]>,
    Extension<[]>,
  ];

  let hasFolder = 1;
  let hasVerifier = 1;

  let extraClassDeclaration = [{
    /// Returns true when two result types are compatible for this op;
    /// Method used by InferTypeOpInterface.
    static bool isCompatibleReturnTypes(TypeRange l, TypeRange r);

    /// Return the OR result between two integer operands
    static inline APInt calcOneElement(APInt leftOperand, APInt rightOperand) {
      return leftOperand | rightOperand;
    }
  }];
}

//===----------------------------------------------------------------------===//
// Operator: reduce_max
//===----------------------------------------------------------------------===//
def Tosa_ReduceMaxOp : Tosa_InferTensorTypeOp<"reduce_max"> {
  let summary = "Reduce Max operator";

  let description = [{
    Reduce a tensor along the given axis with a maximum operation
  }];

  let arguments = (ins
    Tosa_Tensor:$input,
    I32Attr:$axis,
    DefaultValuedAttr<Tosa_NanPropagationAttr, "\"PROPAGATE\"">:$nan_mode
  );

  let results = (outs
    Tosa_Tensor:$output
  );

  list<Availability> availability = [
    Profile<[Tosa_PRO_INT, Tosa_PRO_FP]>,
    Extension<[Tosa_EXT_BF16]>,
  ];

  let hasFolder = 1;
  let hasVerifier = 1;

  let extraClassDeclaration = [{
    /// Returns true when two result types are compatible for this op;
    /// Method used by InferTypeOpInterface.
    static bool isCompatibleReturnTypes(TypeRange l, TypeRange r);

    /// Return the max of the two integer operands
    static inline APInt calcOneElement(APInt leftOperand, APInt rightOperand) {
      const llvm::APInt subtractRes = leftOperand - rightOperand;
      return (!subtractRes.isNegative()) ? leftOperand : rightOperand;
    }
  }];
}

//===----------------------------------------------------------------------===//
// Operator: reduce_min
//===----------------------------------------------------------------------===//
def Tosa_ReduceMinOp : Tosa_InferTensorTypeOp<"reduce_min"> {
  let summary = "Reduce Min operator";

  let description = [{
    Reduce a tensor along the given axis with a minimum operation
  }];

  let arguments = (ins
    Tosa_Tensor:$input,
    I32Attr:$axis,
    DefaultValuedAttr<Tosa_NanPropagationAttr, "\"PROPAGATE\"">:$nan_mode
  );

  let results = (outs
    Tosa_Tensor:$output
  );

  list<Availability> availability = [
    Profile<[Tosa_PRO_INT, Tosa_PRO_FP]>,
    Extension<[Tosa_EXT_BF16]>,
  ];

  let hasFolder = 1;
  let hasVerifier = 1;

  let extraClassDeclaration = [{
    /// Returns true when two result types are compatible for this op;
    /// Method used by InferTypeOpInterface.
    static bool isCompatibleReturnTypes(TypeRange l, TypeRange r);

    /// Return the min of the two integer operands
    static inline APInt calcOneElement(APInt leftOperand, APInt rightOperand) {
      const llvm::APInt subtractRes = leftOperand - rightOperand;
      return (!subtractRes.isNegative()) ? rightOperand : leftOperand;
    }
  }];
}

//===----------------------------------------------------------------------===//
// Operator: reduce_prod
//===----------------------------------------------------------------------===//
def Tosa_ReduceProductOp : Tosa_InferTensorTypeOp<"reduce_product"> {
  let summary = "Reduce Product operator";

  let description = [{
    Reduce a tensor along the given axis by computing the product of the axis.
  }];

  let arguments = (ins
    Tosa_Tensor:$input,
    I32Attr:$axis
  );

  let results = (outs
    Tosa_Tensor:$output
  );

  list<Availability> availability = [
    Profile<[Tosa_PRO_FP]>,
    Extension<[Tosa_EXT_BF16]>,
  ];

  let hasFolder = 1;
  let hasVerifier = 1;

  let extraClassDeclaration = [{
    /// Returns true when two result types are compatible for this op;
    /// Method used by InferTypeOpInterface.
    static bool isCompatibleReturnTypes(TypeRange l, TypeRange r);

    /// Return the prod of the two integer operands
    static inline APInt calcOneElement(APInt leftOperand, APInt rightOperand) {
      return leftOperand * rightOperand;
    }
  }];
}

//===----------------------------------------------------------------------===//
// Operator: reduce_sum
//===----------------------------------------------------------------------===//
def Tosa_ReduceSumOp : Tosa_InferTensorTypeOp<"reduce_sum"> {
  let summary = "Reduce Sum operator";

  let description = [{
    Reduce a tensor along the given axis by computing the sum of the axis.
  }];

  let arguments = (ins
    Tosa_Tensor:$input,
    I32Attr:$axis
  );

  let results = (outs
    Tosa_Tensor:$output
  );

  list<Availability> availability = [
    Profile<[Tosa_PRO_INT, Tosa_PRO_FP]>,
    Extension<[Tosa_EXT_BF16]>,
  ];

  let hasFolder = 1;
  let hasVerifier = 1;

  let extraClassDeclaration = [{
    /// Returns true when two result types are compatible for this op;
    /// Method used by InferTypeOpInterface.
    static bool isCompatibleReturnTypes(TypeRange l, TypeRange r);

    /// Return the sum of the two integer operands
    static inline APInt calcOneElement(APInt leftOperand, APInt rightOperand) {
      return leftOperand + rightOperand;
    }
  }];
}

//===----------------------------------------------------------------------===//
// Operator Class: Data Layout / Memory Reinterpretation.
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Operator: concat
//===----------------------------------------------------------------------===//
def Tosa_ConcatOp : Tosa_InferTensorTypeOp<"concat"> {
  let summary = "Concatenates tensors along one dimension.";

  let description = [{
    Concatenate a variadic amount of tensors along a given axis. No data
    conversion happens during a concat operation.
  }];

  let arguments = (ins
    Variadic<Tosa_Tensor>:$input1,
    I32Attr:$axis
  );

  let results = (outs
    Tosa_Tensor:$output
  );

  list<Availability> availability = [
    Profile<[Tosa_PRO_INT, Tosa_PRO_FP]>,
    Extension<[Tosa_EXT_FP8E4M3, Tosa_EXT_FP8E5M2, Tosa_EXT_BF16]>,
  ];

  let hasCanonicalizer = 1;
  let hasFolder = 1;

  let extraClassDeclaration = [{
    /// Returns true when two result types are compatible for this op;
    /// Method used by InferTypeOpInterface.
    static bool isCompatibleReturnTypes(TypeRange l, TypeRange r);
  }];
}

//===----------------------------------------------------------------------===//
// Operator: pad
//===----------------------------------------------------------------------===//
def Tosa_PadOp : Tosa_InferShapedTypeOp<"pad"> {
  let summary = "Pads a tensor with value specified.";

  let description = [{
    Pads a tensor along the borders of each dimension with a supplied value.
    Returns a new tensor with the padding included. The pad_const value includes
    the zero point if the tensor uses a zero point.

    Example:

    ```mlir
    %0 = tosa.const_shape { value = dense<[1, 2, 3, 4]> : tensor<4xindex> } : () -> !tosa.shape<4>
    tosa.pad %arg0, %0 : (tensor<1x2xf32>, !tosa.shape<4>)  -> (tensor<4x9xf32>)
    ```

    Example 2:

    ```mlir
    %0 = tosa.const_shape { value = dense<[-1, 2, 3, 4]> : tensor<4xindex> } : () -> !tosa.shape<4>
    tosa.pad %arg0, %0 : (tensor<1x2xf32>, !tosa.shape<4>)  -> (tensor<?x9xf32>)
    ```
  }];

  let arguments = (ins
    Tosa_RankedTensor:$input1,
    Tosa_Shape:$padding,
    Optional<Tosa_Rank0Tensor>:$pad_const,
    OptionalAttr<I32Attr>:$input_zp
  );

  let results = (outs
    Tosa_RankedTensor:$output
  );

  list<Availability> availability = [
    Profile<[Tosa_PRO_INT, Tosa_PRO_FP]>,
    Extension<[Tosa_EXT_FP8E4M3, Tosa_EXT_FP8E5M2, Tosa_EXT_BF16]>,
  ];

  let builders = [Tosa_PadOpQuantInfoBuilder,
                  Tosa_ExplicitValuePadOpQuantInfoBuilder];

  let hasCanonicalizer = 1;
  let hasFolder = 1;
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// Operator: reshape
//===----------------------------------------------------------------------===//
def Tosa_ReshapeOp : Tosa_InferTensorTypeOp<"reshape"> {
  let summary = "Reshape operator";

  let description = [{
    Returns a tensor with the same type/values as the input, with a new shape
    specified by the shape argument. Reshape may operate on tensors of any rank.
    No data conversion happens during a reshape operation.
  }];

  let hasFolder = 1;
  let hasVerifier = 1;

  let arguments = (ins
    Tosa_Tensor:$input1,
    Tosa_Shape:$shape
  );

  let results = (outs
    Tosa_RankedTensor:$output
  );

  list<Availability> availability = [
    Profile<[Tosa_PRO_INT, Tosa_PRO_FP]>,
    Extension<[Tosa_EXT_FP8E4M3, Tosa_EXT_FP8E5M2, Tosa_EXT_BF16]>,
  ];

  let extraClassDeclaration = [{
    /// Returns true when two result types are compatible for this op;
    /// Method used by InferTypeOpInterface.
    static bool isCompatibleReturnTypes(TypeRange l, TypeRange r);
  }];

  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
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
    Tosa_Tensor:$input1,
    I32Attr:$axis
  );

  let results = (outs
    Tosa_Tensor:$output
  );

  list<Availability> availability = [
    Profile<[Tosa_PRO_INT, Tosa_PRO_FP]>,
    Extension<[Tosa_EXT_FP8E4M3, Tosa_EXT_FP8E5M2, Tosa_EXT_BF16]>,
  ];

  let hasFolder = 1;
  let hasVerifier = 1;

  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Operator: slice
//===----------------------------------------------------------------------===//
def Tosa_SliceOp : Tosa_InferShapedTypeOp<"slice"> {
  let summary = "Slice operator";

  let description = [{
    Extracts a slice of the input1 on the given axis, beginning at the
    start coordinates, and extending for size elements in each direction.  No
    data conversion happens during a slice operation.
  }];

  let arguments = (ins
    Tosa_Tensor:$input1,
    Tosa_Shape:$start,
    Tosa_Shape:$size
  );

  let results = (outs
    Tosa_Tensor:$output
  );

  list<Availability> availability = [
    Profile<[Tosa_PRO_INT, Tosa_PRO_FP]>,
    Extension<[Tosa_EXT_FP8E4M3, Tosa_EXT_FP8E5M2, Tosa_EXT_BF16]>,
  ];

  let hasCanonicalizer = 1;
  let hasFolder = 1;
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// Operator: tile
//===----------------------------------------------------------------------===//
def Tosa_TileOp : Tosa_InferShapedTypeOp<"tile"> {
  let summary = "Tile operator";

  let description = [{
    Replicates input1 multiplies times along each dimension.
  }];

  let arguments = (ins
    Tosa_Tensor:$input1,
    Tosa_Shape:$multiples);

  let results = (outs
    Tosa_Tensor:$output
  );

  list<Availability> availability = [
    Profile<[Tosa_PRO_INT, Tosa_PRO_FP]>,
    Extension<[Tosa_EXT_FP8E4M3, Tosa_EXT_FP8E5M2, Tosa_EXT_BF16]>,
  ];

  let extraClassDeclaration = [{
    LogicalResult getConstantMultiples(llvm::SmallVector<int64_t> &multiples);
  }];

  let hasFolder = 1;
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// Operator: transpose
//===----------------------------------------------------------------------===//
def Tosa_TransposeOp : Tosa_InferShapedTypeOp<"transpose",
                [DeclareOpInterfaceMethods<ReifyRankedShapedTypeOpInterface>,
                 AllElementTypesMatch<["input1", "output"]>]> {
  let summary = "Transpose operator";

  let description = [{
    Permutes the dimensions of the input tensor input1 based on the perms
    argument. Each value in the perms list must be a valid dimension of the
    input tensor and may not be repeated.
  }];

  let arguments = (ins
    Tosa_Tensor:$input1,
    DenseI32ArrayAttr:$perms
  );

  let results = (
    outs Tosa_Tensor:$output
  );

  list<Availability> availability = [
    Profile<[Tosa_PRO_INT, Tosa_PRO_FP]>,
    Extension<[Tosa_EXT_FP8E4M3, Tosa_EXT_FP8E5M2, Tosa_EXT_BF16]>,
  ];

  let hasCanonicalizer = 1;
  let hasFolder = 1;
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// Operator Class: Scatter/gather Operations.
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Operator: gather
//===----------------------------------------------------------------------===//
def Tosa_GatherOp : Tosa_InferShapedTypeOp<"gather"> {
  let summary = "Gather operation,";

  let description = [{
    Generate a tensor for which each element in the output is a subtensor of the
    values tensor based on the indices. N is the number of batches, W the number
    of indices in each batch, K the range of each index and C the number data
    channels for each index.
  }];

  let arguments = (ins
    Tosa_Tensor3D:$values,
    TosaTensorRankOf<[Tosa_Int32], [2]>:$indices
  );

  let results = (outs
    Tosa_Tensor3D:$output
  );

  list<Availability> availability = [
    Profile<[Tosa_PRO_INT, Tosa_PRO_FP]>,
    Extension<[Tosa_EXT_FP8E4M3, Tosa_EXT_FP8E5M2, Tosa_EXT_BF16]>,
  ];
}

//===----------------------------------------------------------------------===//
// Operator: scatter
//===----------------------------------------------------------------------===//
def Tosa_ScatterOp : Tosa_InferShapedTypeOp<"scatter"> {
  let summary = "Scatter operation,";

  let description = [{
    The values_out tensor is set to the values_in tensor with data modified as
    follows: data from the input tensor is inserted at the positions specified
    by the indices tensor. N is the number of batches, W the number of indices
    in each batch, K the range of each index and C the number data channels for
    each index. It is not permitted to repeat the same output index within a
    single SCATTER operation and so each output index occurs at most once. It
    follows that K >= W. In use cases that require multiple updates to the same
    output position, these must be decomposed into multiple SCATTER operations.
  }];

  let arguments = (ins
    Tosa_Tensor3D:$values_in,
    TosaTensorRankOf<[Tosa_Int32], [2]>:$indices,
    Tosa_Tensor3D:$input
  );

  let results = (outs
    Tosa_Tensor3D:$values_out
  );

  list<Availability> availability = [
    Profile<[Tosa_PRO_INT, Tosa_PRO_FP]>,
    Extension<[Tosa_EXT_FP8E4M3, Tosa_EXT_FP8E5M2, Tosa_EXT_BF16]>,
  ];
}

//===----------------------------------------------------------------------===//
// Operator Class: Image Frontend Functions.
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Operator: resize
//===----------------------------------------------------------------------===//
def Tosa_ResizeOp : Tosa_InferShapedTypeOp<"resize"> {
  let summary = "Resize operation, supports various resize/upsample modes";

  let description = [{
    Resizes a tensor. Resize is only allowed in the H and W dimensions.

    The height dimension is scaled by factor (scale_y_n/scale_y_d). The width
    dimension is scaled by factor (scale_x_n/scale_x_d).

    The NEAREST_NEIGHBOR mode returns the value of the input tensor closest to
    the calculated sample position for both floating-point and integer data
    formats.

    Floating-point BILINEAR mode returns a bilinearly interpolated output value
    based on the four closest input sample positions.

    For integer BILINEAR interpolation mode, the output value must be scaled by
    1/(scale_y_n * scale_x_n) in a following operation to complete the
    interpolation (for example with a RESCALE operator).

    The output dimensions can be derived from the input dimensions by inverting
    the scale as described in the pseudocode. The [border_y, border_x] values
    adjust the output size to allow fractional sampling beyond integer input
    position (IH - 1,IW - 1).

    The limit MAX_SCALE is applied to each scale ratio after reduction of the
    ratio. Individual scale numerator and denominaor values are allowed to be
    larger than MAX_SCALE.
  }];

  let arguments = (ins
    Tosa_Tensor4D:$input,
    Rank4TosaShape:$scale,
    Rank2TosaShape:$offset,
    Rank2TosaShape:$border,
    Tosa_ResizeTypeAttr:$mode
  );

  let results = (outs
    Tosa_Tensor4D:$output
  );

  list<Availability> availability = [
    Profile<[Tosa_PRO_INT, Tosa_PRO_FP]>,
    Extension<[Tosa_EXT_INT16, Tosa_EXT_BF16]>,
  ];

  let hasFolder = 1;
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
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

    | Mode                     | Input   | Output  |
    |--------------------------|---------|---------|
    | signed 8 to bool         | int8    | Boolean |
    | signed 16 to bool        | int16   | Boolean |
    | signed 32 to bool        | int32   | Boolean |
    | bool to 8                | Boolean | int8    |
    | bool to 16               | Boolean | int16   |
    | bool to 32               | Boolean | int32   |
    | signed 8 to signed 16    | int8    | int16   |
    | signed 8 to signed 32    | int8    | int32   |
    | signed 16 to signed 8    | int16   | int8    |
    | signed 16 to signed 32   | int16   | int32   |
    | signed 32 to signed 8    | int32   | int8    |
    | signed 32 to signed 16   | int32   | int16   |
    | float to signed 8        | float   | int8    |
    | float to signed 16       | float   | int16   |
    | signed 8 to float        | int8    | float   |
    | signed 16 to float       | int16   | float   |
    | float 32 to float 64     | float32 | float64 |
    | float 64 to float 32     | float64 | float32 |
  }];

  let arguments = (ins
    Tosa_Tensor:$input
  );

  let results = (outs
    Tosa_Tensor:$output
  );

  list<Availability> availability = [
    Profile<[Tosa_PRO_INT, Tosa_PRO_FP]>,
    Extension<[Tosa_EXT_INT16, Tosa_EXT_FP8E4M3, Tosa_EXT_FP8E5M2, Tosa_EXT_BF16]>,
  ];

  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";

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

    | Mode                   | Input | Output | Unsigned input | Unsigned output |
    |------------------------|-------|--------|----------------|-----------------|
    | signed 8 to 8          | int8  | int8   |  false         |  false          |
    | signed 8 to 16         | int8  | int16  |  false         |  false          |
    | signed 8 to 32         | int8  | int32  |  false         |  false          |
    | signed 16 to 8         | int16 | int8   |  false         |  false          |
    | signed 16 to 16        | int16 | int16  |  false         |  false          |
    | signed 16 to 32        | int16 | int32  |  false         |  false          |
    | signed 32 to 8         | int32 | int8   |  false         |  false          |
    | signed 32 to 16        | int32 | int16  |  false         |  false          |
    | signed 32 to 32        | int32 | int32  |  false         |  false          |
    | signed 48 to 8         | int48 | int8   |  false         |  false          |
    | signed 48 to 16        | int48 | int16  |  false         |  false          |
    | signed 48 to 32        | int48 | int32  |  false         |  false          |
    | unsigned 8 to signed 8 | uint8 | int8   |  true          |  false          |
    | signed 8 to unsigned 8 | int8  | uint8  |  false         |  true           |
  }];

  let arguments = (ins
    Tosa_Tensor:$input,
    I32Attr:$input_zp,
    I32Attr:$output_zp,
    DenseI32ArrayAttr:$multiplier,
    DenseI8ArrayAttr:$shift,
    BoolAttr:$scale32,
    BoolAttr:$double_round,
    BoolAttr:$per_channel,
    DefaultValuedOptionalAttr<BoolAttr, "false">:$input_unsigned,
    DefaultValuedOptionalAttr<BoolAttr, "false">:$output_unsigned
  );

  let results = (outs
    Tosa_Tensor:$output
  );

  list<Availability> availability = [
    Profile<[Tosa_PRO_INT]>,
    Extension<[Tosa_EXT_INT16]>,
  ];

  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Operator Class: Data Node Ops.
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Operator: const
//===----------------------------------------------------------------------===//
def Tosa_ConstOp : Tosa_Op<"const", [ConstantLike, Pure,
                                     AllShapesMatch<["value", "output"]>,
                                     FirstAttrDerivedResultType]> {
  let summary = "Constant op.";

  let description = [{
    A node containing constant data for use as the input to an operation. May
    hold data in any of the supported data formats.

    Example:

    ```mlir
    // Generic form
    %out = "tosa.const"() {value = dense<0> : tensor<2x3xi32>} : () -> tensor<2x3xi32>
    ```
  }];

  let arguments = (ins
    ElementsAttr:$value
  );

  let results = (outs
    TosaTensorOf<[AnyTypeOf<[Tosa_AnyNumber]>]>:$output
  );

  list<Availability> availability = [
    Profile<[Tosa_PRO_INT, Tosa_PRO_FP]>,
    Extension<[Tosa_EXT_INT4, Tosa_EXT_INT16, Tosa_EXT_FP8E4M3, Tosa_EXT_FP8E5M2, Tosa_EXT_BF16]>,
  ];

  let hasFolder = 1;
  let hasVerifier = 1;
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

  list<Availability> availability = [
    Profile<[Tosa_PRO_INT, Tosa_PRO_FP]>,
    Extension<[Tosa_EXT_INT4, Tosa_EXT_INT16, Tosa_EXT_FP8E4M3, Tosa_EXT_FP8E5M2, Tosa_EXT_BF16]>,
  ];

  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
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

    `operator_name` is a string that tells the backend which custom operator is
    being called.

    `domain_name` is a string identifier which can help avoid name collisions on
    the identifier field.

    `implementation_attrs` is a string which is a backend and identifier specific
    set of attributes to the custom operator.

    `input_list` is the set of tensor inputs to the custom operator.

    `output_list` is the list of tensors returned by the operator. The number of operators
    is backend specific.

    Example:

    ```mlir
    %out = tosa.custom %in {domain_name = "tosa_mlir_test", operator_name =
           "custom_test", implementation_attrs = ""}: (tensor<10xi32>) ->
           (tensor<10xi32>)
    ```
  }];

  let arguments = (ins
    StrAttr:$operator_name,
    StrAttr:$domain_name,
    StrAttr:$implementation_attrs,
    Variadic<Tosa_Tensor>:$input_list
  );

  let results = (outs
    Variadic<Tosa_Tensor>:$output_list
  );

  list<Availability> availability = [
    Profile<[Tosa_PRO_INT, Tosa_PRO_FP]>,
    Extension<[]>,
  ];

  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Operator Class: Control Flow Operators.
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Operator: cond_if
//===----------------------------------------------------------------------===//
//===----------------------------------------------------------------------===//
// Further described in docs/Rationale/RationaleTOSADialect.md .
//===----------------------------------------------------------------------===//
def Tosa_IfOp : Tosa_Op<"cond_if",
       [InferShapedTypeOpAdaptor,
       SingleBlockImplicitTerminator<"YieldOp">,
       RecursiveMemoryEffects]> {
  let summary = "Conditional if operator";

  let description = [{
    Evaluates a Boolean condition and then takes one of two distinct execution
    paths. This implements the semantic If-then-else structure.
  }];

  let arguments = (ins
    Tosa_I1Tensor:$cond,
    Variadic<Tosa_Tensor>:$inputs
  );

  let results = (outs
    Variadic<Tosa_Tensor>:$output
  );

  list<Availability> availability = [
    Profile<[]>,
    Extension<[Tosa_EXT_CONTROLFLOW]>,
  ];

  let regions = (region
    SizedRegion<1>:$then_branch,
    SizedRegion<1>:$else_branch
  );

  let hasCustomAssemblyFormat = 1;
}

//===----------------------------------------------------------------------===//
// Operator: while_loop
//===----------------------------------------------------------------------===//
//===----------------------------------------------------------------------===//
// Further described in docs/Rationale/RationaleTOSADialect.md .
//===----------------------------------------------------------------------===//
def Tosa_WhileOp : Tosa_Op<"while_loop", [
       DeclareOpInterfaceMethods<LoopLikeOpInterface>,
       InferShapedTypeOpAdaptor,
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

  list<Availability> availability = [
    Profile<[]>,
    Extension<[Tosa_EXT_CONTROLFLOW]>,
  ];

  let regions = (region
    SizedRegion<1>:$cond,
    SizedRegion<1>:$body
  );

  let hasCustomAssemblyFormat = 1;
}

include "mlir/Dialect/Tosa/IR/TosaUtilOps.td"

include "mlir/Dialect/Tosa/IR/TosaShapeOps.td"

#endif // TOSA_OPS
