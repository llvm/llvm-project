//===- SPIRVTosaOps.td - TOSA extended insts spec file -----*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the op definition spec of TOSA extension ops.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_SPIRV_IR_TOSA_OPS
#define MLIR_DIALECT_SPIRV_IR_TOSA_OPS

include "mlir/Dialect/SPIRV/IR/SPIRVBase.td"
include "mlir/Dialect/SPIRV/IR/SPIRVGraphOps.td"
include "mlir/Dialect/SPIRV/IR/SPIRVTosaTypes.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// SPIR-V TOSA opcode specification.
//===----------------------------------------------------------------------===//

// Base class for all TOSA ops.
class SPIRV_TosaOp<string mnemonic, int opcode, list<Trait> traits = []> :
  SPIRV_ExtInstOp<mnemonic, "Tosa", "TOSA.001000.1", opcode,
  !listconcat(traits, [InGraphScope])> {

  let availability = [
    MinVersion<SPIRV_V_1_5>,
    MaxVersion<SPIRV_V_1_6>,
    Extension<[SPV_ARM_graph]>,
    Capability<[SPIRV_C_GraphARM]>
  ];

  let hasVerifier = 0;
}

class SPIRV_TosaOpWithResult<string mnemonic, int opcode, list<Trait> traits = []> :
  SPIRV_TosaOp<mnemonic, opcode, traits> {

  code extraBaseClassDeclaration = [{
    ::mlir::spirv::TensorArmType getResultType() {
      return cast<::mlir::spirv::TensorArmType>(getType());
    }
  }];
}

class SPIRV_TosaOpWithComplexResult<string mnemonic, int opcode, list<Trait> traits = []> :
  SPIRV_TosaOp<mnemonic, opcode, traits> {

  code extraBaseClassDeclaration = [{
    ::mlir::spirv::TensorArmType getResultRealType() {
      auto resultType = cast<StructType>(getType());
      return cast<::mlir::spirv::TensorArmType>(resultType.getElementType(0));
    }
    ::mlir::spirv::TensorArmType getResultImagType() {
      auto resultType = cast<StructType>(getType());
      return cast<::mlir::spirv::TensorArmType>(resultType.getElementType(1));
    }
  }];
}

class SPIRV_TosaElementwiseUnaryOp<string mnemonic, int opcode, list<Trait> traits = []> :
  SPIRV_TosaOpWithResult<mnemonic, opcode, !listconcat(traits,
    [AllTypesMatch<["input1", "output"]>])> {

  let extraClassDeclaration = extraBaseClassDeclaration#[{
    ::mlir::spirv::TensorArmType getInput1Type() {
      return cast<::mlir::spirv::TensorArmType>(getInput1().getType());
    }
  }];
}

class SPIRV_TosaFloatElementwiseUnaryOp<string mnemonic, int opcode, list<Trait> traits = []> :
  SPIRV_TosaElementwiseUnaryOp<mnemonic, opcode, !listconcat(traits, [Pure])> {

  let arguments = (ins
    SPIRV_TosaFloat_TensorArm: $input1
  );

  let results = (outs
    SPIRV_TosaFloat_TensorArm: $output
  );

  let assemblyFormat = [{
    $input1
    attr-dict `:` type(operands) `->` type(results)
  }];
}

class SPIRV_TosaBinaryOp<string mnemonic, int opcode, list<Trait> traits = []> :
  SPIRV_TosaOpWithResult<mnemonic, opcode, !listconcat(traits, [
    AllElementTypesMatch<["input1", "input2"]>,
    AllRanksMatch<["input1", "input2", "output"]>,
    MatchBroadcastableShapes<"input1", "input2", "output">])> {

  let extraClassDeclaration = extraBaseClassDeclaration#[{
    ::mlir::spirv::TensorArmType getInput1Type() {
      return cast<::mlir::spirv::TensorArmType>(getInput1().getType());
    }
    ::mlir::spirv::TensorArmType getInput2Type() {
      return cast<::mlir::spirv::TensorArmType>(getInput2().getType());
    }
  }];
}

class SPIRV_TosaElementwiseBinaryOp<string mnemonic, int opcode, list<Trait> traits = []> :
  SPIRV_TosaBinaryOp<mnemonic, opcode, !listconcat(traits, [
    AllElementTypesMatch<["input1", "output"]>])> {
}

class SPIRV_TosaIntegerElementwiseBinaryOp<string mnemonic, int opcode, list<Trait> traits = []> :
  SPIRV_TosaElementwiseBinaryOp<mnemonic, opcode, traits> {

  let arguments = (ins
    SPIRV_TosaInteger_TensorArm: $input1,
    SPIRV_TosaInteger_TensorArm: $input2
  );

  let results = (outs
    SPIRV_TosaInteger_TensorArm: $output
  );

  let assemblyFormat = [{
    $input1 `,`
    $input2
    attr-dict `:` type(operands) `->` type(results)
  }];
}

class SPIRV_TosaBoolElementwiseBinaryOp<string mnemonic, int opcode, list<Trait> traits = []> :
  SPIRV_TosaElementwiseBinaryOp<mnemonic, opcode, !listconcat(traits, [Pure])> {

  let arguments = (ins
    SPIRV_Bool_TensorArm: $input1,
    SPIRV_Bool_TensorArm: $input2
  );

  let results = (outs
    SPIRV_Bool_TensorArm: $output
  );

  let assemblyFormat = [{
    $input1 `,`
    $input2
    attr-dict `:` type(operands) `->` type(results)
  }];
}

class SPIRV_TosaConvolutionOp<string mnemonic, int opcode, list<Trait> traits = []> :
  SPIRV_TosaOpWithResult<mnemonic, opcode, !listconcat(traits, [NoMemoryEffect,
    TypeConstraintImplicationOn<"input", I8, "output", [I32]>,
    TypeConstraintImplicationOn<"input", I16, "output", [I64]>,
    TypeConstraintImplicationOn<"input", BF16, "output", [BF16]>,
    TypeConstraintImplicationOn<"input", F16, "output", [F16]>,
    TypeConstraintImplicationOn<"input", F32, "output", [F32]>,
    TypeConstraintImplicationOn<"input", BF16, "weight", [BF16]>,
    TypeConstraintImplicationOn<"input", F16, "weight", [F16]>,
    TypeConstraintImplicationOn<"input", F32, "weight", [F32]>,
    TypeConstraintImplicationOn<"input", AnyInteger, "input", [I8, I16]>,
    TypeConstraintImplicationOn<"weight", AnyInteger, "weight", [I8]>,
    TypeImpliesAccType<"input", I8, ["INT32"]>,
    TypeImpliesAccType<"input", I16, ["INT48"]>,
    TypeImpliesAccType<"input", F16, ["FP16", "FP32"]>,
    TypeImpliesAccType<"input", BF16, ["FP32"]>,
    TypeImpliesAccType<"input", F32, ["FP32"]>,
    AllElementTypesMatch<["bias", "output"]>,
    AllElementTypesMatch<["input", "input_zp"]>,
    AllElementTypesMatch<["weight", "weight_zp"]>])> {

  let extraClassDeclaration = extraBaseClassDeclaration#[{
    ::mlir::spirv::TensorArmType getInputType() {
      return cast<::mlir::spirv::TensorArmType>(getInput().getType());
    }
    ::mlir::spirv::TensorArmType getWeightType() {
      return cast<::mlir::spirv::TensorArmType>(getWeight().getType());
    }
    ::mlir::spirv::TensorArmType getBiasType() {
      return cast<::mlir::spirv::TensorArmType>(getBias().getType());
    }
  }];
}

class SPIRV_TosaComparisonOp<string mnemonic, int opcode, list<Trait> traits = []> :
  SPIRV_TosaOpWithResult<mnemonic, opcode, !listconcat(traits, [Pure,
  AllElementTypesMatch<["input1", "input2"]>,
  MatchBroadcastableShapes<"input1", "input2", "output">])> {

  let arguments = (ins
    SPIRV_TosaNumerical_TensorArm: $input1,
    SPIRV_TosaNumerical_TensorArm: $input2
  );

  let results = (outs
    SPIRV_Bool_TensorArm: $output
  );

  let assemblyFormat = [{
    $input1 `,`
    $input2
    attr-dict `:` type(operands) `->` type(results)
  }];

  let extraClassDeclaration = extraBaseClassDeclaration#[{
    ::mlir::spirv::TensorArmType getInput1Type() {
      return cast<::mlir::spirv::TensorArmType>(getInput1().getType());
    }
    ::mlir::spirv::TensorArmType getInput2Type() {
      return cast<::mlir::spirv::TensorArmType>(getInput2().getType());
    }
  }];
}

class SPIRV_TosaReductionOp<string mnemonic, int opcode, list<Trait> traits = []> :
  SPIRV_TosaOpWithResult<mnemonic, opcode, !listconcat(traits, [
  AllElementTypesMatch<["input", "output"]>,
  AllRanksMatch<["input", "output"]>,
  AxisValueLessThanRankOf<"input">])> {

  let extraClassDeclaration = extraBaseClassDeclaration#[{
    ::mlir::spirv::TensorArmType getInputType() {
      return cast<::mlir::spirv::TensorArmType>(getInput().getType());
    }
  }];
}

def SPIRV_TosaArgMaxOp : SPIRV_TosaOpWithResult<"ArgMax", 0, [Pure,
  OutputRankIsInputRankMinusOne<"input", "output">,
  AxisValueLessThanRankOf<"input">]> {
  let summary = "Perform argmax on the input.";

  let description = [{
    Returns the index with the largest value across the given axis of the
    input tensor. If multiple locations have equal values, returns the first
    match along the search axis.
    NaN Propagation Mode is ignored for inputs using integer element types.

    References:
      * https://github.khronos.org/SPIRV-Registry/extended/TOSA.001000.1.html#_argmax
      * https://www.mlplatform.org/tosa/tosa_spec_1_0_1.html#_argmax

    #### Example:
    ```mlir
    %2 = spirv.Tosa.ArgMax axis = 3, nan_mode = <Propagate>, %arg0 : !spirv.arm.tensor<3x28x17x17xi8> -> !spirv.arm.tensor<3x28x17xi32>
    %2 = spirv.Tosa.ArgMax axis = 2, nan_mode = <Propagate>, %arg0 : !spirv.arm.tensor<2x2x7x14xf32> -> !spirv.arm.tensor<2x2x14xi32>
    ```
  }];

  let arguments = (ins
    SPIRV_TensorArmAxisAttr: $axis,
    SPIRV_TosaExtNaNPropagationModeAttr: $nan_mode,
    SPIRV_TosaNumerical_TensorArm: $input
  );

  let results = (outs
    SPIRV_Int32_TensorArmUpTo5D: $output
  );

  let assemblyFormat = [{
    `axis` `=` $axis `,`
    `nan_mode` `=` $nan_mode `,`
    $input
    attr-dict `:` type(operands) `->` type(results)
  }];

  let extraClassDeclaration = extraBaseClassDeclaration#[{
    ::mlir::spirv::TensorArmType getInputType() {
      return cast<::mlir::spirv::TensorArmType>(getInput().getType());
    }
  }];
}


def SPIRV_TosaAvgPool2DOp : SPIRV_TosaOpWithResult<"AvgPool2D", 1, [NoMemoryEffect,
  TypeImpliesAccType<"input", I8, ["INT32"]>,
  TypeImpliesAccType<"input", I16, ["INT32"]>,
  TypeImpliesAccType<"input", F16, ["FP16", "FP32"]>,
  TypeImpliesAccType<"input", BF16, ["FP32"]>,
  TypeImpliesAccType<"input", F32, ["FP32"]>,
  AllElementTypesMatch<["input", "input_zp", "output", "output_zp"]>]> {
  let summary = "Performs average pooling on the input.";

  let description = [{
    Performs an average pooling over the given input tensor. A sliding
    window of size given by <kernel size> is passed over the input tensor, with
    the mean value being placed in the output tensor. When calculating the
    average, only the number of valid input tensor values, but not padding, are
    used to calculate the divisor. The behaviour is undefined if the accumulated
    result overflows or underflows.

    References:
      * https://github.khronos.org/SPIRV-Registry/extended/TOSA.001000.1.html#_avg_pool2d
      * https://www.mlplatform.org/tosa/tosa_spec_1_0_1.html#_avg_pool2d

    #### Example:
    ```mlir
    %6 = spirv.Tosa.AvgPool2D kernel = [3, 3], stride = [1, 2], pad = [0, 1, 0, 0], acc_type = <INT32>, %arg0, %4, %5 : !spirv.arm.tensor<1x3x65537x1xi8>, !spirv.arm.tensor<1xi8>, !spirv.arm.tensor<1xi8> -> !spirv.arm.tensor<1x2x32768x1xi8>
    %6 = spirv.Tosa.AvgPool2D kernel = [2, 2], stride = [1, 1], pad = [1, 0, 0, 0], acc_type = <FP32>, %arg0, %4, %5 : !spirv.arm.tensor<1x2x65533x2xf32>, !spirv.arm.tensor<1xf32>, !spirv.arm.tensor<1xf32> -> !spirv.arm.tensor<1x2x65532x2xf32>
    ```
  }];

  let arguments = (ins
    SPIRV_Int32_1DTensorArmOfLength2Attr: $kernel,
    SPIRV_Int32_1DTensorArmOfLength2Attr: $stride,
    SPIRV_Int32_1DTensorArmOfLength4Attr: $pad,
    SPIRV_TosaExtAccTypeAttr: $acc_type,
    SPIRV_TosaNumerical_TensorArm4D: $input,
    SPIRV_TosaNumerical_1DTensorArmOfLength1: $input_zp,
    SPIRV_TosaNumerical_1DTensorArmOfLength1: $output_zp
  );

  let results = (outs
    SPIRV_TosaNumerical_TensorArm4D: $output
  );

  let assemblyFormat = [{
    `kernel` `=` custom<SPIRV_I32_1DArmTensor>($kernel) `,`
    `stride` `=` custom<SPIRV_I32_1DArmTensor>($stride) `,`
    `pad` `=` custom<SPIRV_I32_1DArmTensor>($pad) `,`
    `acc_type` `=` $acc_type `,`
    $input `,`
    $input_zp `,`
    $output_zp
    attr-dict `:` type(operands) `->` type(results)
  }];

  let extraClassDeclaration = extraBaseClassDeclaration#[{
    ::mlir::spirv::TensorArmType getInputType() {
      return cast<::mlir::spirv::TensorArmType>(getInput().getType());
    }
  }];
}


def SPIRV_TosaConv2DOp : SPIRV_TosaConvolutionOp<"Conv2D", 2> {
  let summary = "2D Convolution operator.";

  let description = [{
    Performs a 2D convolution over the given tensor input, using the weight
    tensor. Implementations may choose to skip calculation of multiplies in
    the padding area.

    Input and weight have respective zero point values provided in input_zp and weight_zp.

    The behaviour is undefined if the accumulated result overflows or underflows.

    References:
      * https://github.khronos.org/SPIRV-Registry/extended/TOSA.001000.1.html#_conv2d
      * https://www.mlplatform.org/tosa/tosa_spec_1_0_1.html#_conv2d

    #### Example:
    ```mlir
    %7 = spirv.Tosa.Conv2D pad = [1, 0, 0, 0], stride = [1, 2], dilation = [7, 1], acc_type = <INT32>, local_bound = false, %arg0, %arg1, %arg2, %5, %6 : !spirv.arm.tensor<1x65535x3x1xi8>, !spirv.arm.tensor<7x1x1x1xi8>, !spirv.arm.tensor<1xi32>, !spirv.arm.tensor<1xi8>, !spirv.arm.tensor<1xi8> -> !spirv.arm.tensor<1x65536x2x7xi32>
    %7 = spirv.Tosa.Conv2D pad = [0, 0, 0, 0], stride = [1, 1], dilation = [1, 1], acc_type = <FP16>, local_bound = true, %arg0, %arg1, %arg2, %5, %6 : !spirv.arm.tensor<1x34x18x27xf16>, !spirv.arm.tensor<11x1x1x27xf16>, !spirv.arm.tensor<11xf16>, !spirv.arm.tensor<1xf16>, !spirv.arm.tensor<1xf16> -> !spirv.arm.tensor<1x34x18x11xf16>
    ```
  }];

  let arguments = (ins
    SPIRV_Int32_1DTensorArmOfLength4Attr: $pad,
    SPIRV_Int32_1DTensorArmOfLength2Attr: $stride,
    SPIRV_Int32_1DTensorArmOfLength2Attr: $dilation,
    SPIRV_TosaExtAccTypeAttr: $acc_type,
    SPIRV_BoolConstAttr: $local_bound,
    SPIRV_TosaNumerical_TensorArm4D: $input,
    SPIRV_TosaNumerical_TensorArm4D: $weight,
    SPIRV_TosaNumerical_TensorArm1D: $bias,
    SPIRV_TosaNumerical_1DTensorArmOfLength1: $input_zp,
    SPIRV_TosaNumerical_1DTensorArmOfLength1: $weight_zp
  );

  let results = (outs
    SPIRV_TosaNumerical_TensorArm4D: $output
  );

  let assemblyFormat = [{
    `pad` `=` custom<SPIRV_I32_1DArmTensor>($pad) `,`
    `stride` `=` custom<SPIRV_I32_1DArmTensor>($stride) `,`
    `dilation` `=` custom<SPIRV_I32_1DArmTensor>($dilation) `,`
    `acc_type` `=` $acc_type `,`
    `local_bound` `=` $local_bound `,`
    $input `,`
    $weight `,`
    $bias `,`
    $input_zp `,`
    $weight_zp
    attr-dict `:` type(operands) `->` type(results)
  }];
}


def SPIRV_TosaConv3DOp : SPIRV_TosaConvolutionOp<"Conv3D", 3> {
  let summary = "3D Convolution operator.";

  let description = [{
    Performs a 3D convolution over the given input tensor. Implementations
    may choose to skip calculation of multiplies in the padding area.

    Input and weight have respective zero point values provided in input_zp and weight_zp.

    The behaviour is undefined if the accumulated result overflows or underflows.

    References:
      * https://github.khronos.org/SPIRV-Registry/extended/TOSA.001000.1.html#_conv3d
      * https://www.mlplatform.org/tosa/tosa_spec_1_0_1.html#_conv3d

    #### Example:
    ```mlir
    %7 = spirv.Tosa.Conv3D pad = [0, 0, 0, 0, 0, 0], stride = [1, 1, 1], dilation = [1, 1, 1], acc_type = <INT32>, local_bound = false, %arg0, %arg1, %arg2, %5, %6 : !spirv.arm.tensor<1x9x21x14x1xi8>, !spirv.arm.tensor<2x1x2x1x1xi8>, !spirv.arm.tensor<1xi32>, !spirv.arm.tensor<1xi8>, !spirv.arm.tensor<1xi8> -> !spirv.arm.tensor<1x9x20x14x2xi32>
    %7 = spirv.Tosa.Conv3D pad = [0, 1, 1, 0, 0, 1], stride = [1, 1, 1], dilation = [1, 1, 7], acc_type = <FP32>, local_bound = false, %arg0, %arg1, %arg2, %5, %6 : !spirv.arm.tensor<1x2x65539x1x2xf32>, !spirv.arm.tensor<1x1x1x1x2xf32>, !spirv.arm.tensor<1xf32>, !spirv.arm.tensor<1xf32>, !spirv.arm.tensor<1xf32> -> !spirv.arm.tensor<1x3x65540x2x1xf32>
    ```
  }];

  let arguments = (ins
    SPIRV_Int32_1DTensorArmOfLength6Attr: $pad,
    SPIRV_Int32_1DTensorArmOfLength3Attr: $stride,
    SPIRV_Int32_1DTensorArmOfLength3Attr: $dilation,
    SPIRV_TosaExtAccTypeAttr: $acc_type,
    SPIRV_BoolConstAttr: $local_bound,
    SPIRV_TosaNumerical_TensorArm5D: $input,
    SPIRV_TosaNumerical_TensorArm5D: $weight,
    SPIRV_TosaNumerical_TensorArm1D: $bias,
    SPIRV_TosaNumerical_1DTensorArmOfLength1: $input_zp,
    SPIRV_TosaNumerical_1DTensorArmOfLength1: $weight_zp
  );

  let results = (outs
    SPIRV_TosaNumerical_TensorArm5D: $output
  );

  let assemblyFormat = [{
    `pad` `=` custom<SPIRV_I32_1DArmTensor>($pad) `,`
    `stride` `=` custom<SPIRV_I32_1DArmTensor>($stride) `,`
    `dilation` `=` custom<SPIRV_I32_1DArmTensor>($dilation) `,`
    `acc_type` `=` $acc_type `,`
    `local_bound` `=` $local_bound `,`
    $input `,`
    $weight `,`
    $bias `,`
    $input_zp `,`
    $weight_zp
    attr-dict `:` type(operands) `->` type(results)
  }];
}


def SPIRV_TosaDepthwiseConv2DOp : SPIRV_TosaConvolutionOp<"DepthwiseConv2D", 4> {
  let summary = "Depthwise 2D Convolution operator.";

  let description = [{
    Performs 2D convolutions separately over each channel of the given tensor
    input, using the weight tensor. Implementations may choose to skip
    calculation of multiplies in the padding area.

    Input and weight have respective zero point values provided in input_zp and weight_zp.

    The behaviour is undefined if the accumulated result overflows or underflows.

    References:
      * https://github.khronos.org/SPIRV-Registry/extended/TOSA.001000.1.html#_depthwise_conv2d
      * https://www.mlplatform.org/tosa/tosa_spec_1_0_1.html#_depthwise_conv2d

    #### Example:
    ```mlir
    %7 = spirv.Tosa.DepthwiseConv2D pad = [0, 0, 0, 0], stride = [1, 2], dilation = [7, 7], acc_type = <INT32>, local_bound = false, %arg0, %arg1, %arg2, %5, %6 : !spirv.arm.tensor<1x4x65537x1xi8>, !spirv.arm.tensor<1x3x1x4xi8>, !spirv.arm.tensor<4xi32>, !spirv.arm.tensor<1xi8>, !spirv.arm.tensor<1xi8> -> !spirv.arm.tensor<1x4x32762x4xi32>
    %7 = spirv.Tosa.DepthwiseConv2D pad = [0, 1, 1, 1], stride = [1, 2], dilation = [1, 7], acc_type = <FP32>, local_bound = true, %arg0, %arg1, %arg2, %5, %6 : !spirv.arm.tensor<1x65540x1x3xf32>, !spirv.arm.tensor<1x1x3x1xf32>, !spirv.arm.tensor<1xf32>, !spirv.arm.tensor<1xf32>, !spirv.arm.tensor<1xf32> -> !spirv.arm.tensor<1x65541x2x3xf32>
    ```
  }];

  let arguments = (ins
    SPIRV_Int32_1DTensorArmOfLength4Attr: $pad,
    SPIRV_Int32_1DTensorArmOfLength2Attr: $stride,
    SPIRV_Int32_1DTensorArmOfLength2Attr: $dilation,
    SPIRV_TosaExtAccTypeAttr: $acc_type,
    SPIRV_BoolConstAttr: $local_bound,
    SPIRV_TosaNumerical_TensorArm4D: $input,
    SPIRV_TosaNumerical_TensorArm4D: $weight,
    SPIRV_TosaNumerical_TensorArm1D: $bias,
    SPIRV_TosaNumerical_1DTensorArmOfLength1: $input_zp,
    SPIRV_TosaNumerical_1DTensorArmOfLength1: $weight_zp
  );

  let results = (outs
    SPIRV_TosaNumerical_TensorArm4D: $output
  );

  let assemblyFormat = [{
    `pad` `=` custom<SPIRV_I32_1DArmTensor>($pad) `,`
    `stride` `=` custom<SPIRV_I32_1DArmTensor>($stride) `,`
    `dilation` `=` custom<SPIRV_I32_1DArmTensor>($dilation) `,`
    `acc_type` `=` $acc_type `,`
    `local_bound` `=` $local_bound `,`
    $input `,`
    $weight `,`
    $bias `,`
    $input_zp `,`
    $weight_zp
    attr-dict `:` type(operands) `->` type(results)
  }];
}


def SPIRV_TosaFFT2DOp : SPIRV_TosaOpWithComplexResult<"FFT2D", 5, [Pure]> {
  let summary = "Performs FFT2D operation on the input.";

  let description = [{
    Performs a batched complex 2D Fast Fourier Transform over the input. The
    complex input values are constructed from the corresponding values in the
    input_real and input_imag tensors. The resulting values in the output are
    split into the output_real and output_imag tensors. No normalization is
    applied on either the forward or inverse versions of the operation.

    References:
      * https://github.khronos.org/SPIRV-Registry/extended/TOSA.001000.1.html#_fft2d
      * https://www.mlplatform.org/tosa/tosa_spec_1_0_1.html#_fft2d

    #### Example:
    ```mlir
    %0 = spirv.Tosa.FFT2D inverse = true, local_bound = false, %arg0, %arg1 : !spirv.arm.tensor<1x32x32xf32>, !spirv.arm.tensor<1x32x32xf32> -> !spirv.struct<(!spirv.arm.tensor<1x32x32xf32>, !spirv.arm.tensor<1x32x32xf32>)>
    %1 = spirv.CompositeExtract %0[0 : i32] : !spirv.struct<(!spirv.arm.tensor<1x32x32xf32>, !spirv.arm.tensor<1x32x32xf32>)>
    %2 = spirv.CompositeExtract %0[1 : i32] : !spirv.struct<(!spirv.arm.tensor<1x32x32xf32>, !spirv.arm.tensor<1x32x32xf32>)>
    ```
  }];

  let arguments = (ins
    SPIRV_BoolConstAttr: $inverse,
    SPIRV_BoolConstAttr: $local_bound,
    SPIRV_Float32_TensorArm3D: $input_real,
    SPIRV_Float32_TensorArm3D: $input_imag
  );

  let results = (outs
    SPIRV_Struct_2_Float32_TensorArm3D: $output
  );

  let assemblyFormat = [{
    `inverse` `=` $inverse `,`
    `local_bound` `=` $local_bound `,`
    $input_real `,`
    $input_imag
    attr-dict `:` type(operands) `->` type(results)
  }];

  let extraClassDeclaration = extraBaseClassDeclaration#[{
    ::mlir::spirv::TensorArmType getInputRealType() {
      return cast<::mlir::spirv::TensorArmType>(getInputReal().getType());
    }
    ::mlir::spirv::TensorArmType getInputImagType() {
      return cast<::mlir::spirv::TensorArmType>(getInputImag().getType());
    }
  }];
}


def SPIRV_TosaMatMulOp : SPIRV_TosaOpWithResult<"MatMul", 6, [NoMemoryEffect,
  TypeConstraintImplicationOn<"A", I8, "output", [I32]>,
  TypeConstraintImplicationOn<"A", I16, "output", [I64]>,
  TypeConstraintImplicationOn<"A", BF16, "output", [F32]>,
  TypeConstraintImplicationOn<"A", F16, "output", [F16, F32]>,
  TypeConstraintImplicationOn<"A", F32, "output", [F32]>,
  AllElementTypesMatch<["A", "A_zp", "B", "B_zp"]>]> {
  let summary = "Matrix Multiplication operator.";

  let description = [{
    Performs two dimensional matrix multiplications.

    A, B are the inputs with respective zero point values in A_zp, B_zp.

    The behaviour is undefined if the accumulated result overflows or underflows.

    References:
      * https://github.khronos.org/SPIRV-Registry/extended/TOSA.001000.1.html#_matmul
      * https://www.mlplatform.org/tosa/tosa_spec_1_0_1.html#_matmul

    #### Example:
    ```mlir
    %2 = spirv.Tosa.MatMul %arg0, %arg1, %0, %1 : !spirv.arm.tensor<8x2x3xi8>, !spirv.arm.tensor<8x3x8xi8>, !spirv.arm.tensor<1xi8>, !spirv.arm.tensor<1xi8> -> !spirv.arm.tensor<8x2x8xi32>
    %2 = spirv.Tosa.MatMul %arg0, %arg1, %0, %1 : !spirv.arm.tensor<15x39x50xf16>, !spirv.arm.tensor<15x50x24xf16>, !spirv.arm.tensor<1xf16>, !spirv.arm.tensor<1xf16> -> !spirv.arm.tensor<15x39x24xf16>
    ```
  }];

  let arguments = (ins
    SPIRV_TosaNumerical_TensorArm3D: $A,
    SPIRV_TosaNumerical_TensorArm3D: $B,
    SPIRV_TosaNumerical_1DTensorArmOfLength1: $A_zp,
    SPIRV_TosaNumerical_1DTensorArmOfLength1: $B_zp
  );

  let results = (outs
    SPIRV_TosaNumerical_TensorArm3D: $output
  );

  let assemblyFormat = [{
    $A `,`
    $B `,`
    $A_zp `,`
    $B_zp
    attr-dict `:` type(operands) `->` type(results)
  }];

  let extraClassDeclaration = extraBaseClassDeclaration#[{
    ::mlir::spirv::TensorArmType getAType() {
      return cast<::mlir::spirv::TensorArmType>(getA().getType());
    }
    ::mlir::spirv::TensorArmType getBType() {
      return cast<::mlir::spirv::TensorArmType>(getB().getType());
    }
  }];
}


def SPIRV_TosaMaxPool2DOp : SPIRV_TosaOpWithResult<"MaxPool2D", 7, [Pure,
  AllElementTypesMatch<["input", "output"]>]> {
  let summary = "Performs max pooling on the input.";

  let description = [{
    Performs a max pooling over the given input tensor. A sliding window of
    size given by <kernel size> is passed over the input tensor, with the
    maximum value being placed in the output tensor.
    NaN Propagation Mode is ignored for inputs using integer element types.

    References:
      * https://github.khronos.org/SPIRV-Registry/extended/TOSA.001000.1.html#_max_pool2d
      * https://www.mlplatform.org/tosa/tosa_spec_1_0_1.html#_max_pool2d

    #### Example:
    ```mlir
    %4 = spirv.Tosa.MaxPool2D kernel = [3, 2], stride = [1, 2], pad = [1, 0, 0, 1], nan_mode = <Propagate>, %arg0 : !spirv.arm.tensor<1x3x65537x1xi8> -> !spirv.arm.tensor<1x2x32769x1xi8>
    %4 = spirv.Tosa.MaxPool2D kernel = [3, 2], stride = [2, 2], pad = [1, 0, 1, 1], nan_mode = <Propagate>, %arg0 : !spirv.arm.tensor<1x6x65536x1xf32> -> !spirv.arm.tensor<1x3x32769x1xf32>
    ```
  }];

  let arguments = (ins
    SPIRV_Int32_1DTensorArmOfLength2Attr: $kernel,
    SPIRV_Int32_1DTensorArmOfLength2Attr: $stride,
    SPIRV_Int32_1DTensorArmOfLength4Attr: $pad,
    SPIRV_TosaExtNaNPropagationModeAttr: $nan_mode,
    SPIRV_TosaNumerical_TensorArm4D: $input
  );

  let results = (outs
    SPIRV_TosaNumerical_TensorArm4D: $output
  );

  let assemblyFormat = [{
    `kernel` `=` custom<SPIRV_I32_1DArmTensor>($kernel) `,`
    `stride` `=` custom<SPIRV_I32_1DArmTensor>($stride) `,`
    `pad` `=` custom<SPIRV_I32_1DArmTensor>($pad) `,`
    `nan_mode` `=` $nan_mode `,`
    $input
    attr-dict `:` type(operands) `->` type(results)
  }];

  let extraClassDeclaration = extraBaseClassDeclaration#[{
    ::mlir::spirv::TensorArmType getInputType() {
      return cast<::mlir::spirv::TensorArmType>(getInput().getType());
    }
  }];
}


def SPIRV_TosaRFFT2DOp : SPIRV_TosaOpWithComplexResult<"RFFT2D", 8, [Pure]> {
  let summary = "Performs RFFT2D operation on the input.";

  let description = [{
    Performs a batched 2D real-valued Fast Fourier Transform over the input where
    the input tensor consists of real values producing complex valued output. The
    complex output values will be split into the output_real and output_imag
    tensor arguments. This operator takes advantage of Hermitian symmetry to only
    calculate the first half of the final output axis. Implementations may choose
    to skip calculation of the imaginary values at (0,0), (0,W/2), (H/2,0), and
    (H/2, W/2). If the calculation is skipped, the result at that location must be
    zero.

    References:
      * https://github.khronos.org/SPIRV-Registry/extended/TOSA.001000.1.html#_rfft2d
      * https://www.mlplatform.org/tosa/tosa_spec_1_0_1.html#_rfft2d

    #### Example:
    ```mlir
    %0 = spirv.Tosa.RFFT2D local_bound = false, %arg0 : !spirv.arm.tensor<1x32x32xf32> -> !spirv.struct<(!spirv.arm.tensor<1x32x17xf32>, !spirv.arm.tensor<1x32x17xf32>)>
    %1 = spirv.CompositeExtract %0[0 : i32] : !spirv.struct<(!spirv.arm.tensor<1x32x17xf32>, !spirv.arm.tensor<1x32x17xf32>)>
    %2 = spirv.CompositeExtract %0[1 : i32] : !spirv.struct<(!spirv.arm.tensor<1x32x17xf32>, !spirv.arm.tensor<1x32x17xf32>)>
    ```
  }];

  let arguments = (ins
    SPIRV_BoolConstAttr: $local_bound,
    SPIRV_Float32_TensorArm3D: $input_real
  );

  let results = (outs
    SPIRV_Struct_2_Float32_TensorArm3D: $output
  );

  let assemblyFormat = [{
    `local_bound` `=` $local_bound `,`
    $input_real
    attr-dict `:` type(operands) `->` type(results)
  }];

  let extraClassDeclaration = extraBaseClassDeclaration#[{
    ::mlir::spirv::TensorArmType getInputRealType() {
      return cast<::mlir::spirv::TensorArmType>(getInputReal().getType());
    }
  }];
}


def SPIRV_TosaTransposeConv2DOp : SPIRV_TosaConvolutionOp<"TransposeConv2D", 9> {
  let summary = "Transpose 2D Convolution operator.";

  let description = [{
    Performs a 2D transposed convolution over the given tensor input, using the
    weights tensor. Implementations may choose to skip calculation of multiplies
    by zero at fractional input positions.

    Input and weight have respective zero point values provided in input_zp and weight_zp.

    The behaviour is undefined if the accumulated result overflows or underflows.

    References:
      * https://github.khronos.org/SPIRV-Registry/extended/TOSA.001000.1.html#_transpose_conv2d
      * https://www.mlplatform.org/tosa/tosa_spec_1_0_1.html#_transpose_conv2d

    #### Example:
    ```mlir
    %6 = spirv.Tosa.TransposeConv2D out_pad = [0, 0, 0, 0], stride = [1, 1], acc_type = <INT48>, local_bound = false, %arg0, %arg1, %arg2, %4, %5 : !spirv.arm.tensor<1x13x33x3xi16>, !spirv.arm.tensor<11x1x3x3xi8>, !spirv.arm.tensor<1xi64>, !spirv.arm.tensor<1xi16>, !spirv.arm.tensor<1xi8> -> !spirv.arm.tensor<1x13x35x11xi64>
    %6 = spirv.Tosa.TransposeConv2D out_pad = [0, 1, 0, 0], stride = [1, 8], acc_type = <FP16>, local_bound = true, %arg0, %arg1, %arg2, %4, %5 : !spirv.arm.tensor<10x24x9x13xf16>, !spirv.arm.tensor<14x1x1x13xf16>, !spirv.arm.tensor<14xf16>, !spirv.arm.tensor<1xf16>, !spirv.arm.tensor<1xf16> -> !spirv.arm.tensor<10x25x65x14xf16>
    ```
  }];

  let arguments = (ins
    SPIRV_Int32_1DTensorArmOfLength4Attr: $out_pad,
    SPIRV_Int32_1DTensorArmOfLength2Attr: $stride,
    SPIRV_TosaExtAccTypeAttr: $acc_type,
    SPIRV_BoolConstAttr: $local_bound,
    SPIRV_TosaNumerical_TensorArm4D: $input,
    SPIRV_TosaNumerical_TensorArm4D: $weight,
    SPIRV_TosaNumerical_TensorArm1D: $bias,
    SPIRV_TosaNumerical_1DTensorArmOfLength1: $input_zp,
    SPIRV_TosaNumerical_1DTensorArmOfLength1: $weight_zp
  );

  let results = (outs
    SPIRV_TosaNumerical_TensorArm4D: $output
  );

  let assemblyFormat = [{
    `out_pad` `=` custom<SPIRV_I32_1DArmTensor>($out_pad) `,`
    `stride` `=` custom<SPIRV_I32_1DArmTensor>($stride) `,`
    `acc_type` `=` $acc_type `,`
    `local_bound` `=` $local_bound `,`
    $input `,`
    $weight `,`
    $bias `,`
    $input_zp `,`
    $weight_zp
    attr-dict `:` type(operands) `->` type(results)
  }];
}


def SPIRV_TosaClampOp : SPIRV_TosaOpWithResult<"Clamp", 10, [Pure,
  AllTypesMatch<["input", "output"]>,
  AllElementTypesMatch<["input", "output", "min_val", "max_val"]>]> {
  let summary = "Computes Clamp(min, max).";

  let description = [{
    Clamp to an arbitrary minimum and maximum value. Maximum and minimum values
    are specified as values in the range of the input type. Integer element
    types are always interpreted as signed.
    No zero point subtraction is done to the values, thus to clamp to the zero
    point value, the zero point itself should be supplied as the minimum value.
    NaN Propagation Mode is ignored for inputs using integer element types.

    References:
      * https://github.khronos.org/SPIRV-Registry/extended/TOSA.001000.1.html#_clamp
      * https://www.mlplatform.org/tosa/tosa_spec_1_0_1.html#_clamp

    #### Example:
    ```mlir
    %3 = spirv.Tosa.Clamp min_val = -102 : i8, max_val = -100 : i8, nan_mode = <Propagate>, %arg0 : !spirv.arm.tensor<27x44x55xi8> -> !spirv.arm.tensor<27x44x55xi8>
    %3 = spirv.Tosa.Clamp min_val = -1.19339396E+38 : f32, max_val = 2.38255944E+38 : f32, nan_mode = <Ignore>, %arg0 : !spirv.arm.tensor<18x5x17x6xf32> -> !spirv.arm.tensor<18x5x17x6xf32>
    ```
  }];

  let arguments = (ins
    SPIRV_TosaNumericalAttr: $min_val,
    SPIRV_TosaNumericalAttr: $max_val,
    SPIRV_TosaExtNaNPropagationModeAttr: $nan_mode,
    SPIRV_TosaNumerical_TensorArm: $input
  );

  let results = (outs
    SPIRV_TosaNumerical_TensorArm: $output
  );

  let assemblyFormat = [{
    `min_val` `=` $min_val `,`
    `max_val` `=` $max_val `,`
    `nan_mode` `=` $nan_mode `,`
    $input
    attr-dict `:` type(operands) `->` type(results)
  }];

  let extraClassDeclaration = extraBaseClassDeclaration#[{
    ::mlir::spirv::TensorArmType getInputType() {
      return cast<::mlir::spirv::TensorArmType>(getInput().getType());
    }
  }];
}


def SPIRV_TosaErfOp : SPIRV_TosaOpWithResult<"Erf", 11, [Pure,
  AllTypesMatch<["input", "output"]>]> {
  let summary = "Gauss Error Function.";

  let description = [{
    Gauss Error Function: $ erf(x) = \frac{2}{\sqrt{\pi}} \int_{0}^{x} e^{-t^2} dt $
    For quantized integer data types, the `spirv.Tosa.Table` operator should be used instead.

    References:
      * https://github.khronos.org/SPIRV-Registry/extended/TOSA.001000.1.html#_erf
      * https://www.mlplatform.org/tosa/tosa_spec_1_0_1.html#_erf

    #### Example:
    ```mlir
    %0 = spirv.Tosa.Erf %arg0 : !spirv.arm.tensor<47x38x51xf32> -> !spirv.arm.tensor<47x38x51xf32>
    ```
  }];

  let arguments = (ins
    SPIRV_TosaFloat_TensorArm: $input
  );

  let results = (outs
    SPIRV_TosaFloat_TensorArm: $output
  );

  let assemblyFormat = [{
    $input
    attr-dict `:` type(operands) `->` type(results)
  }];

  let extraClassDeclaration = extraBaseClassDeclaration#[{
    ::mlir::spirv::TensorArmType getInputType() {
      return cast<::mlir::spirv::TensorArmType>(getInput().getType());
    }
  }];
}


def SPIRV_TosaSigmoidOp : SPIRV_TosaOpWithResult<"Sigmoid", 12, [Pure,
  AllTypesMatch<["input", "output"]>]> {
  let summary = "Sigmoid operator.";

  let description = [{
    Applies the sigmoid logistic function to each element of the input tensor:
    $ sigmoid(x) = \frac{1}{1 + e^{-x}} $.

    For quantized integer data types, the `spirv.Tosa.Table` operator should be used instead.

    References:
      * https://github.khronos.org/SPIRV-Registry/extended/TOSA.001000.1.html#_sigmoid
      * https://www.mlplatform.org/tosa/tosa_spec_1_0_1.html#_sigmoid

    #### Example:
    ```mlir
    %0 = spirv.Tosa.Sigmoid %arg0 : !spirv.arm.tensor<28x43x45xf32> -> !spirv.arm.tensor<28x43x45xf32>
    ```
  }];

  let arguments = (ins
    SPIRV_TosaFloat_TensorArm: $input
  );

  let results = (outs
    SPIRV_TosaFloat_TensorArm: $output
  );

  let assemblyFormat = [{
    $input
    attr-dict `:` type(operands) `->` type(results)
  }];

  let extraClassDeclaration = extraBaseClassDeclaration#[{
    ::mlir::spirv::TensorArmType getInputType() {
      return cast<::mlir::spirv::TensorArmType>(getInput().getType());
    }
  }];
}


def SPIRV_TosaTanhOp : SPIRV_TosaOpWithResult<"Tanh", 13, [Pure,
  AllTypesMatch<["input", "output"]>]> {
  let summary = "Hyperbolic Tangent operator.";

  let description = [{
    Elementwise Parameterized Hyperbolic Tangent: $ tanh(x) = \frac{1 - e^{-2x}}{1 + e^{-2x}} $.

    For quantized integer data types, the `spirv.Tosa.Table` operator should be used instead.

    References:
      * https://github.khronos.org/SPIRV-Registry/extended/TOSA.001000.1.html#_tanh
      * https://www.mlplatform.org/tosa/tosa_spec_1_0_1.html#_tanh

    #### Example:
    ```mlir
    %0 = spirv.Tosa.Tanh %arg0 : !spirv.arm.tensor<46x50x36xf16> -> !spirv.arm.tensor<46x50x36xf16>
    ```
  }];

  let arguments = (ins
    SPIRV_TosaFloat_TensorArm: $input
  );

  let results = (outs
    SPIRV_TosaFloat_TensorArm: $output
  );

  let assemblyFormat = [{
    $input
    attr-dict `:` type(operands) `->` type(results)
  }];

  let extraClassDeclaration = extraBaseClassDeclaration#[{
    ::mlir::spirv::TensorArmType getInputType() {
      return cast<::mlir::spirv::TensorArmType>(getInput().getType());
    }
  }];
}


def SPIRV_TosaAddOp : SPIRV_TosaElementwiseBinaryOp<"Add", 14, [NoMemoryEffect]> {
  let summary = "Addition operator.";

  let description = [{
    Elementwise Addition of input1 and input2. Axis of size 1 will be broadcast,
    as necessary. Rank of input tensors must match. Integer element types are
    always interpreted as signed. The behavior is undefined if the integer
    addition overflows or underflows the signed integer range.

    References:
      * https://github.khronos.org/SPIRV-Registry/extended/TOSA.001000.1.html#_add
      * https://www.mlplatform.org/tosa/tosa_spec_1_0_1.html#_add

    #### Example:
    ```mlir
    %0 = spirv.Tosa.Add %arg0, %arg1 : !spirv.arm.tensor<4x7x3x10xi32>, !spirv.arm.tensor<4x7x3x1xi32> -> !spirv.arm.tensor<4x7x3x10xi32>
    %0 = spirv.Tosa.Add %arg0, %arg1 : !spirv.arm.tensor<26x37x18xf16>, !spirv.arm.tensor<1x37x18xf16> -> !spirv.arm.tensor<26x37x18xf16>
    ```
  }];

  let arguments = (ins
    SPIRV_TosaNumerical_TensorArm: $input1,
    SPIRV_TosaNumerical_TensorArm: $input2
  );

  let results = (outs
    SPIRV_TosaNumerical_TensorArm: $output
  );

  let assemblyFormat = [{
    $input1 `,`
    $input2
    attr-dict `:` type(operands) `->` type(results)
  }];
}


def SPIRV_TosaArithmeticRightShiftOp : SPIRV_TosaElementwiseBinaryOp<"ArithmeticRightShift", 15, [NoMemoryEffect]> {
  let summary = "Arithmetic Right Shift.";

  let description = [{
    Elementwise Arithmetic Right Shift of input1 by the amount specified in
    input2. Axis of size 1 will be broadcast, as necessary. Rank of input
    tensors must match. Integer element types are always interpreted as signed.
    The behavior is undefined if the shift value is negative or greater or
    equal to the bitwidth of the element type.

    References:
      * https://github.khronos.org/SPIRV-Registry/extended/TOSA.001000.1.html#_arithmetic_right_shift
      * https://www.mlplatform.org/tosa/tosa_spec_1_0_1.html#_arithmetic_right_shift

    #### Example:
    ```mlir
    %1 = spirv.Tosa.ArithmeticRightShift round = true, %arg0, %arg1 : !spirv.arm.tensor<1x47x22xi16>, !spirv.arm.tensor<49x47x22xi16> -> !spirv.arm.tensor<49x47x22xi16>
    ```
  }];

  let arguments = (ins
    SPIRV_BoolConstAttr: $round,
    SPIRV_TosaInteger_TensorArm: $input1,
    SPIRV_TosaInteger_TensorArm: $input2
  );

  let results = (outs
    SPIRV_TosaInteger_TensorArm: $output
  );

  let assemblyFormat = [{
    `round` `=` $round `,`
    $input1 `,`
    $input2
    attr-dict `:` type(operands) `->` type(results)
  }];
}


def SPIRV_TosaBitwiseAndOp : SPIRV_TosaIntegerElementwiseBinaryOp<"BitwiseAnd", 16, [Pure]> {
  let summary = "Bitwise AND operator.";

  let description = [{
    Elementwise Bitwise AND of input1 and input2. Axis of size 1 will be
    broadcast as necessary. Rank of input tensors must match. Integer element
    types are always interpreted as signed.

    References:
      * https://github.khronos.org/SPIRV-Registry/extended/TOSA.001000.1.html#_bitwise_and
      * https://www.mlplatform.org/tosa/tosa_spec_1_0_1.html#_bitwise_and

    #### Example:
    ```mlir
    %0 = spirv.Tosa.BitwiseAnd %arg0, %arg1 : !spirv.arm.tensor<4x1x7x12xi16>, !spirv.arm.tensor<4x13x7x12xi16> -> !spirv.arm.tensor<4x13x7x12xi16>
    ```
  }];
}


def SPIRV_TosaBitwiseOrOp : SPIRV_TosaIntegerElementwiseBinaryOp<"BitwiseOr", 17, [Pure]> {
  let summary = "Bitwise OR operator.";

  let description = [{
    Elementwise Bitwise OR of input1 and input2. Axis of size 1 will be
    broadcast as necessary. Rank of input tensors must match. Integer element
    types are always interpreted as signed.

    References:
      * https://github.khronos.org/SPIRV-Registry/extended/TOSA.001000.1.html#_bitwise_or
      * https://www.mlplatform.org/tosa/tosa_spec_1_0_1.html#_bitwise_or

    #### Example:
    ```mlir
    %0 = spirv.Tosa.BitwiseOr %arg0, %arg1 : !spirv.arm.tensor<11x30x23xi32>, !spirv.arm.tensor<1x30x23xi32> -> !spirv.arm.tensor<11x30x23xi32>
    ```
  }];
}


def SPIRV_TosaBitwiseXorOp : SPIRV_TosaIntegerElementwiseBinaryOp<"BitwiseXor", 18, [Pure]> {
  let summary = "Bitwise XOR operator.";

  let description = [{
    Elementwise Bitwise XOR of input1 and input2. Axis of size 1 will be
    broadcast as necessary. Rank of input tensors must match. Integer element
    types are always interpreted as signed.

    References:
      * https://github.khronos.org/SPIRV-Registry/extended/TOSA.001000.1.html#_bitwise_xor
      * https://www.mlplatform.org/tosa/tosa_spec_1_0_1.html#_bitwise_xor

    #### Example:
    ```mlir
    %0 = spirv.Tosa.BitwiseXor %arg0, %arg1 : !spirv.arm.tensor<4x8x13x9xi16>, !spirv.arm.tensor<4x8x1x9xi16> -> !spirv.arm.tensor<4x8x13x9xi16>
    ```
  }];
}


def SPIRV_TosaIntDivOp : SPIRV_TosaElementwiseBinaryOp<"IntDiv", 19, [NoMemoryEffect]> {
  let summary = "Integer Divide operator.";

  let description = [{
    Elementwise Integer Divide of input1 by input2. Axis of size 1 will be
    broadcast as necessary. Rank of input tensors must match.  Integer element
    types are always interpreted as signed. The behavior is undefined if the
    divisor value is equal to zero and if the divisor value is min_value<si32>
    and the dividend is -1.

    The result of the divide is truncated towards zero. Expected use is for
    operations on non-scaled integers. Floating point divide should use
    `spirv.Tosa.Reciprocal` and `spirv.Tosa.Mul`. Quantized integer divide
    should use `spirv.Tosa.Table`(for $ 1/x $) and `spirv.Tosa.Mul`.

    References:
      * https://github.khronos.org/SPIRV-Registry/extended/TOSA.001000.1.html#_intdiv
      * https://www.mlplatform.org/tosa/tosa_spec_1_0_1.html#_intdiv

    #### Example:
    ```mlir
    %0 = spirv.Tosa.IntDiv %arg0, %arg1 : !spirv.arm.tensor<1x65533x1xi32>, !spirv.arm.tensor<2x65533x1xi32> -> !spirv.arm.tensor<2x65533x1xi32>
    ```
  }];

  let arguments = (ins
    SPIRV_Int32_TensorArm: $input1,
    SPIRV_Int32_TensorArm: $input2
  );

  let results = (outs
    SPIRV_Int32_TensorArm: $output
  );

  let assemblyFormat = [{
    $input1 `,`
    $input2
    attr-dict `:` type(operands) `->` type(results)
  }];
}


def SPIRV_TosaLogicalAndOp : SPIRV_TosaBoolElementwiseBinaryOp<"LogicalAnd", 20> {
  let summary = "Logical AND operator.";

  let description = [{
    Elementwise Logical AND of input1 and input2. Axis of size 1 will be
    broadcast, as necessary. Rank of input tensors must match.

    References:
      * https://github.khronos.org/SPIRV-Registry/extended/TOSA.001000.1.html#_logical_and
      * https://www.mlplatform.org/tosa/tosa_spec_1_0_1.html#_logical_and

    #### Example:
    ```mlir
    %0 = spirv.Tosa.LogicalAnd %arg0, %arg1 : !spirv.arm.tensor<2x1x7x11xi1>, !spirv.arm.tensor<2x4x7x11xi1> -> !spirv.arm.tensor<2x4x7x11xi1>
    ```
  }];
}


def SPIRV_TosaLogicalLeftShiftOp : SPIRV_TosaIntegerElementwiseBinaryOp<"LogicalLeftShift", 21, [NoMemoryEffect]> {
  let summary = "Logical Left Shift operator.";

  let description = [{
    Elementwise Logical Left Shift of input1 by the amount specified in input2.
    Axis of size 1 will be broadcast, as necessary. Rank of input tensors
    must match.  Integer element types are always interpreted as signed. The
    behavior is undefined if the shift value is negative or greater or equal to
    the bitwidth of the element type.

    References:
      * https://github.khronos.org/SPIRV-Registry/extended/TOSA.001000.1.html#_logical_left_shift
      * https://www.mlplatform.org/tosa/tosa_spec_1_0_1.html#_logical_left_shift

    #### Example:
    ```mlir
    %0 = spirv.Tosa.LogicalLeftShift %arg0, %arg1 : !spirv.arm.tensor<7x1x11x4xi8>, !spirv.arm.tensor<7x8x11x4xi8> -> !spirv.arm.tensor<7x8x11x4xi8>
    ```
  }];
}


def SPIRV_TosaLogicalRightShiftOp : SPIRV_TosaIntegerElementwiseBinaryOp<"LogicalRightShift", 22, [NoMemoryEffect]> {
  let summary = "Logical Right Shift operator.";

  let description = [{
    Elementwise Logical Right Shift of input1 by the amount specified in input2.
    Axis of size 1 will be broadcast, as necessary. Rank of input tensors must
    match.  Integer element types are always interpreted as signed. The
    behavior is undefined if the shift value is negative or greater or equal to
    the bitwidth of the element type.

    References:
      * https://github.khronos.org/SPIRV-Registry/extended/TOSA.001000.1.html#_logical_right_shift
      * https://www.mlplatform.org/tosa/tosa_spec_1_0_1.html#_logical_right_shift

    #### Example:
    ```mlir
    %0 = spirv.Tosa.LogicalRightShift %arg0, %arg1 : !spirv.arm.tensor<6x13x1x19xi8>, !spirv.arm.tensor<6x13x6x19xi8> -> !spirv.arm.tensor<6x13x6x19xi8>
    ```
  }];
}


def SPIRV_TosaLogicalOrOp : SPIRV_TosaBoolElementwiseBinaryOp<"LogicalOr", 23> {
  let summary = "Logical OR operator.";

  let description = [{
    Elementwise logical OR of input1 and input2. Axis of size 1 will be
    broadcast as necessary. Rank of input tensors must match.

    References:
      * https://github.khronos.org/SPIRV-Registry/extended/TOSA.001000.1.html#_logical_or
      * https://www.mlplatform.org/tosa/tosa_spec_1_0_1.html#_logical_or

    #### Example:
    ```mlir
    %0 = spirv.Tosa.LogicalOr %arg0, %arg1 : !spirv.arm.tensor<3x6x12x5xi1>, !spirv.arm.tensor<3x6x1x5xi1> -> !spirv.arm.tensor<3x6x12x5xi1>
    ```
  }];
}


def SPIRV_TosaLogicalXorOp : SPIRV_TosaBoolElementwiseBinaryOp<"LogicalXor", 24> {
  let summary = "Logical XOR operator.";

  let description = [{
    Elementwise logical XOR of input1 and input2. Axis of size 1 will be
    broadcast as necessary. Rank of input tensors must match.

    References:
      * https://github.khronos.org/SPIRV-Registry/extended/TOSA.001000.1.html#_logical_xor
      * https://www.mlplatform.org/tosa/tosa_spec_1_0_1.html#_logical_xor

    #### Example:
    ```mlir
    %0 = spirv.Tosa.LogicalXor %arg0, %arg1 : !spirv.arm.tensor<11x4x9x12xi1>, !spirv.arm.tensor<11x4x9x1xi1> -> !spirv.arm.tensor<11x4x9x12xi1>
    ```
  }];
}


def SPIRV_TosaMaximumOp : SPIRV_TosaElementwiseBinaryOp<"Maximum", 25, [Pure,
  TypeConstraintImplicationOn<"input1", AnyInteger, "input1", [I32]>,
  TypeConstraintImplicationOn<"input2", AnyInteger, "input2", [I32]>,
  TypeConstraintImplicationOn<"output", AnyInteger, "output", [I32]>]> {
  let summary = "Maximum.";

  let description = [{
    Elementwise maximum of input1 and input2. Axis of size 1 will be broadcast,
    as necessary. Rank of input tensors must match. Integer element types are
    always interpreted as signed.
    NaN Propagation Mode is ignored for inputs using integer element types.

    References:
      * https://github.khronos.org/SPIRV-Registry/extended/TOSA.001000.1.html#_maximum
      * https://www.mlplatform.org/tosa/tosa_spec_1_0_1.html#_maximum

    #### Example:
    ```mlir
    %1 = spirv.Tosa.Maximum nan_mode = <Propagate>, %arg0, %arg1 : !spirv.arm.tensor<1x2x65533x1xi32>, !spirv.arm.tensor<1x2x65533x2xi32> -> !spirv.arm.tensor<1x2x65533x2xi32>
    %1 = spirv.Tosa.Maximum nan_mode = <Ignore>, %arg0, %arg1 : !spirv.arm.tensor<1x12x14x7xf16>, !spirv.arm.tensor<11x12x14x7xf16> -> !spirv.arm.tensor<11x12x14x7xf16>
    ```
  }];

  let arguments = (ins
    SPIRV_TosaExtNaNPropagationModeAttr: $nan_mode,
    SPIRV_TosaNumerical_TensorArm: $input1,
    SPIRV_TosaNumerical_TensorArm: $input2
  );

  let results = (outs
    SPIRV_TosaNumerical_TensorArm: $output
  );

  let assemblyFormat = [{
    `nan_mode` `=` $nan_mode `,`
    $input1 `,`
    $input2
    attr-dict `:` type(operands) `->` type(results)
  }];
}


def SPIRV_TosaMinimumOp : SPIRV_TosaElementwiseBinaryOp<"Minimum", 26, [Pure,
  TypeConstraintImplicationOn<"input1", AnyInteger, "input1", [I32]>,
  TypeConstraintImplicationOn<"input2", AnyInteger, "input2", [I32]>,
  TypeConstraintImplicationOn<"output", AnyInteger, "output", [I32]>]> {
  let summary = "Minimum.";

  let description = [{
    Elementwise minimum of input1 and input2. Axis of size 1 will be broadcast,
    as necessary. Rank of input tensors must match. Integer element types are
    always interpreted as signed.
    NaN Propagation Mode is ignored for inputs using integer element types.

    References:
      * https://github.khronos.org/SPIRV-Registry/extended/TOSA.001000.1.html#_minimum
      * https://www.mlplatform.org/tosa/tosa_spec_1_0_1.html#_minimum

    #### Example:
    ```mlir
    %1 = spirv.Tosa.Minimum nan_mode = <Propagate>, %arg0, %arg1 : !spirv.arm.tensor<15x2x10x11xi32>, !spirv.arm.tensor<15x1x10x11xi32> -> !spirv.arm.tensor<15x2x10x11xi32>
    %1 = spirv.Tosa.Minimum nan_mode = <Propagate>, %arg0, %arg1 : !spirv.arm.tensor<1x65531x2x1xf32>, !spirv.arm.tensor<1x1x2x1xf32> -> !spirv.arm.tensor<1x65531x2x1xf32>
    ```
  }];

  let arguments = (ins
    SPIRV_TosaExtNaNPropagationModeAttr: $nan_mode,
    SPIRV_TosaNumerical_TensorArm: $input1,
    SPIRV_TosaNumerical_TensorArm: $input2
  );

  let results = (outs
    SPIRV_TosaNumerical_TensorArm: $output
  );

  let assemblyFormat = [{
    `nan_mode` `=` $nan_mode `,`
    $input1 `,`
    $input2
    attr-dict `:` type(operands) `->` type(results)
  }];
}


def SPIRV_TosaMulOp : SPIRV_TosaBinaryOp<"Mul", 27, [NoMemoryEffect,
  AllElementTypesMatch<["input1", "input2"]>,
  AllRanksMatch<["input1", "input2", "output"]>,
  TypeConstraintImplicationOn<"input1", F16, "output", [F16]>,
  TypeConstraintImplicationOn<"input1", F32, "output", [F32]>,
  TypeConstraintImplicationOn<"input1", BF16, "output", [BF16]>,
  TypeConstraintImplicationOn<"input1", AnyInteger, "output", [I32]>]> {
  let summary = "Multiplication operator.";

  let description = [{
    Elementwise Multiplication (Hadamard product) of input1 and input2.
    Axis of size 1 will be broadcast, as necessary. Rank of input tensors must
    match.  Integer element types are always interpreted as signed.
    The shift value is used to right shift the result (rounded up) and
    it is used only if input element type is i32 and it must be zero for the
    other element types. The behavior is undefined if the integer
    multiplication overflows or underflows the signed integer range or the
    shift value exceeds the bitwidth of i32.

    References:
      * https://github.khronos.org/SPIRV-Registry/extended/TOSA.001000.1.html#_mul
      * https://www.mlplatform.org/tosa/tosa_spec_1_0_1.html#_mul

    #### Example:
    ```mlir
    %1 = spirv.Tosa.Mul %arg0, %arg1, %0 : !spirv.arm.tensor<34x21x39xi32>, !spirv.arm.tensor<34x21x1xi32>, !spirv.arm.tensor<1xi8> -> !spirv.arm.tensor<34x21x39xi32>
    %1 = spirv.Tosa.Mul %arg0, %arg1, %0 : !spirv.arm.tensor<57x1x55xf16>, !spirv.arm.tensor<57x37x55xf16>, !spirv.arm.tensor<1xi8> -> !spirv.arm.tensor<57x37x55xf16>
    ```
  }];

  let arguments = (ins
    SPIRV_TosaNumerical_TensorArm: $input1,
    SPIRV_TosaNumerical_TensorArm: $input2,
    SPIRV_Int8_1DTensorArmOfLength1: $shift
  );

  let results = (outs
    SPIRV_TosaNumerical_TensorArm: $output
  );

  let assemblyFormat = [{
    $input1 `,`
    $input2 `,`
    $shift
    attr-dict `:` type(operands) `->` type(results)
  }];
}


def SPIRV_TosaPowOp : SPIRV_TosaElementwiseBinaryOp<"Pow", 28, [NoMemoryEffect]> {
  let summary = "Power operator.";

  let description = [{
    Elementwise input1 value raised to the Power of input2.
    Axis of size 1 will be broadcast, as necessary. Rank of input tensors must
    match. The behavior is undefined if the the input1 value is negative, or
    both input values are zero or negative, or both input values are NaN or
    infinite.

    References:
      * https://github.khronos.org/SPIRV-Registry/extended/TOSA.001000.1.html#_pow
      * https://www.mlplatform.org/tosa/tosa_spec_1_0_1.html#_pow

    #### Example:
    ```mlir
    %0 = spirv.Tosa.Pow %arg0, %arg1 : !spirv.arm.tensor<1x52x53xf16>, !spirv.arm.tensor<44x52x53xf16> -> !spirv.arm.tensor<44x52x53xf16>
    ```
  }];

  let arguments = (ins
    SPIRV_TosaFloat_TensorArm: $input1,
    SPIRV_TosaFloat_TensorArm: $input2
  );

  let results = (outs
    SPIRV_TosaFloat_TensorArm: $output
  );

  let assemblyFormat = [{
    $input1 `,`
    $input2
    attr-dict `:` type(operands) `->` type(results)
  }];
}


def SPIRV_TosaSubOp : SPIRV_TosaElementwiseBinaryOp<"Sub", 29, [NoMemoryEffect,
  TypeConstraintImplicationOn<"input1", AnyInteger, "input1", [I32]>,
  TypeConstraintImplicationOn<"input2", AnyInteger, "input2", [I32]>,
  TypeConstraintImplicationOn<"output", AnyInteger, "output", [I32]>]> {
  let summary = "Subtraction operator.";

  let description = [{
    Elementwise Subtraction of input1 and input2. Axis of size 1 will be
    broadcast as necessary. Rank of input tensors must match. Integer element
    types are always interpreted as signed. The behavior is undefined if the
    integer subtraction overflows or underflows the signed integer range.

    References:
      * https://github.khronos.org/SPIRV-Registry/extended/TOSA.001000.1.html#_sub
      * https://www.mlplatform.org/tosa/tosa_spec_1_0_1.html#_sub

    #### Example:
    ```mlir
    %0 = spirv.Tosa.Sub %arg0, %arg1 : !spirv.arm.tensor<6x10x6x6xi32>, !spirv.arm.tensor<1x10x6x6xi32> -> !spirv.arm.tensor<6x10x6x6xi32>
    %0 = spirv.Tosa.Sub %arg0, %arg1 : !spirv.arm.tensor<1x10x13x12xf16>, !spirv.arm.tensor<6x10x13x12xf16> -> !spirv.arm.tensor<6x10x13x12xf16>
    ```
  }];

  let arguments = (ins
    SPIRV_TosaNumerical_TensorArm: $input1,
    SPIRV_TosaNumerical_TensorArm: $input2
  );

  let results = (outs
    SPIRV_TosaNumerical_TensorArm: $output
  );

  let assemblyFormat = [{
    $input1 `,`
    $input2
    attr-dict `:` type(operands) `->` type(results)
  }];
}


def SPIRV_TosaTableOp : SPIRV_TosaOpWithResult<"Table", 30, [NoMemoryEffect,
  AllElementTypesMatch<["input1", "table"]>,
  AllShapesMatch<["input1", "output"]>,
  TypeConstraintImplicationOn<"input1", I8, "output", [I8]>,
  TypeConstraintImplicationOn<"input1", I16, "output", [I32]>,
  TableSizeConstraint<"input1", I8, 256>,
  TableSizeConstraint<"input1", I16, 513>]> {
  let summary = "Table lookup operator.";

  let description = [{
    Table lookup operation. For int8_t, perform a 256 entry table lookup
    returning an int8_t value. For int16_t tables, the int16_t input is treated
    as a fixed-point 9.7 value. The most significant 9 bits are used to index
    into the table. The fractional 7 bits are used to interpolate based on
    table[index] and table[index+1]. The behavior is undefined if during the
    interpolation step, the integer subtraction table[index+1] - table[index]
    overflows or underflows the signed int16 range.

    For int16_t inputs, this operator returns a 16.7 interpolated value in an
    int32_t. This value can then be input to the `spirv.Tosa.Rescale` operator
    to scale to the required output data type. Note that int16_t table has 513
    values to handle table[index+1] when index=511.

    An int16_t to int16_t table lookup can be constructed as follows:
    * Use the table operator to produce a fixed point 16.7 interpolated result
    * Use `spirv.Tosa.Rescale` (in_t=int32_t, out_t=int16_t, scale=1<<14, shift=21)
      to scale the output to int16_t range (or alternate scale as required)

    References:
      * https://github.khronos.org/SPIRV-Registry/extended/TOSA.001000.1.html#_table
      * https://www.mlplatform.org/tosa/tosa_spec_1_0_1.html#_table

    #### Example:
    ```mlir
    %1 = spirv.Tosa.Table %arg0, %0 : !spirv.arm.tensor<3x2x15x7xi8>, !spirv.arm.tensor<256xi8> -> !spirv.arm.tensor<3x2x15x7xi8>
    ```
  }];

  let arguments = (ins
    SPIRV_TosaInteger_TensorArm: $input1,
    SPIRV_TosaInteger_TensorArm1D: $table
  );

  let results = (outs
    SPIRV_TosaInteger_TensorArm: $output
  );

  let assemblyFormat = [{
    $input1 `,`
    $table
    attr-dict `:` type(operands) `->` type(results)
  }];

  let extraClassDeclaration = extraBaseClassDeclaration#[{
    ::mlir::spirv::TensorArmType getInput1Type() {
      return cast<::mlir::spirv::TensorArmType>(getInput1().getType());
    }
  }];
}


def SPIRV_TosaAbsOp : SPIRV_TosaElementwiseUnaryOp<"Abs", 31, [NoMemoryEffect]> {
  let summary = "Absolute value operation.";

  let description = [{
    Elementwise Absolute value operation. Input and Output types must match.
    Integer element types are always interpreted as signed. The behaviour is
    undefined if the input value is min_value<si32>.

    References:
      * https://github.khronos.org/SPIRV-Registry/extended/TOSA.001000.1.html#_abs
      * https://www.mlplatform.org/tosa/tosa_spec_1_0_1.html#_abs

    #### Example:
    ```mlir
    %0 = spirv.Tosa.Abs %arg0 : !spirv.arm.tensor<5x1x4x4xi32> -> !spirv.arm.tensor<5x1x4x4xi32>
    %0 = spirv.Tosa.Abs %arg0 : !spirv.arm.tensor<3x6x14x8xf16> -> !spirv.arm.tensor<3x6x14x8xf16>
    ```
  }];

  let arguments = (ins
    SPIRV_TosaNumerical_TensorArm: $input1
  );

  let results = (outs
    SPIRV_TosaNumerical_TensorArm: $output
  );

  let assemblyFormat = [{
    $input1
    attr-dict `:` type(operands) `->` type(results)
  }];
}


def SPIRV_TosaBitwiseNotOp : SPIRV_TosaElementwiseUnaryOp<"BitwiseNot", 32, [Pure]> {
  let summary = "Bitwise NOT operator.";

  let description = [{
    Elementwise Bitwise NOT of input tensor. Input and Output types must match.

    References:
      * https://github.khronos.org/SPIRV-Registry/extended/TOSA.001000.1.html#_bitwise_not
      * https://www.mlplatform.org/tosa/tosa_spec_1_0_1.html#_bitwise_not

    #### Example:
    ```mlir
    %0 = spirv.Tosa.BitwiseNot %arg0 : !spirv.arm.tensor<12x56x50xi32> -> !spirv.arm.tensor<12x56x50xi32>
    ```
  }];

  let arguments = (ins
    SPIRV_TosaInteger_TensorArm: $input1
  );

  let results = (outs
    SPIRV_TosaInteger_TensorArm: $output
  );

  let assemblyFormat = [{
    $input1
    attr-dict `:` type(operands) `->` type(results)
  }];
}


def SPIRV_TosaCeilOp : SPIRV_TosaFloatElementwiseUnaryOp<"Ceil", 33> {
  let summary = "Ceil operator.";

  let description = [{
    Elementwise Ceiling operation. Input and Output types must match.

    References:
      * https://github.khronos.org/SPIRV-Registry/extended/TOSA.001000.1.html#_ceil
      * https://www.mlplatform.org/tosa/tosa_spec_1_0_1.html#_ceil

    #### Example:
    ```mlir
    %0 = spirv.Tosa.Ceil %arg0 : !spirv.arm.tensor<46x55x53xf16> -> !spirv.arm.tensor<46x55x53xf16>
    ```
  }];
}


def SPIRV_TosaClzOp : SPIRV_TosaElementwiseUnaryOp<"Clz", 34, [Pure]> {
  let summary = "Count Leading Zero operator.";

  let description = [{
    Elementwise Count Leading Zeros operation. Input and Output types must match.

    References:
      * https://github.khronos.org/SPIRV-Registry/extended/TOSA.001000.1.html#_clz
      * https://www.mlplatform.org/tosa/tosa_spec_1_0_1.html#_clz

    #### Example:
    ```mlir
    %0 = spirv.Tosa.Clz %arg0 : !spirv.arm.tensor<14x10x7x5xi32> -> !spirv.arm.tensor<14x10x7x5xi32>
    ```
  }];

  let arguments = (ins
    SPIRV_Int32_TensorArm: $input1
  );

  let results = (outs
    SPIRV_Int32_TensorArm: $output
  );

  let assemblyFormat = [{
    $input1
    attr-dict `:` type(operands) `->` type(results)
  }];
}


def SPIRV_TosaCosOp : SPIRV_TosaFloatElementwiseUnaryOp<"Cos", 35> {
  let summary = "Cosine operator.";

  let description = [{
    Elementwise Cosine operation for values given in radians. Input and Output types must match.

    References:
      * https://github.khronos.org/SPIRV-Registry/extended/TOSA.001000.1.html#_cos
      * https://www.mlplatform.org/tosa/tosa_spec_1_0_1.html#_cos

    #### Example:
    ```mlir
    %0 = spirv.Tosa.Cos %arg0 : !spirv.arm.tensor<44x49x51xf32> -> !spirv.arm.tensor<44x49x51xf32>
    ```
  }];
}


def SPIRV_TosaExpOp : SPIRV_TosaFloatElementwiseUnaryOp<"Exp", 36> {
  let summary = "Exp operator.";

  let description = [{
    Elementwise e to the power of x operation. Input and Output types must match.

    References:
      * https://github.khronos.org/SPIRV-Registry/extended/TOSA.001000.1.html#_exp
      * https://www.mlplatform.org/tosa/tosa_spec_1_0_1.html#_exp

    #### Example:
    ```mlir
    %0 = spirv.Tosa.Exp %arg0 : !spirv.arm.tensor<37x53x47xf32> -> !spirv.arm.tensor<37x53x47xf32>
    ```
  }];
}


def SPIRV_TosaFloorOp : SPIRV_TosaFloatElementwiseUnaryOp<"Floor", 37> {
  let summary = "Floor operator.";

  let description = [{
    Elementwise Floor operation. Input and Output types must match.

    References:
      * https://github.khronos.org/SPIRV-Registry/extended/TOSA.001000.1.html#_floor
      * https://www.mlplatform.org/tosa/tosa_spec_1_0_1.html#_floor

    #### Example:
    ```mlir
    %0 = spirv.Tosa.Floor %arg0 : !spirv.arm.tensor<40x52x42xf32> -> !spirv.arm.tensor<40x52x42xf32>
    ```
  }];
}


def SPIRV_TosaLogOp : SPIRV_TosaFloatElementwiseUnaryOp<"Log", 38> {
  let summary = "Log operator.";

  let description = [{
    Elementwise Natural Logarithm operation. Input and Output types must match.

    References:
      * https://github.khronos.org/SPIRV-Registry/extended/TOSA.001000.1.html#_log
      * https://www.mlplatform.org/tosa/tosa_spec_1_0_1.html#_log

    #### Example:
    ```mlir
    %0 = spirv.Tosa.Log %arg0 : !spirv.arm.tensor<45x43x36xf16> -> !spirv.arm.tensor<45x43x36xf16>
    ```
  }];
}


def SPIRV_TosaLogicalNotOp : SPIRV_TosaElementwiseUnaryOp<"LogicalNot", 39, [Pure]> {
  let summary = "Logical NOT operator.";

  let description = [{
    Elementwise Logical NOT of input. Input and Output types must match.

    References:
      * https://github.khronos.org/SPIRV-Registry/extended/TOSA.001000.1.html#_logical_not
      * https://www.mlplatform.org/tosa/tosa_spec_1_0_1.html#_logical_not

    #### Example:
    ```mlir
    %0 = spirv.Tosa.LogicalNot %arg0 : !spirv.arm.tensor<54x26x10xi1> -> !spirv.arm.tensor<54x26x10xi1>
    ```
  }];

  let arguments = (ins
    SPIRV_Bool_TensorArm: $input1
  );

  let results = (outs
    SPIRV_Bool_TensorArm: $output
  );

  let assemblyFormat = [{
    $input1
    attr-dict `:` type(operands) `->` type(results)
  }];
}


def SPIRV_TosaNegateOp : SPIRV_TosaElementwiseUnaryOp<"Negate", 40, [NoMemoryEffect,
  AllElementTypesMatch<["input1", "output", "input1_zp", "output_zp"]>]> {
  let summary = "Negate operator.";

  let description = [{
    Elementwise Negation operation. Input and Output types must match. Integer
    element types are always interpreted as signed. The behaviour is undefined
    if the input value is min_value<si32>.

    References:
      * https://github.khronos.org/SPIRV-Registry/extended/TOSA.001000.1.html#_negate
      * https://www.mlplatform.org/tosa/tosa_spec_1_0_1.html#_negate

    #### Example:
    ```mlir
    %2 = spirv.Tosa.Negate %arg0, %0, %1 : !spirv.arm.tensor<3x1x65540x1xi8>, !spirv.arm.tensor<1xi8>, !spirv.arm.tensor<1xi8> -> !spirv.arm.tensor<3x1x65540x1xi8>
    %2 = spirv.Tosa.Negate %arg0, %0, %1 : !spirv.arm.tensor<2x7x15x13xf16>, !spirv.arm.tensor<1xf16>, !spirv.arm.tensor<1xf16> -> !spirv.arm.tensor<2x7x15x13xf16>
    ```
  }];

  let arguments = (ins
    SPIRV_TosaNumerical_TensorArm: $input1,
    SPIRV_TosaNumerical_1DTensorArmOfLength1: $input1_zp,
    SPIRV_TosaNumerical_1DTensorArmOfLength1: $output_zp
  );

  let results = (outs
    SPIRV_TosaNumerical_TensorArm: $output
  );

  let assemblyFormat = [{
    $input1 `,`
    $input1_zp `,`
    $output_zp
    attr-dict `:` type(operands) `->` type(results)
  }];
}


def SPIRV_TosaReciprocalOp : SPIRV_TosaFloatElementwiseUnaryOp<"Reciprocal", 41> {
  let summary = "Reciprocal operator.";

  let description = [{
    Elementwise Reciprocal operation. Input and Output types must match. For
    integer operation, a table should be used with the appropriate ranges.

    References:
      * https://github.khronos.org/SPIRV-Registry/extended/TOSA.001000.1.html#_reciprocal
      * https://www.mlplatform.org/tosa/tosa_spec_1_0_1.html#_reciprocal

    #### Example:
    ```mlir
    %0 = spirv.Tosa.Reciprocal %arg0 : !spirv.arm.tensor<38x47x44xf32> -> !spirv.arm.tensor<38x47x44xf32>
    ```
  }];
}


def SPIRV_TosaRsqrtOp : SPIRV_TosaFloatElementwiseUnaryOp<"Rsqrt", 42> {
  let summary = "Reverse Square Root operator.";

  let description = [{
    Elementwise Reciprocal Square Root operation ($ 1/sqrt $). Input and
    Output types must match. For integer operation, a table should be used
    with the appropriate ranges.

    References:
      * https://github.khronos.org/SPIRV-Registry/extended/TOSA.001000.1.html#_rsqrt
      * https://www.mlplatform.org/tosa/tosa_spec_1_0_1.html#_rsqrt

    #### Example:
    ```mlir
    %0 = spirv.Tosa.Rsqrt %arg0 : !spirv.arm.tensor<40x57x56xf32> -> !spirv.arm.tensor<40x57x56xf32>
    ```
  }];
}


def SPIRV_TosaSinOp : SPIRV_TosaFloatElementwiseUnaryOp<"Sin", 43> {
  let summary = "Sin operator.";

  let description = [{
    Elementwise Sine operation for values given in radians. Input and Output types must match.

    References:
      * https://github.khronos.org/SPIRV-Registry/extended/TOSA.001000.1.html#_sin
      * https://www.mlplatform.org/tosa/tosa_spec_1_0_1.html#_sin

    #### Example:
    ```mlir
    %0 = spirv.Tosa.Sin %arg0 : !spirv.arm.tensor<49x38x58xf16> -> !spirv.arm.tensor<49x38x58xf16>
    ```
  }];
}


def SPIRV_TosaSelectOp : SPIRV_TosaOpWithResult<"Select", 44, [Pure,
  AllElementTypesMatch<["true_value", "false_value", "output"]>,
  AllRanksMatch<["condition", "true_value", "false_value", "output"]>,
  DeclareOpInterfaceMethods<SelectLikeOpInterface>]> {
  let summary = "Select operator.";

  let description = [{
    Elementwise Select of the output based on a condition.

    References:
      * https://github.khronos.org/SPIRV-Registry/extended/TOSA.001000.1.html#_select
      * https://www.mlplatform.org/tosa/tosa_spec_1_0_1.html#_select

    #### Example:
    ```mlir
    %0 = spirv.Tosa.Select %cond, %trueVal, %falseVal : !spirv.arm.tensor<4x1x4x5xi1>, !spirv.arm.tensor<4x6x4x5xi8>, !spirv.arm.tensor<4x6x4x5xi8> -> !spirv.arm.tensor<4x6x4x5xi8>
    %0 = spirv.Tosa.Select %cond, %trueVal, %falseVal : !spirv.arm.tensor<9x2x15x8xi1>, !spirv.arm.tensor<9x2x15x8xf16>, !spirv.arm.tensor<9x1x15x8xf16> -> !spirv.arm.tensor<9x2x15x8xf16>
    ```
  }];

  let arguments = (ins
    SPIRV_Bool_TensorArm: $condition,
    SPIRV_TosaAny_TensorArm: $true_value,
    SPIRV_TosaAny_TensorArm: $false_value
  );

  let results = (outs
    SPIRV_TosaAny_TensorArm: $output
  );

  let hasVerifier = 1;

  let assemblyFormat = [{
    $condition `,`
    $true_value `,`
    $false_value
    attr-dict `:` type(operands) `->` type(results)
  }];

  let extraClassDeclaration = extraBaseClassDeclaration#[{
    ::mlir::spirv::TensorArmType getConditionType() {
      return cast<::mlir::spirv::TensorArmType>(getCondition().getType());
    }
    ::mlir::spirv::TensorArmType getTrueValueType() {
      return cast<::mlir::spirv::TensorArmType>(getTrueValue().getType());
    }
    ::mlir::spirv::TensorArmType getFalseValueType() {
      return cast<::mlir::spirv::TensorArmType>(getFalseValue().getType());
    }
    ::mlir::Value getInput1() {
      return getCondition();
    }
    ::mlir::Value getInput2() {
      return getTrueValue();
    }
    ::mlir::Value getInput3() {
      return getFalseValue();
    }
    ::mlir::spirv::TensorArmType getInput1Type() {
      return getConditionType();
    }
    ::mlir::spirv::TensorArmType getInput2Type() {
      return getTrueValueType();
    }
    ::mlir::spirv::TensorArmType getInput3Type() {
      return getFalseValueType();
    }
  }];
}


def SPIRV_TosaEqualOp : SPIRV_TosaComparisonOp<"Equal", 45> {
  let summary = "Equal comparison operation";

  let description = [{
    Elementwise Equal comparison operation: returns the truth value of
    (input1 == input2) element-wise.

    References:
      * https://github.khronos.org/SPIRV-Registry/extended/TOSA.001000.1.html#_equal
      * https://www.mlplatform.org/tosa/tosa_spec_1_0_1.html#_equal

    #### Example:
    ```mlir
    %0 = spirv.Tosa.Equal %arg0, %arg1 : !spirv.arm.tensor<51x28x59xi32>, !spirv.arm.tensor<51x1x59xi32> -> !spirv.arm.tensor<51x28x59xi1>
    %0 = spirv.Tosa.Equal %arg0, %arg1 : !spirv.arm.tensor<16x11x5x3xf32>, !spirv.arm.tensor<16x1x5x3xf32> -> !spirv.arm.tensor<16x11x5x3xi1>
    ```
  }];
}


def SPIRV_TosaGreaterOp : SPIRV_TosaComparisonOp<"Greater", 46> {
  let summary = "Greater comparison operation";

  let description = [{
    Elementwise Greater than comparison operation: returns the truth value of
    (input1 > input2) element-wise.

    References:
      * https://github.khronos.org/SPIRV-Registry/extended/TOSA.001000.1.html#_greater
      * https://www.mlplatform.org/tosa/tosa_spec_1_0_1.html#_greater

    #### Example:
    ```mlir
    %0 = spirv.Tosa.Greater %arg0, %arg1 : !spirv.arm.tensor<11x10x10x2xi32>, !spirv.arm.tensor<11x10x10x1xi32> -> !spirv.arm.tensor<11x10x10x2xi1>
    %0 = spirv.Tosa.Greater %arg0, %arg1 : !spirv.arm.tensor<6x3x12x4xf16>, !spirv.arm.tensor<6x3x1x4xf16> -> !spirv.arm.tensor<6x3x12x4xi1>
    ```
  }];
}


def SPIRV_TosaGreaterEqualOp : SPIRV_TosaComparisonOp<"GreaterEqual", 47> {
  let summary = "Greater or Equal comparison operation";

  let description = [{
    Elementwise Greater or Equal than comparison operation: returns the truth value of
    (input1 >= input2) element-wise.

    References:
      * https://github.khronos.org/SPIRV-Registry/extended/TOSA.001000.1.html#_greater_equal
      * https://www.mlplatform.org/tosa/tosa_spec_1_0_1.html#_greater_equal

    #### Example:
    ```mlir
    %0 = spirv.Tosa.GreaterEqual %arg0, %arg1 : !spirv.arm.tensor<10x17x7x1xi32>, !spirv.arm.tensor<10x17x7x16xi32> -> !spirv.arm.tensor<10x17x7x16xi1>
    %0 = spirv.Tosa.GreaterEqual %arg0, %arg1 : !spirv.arm.tensor<3x17x6x3xf32>, !spirv.arm.tensor<1x17x6x3xf32> -> !spirv.arm.tensor<3x17x6x3xi1>
    ```
  }];
}


def SPIRV_TosaReduceAllOp : SPIRV_TosaReductionOp<"ReduceAll", 48, [Pure]> {
  let summary = "Reduce All operator.";

  let description = [{
    Reduces a tensor along the given axis with a Logical AND operation.

    References:
      * https://github.khronos.org/SPIRV-Registry/extended/TOSA.001000.1.html#_reduce_all
      * https://www.mlplatform.org/tosa/tosa_spec_1_0_1.html#_reduce_all

    #### Example:
    ```mlir
    %1 = spirv.Tosa.ReduceAll axis = 2, %arg0 : !spirv.arm.tensor<18x22x23x12xi1> -> !spirv.arm.tensor<18x22x1x12xi1>
    ```
  }];

  let arguments = (ins
    SPIRV_TensorArmAxisAttr: $axis,
    SPIRV_Bool_TensorArm: $input
  );

  let results = (outs
    SPIRV_Bool_TensorArm: $output
  );

  let assemblyFormat = [{
    `axis` `=` $axis `,`
    $input
    attr-dict `:` type(operands) `->` type(results)
  }];
}


def SPIRV_TosaReduceAnyOp : SPIRV_TosaReductionOp<"ReduceAny", 49, [Pure]> {
  let summary = "Reduce Any operator.";

  let description = [{
    Reduces a tensor along the given axis with a Logical OR operation.

    References:
      * https://github.khronos.org/SPIRV-Registry/extended/TOSA.001000.1.html#_reduce_any
      * https://www.mlplatform.org/tosa/tosa_spec_1_0_1.html#_reduce_any

    #### Example:
    ```mlir
    %1 = spirv.Tosa.ReduceAny axis = 2, %arg0 : !spirv.arm.tensor<25x13x30x8xi1> -> !spirv.arm.tensor<25x13x1x8xi1>
    ```
  }];

  let arguments = (ins
    SPIRV_TensorArmAxisAttr: $axis,
    SPIRV_Bool_TensorArm: $input
  );

  let results = (outs
    SPIRV_Bool_TensorArm: $output
  );

  let assemblyFormat = [{
    `axis` `=` $axis `,`
    $input
    attr-dict `:` type(operands) `->` type(results)
  }];
}


def SPIRV_TosaReduceMaxOp : SPIRV_TosaReductionOp<"ReduceMax", 50, [Pure]> {
  let summary = "Reduce Max operator.";

  let description = [{
    Reduces a tensor along the given axis with a Maximum operation.
    NaN Propagation Mode is ignored for inputs using integer element types.

    References:
      * https://github.khronos.org/SPIRV-Registry/extended/TOSA.001000.1.html#_reduce_max
      * https://www.mlplatform.org/tosa/tosa_spec_1_0_1.html#_reduce_max

    #### Example:
    ```mlir
    %2 = spirv.Tosa.ReduceMax axis = 2, nan_mode = <Propagate>, %arg0 : !spirv.arm.tensor<8x30x12x3xi8> -> !spirv.arm.tensor<8x30x1x3xi8>
    %2 = spirv.Tosa.ReduceMax axis = 2, nan_mode = <Propagate>, %arg0 : !spirv.arm.tensor<16x20x10xf16> -> !spirv.arm.tensor<16x20x1xf16>
    ```
  }];

  let arguments = (ins
    SPIRV_TensorArmAxisAttr: $axis,
    SPIRV_TosaExtNaNPropagationModeAttr: $nan_mode,
    SPIRV_TosaNumerical_TensorArm: $input
  );

  let results = (outs
    SPIRV_TosaNumerical_TensorArm: $output
  );

  let assemblyFormat = [{
    `axis` `=` $axis `,`
    `nan_mode` `=` $nan_mode `,`
    $input
    attr-dict `:` type(operands) `->` type(results)
  }];
}


def SPIRV_TosaReduceMinOp : SPIRV_TosaReductionOp<"ReduceMin", 51, [Pure]> {
  let summary = "Reduce Min operator.";

  let description = [{
    Reduces a tensor along the given axis with a Minimum operation.
    NaN Propagation Mode is ignored for inputs using integer element types.

    References:
      * https://github.khronos.org/SPIRV-Registry/extended/TOSA.001000.1.html#_reduce_min
      * https://www.mlplatform.org/tosa/tosa_spec_1_0_1.html#_reduce_min

    #### Example:
    ```mlir
    %2 = spirv.Tosa.ReduceMin axis = 2, nan_mode = <Propagate>, %arg0 : !spirv.arm.tensor<2x5x5x1xi8> -> !spirv.arm.tensor<2x5x1x1xi8>
    %2 = spirv.Tosa.ReduceMin axis = 2, nan_mode = <Propagate>, %arg0 : !spirv.arm.tensor<27x10x25x9xf16> -> !spirv.arm.tensor<27x10x1x9xf16>
    ```
  }];

  let arguments = (ins
    SPIRV_TensorArmAxisAttr: $axis,
    SPIRV_TosaExtNaNPropagationModeAttr: $nan_mode,
    SPIRV_TosaNumerical_TensorArm: $input
  );

  let results = (outs
    SPIRV_TosaNumerical_TensorArm: $output
  );

  let assemblyFormat = [{
    `axis` `=` $axis `,`
    `nan_mode` `=` $nan_mode `,`
    $input
    attr-dict `:` type(operands) `->` type(results)
  }];
}


def SPIRV_TosaReduceProductOp : SPIRV_TosaReductionOp<"ReduceProduct", 52, [Pure]> {
  let summary = "Reduce Product operator.";

  let description = [{
    Reduces a tensor along the given axis by computing the Product of the axis.

    References:
      * https://github.khronos.org/SPIRV-Registry/extended/TOSA.001000.1.html#_reduce_product
      * https://www.mlplatform.org/tosa/tosa_spec_1_0_1.html#_reduce_product

    #### Example:
    ```mlir
    %1 = spirv.Tosa.ReduceProduct axis = 2, %arg0 : !spirv.arm.tensor<2x16x25xf16> -> !spirv.arm.tensor<2x16x1xf16>
    ```
  }];

  let arguments = (ins
    SPIRV_TensorArmAxisAttr: $axis,
    SPIRV_TosaFloat_TensorArm: $input
  );

  let results = (outs
    SPIRV_TosaFloat_TensorArm: $output
  );

  let assemblyFormat = [{
    `axis` `=` $axis `,`
    $input
    attr-dict `:` type(operands) `->` type(results)
  }];
}


def SPIRV_TosaReduceSumOp : SPIRV_TosaReductionOp<"ReduceSum", 53, [NoMemoryEffect]> {
  let summary = "Reduce Sum operator.";

  let description = [{
    Reduces a tensor along the given axis by computing the Sum of the axis.
    Integer element types are always interpreted as signed. The behavior is
    undefined if the integer addition overflows or underflows the signed
    integer range.

    References:
      * https://github.khronos.org/SPIRV-Registry/extended/TOSA.001000.1.html#_reduce_sum
      * https://www.mlplatform.org/tosa/tosa_spec_1_0_1.html#_reduce_sum

    #### Example:
    ```mlir
    %1 = spirv.Tosa.ReduceSum axis = 1, %arg0 : !spirv.arm.tensor<20x24x22xi32> -> !spirv.arm.tensor<20x1x22xi32>
    %1 = spirv.Tosa.ReduceSum axis = 1, %arg0 : !spirv.arm.tensor<32x32x33xf32> -> !spirv.arm.tensor<32x1x33xf32>
    ```
  }];

  let arguments = (ins
    SPIRV_TensorArmAxisAttr: $axis,
    SPIRV_TosaNumerical_TensorArm: $input
  );

  let results = (outs
    SPIRV_TosaNumerical_TensorArm: $output
  );

  let assemblyFormat = [{
    `axis` `=` $axis `,`
    $input
    attr-dict `:` type(operands) `->` type(results)
  }];
}


def SPIRV_TosaConcatOp : SPIRV_TosaOpWithResult<"Concat", 54, [Pure,
  VariadicInputWithMinSize<"input1", 1>,
  VariadicInputAllSameElementType<"output", "input1">,
  VariadicInputAllSameRank<"output", "input1">,
  AxisValueLessThanRankOf<"output">]> {
  let summary = "Concatenates tensors along one dimension.";

  let description = [{
    Concatenates a list of tensors along a given axis.
    No data conversion happens during a concat operation.

    References:
      * https://github.khronos.org/SPIRV-Registry/extended/TOSA.001000.1.html#_concat
      * https://www.mlplatform.org/tosa/tosa_spec_1_0_1.html#_concat

    #### Example:
    ```mlir
    %1 = spirv.Tosa.Concat axis = 2, %arg0, %arg1, %arg2, %arg3 : !spirv.arm.tensor<12x13x3x14xi8>, !spirv.arm.tensor<12x13x3x14xi8>, !spirv.arm.tensor<12x13x3x14xi8>, !spirv.arm.tensor<12x13x3x14xi8> -> !spirv.arm.tensor<12x13x12x14xi8>
    %1 = spirv.Tosa.Concat axis = 1, %arg0, %arg1, %arg2 : !spirv.arm.tensor<40x31x19xf32>, !spirv.arm.tensor<40x15x19xf32>, !spirv.arm.tensor<40x16x19xf32> -> !spirv.arm.tensor<40x62x19xf32>
    ```
  }];

  let arguments = (ins
    SPIRV_TensorArmAxisAttr: $axis,
    Variadic<SPIRV_TosaAny_TensorArm>: $input1
  );

  let results = (outs
    SPIRV_TosaAny_TensorArm: $output
  );

  let assemblyFormat = [{
    `axis` `=` $axis `,`
    $input1
    attr-dict `:` type(operands) `->` type(results)
  }];

  let extraClassDeclaration = extraBaseClassDeclaration#[{
    ::mlir::TypeRange getInput1Types() {
      return getInput1().getTypes();
    }
  }];
}


def SPIRV_TosaPadOp : SPIRV_TosaOpWithResult<"Pad", 55, [Pure,
  AllElementTypesMatch<["input1", "pad_const", "output"]>,
  AllRanksMatch<["input1", "output"]>,
  ShapeConstraintFromInputRank<"input1", "padding", 2>]> {
  let summary = "Pads a tensor with value specified.";

  let description = [{
    Pads a tensor along the borders of each dimension with a supplied value.
    Returns a new tensor with the padding included. The pad_const value includes
    the zero point if the tensor uses a zero point.

    References:
      * https://github.khronos.org/SPIRV-Registry/extended/TOSA.001000.1.html#_pad
      * https://www.mlplatform.org/tosa/tosa_spec_1_0_1.html#_pad

    #### Example:
    ```mlir
    %2 = spirv.Tosa.Pad %arg0, %0, %1 : !spirv.arm.tensor<4x7xi8>, !spirv.arm.tensor<4xi32>, !spirv.arm.tensor<1xi8> -> !spirv.arm.tensor<21x19xi8>
    %2 = spirv.Tosa.Pad %arg0, %0, %1 : !spirv.arm.tensor<2x9x2x3xf32>, !spirv.arm.tensor<8xi32>, !spirv.arm.tensor<1xf32> -> !spirv.arm.tensor<4x9x4x4xf32>
    ```
  }];

  let arguments = (ins
    SPIRV_TosaAny_TensorArm: $input1,
    SPIRV_Int32_1DTensorArmOfEvenLength2To12: $padding,
    SPIRV_TosaAny_1DTensorArmOfLength1: $pad_const
  );

  let results = (outs
    SPIRV_TosaAny_TensorArm: $output
  );

  let assemblyFormat = [{
    $input1 `,`
    $padding `,`
    $pad_const
    attr-dict `:` type(operands) `->` type(results)
  }];

  let extraClassDeclaration = extraBaseClassDeclaration#[{
    ::mlir::spirv::TensorArmType getInput1Type() {
      return cast<::mlir::spirv::TensorArmType>(getInput1().getType());
    }
    ::mlir::spirv::TensorArmType getPaddingType() {
      return cast<::mlir::spirv::TensorArmType>(getPadding().getType());
    }
    ::mlir::spirv::TensorArmType getPadConstType() {
      return cast<::mlir::spirv::TensorArmType>(getPadConst().getType());
    }
  }];
}


def SPIRV_TosaReshapeOp : SPIRV_TosaOpWithResult<"Reshape", 56, [Pure,
  AllElementTypesMatch<["input1", "output"]>,
  AllElementCountsMatch<["input1", "output"]>,
  ShapeConstraintFromInputRank<"output", "shape">]> {
  let summary = "Reshape operator.";

  let description = [{
    Returns a tensor with the same type/values as the input, with a new shape
    specified by the shape argument. Reshape may operate on tensors of any rank.
    No data conversion happens during a reshape operation.

    References:
      * https://github.khronos.org/SPIRV-Registry/extended/TOSA.001000.1.html#_reshape
      * https://www.mlplatform.org/tosa/tosa_spec_1_0_1.html#_reshape

    #### Example:
    ```mlir
    %1 = spirv.Tosa.Reshape %arg0, %newShape : !spirv.arm.tensor<25x6x29x35xi16>, !spirv.arm.tensor<4xi32> -> !spirv.arm.tensor<125x6x7x29xi16>
    %1 = spirv.Tosa.Reshape %arg0, %newShape : !spirv.arm.tensor<1x2x7x2xf32>, !spirv.arm.tensor<3xi32> -> !spirv.arm.tensor<2x1x14xf32>
    ```
  }];

  let arguments = (ins
    SPIRV_TosaAny_TensorArm: $input1,
    SPIRV_Int32_1DTensorArmOfLength1To6: $shape
  );

  let results = (outs
    SPIRV_TosaAny_TensorArm: $output
  );

  let assemblyFormat = [{
    $input1 `,`
    $shape
    attr-dict `:` type(operands) `->` type(results)
  }];

  let extraClassDeclaration = extraBaseClassDeclaration#[{
    ::mlir::spirv::TensorArmType getInput1Type() {
      return cast<::mlir::spirv::TensorArmType>(getInput1().getType());
    }
    ::mlir::spirv::TensorArmType getShapeType() {
      return cast<::mlir::spirv::TensorArmType>(getShape().getType());
    }
  }];
}


def SPIRV_TosaReverseOp : SPIRV_TosaOpWithResult<"Reverse", 57, [Pure,
  AllTypesMatch<["input1", "output"]>,
  AxisValueLessThanRankOf<"input1">]> {
  let summary = "Reverse operator.";

  let description = [{
    Returns a tensor with the same type/values as the input, with the data
    reversed along the given axis. No data conversion happens during a reverse
    operation.

    References:
      * https://github.khronos.org/SPIRV-Registry/extended/TOSA.001000.1.html#_reverse
      * https://www.mlplatform.org/tosa/tosa_spec_1_0_1.html#_reverse

    #### Example:
    ```mlir
    %1 = spirv.Tosa.Reverse axis = 2, %arg0 : !spirv.arm.tensor<20x5x28x31xi32> -> !spirv.arm.tensor<20x5x28x31xi32>
    %1 = spirv.Tosa.Reverse axis = 1, %arg0 : !spirv.arm.tensor<21x34x47xf32> -> !spirv.arm.tensor<21x34x47xf32>
    ```
  }];

  let arguments = (ins
    SPIRV_TensorArmAxisAttr: $axis,
    SPIRV_TosaAny_TensorArm: $input1
  );

  let results = (outs
    SPIRV_TosaAny_TensorArm: $output
  );

  let assemblyFormat = [{
    `axis` `=` $axis `,`
    $input1
    attr-dict `:` type(operands) `->` type(results)
  }];

  let extraClassDeclaration = extraBaseClassDeclaration#[{
    ::mlir::spirv::TensorArmType getInput1Type() {
      return cast<::mlir::spirv::TensorArmType>(getInput1().getType());
    }
  }];
}


def SPIRV_TosaSliceOp : SPIRV_TosaOpWithResult<"Slice", 58, [Pure,
  AllElementTypesMatch<["input1", "output"]>,
  ShapeConstraintFromInputRank<"input1", "start">,
  ShapeConstraintFromInputRank<"input1", "size">]> {
  let summary = "Slice operator.";

  let description = [{
    Extracts a slice of input1, beginning at the start coordinates,
    and extending for size elements in each direction.
    No data conversion happens during a slice operation.

    References:
      * https://github.khronos.org/SPIRV-Registry/extended/TOSA.001000.1.html#_slice
      * https://www.mlplatform.org/tosa/tosa_spec_1_0_1.html#_slice

    #### Example:
    ```mlir
    %2 = spirv.Tosa.Slice %arg0, %start, %size : !spirv.arm.tensor<32x19x41xi8>, !spirv.arm.tensor<3xi32>, !spirv.arm.tensor<3xi32> -> !spirv.arm.tensor<21x5x2xi8>
    %2 = spirv.Tosa.Slice %arg0, %start, %size : !spirv.arm.tensor<30x45x29xf32>, !spirv.arm.tensor<3xi32>, !spirv.arm.tensor<3xi32> -> !spirv.arm.tensor<5x12x11xf32>
    ```
  }];

  let arguments = (ins
    SPIRV_TosaAny_TensorArm: $input1,
    SPIRV_Int32_1DTensorArmOfLength1To6: $start,
    SPIRV_Int32_1DTensorArmOfLength1To6: $size
  );

  let results = (outs
    SPIRV_TosaAny_TensorArm: $output
  );

  let assemblyFormat = [{
    $input1 `,`
    $start `,`
    $size
    attr-dict `:` type(operands) `->` type(results)
  }];

  let extraClassDeclaration = extraBaseClassDeclaration#[{
    ::mlir::spirv::TensorArmType getInput1Type() {
      return cast<::mlir::spirv::TensorArmType>(getInput1().getType());
    }
    ::mlir::spirv::TensorArmType getStartType() {
      return cast<::mlir::spirv::TensorArmType>(getStart().getType());
    }
    ::mlir::spirv::TensorArmType getSizeType() {
      return cast<::mlir::spirv::TensorArmType>(getSize().getType());
    }
  }];
}


def SPIRV_TosaTileOp : SPIRV_TosaOpWithResult<"Tile", 59, [Pure,
  AllElementTypesMatch<["input1", "output"]>,
  AllRanksMatch<["input1", "output"]>,
  ShapeConstraintFromInputRank<"input1", "multiples">]> {
  let summary = "Tile operator.";

  let description = [{
    Replicates input1 multiples times along each dimension.

    References:
      * https://github.khronos.org/SPIRV-Registry/extended/TOSA.001000.1.html#_tile
      * https://www.mlplatform.org/tosa/tosa_spec_1_0_1.html#_tile

    #### Example:
    ```mlir
    %1 = spirv.Tosa.Tile %arg0, %multiples : !spirv.arm.tensor<10x28x21xi16>, !spirv.arm.tensor<3xi32> -> !spirv.arm.tensor<10x28x63xi16>
    %1 = spirv.Tosa.Tile %arg0, %multiples : !spirv.arm.tensor<31x19x5xf16>, !spirv.arm.tensor<3xi32> -> !spirv.arm.tensor<62x57x10xf16>
    ```

    For %multiples = [1, 1, 3], the first two dimesions will be tiled just once, whilst the third will be tiled 3 times.
  }];

  let arguments = (ins
    SPIRV_TosaAny_TensorArm: $input1,
    SPIRV_Int32_1DTensorArmOfLength1To6: $multiples
  );

  let results = (outs
    SPIRV_TosaAny_TensorArm: $output
  );

  let assemblyFormat = [{
    $input1 `,`
    $multiples
    attr-dict `:` type(operands) `->` type(results)
  }];

  let extraClassDeclaration = extraBaseClassDeclaration#[{
    ::mlir::spirv::TensorArmType getInput1Type() {
      return cast<::mlir::spirv::TensorArmType>(getInput1().getType());
    }
    ::mlir::spirv::TensorArmType getMultiplesType() {
      return cast<::mlir::spirv::TensorArmType>(getMultiples().getType());
    }
  }];
}


def SPIRV_TosaTransposeOp : SPIRV_TosaOpWithResult<"Transpose", 60, [Pure,
  AllElementTypesMatch<["input1", "output"]>,
  AllRanksMatch<["input1", "output"]>,
  AllElementCountsMatch<["input1", "output"]>,
  ShapeConstraintFromInputRank<"input1", "perms">]> {
  let summary = "Transpose operator.";

  let description = [{
    Permutes the dimensions of the input tensor input1 based on the perms
    argument. Each value in the perms list must be a valid dimension of the
    input tensor and may not be repeated.

    References:
      * https://github.khronos.org/SPIRV-Registry/extended/TOSA.001000.1.html#_transpose
      * https://www.mlplatform.org/tosa/tosa_spec_1_0_1.html#_transpose

    #### Example:
    ```mlir
    %1 = spirv.Tosa.Transpose perms = [2, 0, 1, 3], %arg0 : !spirv.arm.tensor<14x28x1x61xi16> -> !spirv.arm.tensor<1x14x28x61xi16>
    %1 = spirv.Tosa.Transpose perms = [2, 0, 1], %arg0 : !spirv.arm.tensor<42x22x49xi1> -> !spirv.arm.tensor<49x42x22xi1>
    ```
  }];

  let arguments = (ins
    SPIRV_Int32_1DTensorArmOfLength1To6Attr: $perms,
    SPIRV_TosaAny_TensorArm: $input1
  );

  let results = (outs
    SPIRV_TosaAny_TensorArm: $output
  );

  let assemblyFormat = [{
    `perms` `=` custom<SPIRV_I32_1DArmTensor>($perms) `,`
    $input1
    attr-dict `:` type(operands) `->` type(results)
  }];

  let extraClassDeclaration = extraBaseClassDeclaration#[{
    ::mlir::spirv::TensorArmType getInput1Type() {
      return cast<::mlir::spirv::TensorArmType>(getInput1().getType());
    }
  }];
}


def SPIRV_TosaGatherOp : SPIRV_TosaOpWithResult<"Gather", 61, [NoMemoryEffect,
  AllElementTypesMatch<["values", "output"]>,
  ValuesIndicesShapesMatch<"values", "indices", "output">]> {
  let summary = "Gather operation.";

  let description = [{
    Generate a tensor for which each element in the output is a subtensor of the
    values tensor based on the indices. Undefined behaviour may occur if the
    specified indices are out of range.

    References:
      * https://github.khronos.org/SPIRV-Registry/extended/TOSA.001000.1.html#_gather
      * https://www.mlplatform.org/tosa/tosa_spec_1_0_1.html#_gather

    #### Example:
    ```mlir
    %0 = spirv.Tosa.Gather %values, %indices : !spirv.arm.tensor<31x11x45xi32>, !spirv.arm.tensor<31x15xi32> -> !spirv.arm.tensor<31x15x45xi32>
    %0 = spirv.Tosa.Gather %values, %indices : !spirv.arm.tensor<59x61x19xf32>, !spirv.arm.tensor<59x65xi32> -> !spirv.arm.tensor<59x65x19xf32>
    ```
  }];

  let arguments = (ins
    SPIRV_TosaNumerical_TensorArm3D: $values,
    SPIRV_Int32_TensorArm2D: $indices
  );

  let results = (outs
    SPIRV_TosaNumerical_TensorArm3D: $output
  );

  let assemblyFormat = [{
    $values `,`
    $indices
    attr-dict `:` type(operands) `->` type(results)
  }];

  let extraClassDeclaration = extraBaseClassDeclaration#[{
    ::mlir::spirv::TensorArmType getValuesType() {
      return cast<::mlir::spirv::TensorArmType>(getValues().getType());
    }
    ::mlir::spirv::TensorArmType getIndicesType() {
      return cast<::mlir::spirv::TensorArmType>(getIndices().getType());
    }
  }];
}


def SPIRV_TosaScatterOp : SPIRV_TosaOpWithResult<"Scatter", 62, [NoMemoryEffect,
  AllElementTypesMatch<["values_in", "input", "values_out"]>,
  AllTypesMatch<["values_in", "values_out"]>,
  ValuesIndicesShapesMatch<"values_in", "indices", "input">]> {
  let summary = "Scatter operation.";

  let description = [{
    The values_out tensor is set to the values_in tensor with data modified as
    follows: data from the input tensor is inserted at the positions specified
    by the indices tensor. In use cases that require multiple updates to the
    same output position, these must be decomposed into multiple scatter
    operations.  Undefined behaviour may occur if the specified indices are
    out of range or duplicate indices are provided.

    References:
      * https://github.khronos.org/SPIRV-Registry/extended/TOSA.001000.1.html#_scatter
      * https://www.mlplatform.org/tosa/tosa_spec_1_0_1.html#_scatter

    #### Example:
    ```mlir
    %0 = spirv.Tosa.Scatter %values_in, %indices, %arg0 : !spirv.arm.tensor<34x28x54xi32>, !spirv.arm.tensor<34x18xi32>, !spirv.arm.tensor<34x18x54xi32> -> !spirv.arm.tensor<34x28x54xi32>
    %0 = spirv.Tosa.Scatter %values_in, %indices, %arg0 : !spirv.arm.tensor<18x34x25xf16>, !spirv.arm.tensor<18x20xi32>, !spirv.arm.tensor<18x20x25xf16> -> !spirv.arm.tensor<18x34x25xf16>
    ```
  }];

  let arguments = (ins
    SPIRV_TosaNumerical_TensorArm3D: $values_in,
    SPIRV_Int32_TensorArm2D: $indices,
    SPIRV_TosaNumerical_TensorArm3D: $input
  );

  let results = (outs
    SPIRV_TosaNumerical_TensorArm3D: $values_out
  );

  let assemblyFormat = [{
    $values_in `,`
    $indices `,`
    $input
    attr-dict `:` type(operands) `->` type(results)
  }];

  let extraClassDeclaration = extraBaseClassDeclaration#[{
    ::mlir::spirv::TensorArmType getValuesInType() {
      return cast<::mlir::spirv::TensorArmType>(getValuesIn().getType());
    }
    ::mlir::spirv::TensorArmType getIndicesType() {
      return cast<::mlir::spirv::TensorArmType>(getIndices().getType());
    }
    ::mlir::spirv::TensorArmType getInputType() {
      return cast<::mlir::spirv::TensorArmType>(getInput().getType());
    }
  }];
}


def SPIRV_TosaResizeOp : SPIRV_TosaOpWithResult<"Resize", 63, [Pure,
  TypeConstraintImplicationOn<"input", F16, "output", [F16]>,
  TypeConstraintImplicationOn<"input", F32, "output", [F32]>,
  TypeConstraintImplicationOn<"input", BF16, "output", [BF16]>,
  TypeConstraintImplicationOn<"input", I8, "output", [I8, I32]>,
  TypeConstraintImplicationOn<"input", I16, "output", [I16, I64]>]> {
  let summary = "Resize operation, supports various resize/upsample modes.";

  let description = [{
    Resizes a tensor. Resize is only allowed in the H and W dimensions, given the input
    shape = [N,H,W,C].

    The height dimension (H) is scaled by factor ($ scale_y_n/scale_y_d $). The width
    dimension (W) is scaled by factor ($ scale_x_n/scale_x_d $).

    The NearestNeighbor mode returns the value of the input tensor closest to
    the calculated sample position for both floating-point and integer data
    formats.

    Floating-point Bilinear mode returns a bilinearly interpolated output value
    based on the four closest input sample positions.

    For integer Bilinear interpolation mode, the output value must be scaled by
    $ 1/(scale_y_n * scale_x_n) $ in a following operation to complete the
    interpolation (for example with a rescale operator).

    The output dimensions can be derived from the input dimensions by inverting
    the scale. The [border_y, border_x] values adjust the output size to allow
    fractional sampling beyond integer input position (H - 1,W - 1).

    The limit MAX_SCALE=256 is applied to each scale ratio after reduction of the
    ratio. Individual scale numerator and denominator values are allowed to be
    larger than MAX_SCALE.

    References:
      * https://github.khronos.org/SPIRV-Registry/extended/TOSA.001000.1.html#_resize
      * https://www.mlplatform.org/tosa/tosa_spec_1_0_1.html#_resize

    #### Example:
    ```mlir
    %4 = spirv.Tosa.Resize mode = <NearestNeighbor>, %arg0, %scale, %offset, %border : !spirv.arm.tensor<1x1x31x55xi8>, !spirv.arm.tensor<4xi32>, !spirv.arm.tensor<2xi32>, !spirv.arm.tensor<2xi32> -> !spirv.arm.tensor<1x1x278x55xi8>
    %4 = spirv.Tosa.Resize mode = <Bilinear>, %arg0, %scale, %offset, %border : !spirv.arm.tensor<1x48x33x63xf32>, !spirv.arm.tensor<4xi32>, !spirv.arm.tensor<2xi32>, !spirv.arm.tensor<2xi32> -> !spirv.arm.tensor<1x753x297x63xf32>
    ```
  }];

  let arguments = (ins
    SPIRV_TosaExtResizeModeAttr: $mode,
    SPIRV_TosaNumerical_TensorArm4D: $input,
    SPIRV_Int32_1DTensorArmOfLength4: $scale,
    SPIRV_Int32_1DTensorArmOfLength2: $offset,
    SPIRV_Int32_1DTensorArmOfLength2: $border
  );

  let results = (outs
    SPIRV_TosaNumerical_TensorArm4D: $output
  );

  let assemblyFormat = [{
    `mode` `=` $mode `,`
    $input `,`
    $scale `,`
    $offset `,`
    $border
    attr-dict `:` type(operands) `->` type(results)
  }];

  let extraClassDeclaration = extraBaseClassDeclaration#[{
    ::mlir::spirv::TensorArmType getInputType() {
      return cast<::mlir::spirv::TensorArmType>(getInput().getType());
    }
    ::mlir::spirv::TensorArmType getScaleType() {
      return cast<::mlir::spirv::TensorArmType>(getScale().getType());
    }
    ::mlir::spirv::TensorArmType getOffsetType() {
      return cast<::mlir::spirv::TensorArmType>(getOffset().getType());
    }
    ::mlir::spirv::TensorArmType getBorderType() {
      return cast<::mlir::spirv::TensorArmType>(getBorder().getType());
    }
  }];
}


def SPIRV_TosaCastOp : SPIRV_TosaOpWithResult<"Cast", 64, [Pure,
  AllShapesMatch<["input", "output"]>]> {
  let summary = "Cast operation.";

  let description = [{
    Casts a tensor from one data type to another.

    References:
      * https://github.khronos.org/SPIRV-Registry/extended/TOSA.001000.1.html#_cast
      * https://www.mlplatform.org/tosa/tosa_spec_1_0_1.html#_cast

    #### Example:
    ```mlir
    %0 = spirv.Tosa.Cast %arg0 : !spirv.arm.tensor<1x65538x1x2xi8> -> !spirv.arm.tensor<1x65538x1x2xi32>
    %0 = spirv.Tosa.Cast %arg0 : !spirv.arm.tensor<11x5x14x4xf32> -> !spirv.arm.tensor<11x5x14x4xf16>
    ```
  }];

  let arguments = (ins
    SPIRV_TosaAny_TensorArm: $input
  );

  let results = (outs
    SPIRV_TosaAny_TensorArm: $output
  );

  let assemblyFormat = [{
    $input
    attr-dict `:` type(operands) `->` type(results)
  }];

  let extraClassDeclaration = extraBaseClassDeclaration#[{
    ::mlir::spirv::TensorArmType getInputType() {
      return cast<::mlir::spirv::TensorArmType>(getInput().getType());
    }
  }];
}


def SPIRV_TosaRescaleOp : SPIRV_TosaOpWithResult<"Rescale", 65, [NoMemoryEffect,
  AllShapesMatch<["input", "output"]>,
  AllElementTypesMatch<["input", "input_zp"]>,
  AllElementTypesMatch<["output", "output_zp"]>,
  ElementTypeMatchesScale32<"multiplier">,
  TensorLengthMatchesPerChannel<"multiplier">,
  TensorLengthMatchesPerChannel<"shift">,
  TypeConstraintImplicationOn<"input", I8, "output", [I8, I16, I32]>,
  TypeConstraintImplicationOn<"input", I16, "output", [I8, I16, I32]>,
  TypeConstraintImplicationOn<"input", I32, "output", [I8, I16, I32]>,
  TypeConstraintImplicationOn<"input", I64, "output", [I8, I16, I32]>,
  BoolAttrTypeConstraintImplicationOn<"input_unsigned", "input", [I8, I16]>,
  BoolAttrTypeConstraintImplicationOn<"output_unsigned", "input", [I8, I16]>,
  BoolAttrTypeConstraintImplicationOn<"input_unsigned", "output", [I8, I16]>,
  BoolAttrTypeConstraintImplicationOn<"output_unsigned", "output", [I8, I16]>]> {
  let summary = "Rescale operator.";

  let description = [{
    Rescale is defined using an integer multiply, add, and shift.

    Rescale supports two precisions of multiplier: 16-bit and 32-bit. The
    32-bit multiplier version supports two rounding modes to enable simpler
    lowering of existing frameworks that use two stage rounding. All arithmetic
    is designed so that it does not overflow a 64-bit accumulator and that the
    result fits in 32 bits. In particular, a 48-bit value (represented as a
    64-bit value in SPIR-V) cannot be scaled with the 32-bit multiplier because
    the accumulator would need to have 80 bits.

    The shift and value range are limited to allow a variety of implementations.
    The limit of 62 on shift allows the shift to be decomposed as two right
    shifts of 31.

    Unsigned 8- and 16-bit values are only allowed in the Rescale operation,
    to allow for compatibility with networks which expect unsigned 8-bit or
    16-bit tensors for input and output.

    Undefined behaviour may occur if the calculated result underflows or overflows
    their integer ranges.

    References:
      * https://github.khronos.org/SPIRV-Registry/extended/TOSA.001000.1.html#_rescale
      * https://www.mlplatform.org/tosa/tosa_spec_1_0_1.html#_rescale

    #### Example:
    ```mlir
    %9 = spirv.Tosa.Rescale scale32 = true, rounding_mode = <DoubleRound>, per_channel = false, input_unsigned = false, output_unsigned = true, %arg0, %multiplier, %shift, %input_zp, %output_zp : !spirv.arm.tensor<17x29x19xi16>, !spirv.arm.tensor<1xi32>, !spirv.arm.tensor<1xi8>, !spirv.arm.tensor<1xi16>, !spirv.arm.tensor<1xi16> -> !spirv.arm.tensor<17x29x19xi16>
    ```
  }];

  let arguments = (ins
    SPIRV_BoolConstAttr: $scale32,
    SPIRV_TosaExtRoundingModeAttr: $rounding_mode,
    SPIRV_BoolConstAttr: $per_channel,
    SPIRV_BoolConstAttr: $input_unsigned,
    SPIRV_BoolConstAttr: $output_unsigned,
    SPIRV_TosaInteger_TensorArm: $input,
    SPIRV_Int16OrInt32_TensorArm1D: $multiplier,
    SPIRV_Int8_TensorArm1D: $shift,
    SPIRV_TosaInteger_1DTensorArmOfLength1: $input_zp,
    SPIRV_TosaInteger_1DTensorArmOfLength1: $output_zp
  );

  let results = (outs
    SPIRV_TosaInteger_TensorArm: $output
  );

  let assemblyFormat = [{
    `scale32` `=` $scale32 `,`
    `rounding_mode` `=` $rounding_mode `,`
    `per_channel` `=` $per_channel `,`
    `input_unsigned` `=` $input_unsigned `,`
    `output_unsigned` `=` $output_unsigned `,`
    $input `,`
    $multiplier `,`
    $shift `,`
    $input_zp `,`
    $output_zp
    attr-dict `:` type(operands) `->` type(results)
  }];

  let extraClassDeclaration = extraBaseClassDeclaration#[{
    ::mlir::spirv::TensorArmType getInputType() {
      return cast<::mlir::spirv::TensorArmType>(getInput().getType());
    }
    ::mlir::spirv::TensorArmType getMultiplierType() {
      return cast<::mlir::spirv::TensorArmType>(getMultiplier().getType());
    }
    ::mlir::spirv::TensorArmType getShiftType() {
      return cast<::mlir::spirv::TensorArmType>(getShift().getType());
    }
  }];
}


#endif // MLIR_DIALECT_SPIRV_IR_TOSA_OPS
