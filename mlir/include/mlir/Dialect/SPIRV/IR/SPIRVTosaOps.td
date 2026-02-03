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


def SPIRV_TosaArgMaxOp : SPIRV_TosaOpWithResult<"ArgMax", 0, [Pure,
  OutputRankIsInputRankMinusOne<"input", "output">,
  AxisValueLessThanRankOf<"input">]> {
  let summary = "Perform argmax on the input.";

  let description = [{
    Returns the index with the largest value across the given axis of the
    input tensor. If multiple locations have equal values, returns the first
    match along the search axis.

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


def SPIRV_TosaAvgPool2DOp : SPIRV_TosaOpWithResult<"AvgPool2D", 1, [Pure,
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
    used to calculate the divisor.

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


def SPIRV_TosaConv2DOp : SPIRV_TosaOpWithResult<"Conv2D", 2, [Pure,
  TypeConstraintImplicationOn<"input", I8, "output", [I32]>,
  TypeConstraintImplicationOn<"input", I16, "output", [I64]>,
  TypeConstraintImplicationOn<"input", BF16, "output", [BF16]>,
  TypeConstraintImplicationOn<"input", F16, "output", [F16]>,
  TypeConstraintImplicationOn<"input", F32, "output", [F32]>,
  TypeConstraintImplicationOn<"input", AnyInteger, "input", [I8, I16]>,
  TypeConstraintImplicationOn<"weight", AnyInteger, "weight", [I8]>,
  TypeImpliesAccType<"input", I8, ["INT32"]>,
  TypeImpliesAccType<"input", I16, ["INT48"]>,
  TypeImpliesAccType<"input", F16, ["FP16", "FP32"]>,
  TypeImpliesAccType<"input", BF16, ["FP32"]>,
  TypeImpliesAccType<"input", F32, ["FP32"]>,
  AllElementTypesMatch<["bias", "output"]>,
  AllElementTypesMatch<["input", "input_zp"]>,
  AllElementTypesMatch<["weight", "weight_zp"]>]> {
  let summary = "2D Convolution operator.";

  let description = [{
    Performs a 2D convolution over the given tensor input, using the weight
    tensor. Implementations may choose to skip calculation of multiplies in
    the padding area.

    Input and weight have respective zero point values provided in input_zp and weight_zp.

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


def SPIRV_TosaConv3DOp : SPIRV_TosaOpWithResult<"Conv3D", 3, [Pure,
  TypeConstraintImplicationOn<"input", I8, "output", [I32]>,
  TypeConstraintImplicationOn<"input", I16, "output", [I64]>,
  TypeConstraintImplicationOn<"input", BF16, "output", [BF16]>,
  TypeConstraintImplicationOn<"input", F16, "output", [F16]>,
  TypeConstraintImplicationOn<"input", F32, "output", [F32]>,
  TypeConstraintImplicationOn<"input", AnyInteger, "input", [I8, I16]>,
  TypeConstraintImplicationOn<"weight", AnyInteger, "weight", [I8]>,
  TypeImpliesAccType<"input", I8, ["INT32"]>,
  TypeImpliesAccType<"input", I16, ["INT48"]>,
  TypeImpliesAccType<"input", F16, ["FP16", "FP32"]>,
  TypeImpliesAccType<"input", BF16, ["FP32"]>,
  TypeImpliesAccType<"input", F32, ["FP32"]>,
  AllElementTypesMatch<["bias", "output"]>,
  AllElementTypesMatch<["input", "input_zp"]>,
  AllElementTypesMatch<["weight", "weight_zp"]>]> {
  let summary = "3D Convolution operator.";

  let description = [{
    Performs a 3D convolution over the given input tensor. Implementations
    may choose to skip calculation of multiplies in the padding area.

    Input and weight have respective zero point values provided in input_zp and weight_zp.

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


def SPIRV_TosaDepthwiseConv2DOp : SPIRV_TosaOpWithResult<"DepthwiseConv2D", 4, [Pure,
  TypeConstraintImplicationOn<"input", I8, "output", [I32]>,
  TypeConstraintImplicationOn<"input", I16, "output", [I64]>,
  TypeConstraintImplicationOn<"input", BF16, "output", [BF16]>,
  TypeConstraintImplicationOn<"input", F16, "output", [F16]>,
  TypeConstraintImplicationOn<"input", F32, "output", [F32]>,
  TypeConstraintImplicationOn<"input", AnyInteger, "input", [I8, I16]>,
  TypeConstraintImplicationOn<"weight", AnyInteger, "weight", [I8]>,
  TypeImpliesAccType<"input", I8, ["INT32"]>,
  TypeImpliesAccType<"input", I16, ["INT48"]>,
  TypeImpliesAccType<"input", F16, ["FP16", "FP32"]>,
  TypeImpliesAccType<"input", BF16, ["FP32"]>,
  TypeImpliesAccType<"input", F32, ["FP32"]>,
  AllElementTypesMatch<["bias", "output"]>,
  AllElementTypesMatch<["input", "input_zp"]>,
  AllElementTypesMatch<["weight", "weight_zp"]>]> {
  let summary = "Depthwise 2D Convolution operator.";

  let description = [{
    Performs 2D convolutions separately over each channel of the given tensor
    input, using the weight tensor. Implementations may choose to skip
    calculation of multiplies in the padding area.

    Input and weight have respective zero point values provided in input_zp and weight_zp.

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


def SPIRV_TosaMatMulOp : SPIRV_TosaOpWithResult<"MatMul", 6, [Pure,
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


def SPIRV_TosaTransposeConv2DOp : SPIRV_TosaOpWithResult<"TransposeConv2D", 9, [Pure,
  TypeConstraintImplicationOn<"input", I8, "output", [I32]>,
  TypeConstraintImplicationOn<"input", I16, "output", [I64]>,
  TypeConstraintImplicationOn<"input", BF16, "output", [BF16]>,
  TypeConstraintImplicationOn<"input", F16, "output", [F16]>,
  TypeConstraintImplicationOn<"input", F32, "output", [F32]>,
  TypeConstraintImplicationOn<"input", AnyInteger, "input", [I8, I16]>,
  TypeConstraintImplicationOn<"weight", AnyInteger, "weight", [I8]>,
  TypeImpliesAccType<"input", I8, ["INT32"]>,
  TypeImpliesAccType<"input", I16, ["INT48"]>,
  TypeImpliesAccType<"input", F16, ["FP16", "FP32"]>,
  TypeImpliesAccType<"input", BF16, ["FP32"]>,
  TypeImpliesAccType<"input", F32, ["FP32"]>,
  AllElementTypesMatch<["bias", "output"]>,
  AllElementTypesMatch<["input", "input_zp"]>,
  AllElementTypesMatch<["weight", "weight_zp"]>]> {
  let summary = "Transpose 2D Convolution operator.";

  let description = [{
    Performs a 2D transposed convolution over the given tensor input, using the
    weights tensor. Implementations may choose to skip calculation of multiplies
    by zero at fractional input positions.

    Input and weight have respective zero point values provided in input_zp and weight_zp.

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


def SPIRV_TosaClampOp : SPIRV_TosaOpWithResult<"Clamp", 10, [Pure,
  AllTypesMatch<["input", "output"]>,
  AllElementTypesMatch<["input", "output", "min_val", "max_val"]>]> {
  let summary = "Computes Clamp(min, max).";

  let description = [{
    Clamp to an arbitrary minimum and maximum value.
    Maximum and minimum values are specified as values in the range of the
    input type.
    No zero point subtraction is done to the values, thus to clamp to the zero
    point value, the zero point itself should be supplied as the minimum value.

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


#endif // MLIR_DIALECT_SPIRV_IR_TOSA_OPS
