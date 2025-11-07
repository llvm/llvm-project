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
}


def SPIRV_TosaArgMaxOp : SPIRV_TosaOp<"ArgMax", 0, [Pure]> {
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

  let hasVerifier = 1;

  let assemblyFormat = [{
    `axis` `=` $axis `,` `nan_mode` `=` $nan_mode `,`
    $input
    attr-dict `:` type(operands) `->` type(results)
  }];

  let extraClassDeclaration = [{
    ::mlir::spirv::TensorArmType getInputType() {
      return cast<::mlir::spirv::TensorArmType>(getInput().getType());
    }
    ::mlir::spirv::TensorArmType getResultType() {
      return cast<::mlir::spirv::TensorArmType>(getType());
    }
  }];
}


def SPIRV_TosaAvgPool2DOp : SPIRV_TosaOp<"AvgPool2D", 1, [Pure]> {
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
    %6 = spirv.Tosa.AvgPool2D kernel = dense<3> : !spirv.arm.tensor<2xi32>, stride = dense<[1, 2]> : !spirv.arm.tensor<2xi32>, pad = dense<[0, 1, 0, 0]> : !spirv.arm.tensor<4xi32>, acc_type = <INT32>, %arg0, %4, %5 : !spirv.arm.tensor<1x3x65537x1xi8>, !spirv.arm.tensor<1xi8>, !spirv.arm.tensor<1xi8> -> !spirv.arm.tensor<1x2x32768x1xi8>
    %6 = spirv.Tosa.AvgPool2D kernel = dense<2> : !spirv.arm.tensor<2xi32>, stride = dense<1> : !spirv.arm.tensor<2xi32>, pad = dense<[1, 0, 0, 0]> : !spirv.arm.tensor<4xi32>, acc_type = <FP32>, %arg0, %4, %5 : !spirv.arm.tensor<1x2x65533x2xf32>, !spirv.arm.tensor<1xf32>, !spirv.arm.tensor<1xf32> -> !spirv.arm.tensor<1x2x65532x2xf32>
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

  let hasVerifier = 1;

  let assemblyFormat = [{
    `kernel` `=` $kernel `,` `stride` `=` $stride `,`
    `pad` `=` $pad `,` `acc_type` `=` $acc_type `,`
    $input `,` $input_zp `,` $output_zp
    attr-dict `:` type(operands) `->` type(results)
  }];

  let extraClassDeclaration = [{
    ::mlir::spirv::TensorArmType getInputType() {
      return cast<::mlir::spirv::TensorArmType>(getInput().getType());
    }
    ::mlir::spirv::TensorArmType getResultType() {
      return cast<::mlir::spirv::TensorArmType>(getType());
    }
  }];
}


def SPIRV_TosaConv2DOp : SPIRV_TosaOp<"Conv2D", 2, [Pure]> {
  let summary = "2D Convolution operator.";

  let description = [{
    Performs a 2D convolution over the given tensor input, using the weight
    tensor. Implementations may choose to skip calculation of multiplies in
    the padding area.

    References:
      * https://github.khronos.org/SPIRV-Registry/extended/TOSA.001000.1.html#_conv2d
      * https://www.mlplatform.org/tosa/tosa_spec_1_0_1.html#_conv2d

    #### Example:
    ```mlir
    %7 = spirv.Tosa.Conv2D pad = dense<[1, 0, 0, 0]> : !spirv.arm.tensor<4xi32>, stride = dense<[1, 2]> : !spirv.arm.tensor<2xi32>, dilation = dense<[7, 1]> : !spirv.arm.tensor<2xi32>, acc_type = <INT32>, local_bound = false, %arg0, %arg1, %arg2, %5, %6 : !spirv.arm.tensor<1x65535x3x1xi8>, !spirv.arm.tensor<7x1x1x1xi8>, !spirv.arm.tensor<1xi32>, !spirv.arm.tensor<1xi8>, !spirv.arm.tensor<1xi8> -> !spirv.arm.tensor<1x65536x2x7xi32>
    %7 = spirv.Tosa.Conv2D pad = dense<0> : !spirv.arm.tensor<4xi32>, stride = dense<1> : !spirv.arm.tensor<2xi32>, dilation = dense<1> : !spirv.arm.tensor<2xi32>, acc_type = <FP16>, local_bound = true, %arg0, %arg1, %arg2, %5, %6 : !spirv.arm.tensor<1x34x18x27xf16>, !spirv.arm.tensor<11x1x1x27xf16>, !spirv.arm.tensor<11xf16>, !spirv.arm.tensor<1xf16>, !spirv.arm.tensor<1xf16> -> !spirv.arm.tensor<1x34x18x11xf16>
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

  let hasVerifier = 1;

  let assemblyFormat = [{
    `pad` `=` $pad `,` `stride` `=` $stride `,`
    `dilation` `=` $dilation `,` `acc_type` `=` $acc_type `,`
    `local_bound` `=` $local_bound `,`
    $input `,` $weight `,` $bias `,` $input_zp `,` $weight_zp
    attr-dict `:` type(operands) `->` type(results)
  }];

  let extraClassDeclaration = [{
    ::mlir::spirv::TensorArmType getInputType() {
      return cast<::mlir::spirv::TensorArmType>(getInput().getType());
    }
    ::mlir::spirv::TensorArmType getWeightType() {
      return cast<::mlir::spirv::TensorArmType>(getWeight().getType());
    }
    ::mlir::spirv::TensorArmType getBiasType() {
      return cast<::mlir::spirv::TensorArmType>(getBias().getType());
    }
    ::mlir::spirv::TensorArmType getResultType() {
      return cast<::mlir::spirv::TensorArmType>(getType());
    }
  }];
}


def SPIRV_TosaConv3DOp : SPIRV_TosaOp<"Conv3D", 3, [Pure]> {
  let summary = "3D Convolution operator.";

  let description = [{
    Performs a 3D convolution over the given input tensor. Implementations
    may choose to skip calculation of multiplies in the padding area.

    References:
      * https://github.khronos.org/SPIRV-Registry/extended/TOSA.001000.1.html#_conv3d
      * https://www.mlplatform.org/tosa/tosa_spec_1_0_1.html#_conv3d

    #### Example:
    ```mlir
    %7 = spirv.Tosa.Conv3D pad = dense<0> : !spirv.arm.tensor<6xi32>, stride = dense<1> : !spirv.arm.tensor<3xi32>, dilation = dense<1> : !spirv.arm.tensor<3xi32>, acc_type = <INT32>, local_bound = false, %arg0, %arg1, %arg2, %5, %6 : !spirv.arm.tensor<1x9x21x14x1xi8>, !spirv.arm.tensor<2x1x2x1x1xi8>, !spirv.arm.tensor<1xi32>, !spirv.arm.tensor<1xi8>, !spirv.arm.tensor<1xi8> -> !spirv.arm.tensor<1x9x20x14x2xi32>
    %7 = spirv.Tosa.Conv3D pad = dense<[0, 1, 1, 0, 0, 1]> : !spirv.arm.tensor<6xi32>, stride = dense<1> : !spirv.arm.tensor<3xi32>, dilation = dense<[1, 1, 7]> : !spirv.arm.tensor<3xi32>, acc_type = <FP32>, local_bound = false, %arg0, %arg1, %arg2, %5, %6 : !spirv.arm.tensor<1x2x65539x1x2xf32>, !spirv.arm.tensor<1x1x1x1x2xf32>, !spirv.arm.tensor<1xf32>, !spirv.arm.tensor<1xf32>, !spirv.arm.tensor<1xf32> -> !spirv.arm.tensor<1x3x65540x2x1xf32>
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

  let hasVerifier = 1;

  let assemblyFormat = [{
    `pad` `=` $pad `,` `stride` `=` $stride `,`
    `dilation` `=` $dilation `,` `acc_type` `=` $acc_type `,`
    `local_bound` `=` $local_bound `,`
    $input `,` $weight `,` $bias `,` $input_zp `,` $weight_zp
    attr-dict `:` type(operands) `->` type(results)
  }];

  let extraClassDeclaration = [{
    ::mlir::spirv::TensorArmType getInputType() {
      return cast<::mlir::spirv::TensorArmType>(getInput().getType());
    }
    ::mlir::spirv::TensorArmType getWeightType() {
      return cast<::mlir::spirv::TensorArmType>(getWeight().getType());
    }
    ::mlir::spirv::TensorArmType getBiasType() {
      return cast<::mlir::spirv::TensorArmType>(getBias().getType());
    }
    ::mlir::spirv::TensorArmType getResultType() {
      return cast<::mlir::spirv::TensorArmType>(getType());
    }
  }];
}


def SPIRV_TosaDepthwiseConv2DOp : SPIRV_TosaOp<"DepthwiseConv2D", 4, [Pure]> {
  let summary = "Depthwise 2D Convolution operator.";

  let description = [{
    Performs 2D convolutions separately over each channel of the given tensor
    input, using the weight tensor. Implementations may choose to skip
    calculation of multiplies in the padding area.

    References:
      * https://github.khronos.org/SPIRV-Registry/extended/TOSA.001000.1.html#_depthwise_conv2d
      * https://www.mlplatform.org/tosa/tosa_spec_1_0_1.html#_depthwise_conv2d

    #### Example:
    ```mlir
    %7 = spirv.Tosa.DepthwiseConv2D pad = dense<0> : !spirv.arm.tensor<4xi32>, stride = dense<[1, 2]> : !spirv.arm.tensor<2xi32>, dilation = dense<7> : !spirv.arm.tensor<2xi32>, acc_type = <INT32>, local_bound = false, %arg0, %arg1, %arg2, %5, %6 : !spirv.arm.tensor<1x4x65537x1xi8>, !spirv.arm.tensor<1x3x1x4xi8>, !spirv.arm.tensor<4xi32>, !spirv.arm.tensor<1xi8>, !spirv.arm.tensor<1xi8> -> !spirv.arm.tensor<1x4x32762x4xi32>
    %7 = spirv.Tosa.DepthwiseConv2D pad = dense<[0, 1, 1, 1]> : !spirv.arm.tensor<4xi32>, stride = dense<[1, 2]> : !spirv.arm.tensor<2xi32>, dilation = dense<[1, 7]> : !spirv.arm.tensor<2xi32>, acc_type = <FP32>, local_bound = true, %arg0, %arg1, %arg2, %5, %6 : !spirv.arm.tensor<1x65540x1x3xf32>, !spirv.arm.tensor<1x1x3x1xf32>, !spirv.arm.tensor<1xf32>, !spirv.arm.tensor<1xf32>, !spirv.arm.tensor<1xf32> -> !spirv.arm.tensor<1x65541x2x3xf32>
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

  let hasVerifier = 1;

  let assemblyFormat = [{
    `pad` `=` $pad `,` `stride` `=` $stride `,`
    `dilation` `=` $dilation `,` `acc_type` `=` $acc_type `,`
    `local_bound` `=` $local_bound `,`
    $input `,` $weight `,` $bias `,` $input_zp `,` $weight_zp
    attr-dict `:` type(operands) `->` type(results)
  }];

  let extraClassDeclaration = [{
    ::mlir::spirv::TensorArmType getInputType() {
      return cast<::mlir::spirv::TensorArmType>(getInput().getType());
    }
    ::mlir::spirv::TensorArmType getWeightType() {
      return cast<::mlir::spirv::TensorArmType>(getWeight().getType());
    }
    ::mlir::spirv::TensorArmType getBiasType() {
      return cast<::mlir::spirv::TensorArmType>(getBias().getType());
    }
    ::mlir::spirv::TensorArmType getResultType() {
      return cast<::mlir::spirv::TensorArmType>(getType());
    }
  }];
}


def SPIRV_TosaFFT2DOp : SPIRV_TosaOp<"FFT2D", 5, [Pure]> {
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
    SPIRV_TosaFloat_TensorArm3D: $input_real,
    SPIRV_TosaFloat_TensorArm3D: $input_imag
  );

  let results = (outs
    SPIRV_Struct_2_TosaFloat_TensorArm3D: $output
  );

  let hasVerifier = 1;

  let assemblyFormat = [{
    `inverse` `=` $inverse `,` `local_bound` `=` $local_bound `,`
    $input_real `,` $input_imag
    attr-dict `:` type(operands) `->` type(results)
  }];

  let extraClassDeclaration = [{
    ::mlir::spirv::TensorArmType getInputRealType() {
      return cast<::mlir::spirv::TensorArmType>(getInputReal().getType());
    }
    ::mlir::spirv::TensorArmType getInputImagType() {
      return cast<::mlir::spirv::TensorArmType>(getInputImag().getType());
    }
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


def SPIRV_TosaMatMulOp : SPIRV_TosaOp<"MatMul", 6, [Pure]> {
  let summary = "Matrix Multiplication operator.";

  let description = [{
    Performs two dimensional matrix multiplications.

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

  let hasVerifier = 1;

  let assemblyFormat = [{
    $A `,` $B `,` $A_zp `,` $B_zp
    attr-dict `:` type(operands) `->` type(results)
  }];

  let extraClassDeclaration = [{
    ::mlir::spirv::TensorArmType getAType() {
      return cast<::mlir::spirv::TensorArmType>(getA().getType());
    }
    ::mlir::spirv::TensorArmType getBType() {
      return cast<::mlir::spirv::TensorArmType>(getB().getType());
    }
    ::mlir::spirv::TensorArmType getResultType() {
      return cast<::mlir::spirv::TensorArmType>(getType());
    }
  }];
}


def SPIRV_TosaMaxPool2DOp : SPIRV_TosaOp<"MaxPool2D", 7, [Pure]> {
  let summary = "Performs max pooling on the input.";

  let description = [{
    Performs a max pooling over the given input tensor. A sliding window of
    size given by <kernel size> is passed over the input tensor, with the
    maximum value being placed in the
    output tensor.

    References:
      * https://github.khronos.org/SPIRV-Registry/extended/TOSA.001000.1.html#_max_pool2d
      * https://www.mlplatform.org/tosa/tosa_spec_1_0_1.html#_max_pool2d

    #### Example:
    ```mlir
    %4 = spirv.Tosa.MaxPool2D kernel = dense<[3, 2]> : !spirv.arm.tensor<2xi32>, stride = dense<[1, 2]> : !spirv.arm.tensor<2xi32>, pad = dense<[1, 0, 0, 1]> : !spirv.arm.tensor<4xi32>, nan_mode = <Propagate>, %arg0 : !spirv.arm.tensor<1x3x65537x1xi8> -> !spirv.arm.tensor<1x2x32769x1xi8>
    %4 = spirv.Tosa.MaxPool2D kernel = dense<[3, 2]> : !spirv.arm.tensor<2xi32>, stride = dense<2> : !spirv.arm.tensor<2xi32>, pad = dense<[1, 0, 1, 1]> : !spirv.arm.tensor<4xi32>, nan_mode = <Propagate>, %arg0 : !spirv.arm.tensor<1x6x65536x1xf32> -> !spirv.arm.tensor<1x3x32769x1xf32>
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

  let hasVerifier = 1;

  let assemblyFormat = [{
    `kernel` `=` $kernel `,` `stride` `=` $stride `,`
    `pad` `=` $pad `,` `nan_mode` `=` $nan_mode `,`
    $input
    attr-dict `:` type(operands) `->` type(results)
  }];

  let extraClassDeclaration = [{
    ::mlir::spirv::TensorArmType getInputType() {
      return cast<::mlir::spirv::TensorArmType>(getInput().getType());
    }
    ::mlir::spirv::TensorArmType getResultType() {
      return cast<::mlir::spirv::TensorArmType>(getType());
    }
  }];
}


def SPIRV_TosaRFFT2DOp : SPIRV_TosaOp<"RFFT2D", 8, [Pure]> {
  let summary = "Performs RFFT2D operation on the input.";

  let description = [{
    Performs a batched 2D real-valued Fast Fourier Transform over the input where
    the input tensor consists of real values producing complex valued output. The
    complex output values will be split into the output_real and output_imag
    tensor arguments. RFFT2D takes advantage of Hermitian symmetry to only
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
    SPIRV_TosaFloat_TensorArm3D: $input_real
  );

  let results = (outs
    SPIRV_Struct_2_TosaFloat_TensorArm3D: $output
  );

  let hasVerifier = 1;

  let assemblyFormat = [{
    `local_bound` `=` $local_bound `,`
    $input_real
    attr-dict `:` type(operands) `->` type(results)
  }];

  let extraClassDeclaration = [{
    ::mlir::spirv::TensorArmType getInputRealType() {
      return cast<::mlir::spirv::TensorArmType>(getInputReal().getType());
    }
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


def SPIRV_TosaTransposeConv2DOp : SPIRV_TosaOp<"TransposeConv2D", 9, [Pure]> {
  let summary = "Transpose 2D Convolution operator.";

  let description = [{
    Performs a 2D transposed convolution over the given tensor input, using the
    weights tensor. Implementations may choose to skip calculation of multiplies
    by zero at fractional input positions.

    References:
      * https://github.khronos.org/SPIRV-Registry/extended/TOSA.001000.1.html#_transpose_conv2d
      * https://www.mlplatform.org/tosa/tosa_spec_1_0_1.html#_transpose_conv2d

    #### Example:
    ```mlir
    %6 = spirv.Tosa.TransposeConv2D out_pad = dense<0> : !spirv.arm.tensor<4xi32>, stride = dense<1> : !spirv.arm.tensor<2xi32>, acc_type = <INT48>, local_bound = false, %arg0, %arg1, %arg2, %4, %5 : !spirv.arm.tensor<1x13x33x3xi16>, !spirv.arm.tensor<11x1x3x3xi8>, !spirv.arm.tensor<1xi64>, !spirv.arm.tensor<1xi16>, !spirv.arm.tensor<1xi8> -> !spirv.arm.tensor<1x13x35x11xi64>
    %6 = spirv.Tosa.TransposeConv2D out_pad = dense<[0, 1, 0, 0]> : !spirv.arm.tensor<4xi32>, stride = dense<[1, 8]> : !spirv.arm.tensor<2xi32>, acc_type = <FP16>, local_bound = true, %arg0, %arg1, %arg2, %4, %5 : !spirv.arm.tensor<10x24x9x13xf16>, !spirv.arm.tensor<14x1x1x13xf16>, !spirv.arm.tensor<14xf16>, !spirv.arm.tensor<1xf16>, !spirv.arm.tensor<1xf16> -> !spirv.arm.tensor<10x25x65x14xf16>
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

  let hasVerifier = 1;

  let assemblyFormat = [{
    `out_pad` `=` $out_pad `,` `stride` `=` $stride `,`
    `acc_type` `=` $acc_type `,` `local_bound` `=` $local_bound `,`
    $input `,` $weight `,` $bias `,` $input_zp `,` $weight_zp
    attr-dict `:` type(operands) `->` type(results)
  }];

  let extraClassDeclaration = [{
    ::mlir::spirv::TensorArmType getInputType() {
      return cast<::mlir::spirv::TensorArmType>(getInput().getType());
    }
    ::mlir::spirv::TensorArmType getWeightType() {
      return cast<::mlir::spirv::TensorArmType>(getWeight().getType());
    }
    ::mlir::spirv::TensorArmType getBiasType() {
      return cast<::mlir::spirv::TensorArmType>(getBias().getType());
    }
    ::mlir::spirv::TensorArmType getResultType() {
      return cast<::mlir::spirv::TensorArmType>(getType());
    }
  }];
}


def SPIRV_TosaClampOp : SPIRV_TosaOp<"Clamp", 10, [Pure]> {
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

  let hasVerifier = 1;

  let assemblyFormat = [{
    `min_val` `=` $min_val `,` `max_val` `=` $max_val `,`
    `nan_mode` `=` $nan_mode `,`
    $input
    attr-dict `:` type(operands) `->` type(results)
  }];

  let extraClassDeclaration = [{
    ::mlir::spirv::TensorArmType getInputType() {
      return cast<::mlir::spirv::TensorArmType>(getInput().getType());
    }
    ::mlir::spirv::TensorArmType getResultType() {
      return cast<::mlir::spirv::TensorArmType>(getType());
    }
  }];
}


def SPIRV_TosaErfOp : SPIRV_TosaOp<"Erf", 11, [Pure]> {
  let summary = "Computes Gauss Error Function of input.";

  let description = [{
    Gauss Error Function: $ erf(x) = \frac{2}{\sqrt{\pi}} \int_{0}^{x} e^{-t^2} dt $
    For quantized integer data types, the table operator should be used instead
    with the following definition. The ERF table has 513 entries each of
    16-bit precision and covering the input range -4.0 to +4.0 in steps of 1/64.

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

  let hasVerifier = 0;

  let assemblyFormat = [{
    $input
    attr-dict `:` type(operands) `->` type(results)
  }];

  let extraClassDeclaration = [{
    ::mlir::spirv::TensorArmType getInputType() {
      return cast<::mlir::spirv::TensorArmType>(getInput().getType());
    }
    ::mlir::spirv::TensorArmType getResultType() {
      return cast<::mlir::spirv::TensorArmType>(getType());
    }
  }];
}


def SPIRV_TosaSigmoidOp : SPIRV_TosaOp<"Sigmoid", 12, [Pure]> {
  let summary = "Computes elementwise sigmoid of input.";

  let description = [{
    Applies the sigmoid logistic function to each element of the input tensor:
    $ sigmoid(x) = rac{1}{1 + e^{-x}} $.

    For quantized integer data types, the table operator should be used instead.
    Each implementation may choose an appropriate table given the scale and zero
    point of the input data. Eight or sixteen bit precision tables may be used
    based on the input tensor to the sigmoid function.

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

  let hasVerifier = 0;

  let assemblyFormat = [{
    $input
    attr-dict `:` type(operands) `->` type(results)
  }];

  let extraClassDeclaration = [{
    ::mlir::spirv::TensorArmType getInputType() {
      return cast<::mlir::spirv::TensorArmType>(getInput().getType());
    }
    ::mlir::spirv::TensorArmType getResultType() {
      return cast<::mlir::spirv::TensorArmType>(getType());
    }
  }];
}


def SPIRV_TosaTanhOp : SPIRV_TosaOp<"Tanh", 13, [Pure]> {
  let summary = "Computes elementwise Hyperbolic Tangent of input.";

  let description = [{
    Parameterized Hyperbolic Tangent: $ tanh(x) = rac{1 - e^{-2x}}{1 + e^{-2x}} $.

    For quantized integer data types, the table operator should be used instead.
    Each implementation may choose an appropriate table given the scale and zero
    point of the input data. Eight or sixteen bit precision tables may be used
    based on the input tensor to the tanh function.

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

  let hasVerifier = 0;

  let assemblyFormat = [{
    $input
    attr-dict `:` type(operands) `->` type(results)
  }];

  let extraClassDeclaration = [{
    ::mlir::spirv::TensorArmType getInputType() {
      return cast<::mlir::spirv::TensorArmType>(getInput().getType());
    }
    ::mlir::spirv::TensorArmType getResultType() {
      return cast<::mlir::spirv::TensorArmType>(getType());
    }
  }];
}


def SPIRV_TosaAddOp : SPIRV_TosaOp<"Add", 14, [Pure]> {
  let summary = "Elementwise Addition operator.";

  let description = [{
    Elementwise Addition of input1 and input2. Axis of size 1 will be broadcast,
    as necessary. Rank of input tensors must match.

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

  let hasVerifier = 0;

  let assemblyFormat = [{
    $input1 `,` $input2
    attr-dict `:` type(operands) `->` type(results)
  }];

  let extraClassDeclaration = [{
    ::mlir::spirv::TensorArmType getInput1Type() {
      return cast<::mlir::spirv::TensorArmType>(getInput1().getType());
    }
    ::mlir::spirv::TensorArmType getInput2Type() {
      return cast<::mlir::spirv::TensorArmType>(getInput2().getType());
    }
    ::mlir::spirv::TensorArmType getResultType() {
      return cast<::mlir::spirv::TensorArmType>(getType());
    }
  }];
}


def SPIRV_TosaArithmeticRightShiftOp : SPIRV_TosaOp<"ArithmeticRightShift", 15, [Pure]> {
  let summary = "Elementwise Arithmetic Right Shift.";

  let description = [{
    Elementwise Arithmetic Right Shift of input1 by the amount specified in
    input2. Axis of size 1 will be broadcast, as necessary. Rank of input tensors
    must match.

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

  let hasVerifier = 0;

  let assemblyFormat = [{
    `round` `=` $round `,`
    $input1 `,` $input2
    attr-dict `:` type(operands) `->` type(results)
  }];

  let extraClassDeclaration = [{
    ::mlir::spirv::TensorArmType getInput1Type() {
      return cast<::mlir::spirv::TensorArmType>(getInput1().getType());
    }
    ::mlir::spirv::TensorArmType getInput2Type() {
      return cast<::mlir::spirv::TensorArmType>(getInput2().getType());
    }
    ::mlir::spirv::TensorArmType getResultType() {
      return cast<::mlir::spirv::TensorArmType>(getType());
    }
  }];
}


def SPIRV_TosaBitwiseAndOp : SPIRV_TosaOp<"BitwiseAnd", 16, [Pure]> {
  let summary = "Bitwise AND operator.";

  let description = [{
    Elementwise Bitwise AND of input1 and input2. Axis of size 1
    will be broadcast as necessary. Rank of input tensors must match.

    References:
      * https://github.khronos.org/SPIRV-Registry/extended/TOSA.001000.1.html#_bitwise_and
      * https://www.mlplatform.org/tosa/tosa_spec_1_0_1.html#_bitwise_and

    #### Example:
    ```mlir
    %0 = spirv.Tosa.BitwiseAnd %arg0, %arg1 : !spirv.arm.tensor<4x1x7x12xi16>, !spirv.arm.tensor<4x13x7x12xi16> -> !spirv.arm.tensor<4x13x7x12xi16>
    ```
  }];

  let arguments = (ins
    SPIRV_TosaInteger_TensorArm: $input1,
    SPIRV_TosaInteger_TensorArm: $input2
  );

  let results = (outs
    SPIRV_TosaInteger_TensorArm: $output
  );

  let hasVerifier = 0;

  let assemblyFormat = [{
    $input1 `,` $input2
    attr-dict `:` type(operands) `->` type(results)
  }];

  let extraClassDeclaration = [{
    ::mlir::spirv::TensorArmType getInput1Type() {
      return cast<::mlir::spirv::TensorArmType>(getInput1().getType());
    }
    ::mlir::spirv::TensorArmType getInput2Type() {
      return cast<::mlir::spirv::TensorArmType>(getInput2().getType());
    }
    ::mlir::spirv::TensorArmType getResultType() {
      return cast<::mlir::spirv::TensorArmType>(getType());
    }
  }];
}


def SPIRV_TosaBitwiseOrOp : SPIRV_TosaOp<"BitwiseOr", 17, [Pure]> {
  let summary = "Bitwise OR operator.";

  let description = [{
    Elementwise Bitwise OR of input1 and input2. Axis of size 1 will be
    broadcast as necessary. Rank of input tensors must match.

    References:
      * https://github.khronos.org/SPIRV-Registry/extended/TOSA.001000.1.html#_bitwise_or
      * https://www.mlplatform.org/tosa/tosa_spec_1_0_1.html#_bitwise_or

    #### Example:
    ```mlir
    %0 = spirv.Tosa.BitwiseOr %arg0, %arg1 : !spirv.arm.tensor<11x30x23xi32>, !spirv.arm.tensor<1x30x23xi32> -> !spirv.arm.tensor<11x30x23xi32>
    ```
  }];

  let arguments = (ins
    SPIRV_TosaInteger_TensorArm: $input1,
    SPIRV_TosaInteger_TensorArm: $input2
  );

  let results = (outs
    SPIRV_TosaInteger_TensorArm: $output
  );

  let hasVerifier = 0;

  let assemblyFormat = [{
    $input1 `,` $input2
    attr-dict `:` type(operands) `->` type(results)
  }];

  let extraClassDeclaration = [{
    ::mlir::spirv::TensorArmType getInput1Type() {
      return cast<::mlir::spirv::TensorArmType>(getInput1().getType());
    }
    ::mlir::spirv::TensorArmType getInput2Type() {
      return cast<::mlir::spirv::TensorArmType>(getInput2().getType());
    }
    ::mlir::spirv::TensorArmType getResultType() {
      return cast<::mlir::spirv::TensorArmType>(getType());
    }
  }];
}


def SPIRV_TosaBitwiseXorOp : SPIRV_TosaOp<"BitwiseXor", 18, [Pure]> {
  let summary = "Bitwise XOR operator.";

  let description = [{
    Elementwise Bitwise XOR of input1 and input2. Axis of size 1 will be
    broadcast as necessary. Rank of input tensors must match.

    References:
      * https://github.khronos.org/SPIRV-Registry/extended/TOSA.001000.1.html#_bitwise_xor
      * https://www.mlplatform.org/tosa/tosa_spec_1_0_1.html#_bitwise_xor

    #### Example:
    ```mlir
    %0 = spirv.Tosa.BitwiseXor %arg0, %arg1 : !spirv.arm.tensor<4x8x13x9xi16>, !spirv.arm.tensor<4x8x1x9xi16> -> !spirv.arm.tensor<4x8x13x9xi16>
    ```
  }];

  let arguments = (ins
    SPIRV_TosaInteger_TensorArm: $input1,
    SPIRV_TosaInteger_TensorArm: $input2
  );

  let results = (outs
    SPIRV_TosaInteger_TensorArm: $output
  );

  let hasVerifier = 0;

  let assemblyFormat = [{
    $input1 `,` $input2
    attr-dict `:` type(operands) `->` type(results)
  }];

  let extraClassDeclaration = [{
    ::mlir::spirv::TensorArmType getInput1Type() {
      return cast<::mlir::spirv::TensorArmType>(getInput1().getType());
    }
    ::mlir::spirv::TensorArmType getInput2Type() {
      return cast<::mlir::spirv::TensorArmType>(getInput2().getType());
    }
    ::mlir::spirv::TensorArmType getResultType() {
      return cast<::mlir::spirv::TensorArmType>(getType());
    }
  }];
}


def SPIRV_TosaIntDivOp : SPIRV_TosaOp<"IntDiv", 19, [Pure]> {
  let summary = "Integer Divide operator.";

  let description = [{
    Elementwise Integer Divide of input1 by input2. Axis of size 1 will be
    broadcast as necessary. Rank of input tensors must match. The result of the
    divide is truncated towards zero. Expected use is for operations on
    non-scaled integers. Floating point divide should use RECIPROCAL and MUL.
    Quantized integer divide should use table (for 1/x) and MUL.

    References:
      * https://github.khronos.org/SPIRV-Registry/extended/TOSA.001000.1.html#_intdiv
      * https://www.mlplatform.org/tosa/tosa_spec_1_0_1.html#_intdiv

    #### Example:
    ```mlir
    %0 = spirv.Tosa.IntDiv %arg0, %arg1 : !spirv.arm.tensor<1x65533x1xi32>, !spirv.arm.tensor<2x65533x1xi32> -> !spirv.arm.tensor<2x65533x1xi32>
    ```
  }];

  let arguments = (ins
    SPIRV_TosaInteger_TensorArm: $input1,
    SPIRV_TosaInteger_TensorArm: $input2
  );

  let results = (outs
    SPIRV_TosaInteger_TensorArm: $output
  );

  let hasVerifier = 0;

  let assemblyFormat = [{
    $input1 `,` $input2
    attr-dict `:` type(operands) `->` type(results)
  }];

  let extraClassDeclaration = [{
    ::mlir::spirv::TensorArmType getInput1Type() {
      return cast<::mlir::spirv::TensorArmType>(getInput1().getType());
    }
    ::mlir::spirv::TensorArmType getInput2Type() {
      return cast<::mlir::spirv::TensorArmType>(getInput2().getType());
    }
    ::mlir::spirv::TensorArmType getResultType() {
      return cast<::mlir::spirv::TensorArmType>(getType());
    }
  }];
}


def SPIRV_TosaLogicalAndOp : SPIRV_TosaOp<"LogicalAnd", 20, [Pure]> {
  let summary = "Returns the truth value of input1 AND input2 element-wise.";

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

  let arguments = (ins
    SPIRV_Bool_TensorArm: $input1,
    SPIRV_Bool_TensorArm: $input2
  );

  let results = (outs
    SPIRV_Bool_TensorArm: $output
  );

  let hasVerifier = 0;

  let assemblyFormat = [{
    $input1 `,` $input2
    attr-dict `:` type(operands) `->` type(results)
  }];

  let extraClassDeclaration = [{
    ::mlir::spirv::TensorArmType getInput1Type() {
      return cast<::mlir::spirv::TensorArmType>(getInput1().getType());
    }
    ::mlir::spirv::TensorArmType getInput2Type() {
      return cast<::mlir::spirv::TensorArmType>(getInput2().getType());
    }
    ::mlir::spirv::TensorArmType getResultType() {
      return cast<::mlir::spirv::TensorArmType>(getType());
    }
  }];
}


def SPIRV_TosaLogicalLeftShiftOp : SPIRV_TosaOp<"LogicalLeftShift", 21, [Pure]> {
  let summary = "Elementwise Logical Left Shift.";

  let description = [{
    Elementwise Logical Left Shift of input1 by the amount specified in input2.
    Axis of size 1 will be broadcast, as necessary.
    Rank of input tensors must match.

    References:
      * https://github.khronos.org/SPIRV-Registry/extended/TOSA.001000.1.html#_logical_left_shift
      * https://www.mlplatform.org/tosa/tosa_spec_1_0_1.html#_logical_left_shift

    #### Example:
    ```mlir
    %0 = spirv.Tosa.LogicalLeftShift %arg0, %arg1 : !spirv.arm.tensor<7x1x11x4xi8>, !spirv.arm.tensor<7x8x11x4xi8> -> !spirv.arm.tensor<7x8x11x4xi8>
    ```
  }];

  let arguments = (ins
    SPIRV_TosaInteger_TensorArm: $input1,
    SPIRV_TosaInteger_TensorArm: $input2
  );

  let results = (outs
    SPIRV_TosaInteger_TensorArm: $output
  );

  let hasVerifier = 0;

  let assemblyFormat = [{
    $input1 `,` $input2
    attr-dict `:` type(operands) `->` type(results)
  }];

  let extraClassDeclaration = [{
    ::mlir::spirv::TensorArmType getInput1Type() {
      return cast<::mlir::spirv::TensorArmType>(getInput1().getType());
    }
    ::mlir::spirv::TensorArmType getInput2Type() {
      return cast<::mlir::spirv::TensorArmType>(getInput2().getType());
    }
    ::mlir::spirv::TensorArmType getResultType() {
      return cast<::mlir::spirv::TensorArmType>(getType());
    }
  }];
}


def SPIRV_TosaLogicalRightShiftOp : SPIRV_TosaOp<"LogicalRightShift", 22, [Pure]> {
  let summary = "Elementwise Logical Right Shift.";

  let description = [{
    Elementwise Logical Right Shift of input1 by the amount specified in input2.
    Axis of size 1 will be broadcast, as necessary. Rank of input tensors must
    match.

    References:
      * https://github.khronos.org/SPIRV-Registry/extended/TOSA.001000.1.html#_logical_right_shift
      * https://www.mlplatform.org/tosa/tosa_spec_1_0_1.html#_logical_right_shift

    #### Example:
    ```mlir
    %0 = spirv.Tosa.LogicalRightShift %arg0, %arg1 : !spirv.arm.tensor<6x13x1x19xi8>, !spirv.arm.tensor<6x13x6x19xi8> -> !spirv.arm.tensor<6x13x6x19xi8>
    ```
  }];

  let arguments = (ins
    SPIRV_TosaInteger_TensorArm: $input1,
    SPIRV_TosaInteger_TensorArm: $input2
  );

  let results = (outs
    SPIRV_TosaInteger_TensorArm: $output
  );

  let hasVerifier = 0;

  let assemblyFormat = [{
    $input1 `,` $input2
    attr-dict `:` type(operands) `->` type(results)
  }];

  let extraClassDeclaration = [{
    ::mlir::spirv::TensorArmType getInput1Type() {
      return cast<::mlir::spirv::TensorArmType>(getInput1().getType());
    }
    ::mlir::spirv::TensorArmType getInput2Type() {
      return cast<::mlir::spirv::TensorArmType>(getInput2().getType());
    }
    ::mlir::spirv::TensorArmType getResultType() {
      return cast<::mlir::spirv::TensorArmType>(getType());
    }
  }];
}


def SPIRV_TosaLogicalOrOp : SPIRV_TosaOp<"LogicalOr", 23, [Pure]> {
  let summary = "Returns the truth value of x OR y element-wise.";

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

  let arguments = (ins
    SPIRV_Bool_TensorArm: $input1,
    SPIRV_Bool_TensorArm: $input2
  );

  let results = (outs
    SPIRV_Bool_TensorArm: $output
  );

  let hasVerifier = 0;

  let assemblyFormat = [{
    $input1 `,` $input2
    attr-dict `:` type(operands) `->` type(results)
  }];

  let extraClassDeclaration = [{
    ::mlir::spirv::TensorArmType getInput1Type() {
      return cast<::mlir::spirv::TensorArmType>(getInput1().getType());
    }
    ::mlir::spirv::TensorArmType getInput2Type() {
      return cast<::mlir::spirv::TensorArmType>(getInput2().getType());
    }
    ::mlir::spirv::TensorArmType getResultType() {
      return cast<::mlir::spirv::TensorArmType>(getType());
    }
  }];
}


def SPIRV_TosaLogicalXorOp : SPIRV_TosaOp<"LogicalXor", 24, [Pure]> {
  let summary = "Returns the truth value of input1 XOR input2 element-wise.";

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

  let arguments = (ins
    SPIRV_Bool_TensorArm: $input1,
    SPIRV_Bool_TensorArm: $input2
  );

  let results = (outs
    SPIRV_Bool_TensorArm: $output
  );

  let hasVerifier = 0;

  let assemblyFormat = [{
    $input1 `,` $input2
    attr-dict `:` type(operands) `->` type(results)
  }];

  let extraClassDeclaration = [{
    ::mlir::spirv::TensorArmType getInput1Type() {
      return cast<::mlir::spirv::TensorArmType>(getInput1().getType());
    }
    ::mlir::spirv::TensorArmType getInput2Type() {
      return cast<::mlir::spirv::TensorArmType>(getInput2().getType());
    }
    ::mlir::spirv::TensorArmType getResultType() {
      return cast<::mlir::spirv::TensorArmType>(getType());
    }
  }];
}


def SPIRV_TosaMaximumOp : SPIRV_TosaOp<"Maximum", 25, [Pure]> {
  let summary = "Elementwise Maximum.";

  let description = [{
    Elementwise max of input1 and input2. Axis of size 1 will be broadcast, as
    necessary. Rank of input tensors must match.

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

  let hasVerifier = 0;

  let assemblyFormat = [{
    `nan_mode` `=` $nan_mode `,`
    $input1 `,` $input2
    attr-dict `:` type(operands) `->` type(results)
  }];

  let extraClassDeclaration = [{
    ::mlir::spirv::TensorArmType getInput1Type() {
      return cast<::mlir::spirv::TensorArmType>(getInput1().getType());
    }
    ::mlir::spirv::TensorArmType getInput2Type() {
      return cast<::mlir::spirv::TensorArmType>(getInput2().getType());
    }
    ::mlir::spirv::TensorArmType getResultType() {
      return cast<::mlir::spirv::TensorArmType>(getType());
    }
  }];
}


def SPIRV_TosaMinimumOp : SPIRV_TosaOp<"Minimum", 26, [Pure]> {
  let summary = "Elementwise Minimum.";

  let description = [{
    Elementwise Multiplication (Hadamard product) of input1 and input2.
    Axis of size 1 will be broadcast, as necessary. Rank of input tensors must
    match.

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

  let hasVerifier = 0;

  let assemblyFormat = [{
    `nan_mode` `=` $nan_mode `,`
    $input1 `,` $input2
    attr-dict `:` type(operands) `->` type(results)
  }];

  let extraClassDeclaration = [{
    ::mlir::spirv::TensorArmType getInput1Type() {
      return cast<::mlir::spirv::TensorArmType>(getInput1().getType());
    }
    ::mlir::spirv::TensorArmType getInput2Type() {
      return cast<::mlir::spirv::TensorArmType>(getInput2().getType());
    }
    ::mlir::spirv::TensorArmType getResultType() {
      return cast<::mlir::spirv::TensorArmType>(getType());
    }
  }];
}


def SPIRV_TosaMulOp : SPIRV_TosaOp<"Mul", 27, [Pure]> {
  let summary = "Multiplication operator.";

  let description = [{
    Elementwise Multiplication (Hadamard product) of input1 and input2.

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

  let hasVerifier = 1;

  let assemblyFormat = [{
    $input1 `,` $input2 `,` $shift
    attr-dict `:` type(operands) `->` type(results)
  }];

  let extraClassDeclaration = [{
    ::mlir::spirv::TensorArmType getInput1Type() {
      return cast<::mlir::spirv::TensorArmType>(getInput1().getType());
    }
    ::mlir::spirv::TensorArmType getInput2Type() {
      return cast<::mlir::spirv::TensorArmType>(getInput2().getType());
    }
    ::mlir::spirv::TensorArmType getResultType() {
      return cast<::mlir::spirv::TensorArmType>(getType());
    }
  }];
}


def SPIRV_TosaPowOp : SPIRV_TosaOp<"Pow", 28, [Pure]> {
  let summary = "Computes the Power of one value to another.";

  let description = [{
    Elementwise input1 value raised to the Power of input2.
    Axis of size 1 will be broadcast, as necessary. Rank of input tensors must
    match.

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

  let hasVerifier = 0;

  let assemblyFormat = [{
    $input1 `,` $input2
    attr-dict `:` type(operands) `->` type(results)
  }];

  let extraClassDeclaration = [{
    ::mlir::spirv::TensorArmType getInput1Type() {
      return cast<::mlir::spirv::TensorArmType>(getInput1().getType());
    }
    ::mlir::spirv::TensorArmType getInput2Type() {
      return cast<::mlir::spirv::TensorArmType>(getInput2().getType());
    }
    ::mlir::spirv::TensorArmType getResultType() {
      return cast<::mlir::spirv::TensorArmType>(getType());
    }
  }];
}


def SPIRV_TosaSubOp : SPIRV_TosaOp<"Sub", 29, [Pure]> {
  let summary = "Elementwise Subtraction operator.";

  let description = [{
    Elementwise Subtraction of input1 and input2. Axis of size 1 will be
    broadcast as necessary. Rank of input tensors must match.

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

  let hasVerifier = 0;

  let assemblyFormat = [{
    $input1 `,` $input2
    attr-dict `:` type(operands) `->` type(results)
  }];

  let extraClassDeclaration = [{
    ::mlir::spirv::TensorArmType getInput1Type() {
      return cast<::mlir::spirv::TensorArmType>(getInput1().getType());
    }
    ::mlir::spirv::TensorArmType getInput2Type() {
      return cast<::mlir::spirv::TensorArmType>(getInput2().getType());
    }
    ::mlir::spirv::TensorArmType getResultType() {
      return cast<::mlir::spirv::TensorArmType>(getType());
    }
  }];
}


def SPIRV_TosaTableOp : SPIRV_TosaOp<"Table", 30, [Pure]> {
  let summary = "Table lookup operator.";

  let description = [{
    Table lookup operation. For int8_t table operation, perform a 256 entry
    table lookup returning an int8_t value. For int16_t tables, the int16_t
    input is treated as a fixed-point 9.7 value. The most significant 9 bits
    are used to index into the table. The fractional 7 bits are used to
    interpolate based on table[index] and table[index+1]. For int16_t inputs,
    the table operator returns a 16.7 interpolated value in an int32_t. This
    value can then be input to the rescale operator to scale to the required
    output data type. Note that int16_t table has 513 values to handle
    table[index+1] when index=511.

    An int16_t to int16_t table lookup can be constructed as follows:
    * Use the table operator to produce a fixed point 16.7 interpolated result
    * Use rescale (in_t=int32_t, out_t=int16_t, scale=1<<14, shift=21) to
      scale the output to int16_t range (or alternate scale as required)

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

  let hasVerifier = 0;

  let assemblyFormat = [{
    $input1 `,` $table
    attr-dict `:` type(operands) `->` type(results)
  }];

  let extraClassDeclaration = [{
    ::mlir::spirv::TensorArmType getInput1Type() {
      return cast<::mlir::spirv::TensorArmType>(getInput1().getType());
    }
    ::mlir::spirv::TensorArmType getResultType() {
      return cast<::mlir::spirv::TensorArmType>(getType());
    }
  }];
}


def SPIRV_TosaAbsOp : SPIRV_TosaOp<"Abs", 31, [Pure]> {
  let summary = "Elementwise ABS operator.";

  let description = [{
    Elementwise Absolute value operation.

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

  let hasVerifier = 0;

  let assemblyFormat = [{
    $input1
    attr-dict `:` type(operands) `->` type(results)
  }];

  let extraClassDeclaration = [{
    ::mlir::spirv::TensorArmType getInput1Type() {
      return cast<::mlir::spirv::TensorArmType>(getInput1().getType());
    }
    ::mlir::spirv::TensorArmType getResultType() {
      return cast<::mlir::spirv::TensorArmType>(getType());
    }
  }];
}


def SPIRV_TosaBitwiseNotOp : SPIRV_TosaOp<"BitwiseNot", 32, [Pure]> {
  let summary = "Bitwise NOT operator.";

  let description = [{
    Elementwise Bitwise NOT of input tensor.

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

  let hasVerifier = 0;

  let assemblyFormat = [{
    $input1
    attr-dict `:` type(operands) `->` type(results)
  }];

  let extraClassDeclaration = [{
    ::mlir::spirv::TensorArmType getInput1Type() {
      return cast<::mlir::spirv::TensorArmType>(getInput1().getType());
    }
    ::mlir::spirv::TensorArmType getResultType() {
      return cast<::mlir::spirv::TensorArmType>(getType());
    }
  }];
}


def SPIRV_TosaCeilOp : SPIRV_TosaOp<"Ceil", 33, [Pure]> {
  let summary = "Elementwise Ceil operator.";

  let description = [{
    Elementwise Ceiling operation.

    References:
      * https://github.khronos.org/SPIRV-Registry/extended/TOSA.001000.1.html#_ceil
      * https://www.mlplatform.org/tosa/tosa_spec_1_0_1.html#_ceil

    #### Example:
    ```mlir
    %0 = spirv.Tosa.Ceil %arg0 : !spirv.arm.tensor<46x55x53xf16> -> !spirv.arm.tensor<46x55x53xf16>
    ```
  }];

  let arguments = (ins
    SPIRV_TosaFloat_TensorArm: $input1
  );

  let results = (outs
    SPIRV_TosaFloat_TensorArm: $output
  );

  let hasVerifier = 0;

  let assemblyFormat = [{
    $input1
    attr-dict `:` type(operands) `->` type(results)
  }];

  let extraClassDeclaration = [{
    ::mlir::spirv::TensorArmType getInput1Type() {
      return cast<::mlir::spirv::TensorArmType>(getInput1().getType());
    }
    ::mlir::spirv::TensorArmType getResultType() {
      return cast<::mlir::spirv::TensorArmType>(getType());
    }
  }];
}


def SPIRV_TosaClzOp : SPIRV_TosaOp<"Clz", 34, [Pure]> {
  let summary = "Elementwise Count Leading Zero operator.";

  let description = [{
    Elementwise Count Leading Zeros operation.

    References:
      * https://github.khronos.org/SPIRV-Registry/extended/TOSA.001000.1.html#_clz
      * https://www.mlplatform.org/tosa/tosa_spec_1_0_1.html#_clz

    #### Example:
    ```mlir
    %0 = spirv.Tosa.Clz %arg0 : !spirv.arm.tensor<14x10x7x5xi32> -> !spirv.arm.tensor<14x10x7x5xi32>
    ```
  }];

  let arguments = (ins
    SPIRV_TosaInteger_TensorArm: $input1
  );

  let results = (outs
    SPIRV_TosaInteger_TensorArm: $output
  );

  let hasVerifier = 0;

  let assemblyFormat = [{
    $input1
    attr-dict `:` type(operands) `->` type(results)
  }];

  let extraClassDeclaration = [{
    ::mlir::spirv::TensorArmType getInput1Type() {
      return cast<::mlir::spirv::TensorArmType>(getInput1().getType());
    }
    ::mlir::spirv::TensorArmType getResultType() {
      return cast<::mlir::spirv::TensorArmType>(getType());
    }
  }];
}


def SPIRV_TosaCosOp : SPIRV_TosaOp<"Cos", 35, [Pure]> {
  let summary = "Elementwise Cosine operator.";

  let description = [{
    Elementwise Cosine operation for values given in radians.

    References:
      * https://github.khronos.org/SPIRV-Registry/extended/TOSA.001000.1.html#_cos
      * https://www.mlplatform.org/tosa/tosa_spec_1_0_1.html#_cos

    #### Example:
    ```mlir
    %0 = spirv.Tosa.Cos %arg0 : !spirv.arm.tensor<44x49x51xf32> -> !spirv.arm.tensor<44x49x51xf32>
    ```
  }];

  let arguments = (ins
    SPIRV_TosaFloat_TensorArm: $input1
  );

  let results = (outs
    SPIRV_TosaFloat_TensorArm: $output
  );

  let hasVerifier = 0;

  let assemblyFormat = [{
    $input1
    attr-dict `:` type(operands) `->` type(results)
  }];

  let extraClassDeclaration = [{
    ::mlir::spirv::TensorArmType getInput1Type() {
      return cast<::mlir::spirv::TensorArmType>(getInput1().getType());
    }
    ::mlir::spirv::TensorArmType getResultType() {
      return cast<::mlir::spirv::TensorArmType>(getType());
    }
  }];
}


def SPIRV_TosaExpOp : SPIRV_TosaOp<"Exp", 36, [Pure]> {
  let summary = "Elementwise exp operator.";

  let description = [{
    Elementwise e to the power of x operation.

    References:
      * https://github.khronos.org/SPIRV-Registry/extended/TOSA.001000.1.html#_exp
      * https://www.mlplatform.org/tosa/tosa_spec_1_0_1.html#_exp

    #### Example:
    ```mlir
    %0 = spirv.Tosa.Exp %arg0 : !spirv.arm.tensor<37x53x47xf32> -> !spirv.arm.tensor<37x53x47xf32>
    ```
  }];

  let arguments = (ins
    SPIRV_TosaFloat_TensorArm: $input1
  );

  let results = (outs
    SPIRV_TosaFloat_TensorArm: $output
  );

  let hasVerifier = 0;

  let assemblyFormat = [{
    $input1
    attr-dict `:` type(operands) `->` type(results)
  }];

  let extraClassDeclaration = [{
    ::mlir::spirv::TensorArmType getInput1Type() {
      return cast<::mlir::spirv::TensorArmType>(getInput1().getType());
    }
    ::mlir::spirv::TensorArmType getResultType() {
      return cast<::mlir::spirv::TensorArmType>(getType());
    }
  }];
}


def SPIRV_TosaFloorOp : SPIRV_TosaOp<"Floor", 37, [Pure]> {
  let summary = "Elementwise Floor operator.";

  let description = [{
    Elementwise Floor operation.

    References:
      * https://github.khronos.org/SPIRV-Registry/extended/TOSA.001000.1.html#_floor
      * https://www.mlplatform.org/tosa/tosa_spec_1_0_1.html#_floor

    #### Example:
    ```mlir
    %0 = spirv.Tosa.Floor %arg0 : !spirv.arm.tensor<40x52x42xf32> -> !spirv.arm.tensor<40x52x42xf32>
    ```
  }];

  let arguments = (ins
    SPIRV_TosaFloat_TensorArm: $input1
  );

  let results = (outs
    SPIRV_TosaFloat_TensorArm: $output
  );

  let hasVerifier = 0;

  let assemblyFormat = [{
    $input1
    attr-dict `:` type(operands) `->` type(results)
  }];

  let extraClassDeclaration = [{
    ::mlir::spirv::TensorArmType getInput1Type() {
      return cast<::mlir::spirv::TensorArmType>(getInput1().getType());
    }
    ::mlir::spirv::TensorArmType getResultType() {
      return cast<::mlir::spirv::TensorArmType>(getType());
    }
  }];
}


def SPIRV_TosaLogOp : SPIRV_TosaOp<"Log", 38, [Pure]> {
  let summary = "Elementwise Log operator.";

  let description = [{
    Elementwise Natural Logarithm operation.

    References:
      * https://github.khronos.org/SPIRV-Registry/extended/TOSA.001000.1.html#_log
      * https://www.mlplatform.org/tosa/tosa_spec_1_0_1.html#_log

    #### Example:
    ```mlir
    %0 = spirv.Tosa.Log %arg0 : !spirv.arm.tensor<45x43x36xf16> -> !spirv.arm.tensor<45x43x36xf16>
    ```
  }];

  let arguments = (ins
    SPIRV_TosaFloat_TensorArm: $input1
  );

  let results = (outs
    SPIRV_TosaFloat_TensorArm: $output
  );

  let hasVerifier = 0;

  let assemblyFormat = [{
    $input1
    attr-dict `:` type(operands) `->` type(results)
  }];

  let extraClassDeclaration = [{
    ::mlir::spirv::TensorArmType getInput1Type() {
      return cast<::mlir::spirv::TensorArmType>(getInput1().getType());
    }
    ::mlir::spirv::TensorArmType getResultType() {
      return cast<::mlir::spirv::TensorArmType>(getType());
    }
  }];
}


def SPIRV_TosaLogicalNotOp : SPIRV_TosaOp<"LogicalNot", 39, [Pure]> {
  let summary = "Returns the truth value of NOT input1 element-wise.";

  let description = [{
    Elementwise Logical NOT of input.

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

  let hasVerifier = 0;

  let assemblyFormat = [{
    $input1
    attr-dict `:` type(operands) `->` type(results)
  }];

  let extraClassDeclaration = [{
    ::mlir::spirv::TensorArmType getInput1Type() {
      return cast<::mlir::spirv::TensorArmType>(getInput1().getType());
    }
    ::mlir::spirv::TensorArmType getResultType() {
      return cast<::mlir::spirv::TensorArmType>(getType());
    }
  }];
}


def SPIRV_TosaNegateOp : SPIRV_TosaOp<"Negate", 40, [Pure]> {
  let summary = "Elementwise negate operator.";

  let description = [{
    Elementwise Negation operation.

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

  let hasVerifier = 0;

  let assemblyFormat = [{
    $input1 `,` $input1_zp `,` $output_zp
    attr-dict `:` type(operands) `->` type(results)
  }];

  let extraClassDeclaration = [{
    ::mlir::spirv::TensorArmType getInput1Type() {
      return cast<::mlir::spirv::TensorArmType>(getInput1().getType());
    }
    ::mlir::spirv::TensorArmType getResultType() {
      return cast<::mlir::spirv::TensorArmType>(getType());
    }
  }];
}


def SPIRV_TosaReciprocalOp : SPIRV_TosaOp<"Reciprocal", 41, [Pure]> {
  let summary = "Elementwise Reciprocal operator.";

  let description = [{
    Elementwise Reciprocal operation. For integer operation, a table should be
    used with the appropriate ranges.

    References:
      * https://github.khronos.org/SPIRV-Registry/extended/TOSA.001000.1.html#_reciprocal
      * https://www.mlplatform.org/tosa/tosa_spec_1_0_1.html#_reciprocal

    #### Example:
    ```mlir
    %0 = spirv.Tosa.Reciprocal %arg0 : !spirv.arm.tensor<38x47x44xf32> -> !spirv.arm.tensor<38x47x44xf32>
    ```
  }];

  let arguments = (ins
    SPIRV_TosaFloat_TensorArm: $input1
  );

  let results = (outs
    SPIRV_TosaFloat_TensorArm: $output
  );

  let hasVerifier = 0;

  let assemblyFormat = [{
    $input1
    attr-dict `:` type(operands) `->` type(results)
  }];

  let extraClassDeclaration = [{
    ::mlir::spirv::TensorArmType getInput1Type() {
      return cast<::mlir::spirv::TensorArmType>(getInput1().getType());
    }
    ::mlir::spirv::TensorArmType getResultType() {
      return cast<::mlir::spirv::TensorArmType>(getType());
    }
  }];
}


def SPIRV_TosaRsqrtOp : SPIRV_TosaOp<"Rsqrt", 42, [Pure]> {
  let summary = "Elementwise 1/sqrt operator.";

  let description = [{
    Elementwise Reciprocal Square Root operation. For integer operation, a table
    should be used with the appropriate ranges.

    References:
      * https://github.khronos.org/SPIRV-Registry/extended/TOSA.001000.1.html#_rsqrt
      * https://www.mlplatform.org/tosa/tosa_spec_1_0_1.html#_rsqrt

    #### Example:
    ```mlir
    %0 = spirv.Tosa.Rsqrt %arg0 : !spirv.arm.tensor<40x57x56xf32> -> !spirv.arm.tensor<40x57x56xf32>
    ```
  }];

  let arguments = (ins
    SPIRV_TosaFloat_TensorArm: $input1
  );

  let results = (outs
    SPIRV_TosaFloat_TensorArm: $output
  );

  let hasVerifier = 0;

  let assemblyFormat = [{
    $input1
    attr-dict `:` type(operands) `->` type(results)
  }];

  let extraClassDeclaration = [{
    ::mlir::spirv::TensorArmType getInput1Type() {
      return cast<::mlir::spirv::TensorArmType>(getInput1().getType());
    }
    ::mlir::spirv::TensorArmType getResultType() {
      return cast<::mlir::spirv::TensorArmType>(getType());
    }
  }];
}


def SPIRV_TosaSinOp : SPIRV_TosaOp<"Sin", 43, [Pure]> {
  let summary = "Elementwise Sin operator.";

  let description = [{
    Elementwise Sine operation for values given in radians.

    References:
      * https://github.khronos.org/SPIRV-Registry/extended/TOSA.001000.1.html#_sin
      * https://www.mlplatform.org/tosa/tosa_spec_1_0_1.html#_sin

    #### Example:
    ```mlir
    %0 = spirv.Tosa.Sin %arg0 : !spirv.arm.tensor<49x38x58xf16> -> !spirv.arm.tensor<49x38x58xf16>
    ```
  }];

  let arguments = (ins
    SPIRV_TosaFloat_TensorArm: $input1
  );

  let results = (outs
    SPIRV_TosaFloat_TensorArm: $output
  );

  let hasVerifier = 0;

  let assemblyFormat = [{
    $input1
    attr-dict `:` type(operands) `->` type(results)
  }];

  let extraClassDeclaration = [{
    ::mlir::spirv::TensorArmType getInput1Type() {
      return cast<::mlir::spirv::TensorArmType>(getInput1().getType());
    }
    ::mlir::spirv::TensorArmType getResultType() {
      return cast<::mlir::spirv::TensorArmType>(getType());
    }
  }];
}


def SPIRV_TosaSelectOp : SPIRV_TosaOp<"Select", 44, [Pure]> {
  let summary = "Elementwise Select operator.";

  let description = [{
    Elementwise Select of the output based on a condition.

    References:
      * https://github.khronos.org/SPIRV-Registry/extended/TOSA.001000.1.html#_select
      * https://www.mlplatform.org/tosa/tosa_spec_1_0_1.html#_select

    #### Example:
    ```mlir
    %0 = spirv.Tosa.Select %arg0, %arg1, %arg2 : !spirv.arm.tensor<4x1x4x5xi1>, !spirv.arm.tensor<4x6x4x5xi8>, !spirv.arm.tensor<4x6x4x5xi8> -> !spirv.arm.tensor<4x6x4x5xi8>
    %0 = spirv.Tosa.Select %arg0, %arg1, %arg2 : !spirv.arm.tensor<9x2x15x8xi1>, !spirv.arm.tensor<9x2x15x8xf16>, !spirv.arm.tensor<9x1x15x8xf16> -> !spirv.arm.tensor<9x2x15x8xf16>
    ```
  }];

  let arguments = (ins
    SPIRV_Bool_TensorArm: $input1,
    SPIRV_TosaAny_TensorArm: $input2,
    SPIRV_TosaAny_TensorArm: $input3
  );

  let results = (outs
    SPIRV_TosaAny_TensorArm: $output
  );

  let hasVerifier = 1;

  let assemblyFormat = [{
    $input1 `,` $input2 `,` $input3
    attr-dict `:` type(operands) `->` type(results)
  }];

  let extraClassDeclaration = [{
    ::mlir::spirv::TensorArmType getInput1Type() {
      return cast<::mlir::spirv::TensorArmType>(getInput1().getType());
    }
    ::mlir::spirv::TensorArmType getInput2Type() {
      return cast<::mlir::spirv::TensorArmType>(getInput2().getType());
    }
    ::mlir::spirv::TensorArmType getInput3Type() {
      return cast<::mlir::spirv::TensorArmType>(getInput3().getType());
    }
    ::mlir::spirv::TensorArmType getResultType() {
      return cast<::mlir::spirv::TensorArmType>(getType());
    }
  }];
}


def SPIRV_TosaEqualOp : SPIRV_TosaOp<"Equal", 45, [Pure]> {
  let summary = "Returns the truth value of (input1 == input2) element-wise.";

  let description = [{
    Elementwise Equal comparison operation.

    References:
      * https://github.khronos.org/SPIRV-Registry/extended/TOSA.001000.1.html#_equal
      * https://www.mlplatform.org/tosa/tosa_spec_1_0_1.html#_equal

    #### Example:
    ```mlir
    %0 = spirv.Tosa.Equal %arg0, %arg1 : !spirv.arm.tensor<51x28x59xi32>, !spirv.arm.tensor<51x1x59xi32> -> !spirv.arm.tensor<51x28x59xi1>
    %0 = spirv.Tosa.Equal %arg0, %arg1 : !spirv.arm.tensor<16x11x5x3xf32>, !spirv.arm.tensor<16x1x5x3xf32> -> !spirv.arm.tensor<16x11x5x3xi1>
    ```
  }];

  let arguments = (ins
    SPIRV_TosaNumerical_TensorArm: $input1,
    SPIRV_TosaNumerical_TensorArm: $input2
  );

  let results = (outs
    SPIRV_Bool_TensorArm: $output
  );

  let hasVerifier = 0;

  let assemblyFormat = [{
    $input1 `,` $input2
    attr-dict `:` type(operands) `->` type(results)
  }];

  let extraClassDeclaration = [{
    ::mlir::spirv::TensorArmType getInput1Type() {
      return cast<::mlir::spirv::TensorArmType>(getInput1().getType());
    }
    ::mlir::spirv::TensorArmType getInput2Type() {
      return cast<::mlir::spirv::TensorArmType>(getInput2().getType());
    }
    ::mlir::spirv::TensorArmType getResultType() {
      return cast<::mlir::spirv::TensorArmType>(getType());
    }
  }];
}


def SPIRV_TosaGreaterOp : SPIRV_TosaOp<"Greater", 46, [Pure]> {
  let summary = "Returns the truth value of (input1 > input2) element-wise.";

  let description = [{
    Elementwise Greater than comparison operation.

    References:
      * https://github.khronos.org/SPIRV-Registry/extended/TOSA.001000.1.html#_greater
      * https://www.mlplatform.org/tosa/tosa_spec_1_0_1.html#_greater

    #### Example:
    ```mlir
    %0 = spirv.Tosa.Greater %arg0, %arg1 : !spirv.arm.tensor<11x10x10x2xi32>, !spirv.arm.tensor<11x10x10x1xi32> -> !spirv.arm.tensor<11x10x10x2xi1>
    %0 = spirv.Tosa.Greater %arg0, %arg1 : !spirv.arm.tensor<6x3x12x4xf16>, !spirv.arm.tensor<6x3x1x4xf16> -> !spirv.arm.tensor<6x3x12x4xi1>
    ```
  }];

  let arguments = (ins
    SPIRV_TosaNumerical_TensorArm: $input1,
    SPIRV_TosaNumerical_TensorArm: $input2
  );

  let results = (outs
    SPIRV_Bool_TensorArm: $output
  );

  let hasVerifier = 0;

  let assemblyFormat = [{
    $input1 `,` $input2
    attr-dict `:` type(operands) `->` type(results)
  }];

  let extraClassDeclaration = [{
    ::mlir::spirv::TensorArmType getInput1Type() {
      return cast<::mlir::spirv::TensorArmType>(getInput1().getType());
    }
    ::mlir::spirv::TensorArmType getInput2Type() {
      return cast<::mlir::spirv::TensorArmType>(getInput2().getType());
    }
    ::mlir::spirv::TensorArmType getResultType() {
      return cast<::mlir::spirv::TensorArmType>(getType());
    }
  }];
}


def SPIRV_TosaGreaterEqualOp : SPIRV_TosaOp<"GreaterEqual", 47, [Pure]> {
  let summary = "Returns the truth value of (input1 >= input2) element-wise.";

  let description = [{
    Elementwise Greater or Equal than comparison operation.

    References:
      * https://github.khronos.org/SPIRV-Registry/extended/TOSA.001000.1.html#_greater_equal
      * https://www.mlplatform.org/tosa/tosa_spec_1_0_1.html#_greater_equal

    #### Example:
    ```mlir
    %0 = spirv.Tosa.GreaterEqual %arg0, %arg1 : !spirv.arm.tensor<10x17x7x1xi32>, !spirv.arm.tensor<10x17x7x16xi32> -> !spirv.arm.tensor<10x17x7x16xi1>
    %0 = spirv.Tosa.GreaterEqual %arg0, %arg1 : !spirv.arm.tensor<3x17x6x3xf32>, !spirv.arm.tensor<1x17x6x3xf32> -> !spirv.arm.tensor<3x17x6x3xi1>
    ```
  }];

  let arguments = (ins
    SPIRV_TosaNumerical_TensorArm: $input1,
    SPIRV_TosaNumerical_TensorArm: $input2
  );

  let results = (outs
    SPIRV_Bool_TensorArm: $output
  );

  let hasVerifier = 0;

  let assemblyFormat = [{
    $input1 `,` $input2
    attr-dict `:` type(operands) `->` type(results)
  }];

  let extraClassDeclaration = [{
    ::mlir::spirv::TensorArmType getInput1Type() {
      return cast<::mlir::spirv::TensorArmType>(getInput1().getType());
    }
    ::mlir::spirv::TensorArmType getInput2Type() {
      return cast<::mlir::spirv::TensorArmType>(getInput2().getType());
    }
    ::mlir::spirv::TensorArmType getResultType() {
      return cast<::mlir::spirv::TensorArmType>(getType());
    }
  }];
}


def SPIRV_TosaReduceAllOp : SPIRV_TosaOp<"ReduceAll", 48, [Pure]> {
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

  let hasVerifier = 0;

  let assemblyFormat = [{
    `axis` `=` $axis `,`
    $input
    attr-dict `:` type(operands) `->` type(results)
  }];

  let extraClassDeclaration = [{
    ::mlir::spirv::TensorArmType getInputType() {
      return cast<::mlir::spirv::TensorArmType>(getInput().getType());
    }
    ::mlir::spirv::TensorArmType getResultType() {
      return cast<::mlir::spirv::TensorArmType>(getType());
    }
  }];
}


def SPIRV_TosaReduceAnyOp : SPIRV_TosaOp<"ReduceAny", 49, [Pure]> {
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

  let hasVerifier = 0;

  let assemblyFormat = [{
    `axis` `=` $axis `,`
    $input
    attr-dict `:` type(operands) `->` type(results)
  }];

  let extraClassDeclaration = [{
    ::mlir::spirv::TensorArmType getInputType() {
      return cast<::mlir::spirv::TensorArmType>(getInput().getType());
    }
    ::mlir::spirv::TensorArmType getResultType() {
      return cast<::mlir::spirv::TensorArmType>(getType());
    }
  }];
}


def SPIRV_TosaReduceMaxOp : SPIRV_TosaOp<"ReduceMax", 50, [Pure]> {
  let summary = "Reduce Max operator.";

  let description = [{
    Reduces a tensor along the given axis with a Maximum operation.

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

  let hasVerifier = 0;

  let assemblyFormat = [{
    `axis` `=` $axis `,` `nan_mode` `=` $nan_mode `,`
    $input
    attr-dict `:` type(operands) `->` type(results)
  }];

  let extraClassDeclaration = [{
    ::mlir::spirv::TensorArmType getInputType() {
      return cast<::mlir::spirv::TensorArmType>(getInput().getType());
    }
    ::mlir::spirv::TensorArmType getResultType() {
      return cast<::mlir::spirv::TensorArmType>(getType());
    }
  }];
}


def SPIRV_TosaReduceMinOp : SPIRV_TosaOp<"ReduceMin", 51, [Pure]> {
  let summary = "Reduce Min operator.";

  let description = [{
    Reduces a tensor along the given axis with a Minimum operation.

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

  let hasVerifier = 0;

  let assemblyFormat = [{
    `axis` `=` $axis `,` `nan_mode` `=` $nan_mode `,`
    $input
    attr-dict `:` type(operands) `->` type(results)
  }];

  let extraClassDeclaration = [{
    ::mlir::spirv::TensorArmType getInputType() {
      return cast<::mlir::spirv::TensorArmType>(getInput().getType());
    }
    ::mlir::spirv::TensorArmType getResultType() {
      return cast<::mlir::spirv::TensorArmType>(getType());
    }
  }];
}


def SPIRV_TosaReduceProductOp : SPIRV_TosaOp<"ReduceProduct", 52, [Pure]> {
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

  let hasVerifier = 0;

  let assemblyFormat = [{
    `axis` `=` $axis `,`
    $input
    attr-dict `:` type(operands) `->` type(results)
  }];

  let extraClassDeclaration = [{
    ::mlir::spirv::TensorArmType getInputType() {
      return cast<::mlir::spirv::TensorArmType>(getInput().getType());
    }
    ::mlir::spirv::TensorArmType getResultType() {
      return cast<::mlir::spirv::TensorArmType>(getType());
    }
  }];
}


def SPIRV_TosaReduceSumOp : SPIRV_TosaOp<"ReduceSum", 53, [Pure]> {
  let summary = "Reduce Sum operator.";

  let description = [{
    Reduces a tensor along the given axis by computing the Sum of the axis.

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

  let hasVerifier = 0;

  let assemblyFormat = [{
    `axis` `=` $axis `,`
    $input
    attr-dict `:` type(operands) `->` type(results)
  }];

  let extraClassDeclaration = [{
    ::mlir::spirv::TensorArmType getInputType() {
      return cast<::mlir::spirv::TensorArmType>(getInput().getType());
    }
    ::mlir::spirv::TensorArmType getResultType() {
      return cast<::mlir::spirv::TensorArmType>(getType());
    }
  }];
}


def SPIRV_TosaConcatOp : SPIRV_TosaOp<"Concat", 54, [Pure]> {
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

  let hasVerifier = 1;

  let assemblyFormat = [{
    `axis` `=` $axis `,`
    $input1
    attr-dict `:` type(operands) `->` type(results)
  }];

  let extraClassDeclaration = [{
    ::mlir::spirv::TensorArmType getResultType() {
      return cast<::mlir::spirv::TensorArmType>(getType());
    }
  }];
}


def SPIRV_TosaPadOp : SPIRV_TosaOp<"Pad", 55, [Pure]> {
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

  let hasVerifier = 1;

  let assemblyFormat = [{
    $input1 `,` $padding `,` $pad_const
    attr-dict `:` type(operands) `->` type(results)
  }];

  let extraClassDeclaration = [{
    ::mlir::spirv::TensorArmType getInput1Type() {
      return cast<::mlir::spirv::TensorArmType>(getInput1().getType());
    }
    ::mlir::spirv::TensorArmType getResultType() {
      return cast<::mlir::spirv::TensorArmType>(getType());
    }
  }];
}


def SPIRV_TosaReshapeOp : SPIRV_TosaOp<"Reshape", 56, [Pure]> {
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
    %1 = spirv.Tosa.Reshape %arg0, %0 : !spirv.arm.tensor<25x6x29x35xi16>, !spirv.arm.tensor<4xi32> -> !spirv.arm.tensor<125x6x7x29xi16>
    %1 = spirv.Tosa.Reshape %arg0, %0 : !spirv.arm.tensor<1x2x7x2xf32>, !spirv.arm.tensor<3xi32> -> !spirv.arm.tensor<2x1x14xf32>
    ```
  }];

  let arguments = (ins
    SPIRV_TosaAny_TensorArm: $input1,
    SPIRV_Int32_1DTensorArmOfLength1To6: $shape
  );

  let results = (outs
    SPIRV_TosaAny_TensorArm: $output
  );

  let hasVerifier = 1;

  let assemblyFormat = [{
    $input1 `,` $shape
    attr-dict `:` type(operands) `->` type(results)
  }];

  let extraClassDeclaration = [{
    ::mlir::spirv::TensorArmType getInput1Type() {
      return cast<::mlir::spirv::TensorArmType>(getInput1().getType());
    }
    ::mlir::spirv::TensorArmType getResultType() {
      return cast<::mlir::spirv::TensorArmType>(getType());
    }
  }];
}


def SPIRV_TosaReverseOp : SPIRV_TosaOp<"Reverse", 57, [Pure]> {
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

  let hasVerifier = 1;

  let assemblyFormat = [{
    `axis` `=` $axis `,`
    $input1
    attr-dict `:` type(operands) `->` type(results)
  }];

  let extraClassDeclaration = [{
    ::mlir::spirv::TensorArmType getInput1Type() {
      return cast<::mlir::spirv::TensorArmType>(getInput1().getType());
    }
    ::mlir::spirv::TensorArmType getResultType() {
      return cast<::mlir::spirv::TensorArmType>(getType());
    }
  }];
}


def SPIRV_TosaSliceOp : SPIRV_TosaOp<"Slice", 58, [Pure]> {
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
    %2 = spirv.Tosa.Slice %arg0, %0, %1 : !spirv.arm.tensor<32x19x41xi8>, !spirv.arm.tensor<3xi32>, !spirv.arm.tensor<3xi32> -> !spirv.arm.tensor<21x5x2xi8>
    %2 = spirv.Tosa.Slice %arg0, %0, %1 : !spirv.arm.tensor<30x45x29xf32>, !spirv.arm.tensor<3xi32>, !spirv.arm.tensor<3xi32> -> !spirv.arm.tensor<5x12x11xf32>
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

  let hasVerifier = 1;

  let assemblyFormat = [{
    $input1 `,` $start `,` $size
    attr-dict `:` type(operands) `->` type(results)
  }];

  let extraClassDeclaration = [{
    ::mlir::spirv::TensorArmType getInput1Type() {
      return cast<::mlir::spirv::TensorArmType>(getInput1().getType());
    }
    ::mlir::spirv::TensorArmType getResultType() {
      return cast<::mlir::spirv::TensorArmType>(getType());
    }
  }];
}


def SPIRV_TosaTileOp : SPIRV_TosaOp<"Tile", 59, [Pure]> {
  let summary = "Tile operator.";

  let description = [{
    Replicates input1 multiples times along each dimension.

    References:
      * https://github.khronos.org/SPIRV-Registry/extended/TOSA.001000.1.html#_tile
      * https://www.mlplatform.org/tosa/tosa_spec_1_0_1.html#_tile

    #### Example:
    ```mlir
    %1 = spirv.Tosa.Tile %arg0, %0 : !spirv.arm.tensor<10x28x21xi16>, !spirv.arm.tensor<3xi32> -> !spirv.arm.tensor<10x28x63xi16>
    %1 = spirv.Tosa.Tile %arg0, %0 : !spirv.arm.tensor<31x19x5xf16>, !spirv.arm.tensor<3xi32> -> !spirv.arm.tensor<62x57x10xf16>
    ```
  }];

  let arguments = (ins
    SPIRV_TosaAny_TensorArm: $input1,
    SPIRV_Int32_1DTensorArmOfLength1To6: $multiples
  );

  let results = (outs
    SPIRV_TosaAny_TensorArm: $output
  );

  let hasVerifier = 1;

  let assemblyFormat = [{
    $input1 `,` $multiples
    attr-dict `:` type(operands) `->` type(results)
  }];

  let extraClassDeclaration = [{
    ::mlir::spirv::TensorArmType getInput1Type() {
      return cast<::mlir::spirv::TensorArmType>(getInput1().getType());
    }
    ::mlir::spirv::TensorArmType getResultType() {
      return cast<::mlir::spirv::TensorArmType>(getType());
    }
  }];
}


def SPIRV_TosaTransposeOp : SPIRV_TosaOp<"Transpose", 60, [Pure]> {
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
    %1 = spirv.Tosa.Transpose perms = dense<[2, 0, 1, 3]> : !spirv.arm.tensor<4xi32>, %arg0 : !spirv.arm.tensor<14x28x1x61xi16> -> !spirv.arm.tensor<1x14x28x61xi16>
    %1 = spirv.Tosa.Transpose perms = dense<[2, 0, 1]> : !spirv.arm.tensor<3xi32>, %arg0 : !spirv.arm.tensor<42x22x49xi1> -> !spirv.arm.tensor<49x42x22xi1>
    ```
  }];

  let arguments = (ins
    SPIRV_Int32_1DTensorArmOfLength1To6Attr: $perms,
    SPIRV_TosaAny_TensorArm: $input1
  );

  let results = (outs
    SPIRV_TosaAny_TensorArm: $output
  );

  let hasVerifier = 1;

  let assemblyFormat = [{
    `perms` `=` $perms `,`
    $input1
    attr-dict `:` type(operands) `->` type(results)
  }];

  let extraClassDeclaration = [{
    ::mlir::spirv::TensorArmType getInput1Type() {
      return cast<::mlir::spirv::TensorArmType>(getInput1().getType());
    }
    ::mlir::spirv::TensorArmType getResultType() {
      return cast<::mlir::spirv::TensorArmType>(getType());
    }
  }];
}


def SPIRV_TosaGatherOp : SPIRV_TosaOp<"Gather", 61, [Pure]> {
  let summary = "Gather operation.";

  let description = [{
    Generate a tensor for which each element in the output is a subtensor of the
    values tensor based on the indices. Let input shape = [N,K,C],
    the indices shape = [N,W], and the output shape = [N,W,C].
    Where N is the number of batches, W the number of indices in each batch,
    K the range of each index and C the number data channels for each index.

    References:
      * https://github.khronos.org/SPIRV-Registry/extended/TOSA.001000.1.html#_gather
      * https://www.mlplatform.org/tosa/tosa_spec_1_0_1.html#_gather

    #### Example:
    ```mlir
    %0 = spirv.Tosa.Gather %arg0, %arg1 : !spirv.arm.tensor<31x11x45xi32>, !spirv.arm.tensor<31x15xi32> -> !spirv.arm.tensor<31x15x45xi32>
    %0 = spirv.Tosa.Gather %arg0, %arg1 : !spirv.arm.tensor<59x61x19xf32>, !spirv.arm.tensor<59x65xi32> -> !spirv.arm.tensor<59x65x19xf32>
    ```
  }];

  let arguments = (ins
    SPIRV_TosaNumerical_TensorArm3D: $values,
    SPIRV_Int32_TensorArm2D: $indices
  );

  let results = (outs
    SPIRV_TosaNumerical_TensorArm3D: $output
  );

  let hasVerifier = 1;

  let assemblyFormat = [{
    $values `,` $indices
    attr-dict `:` type(operands) `->` type(results)
  }];

  let extraClassDeclaration = [{
    ::mlir::spirv::TensorArmType getResultType() {
      return cast<::mlir::spirv::TensorArmType>(getType());
    }
  }];
}


def SPIRV_TosaScatterOp : SPIRV_TosaOp<"Scatter", 62, [Pure]> {
  let summary = "Scatter operation.";

  let description = [{
    The values_out tensor is set to the values_in tensor with data modified as
    follows: data from the input tensor is inserted at the positions specified
    by the indices tensor. Let values_in shape = [N,K,C], input shape = [N,W,C],
    the indices shape = [N,W], and the output shape = [N,K,C].
    Where N is the number of batches, W the number of indices
    in each batch, K the range of each index and C the number data channels for
    each index. It is not permitted to repeat the same output index within a
    single scatter operation and so each output index occurs at most once. It
    follows that K >= W. In use cases that require multiple updates to the same
    output position, these must be decomposed into multiple scatter operations.

    References:
      * https://github.khronos.org/SPIRV-Registry/extended/TOSA.001000.1.html#_scatter
      * https://www.mlplatform.org/tosa/tosa_spec_1_0_1.html#_scatter

    #### Example:
    ```mlir
    %0 = spirv.Tosa.Scatter %arg0, %arg1, %arg2 : !spirv.arm.tensor<34x28x54xi32>, !spirv.arm.tensor<34x18xi32>, !spirv.arm.tensor<34x18x54xi32> -> !spirv.arm.tensor<34x28x54xi32>
    %0 = spirv.Tosa.Scatter %arg0, %arg1, %arg2 : !spirv.arm.tensor<18x34x25xf16>, !spirv.arm.tensor<18x20xi32>, !spirv.arm.tensor<18x20x25xf16> -> !spirv.arm.tensor<18x34x25xf16>
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

  let hasVerifier = 1;

  let assemblyFormat = [{
    $values_in `,` $indices `,` $input
    attr-dict `:` type(operands) `->` type(results)
  }];

  let extraClassDeclaration = [{
    ::mlir::spirv::TensorArmType getInputType() {
      return cast<::mlir::spirv::TensorArmType>(getInput().getType());
    }
    ::mlir::spirv::TensorArmType getResultType() {
      return cast<::mlir::spirv::TensorArmType>(getType());
    }
  }];
}


def SPIRV_TosaResizeOp : SPIRV_TosaOp<"Resize", 63, [Pure]> {
  let summary = "Resize operation, supports various resize/upsample modes.";

  let description = [{
    Resizes a tensor. Resize is only allowed in the H and W dimensions, given the input
    shape = [N,H,W,C].

    The height dimension (H) is scaled by factor (scale_y_n/scale_y_d). The width
    dimension (W) is scaled by factor (scale_x_n/scale_x_d).

    The NearestNeighbor mode returns the value of the input tensor closest to
    the calculated sample position for both floating-point and integer data
    formats.

    Floating-point Bilinear mode returns a bilinearly interpolated output value
    based on the four closest input sample positions.

    For integer Bilinear interpolation mode, the output value must be scaled by
    1/(scale_y_n * scale_x_n) in a following operation to complete the
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
    %4 = spirv.Tosa.Resize mode = <NearestNeighbor>, %arg0, %1, %2, %3 : !spirv.arm.tensor<1x1x31x55xi8>, !spirv.arm.tensor<4xi32>, !spirv.arm.tensor<2xi32>, !spirv.arm.tensor<2xi32> -> !spirv.arm.tensor<1x1x278x55xi8>
    %4 = spirv.Tosa.Resize mode = <Bilinear>, %arg0, %1, %2, %3 : !spirv.arm.tensor<1x48x33x63xf32>, !spirv.arm.tensor<4xi32>, !spirv.arm.tensor<2xi32>, !spirv.arm.tensor<2xi32> -> !spirv.arm.tensor<1x753x297x63xf32>
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

  let hasVerifier = 0;

  let assemblyFormat = [{
    `mode` `=` $mode `,`
    $input `,` $scale `,` $offset `,` $border
    attr-dict `:` type(operands) `->` type(results)
  }];

  let extraClassDeclaration = [{
    ::mlir::spirv::TensorArmType getInputType() {
      return cast<::mlir::spirv::TensorArmType>(getInput().getType());
    }
    ::mlir::spirv::TensorArmType getResultType() {
      return cast<::mlir::spirv::TensorArmType>(getType());
    }
  }];
}


def SPIRV_TosaCastOp : SPIRV_TosaOp<"Cast", 64, [Pure]> {
  let summary = "Cast operation.";

  let description = [{
    Casts a tensor from one data type to another.
    * This table is showing the supported conversions from the TOSA Specification.
    * The MLIR dialect here can be used to represent other conversions.

    | Mode                     | Input   | Output  |
    |--------------------------|---------|---------|
    | fp16 to fp32             | float16 | float32 |
    | fp16 to int 16           | float16 | int16   |
    | fp16 to int 32           | float16 | int32   |
    | fp16 to int 8            | float16 | int8    |
    | fp32 to fp16             | float32 | float16 |
    | fp32 to int 16           | float32 | int16   |
    | fp32 to int 32           | float32 | int32   |
    | fp32 to int 8            | float32 | int8    |
    | int 16 to fp16           | int16   | float16 |
    | int 16 to fp32           | int16   | float32 |
    | int 32 to fp16           | int32   | float16 |
    | int 32 to fp32           | int32   | float32 |
    | int 8 to fp16            | int8    | float16 |
    | int 8 to fp32            | int8    | float32 |
    | bool to int 16           | Boolean | int16   |
    | bool to int 32           | Boolean | int32   |
    | bool to int 8            | Boolean | int8    |
    | int 16 to bool           | int16   | Boolean |
    | int 16 to int 32         | int16   | int32   |
    | int 16 to int 8          | int16   | int8    |
    | int 32 to bool           | int32   | Boolean |
    | int 32 to int 16         | int32   | int16   |
    | int 32 to int 8          | int32   | int8    |
    | int 8 to bool            | int8    | Boolean |
    | int 8 to int 16          | int8    | int16   |
    | int 8 to int 32          | int8    | int32   |
    | bf16 to fp32             | bf16    | float32 |
    | bf16 to int 16           | bf16    | int16   |
    | bf16 to int 32           | bf16    | int32   |
    | bf16 to int 8            | bf16    | int8    |
    | fp32 to bf16             | float32 | bf16    |
    | int 16 to bf16           | int16   | bf16    |
    | int 32 to bf16           | int32   | bf16    |
    | int 8 to bf16            | int8    | bf16    |
    | bf16 to fp8e4m3          | bf16    | fp8e4m3 |
    | fp8e4m3 to bf16          | fp8e4m3 | bf16    |
    | bf16 to fp8e5m2          | bf16    | fp8e5m2 |
    | fp8e5m2 to bf16          | fp8e5m2 | bf16    |
    | fp16 to fp8e4m3          | float16 | fp8e4m3 |
    | fp32 to fp8e4m3          | float32 | fp8e4m3 |
    | fp8e4m3 to fp16          | fp8e4m3 | float16 |
    | fp8e4m3 to fp32          | fp8e4m3 | float32 |
    | fp16 to fp8e5m2          | float16 | fp8e5m2 |
    | fp32 to fp8e5m2          | float32 | fp8e5m2 |
    | fp8e5m2 to fp16          | fp8e5m2 | float16 |
    | fp8e5m2 to fp32          | fp8e5m2 | float32 |

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

  let hasVerifier = 1;

  let assemblyFormat = [{
    $input
    attr-dict `:` type(operands) `->` type(results)
  }];

  let extraClassDeclaration = [{
    ::mlir::spirv::TensorArmType getInputType() {
      return cast<::mlir::spirv::TensorArmType>(getInput().getType());
    }
    ::mlir::spirv::TensorArmType getResultType() {
      return cast<::mlir::spirv::TensorArmType>(getType());
    }
  }];
}


def SPIRV_TosaRescaleOp : SPIRV_TosaOp<"Rescale", 65, [Pure]> {
  let summary = "Rescale operator.";

  let description = [{
    Rescale is defined using an integer multiply, add, and shift.

    Rescale supports two precisions of multiplier: 16-bit and 32-bit. The 32-bit multiplier
    version supports two rounding modes to enable simpler lowering of existing frameworks
    that use two stage rounding. All arithmetic is designed so that it does not overflow a
    64-bit accumulator and that the result fits in 32 bits. In particular, a 48-bit value
    cannot be scaled with the 32-bit multiplier because the accumulator would need to have
    80 bits.

    The shift and value range are limited to allow a variety of implementations. The limit
    of 62 on shift allows the shift to be decomposed as two right shifts of 31.

    Supported rescalings:
    * This table is showing the supported conversions from the TOSA Specification.
    * The MLIR dialect here can be used to represent other conversions.

    | Mode                   | Input | Output | Unsigned input | Unsigned output |
    |------------------------|-------|--------|----------------|-----------------|
    | signed 16 to 16        | int16 | int16  |  false         |  false          |
    | signed 16 to 32        | int16 | int32  |  false         |  false          |
    | signed 16 to 8         | int16 | int8   |  false         |  false          |
    | signed 32 to 16        | int32 | int16  |  false         |  false          |
    | signed 32 to 32        | int32 | int32  |  false         |  false          |
    | signed 32 to 8         | int32 | int8   |  false         |  false          |
    | signed 8 to 16         | int8  | int16  |  false         |  false          |
    | signed 8 to 32         | int8  | int32  |  false         |  false          |
    | signed 8 to 8          | int8  | int8   |  false         |  false          |
    | signed 48 to 16        | int48 | int16  |  false         |  false          |
    | signed 48 to 32        | int48 | int32  |  false         |  false          |
    | signed 48 to 8         | int48 | int8   |  false         |  false          |
    | unsigned 8 to signed 8 | uint8 | int8   |  true          |  false          |
    | signed 8 to unsigned 8 | int8  | uint8  |  false         |  true           |

    Note: int48 is represented as an int64 type in SPIR-V but the contained values
    are still in the int48 range.

    References:
      * https://github.khronos.org/SPIRV-Registry/extended/TOSA.001000.1.html#_rescale
      * https://www.mlplatform.org/tosa/tosa_spec_1_0_1.html#_rescale

    #### Example:
    ```mlir
    %9 = spirv.Tosa.Rescale scale32 = true, rounding_mode = <DoubleRound>, per_channel = false, input_unsigned = false, output_unsigned = true, %arg0, %5, %6, %7, %8 : !spirv.arm.tensor<17x29x19xi16>, !spirv.arm.tensor<1xi32>, !spirv.arm.tensor<1xi8>, !spirv.arm.tensor<1xi16>, !spirv.arm.tensor<1xi16> -> !spirv.arm.tensor<17x29x19xi16>
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

  let hasVerifier = 1;

  let assemblyFormat = [{
    `scale32` `=` $scale32 `,` `rounding_mode` `=` $rounding_mode `,`
    `per_channel` `=` $per_channel `,` `input_unsigned` `=` $input_unsigned `,`
    `output_unsigned` `=` $output_unsigned `,`
    $input `,` $multiplier `,` $shift `,` $input_zp `,` $output_zp
    attr-dict `:` type(operands) `->` type(results)
  }];

  let extraClassDeclaration = [{
    ::mlir::spirv::TensorArmType getInputType() {
      return cast<::mlir::spirv::TensorArmType>(getInput().getType());
    }
    ::mlir::spirv::TensorArmType getResultType() {
      return cast<::mlir::spirv::TensorArmType>(getType());
    }
  }];
}


#endif // MLIR_DIALECT_SPIRV_IR_TOSA_OPS
