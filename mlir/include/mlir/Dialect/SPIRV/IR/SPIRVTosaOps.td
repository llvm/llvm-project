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


def SPIRV_TosaConv2DOp : SPIRV_TosaOp<"Conv2D", 2, [Pure,
  AllElementTypesMatch<["bias", "output"]>,
  AllElementTypesMatch<["input", "input_zp"]>,
  AllElementTypesMatch<["weight", "weight_zp"]>]> {
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

  let hasVerifier = 1;

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


def SPIRV_TosaConv3DOp : SPIRV_TosaOp<"Conv3D", 3, [Pure,
  AllElementTypesMatch<["bias", "output"]>,
  AllElementTypesMatch<["input", "input_zp"]>,
  AllElementTypesMatch<["weight", "weight_zp"]>]> {
  let summary = "3D Convolution operator.";

  let description = [{
    Performs a 3D convolution over the given input tensor. Implementations
    may choose to skip calculation of multiplies in the padding area.

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

  let hasVerifier = 1;

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


def SPIRV_TosaDepthwiseConv2DOp : SPIRV_TosaOp<"DepthwiseConv2D", 4, [Pure,
  AllElementTypesMatch<["bias", "output"]>,
  AllElementTypesMatch<["input", "input_zp"]>,
  AllElementTypesMatch<["weight", "weight_zp"]>]> {
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

  let hasVerifier = 1;

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


def SPIRV_TosaTransposeConv2DOp : SPIRV_TosaOp<"TransposeConv2D", 9, [Pure,
  AllElementTypesMatch<["bias", "output"]>,
  AllElementTypesMatch<["input", "input_zp"]>,
  AllElementTypesMatch<["weight", "weight_zp"]>]> {
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

  let hasVerifier = 1;

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


#endif // MLIR_DIALECT_SPIRV_IR_TOSA_OPS
