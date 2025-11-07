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
  SPIRV_ExtInstOp<mnemonic, "Tosa", "TOSA.001000.1", opcode, !listconcat(traits, [InGraphScope])> {

  let availability = [
    MinVersion<SPIRV_V_1_5>,
    MaxVersion<SPIRV_V_1_6>,
    Extension<[SPV_ARM_graph, SPV_ARM_tensors]>,
    Capability<[SPIRV_C_GraphARM]>
  ];
}


def SPIRV_TosaArgMaxOp : SPIRV_TosaOp<"ArgMax", 0, [Pure]> {
 let summary = "ArgMax - TOSA extended instruction set 001000.1";

  let description = [{
    https://github.khronos.org/SPIRV-Registry/extended/TOSA.001000.1.html#_argmax
  }];


  let arguments = (ins
    SPIRV_Int32: $axis,
    SPIRV_Int32: $nan_mode,
    SPIRV_TosaNumerical_TensorArm1DTo6D: $input
  );


  let results = (outs
    SPIRV_TosaInteger_TensorArmUpTo5D: $output
  );

  let hasVerifier = 1;

  let assemblyFormat = [{
    operands attr-dict `:` `(` type(operands) `)` `->` type(results)
  }];
}


def SPIRV_TosaAvgPool2DOp : SPIRV_TosaOp<"AvgPool2D", 1, [Pure]> {
 let summary = "AvgPool2D - TOSA extended instruction set 001000.1";

  let description = [{
    https://github.khronos.org/SPIRV-Registry/extended/TOSA.001000.1.html#_avg_pool2d
  }];


  let arguments = (ins
    SPIRV_Int32_1DTensorArmOfLength2: $kernel,
    SPIRV_Int32_1DTensorArmOfLength2: $stride,
    SPIRV_Int32_1DTensorArmOfLength4: $pad,
    SPIRV_Int32: $acc_type,
    SPIRV_TosaNumerical_TensorArm4D: $input,
    SPIRV_TosaNumerical_1DTensorArmOfLength1: $input_zp,
    SPIRV_TosaNumerical_1DTensorArmOfLength1: $output_zp
  );


  let results = (outs
    SPIRV_TosaNumerical_TensorArm4D: $output
  );

  let hasVerifier = 1;

  let assemblyFormat = [{
    operands attr-dict `:` `(` type(operands) `)` `->` type(results)
  }];
}


def SPIRV_TosaConv2DOp : SPIRV_TosaOp<"Conv2D", 2, [Pure]> {
 let summary = "Conv2D - TOSA extended instruction set 001000.1";

  let description = [{
    https://github.khronos.org/SPIRV-Registry/extended/TOSA.001000.1.html#_conv2d
  }];


  let arguments = (ins
    SPIRV_Int32_1DTensorArmOfLength4: $pad,
    SPIRV_Int32_1DTensorArmOfLength2: $stride,
    SPIRV_Int32_1DTensorArmOfLength2: $dilation,
    SPIRV_Int32: $acc_type,
    SPIRV_Bool: $local_bound,
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
    operands attr-dict `:` `(` type(operands) `)` `->` type(results)
  }];
}


def SPIRV_TosaConv3DOp : SPIRV_TosaOp<"Conv3D", 3, [Pure]> {
 let summary = "Conv3D - TOSA extended instruction set 001000.1";

  let description = [{
    https://github.khronos.org/SPIRV-Registry/extended/TOSA.001000.1.html#_conv3d
  }];


  let arguments = (ins
    SPIRV_Int32_1DTensorArmOfLength6: $pad,
    SPIRV_Int32_1DTensorArmOfLength3: $stride,
    SPIRV_Int32_1DTensorArmOfLength3: $dilation,
    SPIRV_Int32: $acc_type,
    SPIRV_Bool: $local_bound,
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
    operands attr-dict `:` `(` type(operands) `)` `->` type(results)
  }];
}


def SPIRV_TosaDepthwiseConv2DOp : SPIRV_TosaOp<"DepthwiseConv2D", 4, [Pure]> {
 let summary = "DepthwiseConv2D - TOSA extended instruction set 001000.1";

  let description = [{
    https://github.khronos.org/SPIRV-Registry/extended/TOSA.001000.1.html#_depthwise_conv2d
  }];


  let arguments = (ins
    SPIRV_Int32_1DTensorArmOfLength4: $pad,
    SPIRV_Int32_1DTensorArmOfLength2: $stride,
    SPIRV_Int32_1DTensorArmOfLength2: $dilation,
    SPIRV_Int32: $acc_type,
    SPIRV_Bool: $local_bound,
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
    operands attr-dict `:` `(` type(operands) `)` `->` type(results)
  }];
}


def SPIRV_TosaFFT2DOp : SPIRV_TosaOp<"FFT2D", 5, [Pure]> {
 let summary = "FFT2D - TOSA extended instruction set 001000.1";

  let description = [{
    https://github.khronos.org/SPIRV-Registry/extended/TOSA.001000.1.html#_fft2d
  }];


  let arguments = (ins
    SPIRV_Bool: $inverse,
    SPIRV_Bool: $local_bound,
    SPIRV_TosaFloat_TensorArm3D: $input_real,
    SPIRV_TosaFloat_TensorArm3D: $input_imag
  );


  let results = (outs
    SPIRV_Struct_2_TosaFloat_TensorArm3D: $output
  );

  let hasVerifier = 1;

  let assemblyFormat = [{
    operands attr-dict `:` `(` type(operands) `)` `->` type(results)
  }];
}


def SPIRV_TosaMatMulOp : SPIRV_TosaOp<"MatMul", 6, [Pure]> {
 let summary = "MatMul - TOSA extended instruction set 001000.1";

  let description = [{
    https://github.khronos.org/SPIRV-Registry/extended/TOSA.001000.1.html#_matmul
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
    operands attr-dict `:` `(` type(operands) `)` `->` type(results)
  }];
}


def SPIRV_TosaMaxPool2DOp : SPIRV_TosaOp<"MaxPool2D", 7, [Pure]> {
 let summary = "MaxPool2D - TOSA extended instruction set 001000.1";

  let description = [{
    https://github.khronos.org/SPIRV-Registry/extended/TOSA.001000.1.html#_max_pool2d
  }];


  let arguments = (ins
    SPIRV_Int32_1DTensorArmOfLength2: $kernel,
    SPIRV_Int32_1DTensorArmOfLength2: $stride,
    SPIRV_Int32_1DTensorArmOfLength4: $pad,
    SPIRV_Int32: $nan_mode,
    SPIRV_TosaNumerical_TensorArm4D: $input
  );


  let results = (outs
    SPIRV_TosaNumerical_TensorArm4D: $output
  );

  let hasVerifier = 1;

  let assemblyFormat = [{
    operands attr-dict `:` `(` type(operands) `)` `->` type(results)
  }];
}


def SPIRV_TosaRFFT2DOp : SPIRV_TosaOp<"RFFT2D", 8, [Pure]> {
 let summary = "RFFT2D - TOSA extended instruction set 001000.1";

  let description = [{
    https://github.khronos.org/SPIRV-Registry/extended/TOSA.001000.1.html#_rfft2d
  }];


  let arguments = (ins
    SPIRV_Bool: $local_bound,
    SPIRV_TosaFloat_TensorArm3D: $input_real
  );


  let results = (outs
    SPIRV_Struct_2_TosaFloat_TensorArm3D: $output
  );

  let hasVerifier = 1;

  let assemblyFormat = [{
    operands attr-dict `:` `(` type(operands) `)` `->` type(results)
  }];
}


def SPIRV_TosaTransposeConv2DOp : SPIRV_TosaOp<"TransposeConv2D", 9, [Pure]> {
 let summary = "TransposeConv2D - TOSA extended instruction set 001000.1";

  let description = [{
    https://github.khronos.org/SPIRV-Registry/extended/TOSA.001000.1.html#_transpose_conv2d
  }];


  let arguments = (ins
    SPIRV_Int32_1DTensorArmOfLength4: $out_pad,
    SPIRV_Int32_1DTensorArmOfLength2: $stride,
    SPIRV_Int32: $acc_type,
    SPIRV_Bool: $local_bound,
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
    operands attr-dict `:` `(` type(operands) `)` `->` type(results)
  }];
}


def SPIRV_TosaClampOp : SPIRV_TosaOp<"Clamp", 10, [Pure]> {
 let summary = "Clamp - TOSA extended instruction set 001000.1";

  let description = [{
    https://github.khronos.org/SPIRV-Registry/extended/TOSA.001000.1.html#_clamp
  }];


  let arguments = (ins
    SPIRV_TosaNumerical: $min_val,
    SPIRV_TosaNumerical: $max_val,
    SPIRV_Int32: $nan_mode,
    SPIRV_TosaNumerical_TensorArmUpTo6D: $input
  );


  let results = (outs
    SPIRV_TosaNumerical_TensorArmUpTo6D: $output
  );

  let hasVerifier = 1;

  let assemblyFormat = [{
    operands attr-dict `:` `(` type(operands) `)` `->` type(results)
  }];
}


def SPIRV_TosaErfOp : SPIRV_TosaOp<"Erf", 11, [Pure]> {
 let summary = "Erf - TOSA extended instruction set 001000.1";

  let description = [{
    https://github.khronos.org/SPIRV-Registry/extended/TOSA.001000.1.html#_erf
  }];


  let arguments = (ins
    SPIRV_TosaFloat_TensorArmUpTo6D: $input
  );


  let results = (outs
    SPIRV_TosaFloat_TensorArmUpTo6D: $output
  );

  let hasVerifier = 0;

  let assemblyFormat = [{
    operands attr-dict `:` `(` type(operands) `)` `->` type(results)
  }];
}


def SPIRV_TosaSigmoidOp : SPIRV_TosaOp<"Sigmoid", 12, [Pure]> {
 let summary = "Sigmoid - TOSA extended instruction set 001000.1";

  let description = [{
    https://github.khronos.org/SPIRV-Registry/extended/TOSA.001000.1.html#_sigmoid
  }];


  let arguments = (ins
    SPIRV_TosaFloat_TensorArmUpTo6D: $input
  );


  let results = (outs
    SPIRV_TosaFloat_TensorArmUpTo6D: $output
  );

  let hasVerifier = 0;

  let assemblyFormat = [{
    operands attr-dict `:` `(` type(operands) `)` `->` type(results)
  }];
}


def SPIRV_TosaTanhOp : SPIRV_TosaOp<"Tanh", 13, [Pure]> {
 let summary = "Tanh - TOSA extended instruction set 001000.1";

  let description = [{
    https://github.khronos.org/SPIRV-Registry/extended/TOSA.001000.1.html#_tanh
  }];


  let arguments = (ins
    SPIRV_TosaFloat_TensorArmUpTo6D: $input
  );


  let results = (outs
    SPIRV_TosaFloat_TensorArmUpTo6D: $output
  );

  let hasVerifier = 0;

  let assemblyFormat = [{
    operands attr-dict `:` `(` type(operands) `)` `->` type(results)
  }];
}


def SPIRV_TosaAddOp : SPIRV_TosaOp<"Add", 14, [Pure]> {
 let summary = "Add - TOSA extended instruction set 001000.1";

  let description = [{
    https://github.khronos.org/SPIRV-Registry/extended/TOSA.001000.1.html#_add
  }];


  let arguments = (ins
    SPIRV_TosaNumerical_TensorArmUpTo6D: $input1,
    SPIRV_TosaNumerical_TensorArmUpTo6D: $input2
  );


  let results = (outs
    SPIRV_TosaNumerical_TensorArmUpTo6D: $output
  );

  let hasVerifier = 0;

  let assemblyFormat = [{
    operands attr-dict `:` `(` type(operands) `)` `->` type(results)
  }];
}


def SPIRV_TosaArithmeticRightShiftOp : SPIRV_TosaOp<"ArithmeticRightShift", 15, [Pure]> {
 let summary = "ArithmeticRightShift - TOSA extended instruction set 001000.1";

  let description = [{
    https://github.khronos.org/SPIRV-Registry/extended/TOSA.001000.1.html#_arithmetic_right_shift
  }];


  let arguments = (ins
    SPIRV_Bool: $round,
    SPIRV_TosaInteger_TensorArmUpTo6D: $input1,
    SPIRV_TosaInteger_TensorArmUpTo6D: $input2
  );


  let results = (outs
    SPIRV_TosaInteger_TensorArmUpTo6D: $output
  );

  let hasVerifier = 0;

  let assemblyFormat = [{
    operands attr-dict `:` `(` type(operands) `)` `->` type(results)
  }];
}


def SPIRV_TosaBitwiseAndOp : SPIRV_TosaOp<"BitwiseAnd", 16, [Pure]> {
 let summary = "BitwiseAnd - TOSA extended instruction set 001000.1";

  let description = [{
    https://github.khronos.org/SPIRV-Registry/extended/TOSA.001000.1.html#_bitwise_and
  }];


  let arguments = (ins
    SPIRV_TosaInteger_TensorArmUpTo6D: $input1,
    SPIRV_TosaInteger_TensorArmUpTo6D: $input2
  );


  let results = (outs
    SPIRV_TosaInteger_TensorArmUpTo6D: $output
  );

  let hasVerifier = 0;

  let assemblyFormat = [{
    operands attr-dict `:` `(` type(operands) `)` `->` type(results)
  }];
}


def SPIRV_TosaBitwiseOrOp : SPIRV_TosaOp<"BitwiseOr", 17, [Pure]> {
 let summary = "BitwiseOr - TOSA extended instruction set 001000.1";

  let description = [{
    https://github.khronos.org/SPIRV-Registry/extended/TOSA.001000.1.html#_bitwise_or
  }];


  let arguments = (ins
    SPIRV_TosaInteger_TensorArmUpTo6D: $input1,
    SPIRV_TosaInteger_TensorArmUpTo6D: $input2
  );


  let results = (outs
    SPIRV_TosaInteger_TensorArmUpTo6D: $output
  );

  let hasVerifier = 0;

  let assemblyFormat = [{
    operands attr-dict `:` `(` type(operands) `)` `->` type(results)
  }];
}


def SPIRV_TosaBitwiseXorOp : SPIRV_TosaOp<"BitwiseXor", 18, [Pure]> {
 let summary = "BitwiseXor - TOSA extended instruction set 001000.1";

  let description = [{
    https://github.khronos.org/SPIRV-Registry/extended/TOSA.001000.1.html#_bitwise_xor
  }];


  let arguments = (ins
    SPIRV_TosaInteger_TensorArmUpTo6D: $input1,
    SPIRV_TosaInteger_TensorArmUpTo6D: $input2
  );


  let results = (outs
    SPIRV_TosaInteger_TensorArmUpTo6D: $output
  );

  let hasVerifier = 0;

  let assemblyFormat = [{
    operands attr-dict `:` `(` type(operands) `)` `->` type(results)
  }];
}


def SPIRV_TosaIntDivOp : SPIRV_TosaOp<"IntDiv", 19, [Pure]> {
 let summary = "IntDiv - TOSA extended instruction set 001000.1";

  let description = [{
    https://github.khronos.org/SPIRV-Registry/extended/TOSA.001000.1.html#_intdiv
  }];


  let arguments = (ins
    SPIRV_TosaInteger_TensorArmUpTo6D: $input1,
    SPIRV_TosaInteger_TensorArmUpTo6D: $input2
  );


  let results = (outs
    SPIRV_TosaInteger_TensorArmUpTo6D: $output
  );

  let hasVerifier = 0;

  let assemblyFormat = [{
    operands attr-dict `:` `(` type(operands) `)` `->` type(results)
  }];
}


def SPIRV_TosaLogicalAndOp : SPIRV_TosaOp<"LogicalAnd", 20, [Pure]> {
 let summary = "LogicalAnd - TOSA extended instruction set 001000.1";

  let description = [{
    https://github.khronos.org/SPIRV-Registry/extended/TOSA.001000.1.html#_logical_and
  }];


  let arguments = (ins
    SPIRV_Bool_TensorArmUpTo6D: $input1,
    SPIRV_Bool_TensorArmUpTo6D: $input2
  );


  let results = (outs
    SPIRV_Bool_TensorArmUpTo6D: $output
  );

  let hasVerifier = 0;

  let assemblyFormat = [{
    operands attr-dict `:` `(` type(operands) `)` `->` type(results)
  }];
}


def SPIRV_TosaLogicalLeftShiftOp : SPIRV_TosaOp<"LogicalLeftShift", 21, [Pure]> {
 let summary = "LogicalLeftShift - TOSA extended instruction set 001000.1";

  let description = [{
    https://github.khronos.org/SPIRV-Registry/extended/TOSA.001000.1.html#_logical_left_shift
  }];


  let arguments = (ins
    SPIRV_TosaInteger_TensorArmUpTo6D: $input1,
    SPIRV_TosaInteger_TensorArmUpTo6D: $input2
  );


  let results = (outs
    SPIRV_TosaInteger_TensorArmUpTo6D: $output
  );

  let hasVerifier = 0;

  let assemblyFormat = [{
    operands attr-dict `:` `(` type(operands) `)` `->` type(results)
  }];
}


def SPIRV_TosaLogicalRightShiftOp : SPIRV_TosaOp<"LogicalRightShift", 22, [Pure]> {
 let summary = "LogicalRightShift - TOSA extended instruction set 001000.1";

  let description = [{
    https://github.khronos.org/SPIRV-Registry/extended/TOSA.001000.1.html#_logical_right_shift
  }];


  let arguments = (ins
    SPIRV_TosaInteger_TensorArmUpTo6D: $input1,
    SPIRV_TosaInteger_TensorArmUpTo6D: $input2
  );


  let results = (outs
    SPIRV_TosaInteger_TensorArmUpTo6D: $output
  );

  let hasVerifier = 0;

  let assemblyFormat = [{
    operands attr-dict `:` `(` type(operands) `)` `->` type(results)
  }];
}


def SPIRV_TosaLogicalOrOp : SPIRV_TosaOp<"LogicalOr", 23, [Pure]> {
 let summary = "LogicalOr - TOSA extended instruction set 001000.1";

  let description = [{
    https://github.khronos.org/SPIRV-Registry/extended/TOSA.001000.1.html#_logical_or
  }];


  let arguments = (ins
    SPIRV_Bool_TensorArmUpTo6D: $input1,
    SPIRV_Bool_TensorArmUpTo6D: $input2
  );


  let results = (outs
    SPIRV_Bool_TensorArmUpTo6D: $output
  );

  let hasVerifier = 0;

  let assemblyFormat = [{
    operands attr-dict `:` `(` type(operands) `)` `->` type(results)
  }];
}


def SPIRV_TosaLogicalXorOp : SPIRV_TosaOp<"LogicalXor", 24, [Pure]> {
 let summary = "LogicalXor - TOSA extended instruction set 001000.1";

  let description = [{
    https://github.khronos.org/SPIRV-Registry/extended/TOSA.001000.1.html#_logical_xor
  }];


  let arguments = (ins
    SPIRV_Bool_TensorArmUpTo6D: $input1,
    SPIRV_Bool_TensorArmUpTo6D: $input2
  );


  let results = (outs
    SPIRV_Bool_TensorArmUpTo6D: $output
  );

  let hasVerifier = 0;

  let assemblyFormat = [{
    operands attr-dict `:` `(` type(operands) `)` `->` type(results)
  }];
}


def SPIRV_TosaMaximumOp : SPIRV_TosaOp<"Maximum", 25, [Pure]> {
 let summary = "Maximum - TOSA extended instruction set 001000.1";

  let description = [{
    https://github.khronos.org/SPIRV-Registry/extended/TOSA.001000.1.html#_maximum
  }];


  let arguments = (ins
    SPIRV_Int32: $nan_mode,
    SPIRV_TosaNumerical_TensorArmUpTo6D: $input1,
    SPIRV_TosaNumerical_TensorArmUpTo6D: $input2
  );


  let results = (outs
    SPIRV_TosaNumerical_TensorArmUpTo6D: $output
  );

  let hasVerifier = 0;

  let assemblyFormat = [{
    operands attr-dict `:` `(` type(operands) `)` `->` type(results)
  }];
}


def SPIRV_TosaMinimumOp : SPIRV_TosaOp<"Minimum", 26, [Pure]> {
 let summary = "Minimum - TOSA extended instruction set 001000.1";

  let description = [{
    https://github.khronos.org/SPIRV-Registry/extended/TOSA.001000.1.html#_minimum
  }];


  let arguments = (ins
    SPIRV_Int32: $nan_mode,
    SPIRV_TosaNumerical_TensorArmUpTo6D: $input1,
    SPIRV_TosaNumerical_TensorArmUpTo6D: $input2
  );


  let results = (outs
    SPIRV_TosaNumerical_TensorArmUpTo6D: $output
  );

  let hasVerifier = 0;

  let assemblyFormat = [{
    operands attr-dict `:` `(` type(operands) `)` `->` type(results)
  }];
}


def SPIRV_TosaMulOp : SPIRV_TosaOp<"Mul", 27, [Pure]> {
 let summary = "Mul - TOSA extended instruction set 001000.1";

  let description = [{
    https://github.khronos.org/SPIRV-Registry/extended/TOSA.001000.1.html#_mul
  }];


  let arguments = (ins
    SPIRV_TosaNumerical_TensorArmUpTo6D: $input1,
    SPIRV_TosaNumerical_TensorArmUpTo6D: $input2,
    SPIRV_Int8_1DTensorArmOfLength1: $shift
  );


  let results = (outs
    SPIRV_TosaNumerical_TensorArmUpTo6D: $output
  );

  let hasVerifier = 1;

  let assemblyFormat = [{
    operands attr-dict `:` `(` type(operands) `)` `->` type(results)
  }];
}


def SPIRV_TosaPowOp : SPIRV_TosaOp<"Pow", 28, [Pure]> {
 let summary = "Pow - TOSA extended instruction set 001000.1";

  let description = [{
    https://github.khronos.org/SPIRV-Registry/extended/TOSA.001000.1.html#_pow
  }];


  let arguments = (ins
    SPIRV_TosaFloat_TensorArmUpTo6D: $input1,
    SPIRV_TosaFloat_TensorArmUpTo6D: $input2
  );


  let results = (outs
    SPIRV_TosaFloat_TensorArmUpTo6D: $output
  );

  let hasVerifier = 0;

  let assemblyFormat = [{
    operands attr-dict `:` `(` type(operands) `)` `->` type(results)
  }];
}


def SPIRV_TosaSubOp : SPIRV_TosaOp<"Sub", 29, [Pure]> {
 let summary = "Sub - TOSA extended instruction set 001000.1";

  let description = [{
    https://github.khronos.org/SPIRV-Registry/extended/TOSA.001000.1.html#_sub
  }];


  let arguments = (ins
    SPIRV_TosaNumerical_TensorArmUpTo6D: $input1,
    SPIRV_TosaNumerical_TensorArmUpTo6D: $input2
  );


  let results = (outs
    SPIRV_TosaNumerical_TensorArmUpTo6D: $output
  );

  let hasVerifier = 0;

  let assemblyFormat = [{
    operands attr-dict `:` `(` type(operands) `)` `->` type(results)
  }];
}


def SPIRV_TosaTableOp : SPIRV_TosaOp<"Table", 30, [Pure]> {
 let summary = "Table - TOSA extended instruction set 001000.1";

  let description = [{
    https://github.khronos.org/SPIRV-Registry/extended/TOSA.001000.1.html#_table
  }];


  let arguments = (ins
    SPIRV_TosaInteger_TensorArmUpTo6D: $input1,
    SPIRV_TosaInteger_TensorArm1D: $table
  );


  let results = (outs
    SPIRV_TosaInteger_TensorArmUpTo6D: $output
  );

  let hasVerifier = 0;

  let assemblyFormat = [{
    operands attr-dict `:` `(` type(operands) `)` `->` type(results)
  }];
}


def SPIRV_TosaAbsOp : SPIRV_TosaOp<"Abs", 31, [Pure]> {
 let summary = "Abs - TOSA extended instruction set 001000.1";

  let description = [{
    https://github.khronos.org/SPIRV-Registry/extended/TOSA.001000.1.html#_abs
  }];


  let arguments = (ins
    SPIRV_TosaNumerical_TensorArmUpTo6D: $input1
  );


  let results = (outs
    SPIRV_TosaNumerical_TensorArmUpTo6D: $output
  );

  let hasVerifier = 0;

  let assemblyFormat = [{
    operands attr-dict `:` `(` type(operands) `)` `->` type(results)
  }];
}


def SPIRV_TosaBitwiseNotOp : SPIRV_TosaOp<"BitwiseNot", 32, [Pure]> {
 let summary = "BitwiseNot - TOSA extended instruction set 001000.1";

  let description = [{
    https://github.khronos.org/SPIRV-Registry/extended/TOSA.001000.1.html#_bitwise_not
  }];


  let arguments = (ins
    SPIRV_TosaInteger_TensorArmUpTo6D: $input1
  );


  let results = (outs
    SPIRV_TosaInteger_TensorArmUpTo6D: $output
  );

  let hasVerifier = 0;

  let assemblyFormat = [{
    operands attr-dict `:` `(` type(operands) `)` `->` type(results)
  }];
}


def SPIRV_TosaCeilOp : SPIRV_TosaOp<"Ceil", 33, [Pure]> {
 let summary = "Ceil - TOSA extended instruction set 001000.1";

  let description = [{
    https://github.khronos.org/SPIRV-Registry/extended/TOSA.001000.1.html#_ceil
  }];


  let arguments = (ins
    SPIRV_TosaFloat_TensorArmUpTo6D: $input1
  );


  let results = (outs
    SPIRV_TosaFloat_TensorArmUpTo6D: $output
  );

  let hasVerifier = 0;

  let assemblyFormat = [{
    operands attr-dict `:` `(` type(operands) `)` `->` type(results)
  }];
}


def SPIRV_TosaClzOp : SPIRV_TosaOp<"Clz", 34, [Pure]> {
 let summary = "Clz - TOSA extended instruction set 001000.1";

  let description = [{
    https://github.khronos.org/SPIRV-Registry/extended/TOSA.001000.1.html#_clz
  }];


  let arguments = (ins
    SPIRV_TosaInteger_TensorArmUpTo6D: $input1
  );


  let results = (outs
    SPIRV_TosaInteger_TensorArmUpTo6D: $output
  );

  let hasVerifier = 0;

  let assemblyFormat = [{
    operands attr-dict `:` `(` type(operands) `)` `->` type(results)
  }];
}


def SPIRV_TosaCosOp : SPIRV_TosaOp<"Cos", 35, [Pure]> {
 let summary = "Cos - TOSA extended instruction set 001000.1";

  let description = [{
    https://github.khronos.org/SPIRV-Registry/extended/TOSA.001000.1.html#_cos
  }];


  let arguments = (ins
    SPIRV_TosaFloat_TensorArmUpTo6D: $input1
  );


  let results = (outs
    SPIRV_TosaFloat_TensorArmUpTo6D: $output
  );

  let hasVerifier = 0;

  let assemblyFormat = [{
    operands attr-dict `:` `(` type(operands) `)` `->` type(results)
  }];
}


def SPIRV_TosaExpOp : SPIRV_TosaOp<"Exp", 36, [Pure]> {
 let summary = "Exp - TOSA extended instruction set 001000.1";

  let description = [{
    https://github.khronos.org/SPIRV-Registry/extended/TOSA.001000.1.html#_exp
  }];


  let arguments = (ins
    SPIRV_TosaFloat_TensorArmUpTo6D: $input1
  );


  let results = (outs
    SPIRV_TosaFloat_TensorArmUpTo6D: $output
  );

  let hasVerifier = 0;

  let assemblyFormat = [{
    operands attr-dict `:` `(` type(operands) `)` `->` type(results)
  }];
}


def SPIRV_TosaFloorOp : SPIRV_TosaOp<"Floor", 37, [Pure]> {
 let summary = "Floor - TOSA extended instruction set 001000.1";

  let description = [{
    https://github.khronos.org/SPIRV-Registry/extended/TOSA.001000.1.html#_floor
  }];


  let arguments = (ins
    SPIRV_TosaFloat_TensorArmUpTo6D: $input1
  );


  let results = (outs
    SPIRV_TosaFloat_TensorArmUpTo6D: $output
  );

  let hasVerifier = 0;

  let assemblyFormat = [{
    operands attr-dict `:` `(` type(operands) `)` `->` type(results)
  }];
}


def SPIRV_TosaLogOp : SPIRV_TosaOp<"Log", 38, [Pure]> {
 let summary = "Log - TOSA extended instruction set 001000.1";

  let description = [{
    https://github.khronos.org/SPIRV-Registry/extended/TOSA.001000.1.html#_log
  }];


  let arguments = (ins
    SPIRV_TosaFloat_TensorArmUpTo6D: $input1
  );


  let results = (outs
    SPIRV_TosaFloat_TensorArmUpTo6D: $output
  );

  let hasVerifier = 0;

  let assemblyFormat = [{
    operands attr-dict `:` `(` type(operands) `)` `->` type(results)
  }];
}


def SPIRV_TosaLogicalNotOp : SPIRV_TosaOp<"LogicalNot", 39, [Pure]> {
 let summary = "LogicalNot - TOSA extended instruction set 001000.1";

  let description = [{
    https://github.khronos.org/SPIRV-Registry/extended/TOSA.001000.1.html#_logical_not
  }];


  let arguments = (ins
    SPIRV_Bool_TensorArmUpTo6D: $input1
  );


  let results = (outs
    SPIRV_Bool_TensorArmUpTo6D: $output
  );

  let hasVerifier = 0;

  let assemblyFormat = [{
    operands attr-dict `:` `(` type(operands) `)` `->` type(results)
  }];
}


def SPIRV_TosaNegateOp : SPIRV_TosaOp<"Negate", 40, [Pure]> {
 let summary = "Negate - TOSA extended instruction set 001000.1";

  let description = [{
    https://github.khronos.org/SPIRV-Registry/extended/TOSA.001000.1.html#_negate
  }];


  let arguments = (ins
    SPIRV_TosaNumerical_TensorArmUpTo6D: $input1,
    SPIRV_TosaNumerical_1DTensorArmOfLength1: $input1_zp,
    SPIRV_TosaNumerical_1DTensorArmOfLength1: $output_zp
  );


  let results = (outs
    SPIRV_TosaNumerical_TensorArmUpTo6D: $output
  );

  let hasVerifier = 0;

  let assemblyFormat = [{
    operands attr-dict `:` `(` type(operands) `)` `->` type(results)
  }];
}


def SPIRV_TosaReciprocalOp : SPIRV_TosaOp<"Reciprocal", 41, [Pure]> {
 let summary = "Reciprocal - TOSA extended instruction set 001000.1";

  let description = [{
    https://github.khronos.org/SPIRV-Registry/extended/TOSA.001000.1.html#_reciprocal
  }];


  let arguments = (ins
    SPIRV_TosaFloat_TensorArmUpTo6D: $input1
  );


  let results = (outs
    SPIRV_TosaFloat_TensorArmUpTo6D: $output
  );

  let hasVerifier = 0;

  let assemblyFormat = [{
    operands attr-dict `:` `(` type(operands) `)` `->` type(results)
  }];
}


def SPIRV_TosaRsqrtOp : SPIRV_TosaOp<"Rsqrt", 42, [Pure]> {
 let summary = "Rsqrt - TOSA extended instruction set 001000.1";

  let description = [{
    https://github.khronos.org/SPIRV-Registry/extended/TOSA.001000.1.html#_rsqrt
  }];


  let arguments = (ins
    SPIRV_TosaFloat_TensorArmUpTo6D: $input1
  );


  let results = (outs
    SPIRV_TosaFloat_TensorArmUpTo6D: $output
  );

  let hasVerifier = 0;

  let assemblyFormat = [{
    operands attr-dict `:` `(` type(operands) `)` `->` type(results)
  }];
}


def SPIRV_TosaSinOp : SPIRV_TosaOp<"Sin", 43, [Pure]> {
 let summary = "Sin - TOSA extended instruction set 001000.1";

  let description = [{
    https://github.khronos.org/SPIRV-Registry/extended/TOSA.001000.1.html#_sin
  }];


  let arguments = (ins
    SPIRV_TosaFloat_TensorArmUpTo6D: $input1
  );


  let results = (outs
    SPIRV_TosaFloat_TensorArmUpTo6D: $output
  );

  let hasVerifier = 0;

  let assemblyFormat = [{
    operands attr-dict `:` `(` type(operands) `)` `->` type(results)
  }];
}


def SPIRV_TosaSelectOp : SPIRV_TosaOp<"Select", 44, [Pure]> {
 let summary = "Select - TOSA extended instruction set 001000.1";

  let description = [{
    https://github.khronos.org/SPIRV-Registry/extended/TOSA.001000.1.html#_select
  }];


  let arguments = (ins
    SPIRV_Bool_TensorArmUpTo6D: $input1,
    SPIRV_TosaAny_TensorArmUpTo6D: $input2,
    SPIRV_TosaAny_TensorArmUpTo6D: $input3
  );


  let results = (outs
    SPIRV_TosaAny_TensorArmUpTo6D: $output
  );

  let hasVerifier = 1;

  let assemblyFormat = [{
    operands attr-dict `:` `(` type(operands) `)` `->` type(results)
  }];
}


def SPIRV_TosaEqualOp : SPIRV_TosaOp<"Equal", 45, [Pure]> {
 let summary = "Equal - TOSA extended instruction set 001000.1";

  let description = [{
    https://github.khronos.org/SPIRV-Registry/extended/TOSA.001000.1.html#_equal
  }];


  let arguments = (ins
    SPIRV_TosaNumerical_TensorArmUpTo6D: $input1,
    SPIRV_TosaNumerical_TensorArmUpTo6D: $input2
  );


  let results = (outs
    SPIRV_Bool_TensorArmUpTo6D: $output
  );

  let hasVerifier = 0;

  let assemblyFormat = [{
    operands attr-dict `:` `(` type(operands) `)` `->` type(results)
  }];
}


def SPIRV_TosaGreaterOp : SPIRV_TosaOp<"Greater", 46, [Pure]> {
 let summary = "Greater - TOSA extended instruction set 001000.1";

  let description = [{
    https://github.khronos.org/SPIRV-Registry/extended/TOSA.001000.1.html#_greater
  }];


  let arguments = (ins
    SPIRV_TosaNumerical_TensorArmUpTo6D: $input1,
    SPIRV_TosaNumerical_TensorArmUpTo6D: $input2
  );


  let results = (outs
    SPIRV_Bool_TensorArmUpTo6D: $output
  );

  let hasVerifier = 0;

  let assemblyFormat = [{
    operands attr-dict `:` `(` type(operands) `)` `->` type(results)
  }];
}


def SPIRV_TosaGreaterEqualOp : SPIRV_TosaOp<"GreaterEqual", 47, [Pure]> {
 let summary = "GreaterEqual - TOSA extended instruction set 001000.1";

  let description = [{
    https://github.khronos.org/SPIRV-Registry/extended/TOSA.001000.1.html#_greater_equal
  }];


  let arguments = (ins
    SPIRV_TosaNumerical_TensorArmUpTo6D: $input1,
    SPIRV_TosaNumerical_TensorArmUpTo6D: $input2
  );


  let results = (outs
    SPIRV_Bool_TensorArmUpTo6D: $output
  );

  let hasVerifier = 0;

  let assemblyFormat = [{
    operands attr-dict `:` `(` type(operands) `)` `->` type(results)
  }];
}


def SPIRV_TosaReduceAllOp : SPIRV_TosaOp<"ReduceAll", 48, [Pure]> {
 let summary = "ReduceAll - TOSA extended instruction set 001000.1";

  let description = [{
    https://github.khronos.org/SPIRV-Registry/extended/TOSA.001000.1.html#_reduce_all
  }];


  let arguments = (ins
    SPIRV_Int32: $axis,
    SPIRV_Bool_TensorArm1DTo6D: $input
  );


  let results = (outs
    SPIRV_Bool_TensorArm1DTo6D: $output
  );

  let hasVerifier = 0;

  let assemblyFormat = [{
    operands attr-dict `:` `(` type(operands) `)` `->` type(results)
  }];
}


def SPIRV_TosaReduceAnyOp : SPIRV_TosaOp<"ReduceAny", 49, [Pure]> {
 let summary = "ReduceAny - TOSA extended instruction set 001000.1";

  let description = [{
    https://github.khronos.org/SPIRV-Registry/extended/TOSA.001000.1.html#_reduce_any
  }];


  let arguments = (ins
    SPIRV_Int32: $axis,
    SPIRV_Bool_TensorArm1DTo6D: $input
  );


  let results = (outs
    SPIRV_Bool_TensorArm1DTo6D: $output
  );

  let hasVerifier = 0;

  let assemblyFormat = [{
    operands attr-dict `:` `(` type(operands) `)` `->` type(results)
  }];
}


def SPIRV_TosaReduceMaxOp : SPIRV_TosaOp<"ReduceMax", 50, [Pure]> {
 let summary = "ReduceMax - TOSA extended instruction set 001000.1";

  let description = [{
    https://github.khronos.org/SPIRV-Registry/extended/TOSA.001000.1.html#_reduce_max
  }];


  let arguments = (ins
    SPIRV_Int32: $axis,
    SPIRV_Int32: $nan_mode,
    SPIRV_TosaNumerical_TensorArm1DTo6D: $input
  );


  let results = (outs
    SPIRV_TosaNumerical_TensorArm1DTo6D: $output
  );

  let hasVerifier = 0;

  let assemblyFormat = [{
    operands attr-dict `:` `(` type(operands) `)` `->` type(results)
  }];
}


def SPIRV_TosaReduceMinOp : SPIRV_TosaOp<"ReduceMin", 51, [Pure]> {
 let summary = "ReduceMin - TOSA extended instruction set 001000.1";

  let description = [{
    https://github.khronos.org/SPIRV-Registry/extended/TOSA.001000.1.html#_reduce_min
  }];


  let arguments = (ins
    SPIRV_Int32: $axis,
    SPIRV_Int32: $nan_mode,
    SPIRV_TosaNumerical_TensorArm1DTo6D: $input
  );


  let results = (outs
    SPIRV_TosaNumerical_TensorArm1DTo6D: $output
  );

  let hasVerifier = 0;

  let assemblyFormat = [{
    operands attr-dict `:` `(` type(operands) `)` `->` type(results)
  }];
}


def SPIRV_TosaReduceProductOp : SPIRV_TosaOp<"ReduceProduct", 52, [Pure]> {
 let summary = "ReduceProduct - TOSA extended instruction set 001000.1";

  let description = [{
    https://github.khronos.org/SPIRV-Registry/extended/TOSA.001000.1.html#_reduce_product
  }];


  let arguments = (ins
    SPIRV_Int32: $axis,
    SPIRV_TosaFloat_TensorArm1DTo6D: $input
  );


  let results = (outs
    SPIRV_TosaFloat_TensorArm1DTo6D: $output
  );

  let hasVerifier = 0;

  let assemblyFormat = [{
    operands attr-dict `:` `(` type(operands) `)` `->` type(results)
  }];
}


def SPIRV_TosaReduceSumOp : SPIRV_TosaOp<"ReduceSum", 53, [Pure]> {
 let summary = "ReduceSum - TOSA extended instruction set 001000.1";

  let description = [{
    https://github.khronos.org/SPIRV-Registry/extended/TOSA.001000.1.html#_reduce_sum
  }];


  let arguments = (ins
    SPIRV_Int32: $axis,
    SPIRV_TosaNumerical_TensorArm1DTo6D: $input
  );


  let results = (outs
    SPIRV_TosaNumerical_TensorArm1DTo6D: $output
  );

  let hasVerifier = 0;

  let assemblyFormat = [{
    operands attr-dict `:` `(` type(operands) `)` `->` type(results)
  }];
}


def SPIRV_TosaConcatOp : SPIRV_TosaOp<"Concat", 54, [Pure]> {
 let summary = "Concat - TOSA extended instruction set 001000.1";

  let description = [{
    https://github.khronos.org/SPIRV-Registry/extended/TOSA.001000.1.html#_concat
  }];


  let arguments = (ins
    SPIRV_Int32: $axis,
    Variadic<SPIRV_TosaAny_TensorArm1DTo6D>: $input1
  );


  let results = (outs
    SPIRV_TosaAny_TensorArm1DTo6D: $output
  );

  let hasVerifier = 1;

  let assemblyFormat = [{
    operands attr-dict `:` `(` type(operands) `)` `->` type(results)
  }];
}


def SPIRV_TosaPadOp : SPIRV_TosaOp<"Pad", 55, [Pure]> {
 let summary = "Pad - TOSA extended instruction set 001000.1";

  let description = [{
    https://github.khronos.org/SPIRV-Registry/extended/TOSA.001000.1.html#_pad
  }];


  let arguments = (ins
    SPIRV_TosaAny_TensorArm1DTo6D: $input1,
    SPIRV_Int32_1DTensorArmOfEvenLength2To12: $padding,
    SPIRV_TosaAny_1DTensorArmOfLength1: $pad_const
  );


  let results = (outs
    SPIRV_TosaAny_TensorArm1DTo6D: $output
  );

  let hasVerifier = 1;

  let assemblyFormat = [{
    operands attr-dict `:` `(` type(operands) `)` `->` type(results)
  }];
}


def SPIRV_TosaReshapeOp : SPIRV_TosaOp<"Reshape", 56, [Pure]> {
 let summary = "Reshape - TOSA extended instruction set 001000.1";

  let description = [{
    https://github.khronos.org/SPIRV-Registry/extended/TOSA.001000.1.html#_reshape
  }];


  let arguments = (ins
    SPIRV_TosaAny_TensorArmUpTo6D: $input1,
    SPIRV_Int32_1DTensorArmOfLength1To6: $shape
  );


  let results = (outs
    SPIRV_TosaAny_TensorArmUpTo6D: $output
  );

  let hasVerifier = 1;

  let assemblyFormat = [{
    operands attr-dict `:` `(` type(operands) `)` `->` type(results)
  }];
}


def SPIRV_TosaReverseOp : SPIRV_TosaOp<"Reverse", 57, [Pure]> {
 let summary = "Reverse - TOSA extended instruction set 001000.1";

  let description = [{
    https://github.khronos.org/SPIRV-Registry/extended/TOSA.001000.1.html#_reverse
  }];


  let arguments = (ins
    SPIRV_Int32: $axis,
    SPIRV_TosaAny_TensorArm1DTo6D: $input1
  );


  let results = (outs
    SPIRV_TosaAny_TensorArm1DTo6D: $output
  );

  let hasVerifier = 1;

  let assemblyFormat = [{
    operands attr-dict `:` `(` type(operands) `)` `->` type(results)
  }];
}


def SPIRV_TosaSliceOp : SPIRV_TosaOp<"Slice", 58, [Pure]> {
 let summary = "Slice - TOSA extended instruction set 001000.1";

  let description = [{
    https://github.khronos.org/SPIRV-Registry/extended/TOSA.001000.1.html#_slice
  }];


  let arguments = (ins
    SPIRV_TosaAny_TensorArm1DTo6D: $input1,
    SPIRV_Int32_1DTensorArmOfLength1To6: $start,
    SPIRV_Int32_1DTensorArmOfLength1To6: $size
  );


  let results = (outs
    SPIRV_TosaAny_TensorArm1DTo6D: $output
  );

  let hasVerifier = 1;

  let assemblyFormat = [{
    operands attr-dict `:` `(` type(operands) `)` `->` type(results)
  }];
}


def SPIRV_TosaTileOp : SPIRV_TosaOp<"Tile", 59, [Pure]> {
 let summary = "Tile - TOSA extended instruction set 001000.1";

  let description = [{
    https://github.khronos.org/SPIRV-Registry/extended/TOSA.001000.1.html#_tile
  }];


  let arguments = (ins
    SPIRV_TosaAny_TensorArm1DTo6D: $input1,
    SPIRV_Int32_1DTensorArmOfLength1To6: $multiples
  );


  let results = (outs
    SPIRV_TosaAny_TensorArm1DTo6D: $output
  );

  let hasVerifier = 1;

  let assemblyFormat = [{
    operands attr-dict `:` `(` type(operands) `)` `->` type(results)
  }];
}


def SPIRV_TosaTransposeOp : SPIRV_TosaOp<"Transpose", 60, [Pure]> {
 let summary = "Transpose - TOSA extended instruction set 001000.1";

  let description = [{
    https://github.khronos.org/SPIRV-Registry/extended/TOSA.001000.1.html#_transpose
  }];


  let arguments = (ins
    SPIRV_Int32_1DTensorArmOfLength1To6: $perms,
    SPIRV_TosaAny_TensorArm1DTo6D: $input1
  );


  let results = (outs
    SPIRV_TosaAny_TensorArm1DTo6D: $output
  );

  let hasVerifier = 1;

  let assemblyFormat = [{
    operands attr-dict `:` `(` type(operands) `)` `->` type(results)
  }];
}


def SPIRV_TosaGatherOp : SPIRV_TosaOp<"Gather", 61, [Pure]> {
 let summary = "Gather - TOSA extended instruction set 001000.1";

  let description = [{
    https://github.khronos.org/SPIRV-Registry/extended/TOSA.001000.1.html#_gather
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
    operands attr-dict `:` `(` type(operands) `)` `->` type(results)
  }];
}


def SPIRV_TosaScatterOp : SPIRV_TosaOp<"Scatter", 62, [Pure]> {
 let summary = "Scatter - TOSA extended instruction set 001000.1";

  let description = [{
    https://github.khronos.org/SPIRV-Registry/extended/TOSA.001000.1.html#_scatter
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
    operands attr-dict `:` `(` type(operands) `)` `->` type(results)
  }];
}


def SPIRV_TosaResizeOp : SPIRV_TosaOp<"Resize", 63, [Pure]> {
 let summary = "Resize - TOSA extended instruction set 001000.1";

  let description = [{
    https://github.khronos.org/SPIRV-Registry/extended/TOSA.001000.1.html#_resize
  }];


  let arguments = (ins
    SPIRV_Int32: $mode,
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
    operands attr-dict `:` `(` type(operands) `)` `->` type(results)
  }];
}


def SPIRV_TosaCastOp : SPIRV_TosaOp<"Cast", 64, [Pure]> {
 let summary = "Cast - TOSA extended instruction set 001000.1";

  let description = [{
    https://github.khronos.org/SPIRV-Registry/extended/TOSA.001000.1.html#_cast
  }];


  let arguments = (ins
    SPIRV_TosaAny_TensorArmUpTo6D: $input
  );


  let results = (outs
    SPIRV_TosaAny_TensorArmUpTo6D: $output
  );

  let hasVerifier = 1;

  let assemblyFormat = [{
    operands attr-dict `:` `(` type(operands) `)` `->` type(results)
  }];
}


def SPIRV_TosaRescaleOp : SPIRV_TosaOp<"Rescale", 65, [Pure]> {
 let summary = "Rescale - TOSA extended instruction set 001000.1";

  let description = [{
    https://github.khronos.org/SPIRV-Registry/extended/TOSA.001000.1.html#_rescale
  }];


  let arguments = (ins
    SPIRV_Bool: $scale32,
    SPIRV_Int32: $rounding_mode,
    SPIRV_Bool: $per_channel,
    SPIRV_Bool: $input_unsigned,
    SPIRV_Bool: $output_unsigned,
    SPIRV_TosaInteger_TensorArmUpTo6D: $input,
    SPIRV_Int16OrInt32_TensorArm1D: $multiplier,
    SPIRV_Int8_TensorArm1D: $shift,
    SPIRV_TosaInteger_1DTensorArmOfLength1: $input_zp,
    SPIRV_TosaInteger_1DTensorArmOfLength1: $output_zp
  );


  let results = (outs
    SPIRV_TosaInteger_TensorArmUpTo6D: $output
  );

  let hasVerifier = 1;

  let assemblyFormat = [{
    operands attr-dict `:` `(` type(operands) `)` `->` type(results)
  }];
}


#endif // MLIR_DIALECT_SPIRV_IR_TOSA_OPS
