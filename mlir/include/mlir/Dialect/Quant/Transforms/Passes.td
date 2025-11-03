//===-- Passes.td - Arith pass definition file --------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_QUANT_TRANSFORMS_PASSES
#define MLIR_DIALECT_QUANT_TRANSFORMS_PASSES

include "mlir/Pass/PassBase.td"

def LowerQuantOps : Pass<"lower-quant-ops", "func::FuncOp"> {
  let summary = "Lower quant.dcast and quant.qcast ops";
  let description = [{
    Lower quantization (`quant.qcast`) and dequantization (`quant.dcast`) ops
    into other core dialects.

    The lowering process generates storage type casts in the form of
    `quant.scast` ops to act as an interface between the original quantized
    types of operands and results and their corresponding storage types used in
    the generated arithmetic computations.
  }];
  let dependentDialects = [
    "arith::ArithDialect",
    "linalg::LinalgDialect",
    "quant::QuantDialect",
    "shape::ShapeDialect",
    "tensor::TensorDialect"
  ];
}

def NormalizeQuantTypes : Pass<"normalize-quant-types", "func::FuncOp"> {
  let summary = "Normalize generic quantized types to specific quantized types";
  let description = [{
    This pass converts generic quantized types in the `quant` dialect to more
    specific types when possible.

    The following conversions are performed:

    1. Sub-channel to per-axis: If the shape of the scales tensor of sub-channel
       quantized type has all but one non-one value, it is converted to a
       per-axis quantized type.
       
       For example:
       
       * `!quant.uniform<i8:f32:{0:1}, {{2.0}, {3.0}}>` 
          -> `!quant.uniform<i8:f32:0, {2.0, 3.0}>`
       * `tensor<?x?x!quant.uniform<i8:f32:{0:1,1:4}, {{2.0}, {3.0}}>>` 
          -> `tensor<?x?x!quant.uniform<i8:f32:0, {2.0, 3.0}>>`

    2. Sub-channel to per-tensor: If a sub-channel quantized type has only 
       one scale or zero-point, it is converted to a per-tensor 
       quantized type.
       
       For example:
       
       * `!quant.uniform<i8:f32:{}, {{2.0}}>`
          -> `!quant.uniform<i8:f32, 2.0>`
       * `tensor<?x?x!quant.uniform<i8:f32:{0:1, 0:4}, {{2.0}}>>`
          -> `tensor<?x?x!quant.uniform<i8:f32, 2.0>>`

    The rationale for these conversions is that the decompositions / handling of
    more precise quantized types tends to be more efficient than treating
    everything as subchannel type.

  }];
  let dependentDialects = ["func::FuncDialect", "quant::QuantDialect"];
}

def StripFuncQuantTypes : Pass<"strip-func-quant-types"> {
  let summary = "Strip quantized types from function headers";
  let description = [{
    Identify occurrences of function arguments using a quantized type and
    replace them with a new value of the corresponding storage (signless
    integer) type. For each converted argument, a `quant.scast` op is introduced
    at the head of the function's entry block converting the new integer
    argument into the original quantized value.
  }];
  let dependentDialects = [
    "func::FuncDialect",
    "quant::QuantDialect"
  ];
}

#endif // MLIR_DIALECT_QUANT_TRANSFORMS_PASSES
