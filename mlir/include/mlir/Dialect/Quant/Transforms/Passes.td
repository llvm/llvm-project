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
