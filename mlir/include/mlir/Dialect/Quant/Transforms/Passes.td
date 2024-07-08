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

def LowerQuantOps : Pass<"lower-quant-ops"> {
  let summary = "Lower 'quant.dcast' and 'quant.qcast' ops.";
  let description = [{
    Lower quantization (`quant.qcast`) and dequantization (`quant.dcast`) ops
    into other core dialects.

    The lowering process generates storage type casts in the form of
    `quant.scast` ops to convert operands and results from quantized types to
    the corresponding storage type, or vice versa.
  }];
  let dependentDialects = ["::mlir::quant::QuantDialect"];
}

#endif // MLIR_DIALECT_QUANT_TRANSFORMS_PASSES
