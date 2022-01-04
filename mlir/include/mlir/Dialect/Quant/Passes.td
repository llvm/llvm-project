//===-- Passes.td - Quant pass definition file -------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_QUANT_PASSES
#define MLIR_DIALECT_QUANT_PASSES

include "mlir/Pass/PassBase.td"

def QuantConvertConst : Pass<"quant-convert-const", "FuncOp"> {
  let summary = "Converts constants followed by qbarrier to actual quantized "
                "values";
  let constructor = "mlir::quant::createConvertConstPass()";
}

def QuantConvertSimulatedQuant
    : Pass<"quant-convert-simulated-quantization", "FuncOp"> {
  let summary = "Converts training-time simulated quantization ops to "
                "corresponding quantize/dequantize casts";
  let constructor = "mlir::quant::createConvertSimulatedQuantPass()";
}

#endif // MLIR_DIALECT_QUANT_PASSES
