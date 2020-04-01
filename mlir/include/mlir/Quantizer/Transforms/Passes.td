//===-- Passes.td - Quantizer pass definition file ---------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_QUANTIZER_TRANSFORMS_PASSES
#define MLIR_QUANTIZER_TRANSFORMS_PASSES

include "mlir/Pass/PassBase.td"

def QuantizerAddDefaultStats : Pass<"quantizer-add-default-stats-test"> {
  let summary = "Add default (dummy) statistics to all ops that can benefit "
                "from runtime statistics";
  let constructor = "mlir::quantizer::createAddDefaultStatsPass()";
}

def QuantizerInferQuantizedTypes : Pass<"quantizer-infer-quantized-types"> {
  let summary = "Infer quantized types for a module";
  let constructor = "mlir::quantizer::createInferQuantizedTypesPass()";
}

def QuantizerRemoveInstrumentation : Pass<"quantizer-remove-instrumentation"> {
  let summary = "Remove instrumentation and hints which have no effect on "
                "final execution";
  let constructor = "mlir::quantizer::createRemoveInstrumentationPass()";
}

#endif // MLIR_QUANTIZER_TRANSFORMS_PASSES
