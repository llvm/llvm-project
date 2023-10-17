//===-- Passes.td - Math pass definition file --------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_MATH_TRANSFORMS_PASSES
#define MLIR_DIALECT_MATH_TRANSFORMS_PASSES

include "mlir/Pass/PassBase.td"

def MathUpliftToFMA : Pass<"math-uplift-to-fma"> {
  let summary = "Uplift arith ops to math.fma.";
  let description = [{
    Uplift sequence of addf and mulf ops to math.fma if fastmath flags allows it.
  }];
  let dependentDialects = ["math::MathDialect"];
}

#endif // MLIR_DIALECT_MATH_TRANSFORMS_PASSES
