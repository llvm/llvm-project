//===-- Passes.td - FxpMath pass definition file -----------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_FXPMATH_PASSES
#define MLIR_DIALECT_FXPMATH_PASSES

include "mlir/Pass/PassBase.td"

def FxpMathLowerUniformCasts : Pass<"fxpmath-lower-uniform-casts"> {
  let summary = "Lowers uniform-quantized casts";
  let constructor = "mlir::fxpmath::createLowerUniformCastsPass()";
}

def FxpMathLowerUniformRealMath : Pass<"fxpmath-lower-uniform-real-math"> {
  let summary = "Lowers uniform-quantized real math ops to integer arithmetic";
  let constructor = "mlir::fxpmath::createLowerUniformRealMathPass()";
}

#endif // MLIR_DIALECT_FXPMATH_PASSES
