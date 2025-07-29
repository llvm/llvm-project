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

def MathExtendToSupportedTypes : Pass<"math-extend-to-supported-types"> {
  let summary = "Legalize floating-point math ops on low-precision floats";
  let description = [{
    On many targets, the math functions are not implemented for floating-point
    types less precise than IEEE single-precision (aka f32), such as half-floats,
    bfloat16, or 8-bit floats.

    This pass explicitly legalizes these math functions by inserting
    `arith.extf` and `arith.truncf` pairs around said op, which preserves
    the original semantics while enabling lowering. The extra supported floating-point
    types for the target are passed as arguments. Types f64 and f32 are implicitly 
    supported.

    As an exception, this pass does not legalize `math.fma`, because
    that is an operation frequently implemented at low precisions.
  }];
  let options = [
    ListOption<"extraTypeStrs", "extra-types", "std::string",
      "MLIR types with arithmetic support on a given target (f64 and f32 are implicitly supported)">,
    Option<"targetTypeStr", "target-type", "std::string", "\"f32\"",
      "MLIR type to convert the unsupported source types to">,
  ];
  let dependentDialects = ["math::MathDialect", "arith::ArithDialect"];
}

def MathExpandOpsPass : Pass<"math-expand-ops"> {
  let summary = "Expand math operations.";
  let description = [{
    Expands some math operations into more fundamental operations, allowing them
    to be subsequently lowered through these. For example, hyperbolic functions
    are transformed into their expanded form containing only `exp` functions.

    The `ops` parameter can be used to apply only a subset of all the
    available expansions, these must correspond to the operation mnemonic.
    For example, `ops=sinh,acosh` will expand only `math.sinh` and
    `math.acosh` operations. If the list is empty, then all expansions are
    applied.
  }];
  let dependentDialects = ["arith::ArithDialect"];
  let options = [
    ListOption<"opMnemonics", "ops", "std::string",
               "Operations to expand.">
  ];
}

#endif // MLIR_DIALECT_MATH_TRANSFORMS_PASSES
