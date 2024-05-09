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

def MathLegalizeToF32 : Pass<"math-legalize-to-f32"> {
  let summary = "Legalize floating-point math ops on low-precision floats";
  let description = [{
    On many targets, the math functions are not implemented for floating-point
    types less precise than IEEE single-precision (aka f32), such as half-floats,
    bfloat16, or 8-bit floats.

    This pass explicitly legalizes these math functions by inserting
    `arith.extf` and `arith.truncf` pairs around said op, which preserves
    the original semantics while enabling lowering.

    As an exception, this pass does not legalize `math.fma`, because
    that is an operation frequently implemented at low precisions.
  }];
  let dependentDialects = ["math::MathDialect", "arith::ArithDialect"];
}

def MathCanonicalizeF32Promotion : Pass<"math-canonicalize-f32-promotion"> {
  let summary = "Eliminate redundant truncf/extf pairs";
  let description = [{
    `legalize-to-f32` pass does f32 promotion for every op belonging to the
    illegal op list. Once there are some consecutive illegal ops, `legalize-to-f32`
    will insert redundant `arith.truncf` and `arith.extf` pairs between the illegal
    ops.
    
    This pass is to eliminate the redundant truncf/extf pairs.

    Example:

    ```mlir
    // the initial func
    func.func @bf16_sin_vector(%arg0: vector<32xbf16>) -> vector<32xbf16> {
        %0 = math.absf %arg0 : vector<32xbf16>
        %1 = math.sin %0 : vector<32xbf16>
        return %1 : vector<32xbf16>
      }
    // after legalize-to-f32
    func.func @bf16_sin_vector(%arg0: vector<32xbf16>) -> vector<32xbf16> {
        %0 = arith.extf %arg0 : vector<32xbf16> to vector<32xf32>
        %1 = math.absf %0 : vector<32xf32>
        %2 = arith.truncf %1 : vector<32xf32> to vector<32xbf16>
        %3 = arith.extf %2 : vector<32xbf16> to vector<32xf32>
        %4 = math.sin %3 : vector<32xf32>
        %5 = arith.truncf %4 : vector<32xf32> to vector<32xbf16>
        return %5 : vector<32xbf16>
      }
    // after canonicalize-f32-promotion
    func.func @bf16_sin_vector(%arg0: vector<32xbf16>) -> vector<32xbf16> {
        %0 = arith.extf %arg0 : vector<32xbf16> to vector<32xf32>
        %1 = math.absf %0 : vector<32xf32>
        %2 = math.sin %1 : vector<32xf32>
        %3 = arith.truncf %2 : vector<32xf32> to vector<32xbf16>
        return %3 : vector<32xbf16>
      }
    ```

  }];
  let dependentDialects = ["math::MathDialect", "arith::ArithDialect"];
}

#endif // MLIR_DIALECT_MATH_TRANSFORMS_PASSES
