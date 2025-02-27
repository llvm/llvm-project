//===-- Passes.td - Arith pass definition file --------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_ARITH_TRANSFORMS_PASSES
#define MLIR_DIALECT_ARITH_TRANSFORMS_PASSES

include "mlir/Pass/PassBase.td"

def ArithExpandOpsPass : Pass<"arith-expand"> {
  let summary = "Legalize Arith ops to be convertible to LLVM.";
  let dependentDialects = ["vector::VectorDialect"];
  let options = [
    Option<"includeBf16", "include-bf16", "bool", /*default=*/"false",
           "Enable the BF16 expansion patterns">,
  ];
}

def ArithUnsignedWhenEquivalent : Pass<"arith-unsigned-when-equivalent"> {
  let summary = "Replace signed ops with unsigned ones where they are proven equivalent";
  let description = [{
    Replace signed ops with their unsigned equivalents when integer range analysis
    determines that their arguments and results are all guaranteed to be
    non-negative when interpreted as signed integers. When this occurs,
    we know that the semantics of the signed and unsigned operations are the same,
    since they share the same behavior when their operands and results  are in the
    range [0, signed_max(type)].

    The affect ops include division, remainder, shifts, min, max, and integer
    comparisons.
  }];
  let constructor = "mlir::arith::createArithUnsignedWhenEquivalentPass()";
}

def ArithIntRangeOpts : Pass<"int-range-optimizations"> {
  let summary = "Do optimizations based on integer range analysis";
  let description = [{
    This pass runs integer range analysis and apllies optimizations based on its
    results. It replaces operations with known-constant results with said constants,
    rewrites `(0 <= %x < D) mod D` to `%x`.
  }];
  // Explicitly depend on "arith" because this pass could create operations in
  // `arith` out of thin air in some cases.
  let dependentDialects = [
    "::mlir::arith::ArithDialect"
  ];
}

def ArithIntRangeNarrowing : Pass<"arith-int-range-narrowing"> {
  let summary = "Reduce integer operations bitwidth based on integer range analysis";
  let description = [{
    This pass runs integer range analysis and tries to narrow arith ops to the
    specified bitwidth based on its results.

    `bitwidthsSupported` assumed to be not wider than `index` type.
    TODO: get index width from DLTI.
  }];

  let options = [
    ListOption<"bitwidthsSupported", "int-bitwidths-supported", "unsigned",
               "Integer bitwidths supported">,
  ];

  // Explicitly depend on "arith" because this pass could create operations in
  // `arith` out of thin air in some cases.
  let dependentDialects = [
    "::mlir::arith::ArithDialect"
  ];
}

def ArithEmulateUnsupportedFloats : Pass<"arith-emulate-unsupported-floats"> {
  let summary = "Emulate operations on unsupported floats with extf/truncf";
  let description = [{
    Emulate arith and vector floating point operations that use float types
    which are unspported on a target by inserting extf/truncf pairs around all
    such operations in order to produce arithmetic that can be performed while
    preserving the original rounding behavior.

    This pass does not attempt to reason about the operations being performed
    to determine when type conversions can be elided.
  }];

  let options = [
    ListOption<"sourceTypeStrs", "source-types", "std::string",
      "MLIR types without arithmetic support on a given target">,
    Option<"targetTypeStr", "target-type", "std::string", "\"f32\"",
      "MLIR type to convert the unsupported source types to">,
  ];

  let dependentDialects = ["vector::VectorDialect"];
}

def ArithEmulateWideInt : Pass<"arith-emulate-wide-int"> {
  let summary = "Emulate 2*N-bit integer operations using N-bit operations";
  let description = [{
    Emulate arith integer operations that use too wide integer types with
    equivalent operations on supported narrow integer types. This is done by
    splitting original integer values into two halves.

    This pass is intended preserve semantics but not necessarily provide the
    most efficient implementation.
    TODO: Optimize op emulation.

    Currently, only power-of-two integer bitwidths are supported.
  }];
  let options = [
    Option<"widestIntSupported", "widest-int-supported", "unsigned",
           /*default=*/"32", "Widest integer type supported by the target">,
  ];
  let dependentDialects = ["vector::VectorDialect"];
}

#endif // MLIR_DIALECT_ARITH_TRANSFORMS_PASSES
