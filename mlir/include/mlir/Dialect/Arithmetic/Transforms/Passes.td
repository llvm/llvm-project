//===-- Passes.td - Arithmetic pass definition file --------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_ARITHMETIC_TRANSFORMS_PASSES
#define MLIR_DIALECT_ARITHMETIC_TRANSFORMS_PASSES

include "mlir/Pass/PassBase.td"

def ArithmeticBufferizePass : Pass<"arith-bufferize", "ModuleOp"> {
  let summary = "Bufferize Arithmetic dialect ops.";
  let description = [{
    This pass bufferizes arith dialect ops.

    This pass needs to be a module pass because it inserts memref.global
    ops into the module, which cannot be done safely from a function pass due to
    multi-threading. Most other bufferization passes can run in parallel at
    function granularity.
  }];
  let constructor = "mlir::arith::createArithmeticBufferizePass()";
  let options = [
    Option<"alignment", "alignment", "unsigned", /*default=*/"0",
           "Create global memrefs with a specified alignment">,
  ];
}

def ArithmeticExpandOpsPass : Pass<"arith-expand"> {
  let summary = "Legalize Arithmetic ops to be convertible to LLVM.";
  let constructor = "mlir::arith::createArithmeticExpandOpsPass()";
}

def ArithmeticUnsignedWhenEquivalentPass : Pass<"arith-unsigned-when-equivalent"> {
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
  let constructor = "mlir::arith::createArithmeticUnsignedWhenEquivalentPass()";
}

#endif // MLIR_DIALECT_ARITHMETIC_TRANSFORMS_PASSES
