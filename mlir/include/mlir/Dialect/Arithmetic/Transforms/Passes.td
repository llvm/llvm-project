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

def ArithmeticBufferize : Pass<"arith-bufferize", "FuncOp"> {
  let summary = "Bufferize Arithmetic dialect ops.";
  let constructor = "mlir::arith::createArithmeticBufferizePass()";
}

def ArithmeticExpandOps : Pass<"arith-expand", "FuncOp"> {
  let summary = "Legalize Arithmetic ops to be convertible to LLVM.";
  let constructor = "mlir::arith::createArithmeticExpandOpsPass()";
  let dependentDialects = ["StandardOpsDialect"];
}

#endif // MLIR_DIALECT_ARITHMETIC_TRANSFORMS_PASSES
