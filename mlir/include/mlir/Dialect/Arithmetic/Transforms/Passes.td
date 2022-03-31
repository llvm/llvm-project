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

def ArithmeticBufferize : Pass<"arith-bufferize", "ModuleOp"> {
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

def ArithmeticExpandOps : Pass<"arith-expand"> {
  let summary = "Legalize Arithmetic ops to be convertible to LLVM.";
  let constructor = "mlir::arith::createArithmeticExpandOpsPass()";
}

#endif // MLIR_DIALECT_ARITHMETIC_TRANSFORMS_PASSES
