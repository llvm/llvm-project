//===-- Passes.td - StandardOps pass definition file -------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_STANDARD_TRANSFORMS_PASSES
#define MLIR_DIALECT_STANDARD_TRANSFORMS_PASSES

include "mlir/Pass/PassBase.td"

def ExpandAtomic : FunctionPass<"expand-atomic"> {
  let summary = "Expands AtomicRMWOp into GenericAtomicRMWOp.";
  let constructor = "mlir::createExpandAtomicPass()";
}

def StdBufferize : FunctionPass<"std-bufferize"> {
  let summary = "Bufferize the std dialect";
  let constructor = "mlir::createStdBufferizePass()";
}

#endif // MLIR_DIALECT_STANDARD_TRANSFORMS_PASSES
