//===-- Passes.td - Vector pass definition file ------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_VECTOR_TRANSFORMS_PASSES
#define MLIR_DIALECT_VECTOR_TRANSFORMS_PASSES

include "mlir/Pass/PassBase.td"

def VectorBufferize : Pass<"vector-bufferize", "FuncOp"> {
  let summary = "Bufferize Vector dialect ops";
  let constructor = "mlir::vector::createVectorBufferizePass()";
}

#endif // MLIR_DIALECT_VECTOR_TRANSFORMS_PASSES
