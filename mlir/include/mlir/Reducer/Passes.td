//===-- Passes.td - MLIR Reduce pass definition file -------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains definitions of the passes for the MLIR Reduce Tool.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_REDUCER_PASSES
#define MLIR_REDUCER_PASSES

include "mlir/Pass/PassBase.td"

def ReductionTree : Pass<"reduction-tree", "ModuleOp"> {
  let summary = "A general reduction tree pass for the MLIR Reduce Tool";
  let constructor = "mlir::createReductionTreePass()";
}

#endif // MLIR_REDUCE_PASSES
