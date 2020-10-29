//===-- Passes.td - TOSA optimization pass declarations *- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the optimization passes for the TOSA Dialect in MLIR.
//
//===----------------------------------------------------------------------===//

include "mlir/Pass/PassBase.td"

def TosaBinaryInputReshapePass : Pass<"tosa-make-broadcastable", "FuncOp"> {
  let summary = "TOSA rank Reshape to enable Broadcasting";
  let constructor = "CreateTosaMakeBroadcastablePass()";
}
