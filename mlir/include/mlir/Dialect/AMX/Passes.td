//===-- Passes.td - AMX pass definition file ------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_AMX_PASSES_TD
#define MLIR_DIALECT_AMX_PASSES_TD

include "mlir/Pass/PassBase.td"

def EnableAMXTileBinding
    : Pass<"enable-amx-tile-binding", "mlir::func::FuncOp"> {
  let summary = "Enable AMX tile register binding";
  let description = [{
    Enables the AMX tile register binding for each func 
    by propagating specified binding information 
    and automatically configuring harware context
  }];
  let dependentDialects = ["func::FuncDialect", "amx::AMXDialect", "LLVM::LLVMDialect"];
}

#endif // MLIR_DIALECT_AMX_PASSES_TD
