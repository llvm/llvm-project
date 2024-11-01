//===- EmitCBase.td - EmitC dialect ------------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines the MLIR EmitC dialect.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_EMITC_IR_EMITCBASE
#define MLIR_DIALECT_EMITC_IR_EMITCBASE

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// EmitC dialect definition
//===----------------------------------------------------------------------===//

def EmitC_Dialect : Dialect {
  let name = "emitc";
  let cppNamespace = "::mlir::emitc";

  let summary = "Dialect to generate C/C++ from MLIR.";
  let description = [{
    [include "Dialects/emitc.md"]
  }];

  let hasConstantMaterializer = 1;
  let useDefaultTypePrinterParser = 1;
  let useDefaultAttributePrinterParser = 1;
  let useFoldAPI = kEmitFoldAdaptorFolder;
}

#endif // MLIR_DIALECT_EMITC_IR_EMITCBASE
