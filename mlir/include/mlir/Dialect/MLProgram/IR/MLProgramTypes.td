//===- MLProgramTypes.td - Type definitions ----------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLPROGRAM_TYPES
#define MLPROGRAM_TYPES

include "mlir/IR/AttrTypeBase.td"
include "mlir/Dialect/MLProgram/IR/MLProgramBase.td"

class MLProgram_Type<string name, list<Trait> traits = [],
                     string baseCppClass = "::mlir::Type">
    : TypeDef<MLProgram_Dialect, name, traits, baseCppClass> {}

def MLProgram_TokenType : MLProgram_Type<"Token"> {
  let summary = "Token for establishing execution ordering in a graph";
  let mnemonic = "token";
}

#endif // MLPROGRAM_TYPES
