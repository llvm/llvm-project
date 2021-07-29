//===-- BuiltinDialect.td - Builtin dialect definition -----*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the definition of the Builtin dialect. This dialect
// contains all of the attributes, operations, and types that are core to MLIR.
//
//===----------------------------------------------------------------------===//

#ifndef BUILTIN_BASE
#define BUILTIN_BASE

include "mlir/IR/OpBase.td"

def Builtin_Dialect : Dialect {
  let summary =
    "A dialect containing the builtin Attributes, Operations, and Types";

  let name = "builtin";
  let cppNamespace = "::mlir";
  let extraClassDeclaration = [{
  private:
    // Register the builtin Attributes.
    void registerAttributes();
    // Register the builtin Location Attributes.
    void registerLocationAttributes();
    // Register the builtin Types.
    void registerTypes();

  public:
  }];
}

#endif // BUILTIN_BASE
