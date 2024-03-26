//===- AsmInterfaces.td - Asm Interfaces -------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains interfaces and other utilities for interacting with the
// AsmParser and AsmPrinter.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_IR_ASMINTERFACES_TD
#define MLIR_IR_ASMINTERFACES_TD

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// AttrAsmAliasAttrInterface
//===----------------------------------------------------------------------===//

def AttrAsmAliasAttrInterface : AttrInterface<"AttrAsmAliasAttrInterface"> {
  let cppNamespace = "::mlir";
  let description = [{
    This interface allows aliasing an attribute between dialects, allowing
    custom printing of an attribute by an external dialect.
  }];
  let methods = [
    InterfaceMethod<[{
        Returns the dialect responsible for printing and parsing the attribute
        instance.
      }],
      "Dialect*", "getAliasDialect", (ins), [{}], [{}]
    >
  ];
}

//===----------------------------------------------------------------------===//
// TypeAsmAliasTypeInterface
//===----------------------------------------------------------------------===//

def TypeAsmAliasTypeInterface : TypeInterface<"TypeAsmAliasTypeInterface"> {
  let cppNamespace = "::mlir";
  let description = [{
    This interface allows aliasing a type between dialects, allowing custom
    printing of a type by an external dialect.
  }];
  let methods = [
    InterfaceMethod<[{
        Returns the dialect responsible for printing and parsing the type
        instance.
      }],
      "Dialect*", "getAliasDialect", (ins), [{}], [{}]
    >
  ];
}

#endif // MLIR_IR_ASMINTERFACES_TD
