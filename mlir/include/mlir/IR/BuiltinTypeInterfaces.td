//===- BuiltinTypeInterfaces.td - Builtin type interfaces --*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains definitions for type interfaces that closely interact with
// attributes, types, and operations in the builtin dialect.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_IR_BUILTINTYPEINTERFACES_TD_
#define MLIR_IR_BUILTINTYPEINTERFACES_TD_

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// MemRefElementTypeInterface
//===----------------------------------------------------------------------===//

def MemRefElementTypeInterface : TypeInterface<"MemRefElementTypeInterface"> {
  let cppNamespace = "::mlir";
  let description = [{
    Indication that this type can be used as element in memref types.

    Implementing this interface establishes a contract between this type and the
    memref type indicating that this type can be used as element of ranked or
    unranked memrefs. The type is expected to:

      - model an entity stored in memory;
      - have non-zero size.

    For example, scalar values such as integers can implement this interface,
    but indicator types such as `void` or `unit` should not.

    The interface currently has no methods and is used by types to opt into
    being memref elements. This may change in the future, in particular to
    require types to provide their size or alignment given a data layout.
  }];
}

#endif // MLIR_IR_BUILTINTYPEINTERFACES_TD_
