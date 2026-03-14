//===-- MyExtensionTypes.td - Transform dialect tutorial ---*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines Transform dialect extension types used in the
// Chapter 3 of the Transform dialect tutorial.
//
//===----------------------------------------------------------------------===//

#ifndef MY_EXTENSIONTYPES
#define MY_EXTENSIONTYPES

include "mlir/IR/AttrTypeBase.td"
include "mlir/Dialect/Transform/IR/TransformDialect.td"
include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.td"

// Transform dialect allows additional types to be defined and injected.
def CallOpInterfaceHandle
  : TypeDef<Transform_Dialect, "CallOpInterfaceHandle",
      // The type must implement `TransformHandleTypeInterface`.
      [DeclareTypeInterfaceMethods<TransformHandleTypeInterface>]> {

  // The usual components of a type such as description, mnemonic and assembly format 
  // should be provided.
  let summary = "handle to payload operations implementing CallOpInterface";
  let mnemonic = "my.call_op_interface";
  let assemblyFormat = "";
}

#endif // MY_EXTENSIONTYPES
