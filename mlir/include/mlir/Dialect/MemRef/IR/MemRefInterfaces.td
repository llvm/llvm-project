//===- MemRefInterfaces.td - Memref dialect interfaces -----*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MEMREF_INTERFACES
#define MEMREF_INTERFACES

include "mlir/IR/OpBase.td"

def AliasAnalysisOpInterface : OpInterface<"AliasAnalysisOpInterface"> {
  let description = [{
    An interface for memory operations that can carry alias analysis metadata.
    It provides setters and getters for the operation's alias analysis
    attributes.
  }];

  let cppNamespace = "::mlir::memref";

  let methods = [
    InterfaceMethod<
      /*desc=*/        "Returns the alias attribute or nullptr",
      /*returnType=*/  "::mlir::memref::AliasingAttr",
      /*methodName=*/  "getAliasingAttr",
      /*args=*/        (ins),
      /*methodBody=*/  [{}],
      /*defaultImpl=*/ [{
        auto op = cast<ConcreteOp>(this->getOperation());
        return op.getAliasAttr();
      }]
      >,
    InterfaceMethod<
      /*desc=*/        "Sets the alias attribute",
      /*returnType=*/  "void",
      /*methodName=*/  "setAliasingAttr",
      /*args=*/        (ins "::mlir::memref::AliasingAttr":$attr),
      /*methodBody=*/  [{}],
      /*defaultImpl=*/ [{
        auto op = cast<ConcreteOp>(this->getOperation());
        op.setAliasAttr(attr);
      }]
      >
  ];
}

#endif // MEMREF_INTERFACES
