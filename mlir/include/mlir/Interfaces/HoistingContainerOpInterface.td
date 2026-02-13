//===- HoistingContainerOpInterface.td - Interface Decl. -*- tablegen -*---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the definition file for the HoistingContainerOpInterface. 
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_INTERFACES_HOISTING_CONTAINER_OP_INTERFACE
#define MLIR_INTERFACES_HOISTING_CONTAINER_OP_INTERFACE

include "mlir/IR/OpBase.td"

def HoistingContainerOpInterface : OpInterface<"HoistingContainerOpInterface"> {
  let description = [{
    This interface models whether an operation's regions are capable of 
    acting as a container for operations hoisted from nested regions.
  }];
  let cppNamespace = "::mlir";
  let methods = [
    InterfaceMethod<
      /*desc=*/[{
        Returns true if this operation's regions can accommodate operations 
        hoisted from its nested scopes.
      }],
      /*retTy=*/"bool",
      /*methodName=*/"canContainHoistedOps",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return true;
      }]
    >
  ];
}

#endif // MLIR_INTERFACES_HOISTING_CONTAINER_OP_INTERFACE
