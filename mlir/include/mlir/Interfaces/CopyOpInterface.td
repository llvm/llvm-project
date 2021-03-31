//===- CopyOpInterface.td - Copy operation interface -------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines the interface for copy-like operations.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_INTERFACES_COPYOPINTERFACE
#define MLIR_INTERFACES_COPYOPINTERFACE

include "mlir/IR/OpBase.td"

def CopyOpInterface : OpInterface<"CopyOpInterface"> {
  let description = [{
    A copy-like operation is one that copies from source value to target value.
  }];
  let cppNamespace = "::mlir";

  let methods = [
    InterfaceMethod<
      /*desc=*/"Returns the source value for this copy operation",
      /*retTy=*/"::mlir::Value",
      /*methodName=*/"getSource"
    >,
    InterfaceMethod<
      /*desc=*/"Returns the target value for this copy operation",
      /*retTy=*/"::mlir::Value",
      /*methodName=*/"getTarget"
    >
  ];
}

#endif // MLIR_INTERFACES_COPYOPINTERFACE
