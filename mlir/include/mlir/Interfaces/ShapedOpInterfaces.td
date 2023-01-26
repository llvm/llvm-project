//===-- ShapedOpInterfaces.td - Interfaces for Shaped Ops --*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains a set of interfaces for ops that operate on shaped values.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_INTERFACES_SHAPEDOPINTERFACES
#define MLIR_INTERFACES_SHAPEDOPINTERFACES

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// ShapedDimOpInterface
//===----------------------------------------------------------------------===//

// Ops that return the dimension of a shaped value.
def ShapedDimOpInterface : OpInterface<"ShapedDimOpInterface"> {
  let description = [{
    An interface for ops that return the dimension of a shaped value (such as a
    tensor or a memref).  It provides access to the source shaped value and to
    the dimension.
  }];
  let cppNamespace = "::mlir";

  let methods = [
    InterfaceMethod<
      /*desc=*/[{
        Return the shaped value operand. This is the value that the dimension
        is taken from.
      }],
      /*retTy=*/"::mlir::Value",
      /*methodName=*/"getShapedValue",
      /*args=*/(ins)
    >,
    InterfaceMethod<
      /*desc=*/[{
        Return the dimension operand. This can be a constant or an SSA value.
      }],
      /*retTy=*/"::mlir::OpFoldResult",
      /*methodName=*/"getDimension",
      /*args=*/(ins)
    >
  ];

  let verify = [{
    return verifyShapedDimOpInterface($_op);
  }];
}

#endif // MLIR_INTERFACES_SHAPEDOPINTERFACES
