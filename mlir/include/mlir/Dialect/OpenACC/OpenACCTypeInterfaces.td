//===-- OpenACCTypeInterfaces.td - OpenACC type interfaces ---*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef OPENACC_TYPE_INTERFACES
#define OPENACC_TYPE_INTERFACES

include "mlir/IR/OpBase.td"

def OpenACC_PointerLikeTypeInterface : TypeInterface<"PointerLikeType"> {
  let cppNamespace = "::mlir::acc";

  let description = [{
    An interface for pointer-like types that point to an OpenACC var.
  }];

  // By convention, any of the pointer types associated with this interface
  // will need to provide getElementType.
  let methods = [
    InterfaceMethod<
      /*description=*/[{
        Returns the pointee type or null if the pointer has no pointee type
      }],
      /*retTy=*/"::mlir::Type",
      /*methodName=*/"getElementType"
    >,
  ];
}

#endif // OPENACC_TYPE_INTERFACES
