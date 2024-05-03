//===-- OpenACCOpsInterfaces.td - OpenACC op interfaces ----*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the OpenACC Dialect interfaces definition file.
//
//===----------------------------------------------------------------------===//

#ifndef OPENACC_OPS_INTERFACES
#define OPENACC_OPS_INTERFACES

include "mlir/IR/OpBase.td"

def RecipeInterface : OpInterface<"RecipeInterface"> {
    let description = [{
      OpenACC operations with one or more regions holding executable code.
    }];
    let cppNamespace = "::mlir::acc";
    let methods = [
      InterfaceMethod<
        /*description=*/[{
          For the given region of the operation return the block
          inside the region, where an alloca-like operation should be inserted.
          The default implementation returns the entry block of the region.
        }],
        /*retTy*/"::mlir::Block *",
        /*methodName=*/"getAllocaBlock",
        /*args=*/(ins "::mlir::Region &":$region),
        /*methodBody=*/"",
        /*defaultImplementation=*/[{
          return &region.front();
        }]
      >,
    ];
}

#endif // OPENACC_OPS_INTERFACES
