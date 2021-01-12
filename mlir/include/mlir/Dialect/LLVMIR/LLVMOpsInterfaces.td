//===-- LLVMOpsInterfaces.td - LLVM op interfaces ----------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the LLVM IR interfaces definition file.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_OPS_INTERFACES
#define LLVM_OPS_INTERFACES

include "mlir/IR/OpBase.td"

def FastmathFlagsInterface : OpInterface<"FastmathFlagsInterface"> {
  let description = [{
    Access to op fastmath flags.
  }];

  let cppNamespace = "::mlir::LLVM";

  let methods = [
    InterfaceMethod<"Get fastmath flags", "::mlir::LLVM::FastmathFlags", "fastmathFlags">,
  ];
}

#endif // LLVM_OPS_INTERFACES
