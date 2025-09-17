//===- AMXInterfaces.td - AMX interfaces -------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines interfaces for the AMX dialect.
//
//===----------------------------------------------------------------------===//

#ifndef AMX_INTERFACES
#define AMX_INTERFACES

include "mlir/IR/Interfaces.td"
include "mlir/Dialect/LLVMIR/LLVMInterfaces.td"

//===----------------------------------------------------------------------===//
// AMX Intrinsic Interface
//===----------------------------------------------------------------------===//

def AMXIntrinsicOpInterface
    : OpInterface<"AMXIntrinsicOp", [OneToOneIntrinsicOpInterface]> {
  let description = [{
    A wrapper interface for operations representing AMX LLVM intrinsics.
  }];
  let cppNamespace = "::mlir::amx";
}

#endif // AMX_INTERFACES
