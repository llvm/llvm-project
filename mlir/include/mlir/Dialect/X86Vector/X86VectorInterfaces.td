//===- X86VectorInterfaces.td - X86Vector interfaces -------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines interfaces for the X86Vector dialect.
//
//===----------------------------------------------------------------------===//

#ifndef X86VECTOR_INTERFACES
#define X86VECTOR_INTERFACES

include "mlir/IR/Interfaces.td"
include "mlir/Dialect/LLVMIR/LLVMInterfaces.td"

//===----------------------------------------------------------------------===//
// X86 Intrinsic Interface
//===----------------------------------------------------------------------===//

def X86IntrinsicOpInterface
    : OpInterface<"X86IntrinsicOp", [OneToOneIntrinsicOpInterface]> {
  let description = [{
    A wrapper interface for operations representing x86 LLVM intrinsics.
  }];
  let cppNamespace = "::mlir::x86vector";
}

#endif // X86VECTOR_INTERFACES
