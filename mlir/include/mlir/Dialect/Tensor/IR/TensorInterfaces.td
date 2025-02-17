//===- TensorInterfaces.td - Tensor Interfaces Declaration -*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the definition file for the structured interface sfor Tensor ops.
//
//===----------------------------------------------------------------------===//

#ifndef TENSOR_IR_TENSORINTERFACES
#define TENSOR_IR_TENSORINTERFACES

include "mlir/Interfaces/DestinationStyleOpInterface.td"
include "mlir/IR/OpBase.td"

def TensorRelayoutOpInterface : OpInterface<"RelayoutOpInterface"> {
  let description = [{
    A Tensor relayout-op is either linalg.pack or linalg.unpack.

    While we could extend this interface with methods from Linalg_RelayoutOp,
    this is currently not needed and left as a TODO.
  }];
  let cppNamespace = "::mlir::tensor";
}

#endif // TENSOR_IR_TENSORINTERFACES

