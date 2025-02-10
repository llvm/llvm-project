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

// TODO: To be moved to LinalgInterfaces.td, see:
//  * https://github.com/llvm/llvm-project/pull/123902
//  * https://discourse.llvm.org/t/rfc-move-tensor-pack-and-tensor-unpack-into-linalg/
def TensorRelayoutOpInterface : OpInterface<"RelayoutOpInterface"> {
  let description = [{
    A Tensor (soon to be Linalg) relayout-op is either tensor.pack or
    tensor.unpack.

    While we could extend this interface with methods from Tensor_RelayoutOp,
    this is currently not needed and left as a TODO.
  }];
  let cppNamespace = "::mlir::tensor";
}

#endif // TENSOR_IR_TENSORINTERFACES
