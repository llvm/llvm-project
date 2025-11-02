//===-- OpenACCOpsInterfaces.td - OpenACC type interfaces ---*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef OPENACC_OPS_INTERFACES
#define OPENACC_OPS_INTERFACES

include "mlir/IR/OpBase.td"

def ComputeRegionOpInterface : OpInterface<"ComputeRegionOpInterface"> {
  let cppNamespace = "::mlir::acc";

  let description = [{
    An interface for compute and loop construct operations.
  }];

  let methods = [
    InterfaceMethod<"Get alloca block", "::mlir::Block*", "getAllocaBlock",
      (ins), [{
        return &$_op.getRegion().front();
      }]>,
  ];
}

def PartialEntityAccessOpInterface : OpInterface<"PartialEntityAccessOpInterface"> {
  let cppNamespace = "::mlir::acc";

  let description = [{
    An interface for operations that access a partial entity such as
    field or array element access.
  }];

  let methods = [
    InterfaceMethod<"Get the base entity being accessed", "::mlir::Value",
      "getBaseEntity", (ins)>,
    InterfaceMethod<"Check if this is a complete view of the entity", "bool",
      "isCompleteView", (ins), [{
        return false;
      }]>,
  ];
}

#endif // OPENACC_OPS_INTERFACES
