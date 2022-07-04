//===-- OpenMPOpsInterfaces.td - OpenMP op interfaces ------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the OpenMP Dialect interfaces definition file.
//
//===----------------------------------------------------------------------===//

#ifndef OpenMP_OPS_INTERFACES
#define OpenMP_OPS_INTERFACES

include "mlir/IR/OpBase.td"

def OutlineableOpenMPOpInterface : OpInterface<"OutlineableOpenMPOpInterface"> {
  let description = [{
    OpenMP operations whose region will be outlined will implement this
    interface. These operations will
  }];

  let cppNamespace = "::mlir::omp";

  let methods = [
    InterfaceMethod<"Get alloca block", "::mlir::Block*", "getAllocaBlock",
      (ins), [{
      return &$_op.getRegion().front();
      }]>,
  ];
}

def ReductionClauseInterface : OpInterface<"ReductionClauseInterface"> {
  let description = [{
    OpenMP operations that support reduction clause have this interface.
  }];

  let cppNamespace = "::mlir::omp";

  let methods = [
    InterfaceMethod<
      "Get reduction vars", "::mlir::SmallVector<::mlir::Value>",
      "getReductionVars">,
  ];
}

#endif // OpenMP_OPS_INTERFACES
