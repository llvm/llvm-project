//===- IRDLInterfaces.td - IRDL Attributes -----------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the interfaces used by IRDL.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_IRDL_IR_IRDLINTERFACES
#define MLIR_DIALECT_IRDL_IR_IRDLINTERFACES

include "mlir/IR/OpBase.td"

class VerifyInterface<string name, string return_type>
      : OpInterface<"Verify" # name # "Interface"> {
  let cppNamespace = "::mlir::irdl";

  let description = "Interface to get an IRDL"
                  # name # "verifier from an operation.";

  let methods = [
    InterfaceMethod<
      [{
        Get an instance of a constraint verifier for the associated operation."
        Returns `nullptr` upon failure.
      }],
      "std::unique_ptr<::mlir::irdl::" # return_type # ">",
      "getVerifier",
      (ins "::mlir::ArrayRef<Value>":$valueToConstr,
      [{::mlir::DenseMap<::mlir::irdl::TypeOp,
        std::unique_ptr<::mlir::DynamicTypeDefinition>> const&}]:$types,
      [{::mlir::DenseMap<::mlir::irdl::AttributeOp,
        std::unique_ptr<::mlir::DynamicAttrDefinition>> const&}]:$attrs)
    >
  ];
}

def VerifyConstraintInterface : VerifyInterface<"Constraint", "Constraint"> {}

def VerifyRegionInterface : VerifyInterface<"Region", "RegionConstraint"> {}

#endif // MLIR_DIALECT_IRDL_IR_IRDLINTERFACES
