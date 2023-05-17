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

def VerifyConstraintInterface : OpInterface<"VerifyConstraintInterface"> {
  let cppNamespace = "::mlir::irdl";

  let description = [{
    Interface to get an IRDL constraint verifier from an operation. 
  }];

  let methods = [
    InterfaceMethod<
      [{
        Get an instance of a constraint verifier for the associated operation."
        Returns `nullptr` upon failure.
      }],
      "std::unique_ptr<::mlir::irdl::Constraint>",
      "getVerifier",
      (ins "::mlir::ArrayRef<Value>":$valueToConstr,
      "::mlir::DenseMap<::mlir::irdl::TypeOp, std::unique_ptr<::mlir::DynamicTypeDefinition>> const&":$types,
      "::mlir::DenseMap<::mlir::irdl::AttributeOp, std::unique_ptr<::mlir::DynamicAttrDefinition>> const&":$attrs)
    >
  ];
}

#endif // MLIR_DIALECT_IRDL_IR_IRDLINTERFACES
