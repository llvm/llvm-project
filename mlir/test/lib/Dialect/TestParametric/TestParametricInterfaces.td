//===-- TestInterfaces.td - Test dialect interfaces --------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TEST_DIALECT_TESTPARAMETRIC_INTERFACES
#define MLIR_TEST_DIALECT_TESTPARAMETRIC_INTERFACES

include "mlir/IR/OpBase.td"

def TestParametricOpInterface : OpInterface<"TestParametricOpInterface"> {
  let cppNamespace = "::mlir";
  let methods = [
    InterfaceMethod<"",
      "LogicalResult", "specializeAttr", (ins
        "::mlir::StringAttr":$parameterName,
        "::mlir::Attribute":$concreteAttr)>,
    InterfaceMethod<"",
      "LogicalResult", "specializeType", (ins
        "::mlir::StringAttr":$parameterName,
        "::mlir::Type":$concreteType)>,
    InterfaceMethod<"",
      "LogicalResult", "checkOperand", (ins
        "::mlir::OpOperand &":$operand,
        "::mlir::Type":$concreteType)>,
  ];
}



#endif // MLIR_TEST_DIALECT_TESTPARAMETRIC_INTERFACES
