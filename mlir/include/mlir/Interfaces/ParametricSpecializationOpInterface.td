//===-- ParametricSpecializationOpInterface.td -------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_INTERFACES_PARAMETRICSPECIALIZATIONOPINTERFACES
#define MLIR_INTERFACES_PARAMETRICSPECIALIZATIONOPINTERFACES

include "mlir/IR/OpBase.td"

def ParametricOpInterface : OpInterface<"ParametricOpInterface"> {
  let cppNamespace = "::mlir";
  let methods = [
    InterfaceMethod<"",
      "::mlir::LogicalResult", "specialize", (ins
        "::mlir::DictionaryAttr":$params)>,
    InterfaceMethod<"",
      "::mlir::LogicalResult", "checkOperand", (ins
        "::mlir::OpOperand &":$operand,
        "::mlir::Type":$concreteType)>,
    InterfaceMethod<"Only for symbol operation which will be cloned, mangle in-place.",
      "::mlir::FailureOr<::mlir::StringAttr>", "getMangledName", (ins
        "::mlir::DictionaryAttr":$metaArgs), "", [{
        return failure();
      }]
>,
  ];
}

def SpecializingOpInterface : OpInterface<"SpecializingOpInterface"> {
 let cppNamespace = "::mlir";
  let methods = [
    InterfaceMethod<"",
      "::mlir::SymbolRefAttr", "getTarget", (ins)>,
    InterfaceMethod<"",
      "::mlir::DictionaryAttr", "getMetaArgs", (ins)>,
    InterfaceMethod<"",
      "::mlir::LogicalResult", "setSpecializedTarget", (ins
        "::mlir::SymbolOpInterface":$target)>,
  ];
}

#endif // MLIR_INTERFACES_PARAMETRICSPECIALIZATIONOPINTERFACES
