//===- TensorEncoding.td - Tensor encoding interfaces ------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines the interfaces associated with tensor encoding attributes.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_IR_TENSORINTERFACES
#define MLIR_IR_TENSORINTERFACES

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// Attribute interface to verify a tensor encoding.
//===----------------------------------------------------------------------===//

def VerifiableTensorEncoding : AttrInterface<"VerifiableTensorEncoding"> {
  let cppNamespace = "::mlir";
  let description = [{
    Verifies an encoding attribute for a tensor.
  }];
  let methods = [
    InterfaceMethod<
      /*desc=*/[{
        Verifies the encoding is valid for a tensor type with the
        given shape and element type. Generates a diagnostic using
        the supplied callback on failure.
      }],
      /*retTy=*/"::llvm::LogicalResult",
      /*methodName=*/"verifyEncoding",
      /*args=*/(ins
        "::mlir::ArrayRef<int64_t>":$shape,
        "::mlir::Type":$elementType,
        "::llvm::function_ref<::mlir::InFlightDiagnostic()>":$emitError)
    >,
  ];
}

#endif // MLIR_IR_TENSORINTERFACES
