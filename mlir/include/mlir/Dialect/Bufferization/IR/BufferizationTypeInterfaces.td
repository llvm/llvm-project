//===- BufferizationTypeInterfaces.td - Type Interfaces ----*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the definition file for type interfaces used in Bufferization.
//
//===----------------------------------------------------------------------===//

#ifndef BUFFERIZATION_TYPE_INTERFACES
#define BUFFERIZATION_TYPE_INTERFACES

include "mlir/IR/OpBase.td"

def Bufferization_TensorLikeTypeInterface
    : TypeInterface<"TensorLikeType"> {
  let cppNamespace = "::mlir::bufferization";
  let description = [{
    Indicates that this type is a tensor type (similarly to a MLIR builtin
    tensor) for bufferization purposes.
  }];

  let methods = [
    InterfaceMethod<[{
        Returns a BufferLike type for this TensorLike type.
      }],
      /*retTy=*/"::mlir::FailureOr<::mlir::bufferization::BufferLikeType>",
      /*methodName=*/"getBufferType",
      /*args=*/(ins
        "const ::mlir::bufferization::BufferizationOptions &":$options,
        "::llvm::function_ref<::mlir::InFlightDiagnostic()>":$emitError
      )
    >,
    InterfaceMethod<[{
        Returns whether a BufferLike type is compatible to this TensorLike type.
        The BufferLike type is assumed to be created by getBufferType().
      }],
      /*retTy=*/"::mlir::LogicalResult",
      /*methodName=*/"verifyCompatibleBufferType",
      /*args=*/(ins
        "::mlir::bufferization::BufferLikeType":$bufferType,
        "::llvm::function_ref<::mlir::InFlightDiagnostic()>":$emitError)
    >
  ];
}

def Bufferization_BufferLikeTypeInterface
    : TypeInterface<"BufferLikeType"> {
  let cppNamespace = "::mlir::bufferization";
  let description = [{
    Indicates that this type is a buffer type (similarly to a MLIR builtin
    memref) for bufferization purposes.

    The interface currently has no methods as it is used by types to opt into
    being supported by the bufferization procedures.
  }];
}

#endif // BUFFERIZATION_TYPE_INTERFACES
