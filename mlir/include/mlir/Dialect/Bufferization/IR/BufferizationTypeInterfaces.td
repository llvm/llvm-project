//===- BufferizationTypeInterfaces.td - Bufferization type interfaces -*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===---------------------------------------------------------------------------------===//
//
// This is the definition file for type interfaces used in Bufferization.
//
//===---------------------------------------------------------------------------------===//

#ifndef BUFFERIZATION_TYPE_INTERFACES
#define BUFFERIZATION_TYPE_INTERFACES

include "mlir/IR/OpBase.td"
include "mlir/IR/BuiltinTypeInterfaces.td"

def Bufferization_TensorLikeTypeInterface
    : TypeInterface<"TensorLikeType", [ShapedTypeInterface]> {
  let cppNamespace = "::mlir::bufferization";
  let description = [{
    Indicates that the type that attaches this interface can be treated as a
    tensor type (similarly to a MLIR builtin tensor) within the Bufferization
    dialect.

    Implementing this interface means that the type also implements
    ShapedTypeInterface.

    The interface currently has no methods as it is used by types to opt into
    being supported by the bufferization procedures.
  }];
}

def Bufferization_MemRefLikeTypeInterface
    : TypeInterface<"MemRefLikeType", [ShapedTypeInterface]> {
  let cppNamespace = "::mlir::bufferization";
  let description = [{
    Indicates that the type that attaches this interface can be treated as a
    memref type (similarly to a MLIR builtin memref) within the Bufferization
    dialect.

    Implementing this interface means that the type also implements
    ShapedTypeInterface.

    The interface currently has no methods as it is used by types to opt into
    being supported by the bufferization procedures.
  }];
}

#endif // BUFFERIZATION_TYPE_INTERFACES
