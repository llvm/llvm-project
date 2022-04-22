//===- BufferizationBase.td - Bufferization dialect base ---*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef BUFFERIZATION_BASE
#define BUFFERIZATION_BASE

include "mlir/IR/OpBase.td"

def Bufferization_Dialect : Dialect {
  let name = "bufferization";
  let cppNamespace = "::mlir::bufferization";
  let description = [{
    Bufferization in MLIR is the process of converting the `tensor` type to the
    `memref` type.
    The `bufferization` dialect is intended to collect operations/interfaces
    specific to the bufferization passes.

    Overview of the bufferization infrastructure and important conceptual
    details related to using the MLIR dialect conversion infrastructure can be
    found in [bufferization](/docs/Bufferization/) and [buffer
    deallocation](/docs/BufferDeallocationInternals/).
  }];
  let dependentDialects = ["memref::MemRefDialect", "tensor::TensorDialect"];

  let extraClassDeclaration = [{
    /// An attribute that can override writability of buffers of tensor function
    /// arguments during One-Shot Module Bufferize.
    constexpr const static ::llvm::StringLiteral
        kWritableAttrName = "bufferization.writable";

    /// Attribute name used to mark the bufferization layout for region
    /// arguments during One-Shot Module Bufferize.
    constexpr const static ::llvm::StringLiteral
        kBufferLayoutAttrName = "bufferization.buffer_layout";
  }];
  let hasOperationAttrVerify = 1;
}

#endif // BUFFERIZATION_BASE
