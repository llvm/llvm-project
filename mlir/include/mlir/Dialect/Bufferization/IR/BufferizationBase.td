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
    Simply put, bufferization is the process of converting computations on the
    mathematical tensor construct to computations on physical memory buffers.
    The `bufferization` dialect contains operations/interfaces specific to the
    bufferization passes.

    An overview of the bufferization infrastructure and important conceptual
    details related to using the MLIR dialect conversion infrastructure can be
    found in [bufferization](/docs/Bufferization/) and [buffer
    deallocation](/docs/BufferDeallocationInternals/).
  }];
  let dependentDialects = [
    "affine::AffineDialect", "memref::MemRefDialect", "tensor::TensorDialect",
    "arith::ArithDialect"
  ];

  let extraClassDeclaration = [{
    /// Verify an attribute from this dialect on the argument at 'argIndex' for
    /// the region at 'regionIndex' on the given operation. Returns failure if
    /// the verification failed, success otherwise. This hook may optionally be
    /// invoked from any operation containing a region.
    LogicalResult verifyRegionArgAttribute(Operation *,
                                           unsigned regionIndex,
                                           unsigned argIndex,
                                           NamedAttribute) override;

    /// An attribute that can override writability of buffers of tensor function
    /// arguments during One-Shot Module Bufferize.
    constexpr const static ::llvm::StringLiteral
        kWritableAttrName = "bufferization.writable";

    /// An attribute for function arguments that describes how the function
    /// accesses the buffer. Can be one "none", "read", "write" or "read-write".
    ///
    /// When no attribute is specified, the analysis tries to infer the access
    /// behavior from its body. In case of external functions, for which no
    /// function body is available, "read-write" is assumed by default.
    constexpr const static ::llvm::StringLiteral
        kBufferAccessAttrName = "bufferization.access";

    /// Attribute name used to mark the bufferization layout for region
    /// arguments during One-Shot Module Bufferize.
    constexpr const static ::llvm::StringLiteral
        kBufferLayoutAttrName = "bufferization.buffer_layout";

    /// Attribute name used to mark escaping behavior of buffer allocations.
    /// Escaping allocations cannot be deallocated in the same block and must
    /// be treated specially: They are currently deallocated with the
    /// BufferDeallocation pass.
    ///
    /// Note: Only ops with at least one OpResult that bufferizes to a buffer
    /// allocation (as per BufferizableOpInterface) may have this attribute.
    constexpr const static ::llvm::StringLiteral
        kEscapeAttrName = "bufferization.escape";
  }];
  let hasOperationAttrVerify = 1;
}

#endif // BUFFERIZATION_BASE
