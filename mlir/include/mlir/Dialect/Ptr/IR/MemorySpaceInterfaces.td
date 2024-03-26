//===-- MemorySpaceInterfaces.td - Memory space interfaces ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines memory space attribute interfaces.
//
//===----------------------------------------------------------------------===//

#ifndef PTR_MEMORYSPACEINTERFACES
#define PTR_MEMORYSPACEINTERFACES

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// Memory space attribute interface.
//===----------------------------------------------------------------------===//

def MemorySpaceAttrInterface : AttrInterface<"MemorySpaceAttrInterface"> {
  let description = [{
    This interface defines a common API for interacting with the memory model of
    a memory space and the operations in the pointer dialect, giving proper
    semantical meaning to the ops.

    Furthermore, this interface allows concepts such as read-only memory to be
    adequately modeled and enforced.
  }];
  let cppNamespace = "::mlir::ptr";
  let methods = [
    InterfaceMethod<
      /*desc=*/        [{
        Returns the dialect implementing the memory space.
      }],
      /*returnType=*/  "::mlir::Dialect*",
      /*methodName=*/  "getMemorySpaceDialect",
      /*args=*/        (ins),
      /*methodBody=*/  [{}],
      /*defaultImpl=*/ [{ return nullptr; }]
    >,
    InterfaceMethod<
      /*desc=*/        [{
        Returns the default memory space as an attribute.
      }],
      /*returnType=*/  "::mlir::Attribute",
      /*methodName=*/  "getDefaultMemorySpace",
      /*args=*/        (ins),
      /*methodBody=*/  [{}],
      /*defaultImpl=*/ [{}]
    >,
    InterfaceMethod<
      /*desc=*/        [{
        Returns the memory space as an integer, or 0 if using the default model.
      }],
      /*returnType=*/  "unsigned",
      /*methodName=*/  "getAddressSpace",
      /*args=*/        (ins),
      /*methodBody=*/  [{}],
      /*defaultImpl=*/ [{ return 0; }]
    >,
    InterfaceMethod<
      /*desc=*/        [{
        This method checks if it's valid to load a value from the memory space
        with a specific type, alignment, and atomic ordering.
        If `diagnosticOp` is non-null then the method might emit diagnostics.
      }],
      /*returnType=*/  "::mlir::LogicalResult",
      /*methodName=*/  "isValidLoad",
      /*args=*/        (ins "::mlir::Type":$type,
                            "::mlir::ptr::AtomicOrdering":$ordering,
                            "::mlir::IntegerAttr":$alignment,
                            "::mlir::Operation*":$diagnosticOp),
      /*methodBody=*/  [{}],
      /*defaultImpl=*/ [{}]
    >,
    InterfaceMethod<
      /*desc=*/        [{
        This method checks if it's valid to store a value in the memory space
        with a specific type, alignment, and atomic ordering.
        If `diagnosticOp` is non-null then the method might emit diagnostics.
      }],
      /*returnType=*/  "::mlir::LogicalResult",
      /*methodName=*/  "isValidStore",
      /*args=*/        (ins "::mlir::Type":$type,
                            "::mlir::ptr::AtomicOrdering":$ordering,
                            "::mlir::IntegerAttr":$alignment,
                            "::mlir::Operation*":$diagnosticOp),
      /*methodBody=*/  [{}],
      /*defaultImpl=*/ [{}]
    >,
    InterfaceMethod<
      /*desc=*/        [{
        This method checks if it's valid to perform an atomic operation in the
        memory space with a specific type, alignment, and atomic ordering.
        If `diagnosticOp` is non-null then the method might emit diagnostics.
      }],
      /*returnType=*/  "::mlir::LogicalResult",
      /*methodName=*/  "isValidAtomicOp",
      /*args=*/        (ins "::mlir::ptr::AtomicBinOp":$op,
                            "::mlir::Type":$type,
                            "::mlir::ptr::AtomicOrdering":$ordering,
                            "::mlir::IntegerAttr":$alignment,
                            "::mlir::Operation*":$diagnosticOp),
      /*methodBody=*/  [{}],
      /*defaultImpl=*/ [{}]
    >,
    InterfaceMethod<
      /*desc=*/        [{
        This method checks if it's valid to perform an atomic exchange operation
        in the memory space with a specific type, alignment, and atomic
        orderings.
        If `diagnosticOp` is non-null then the method might emit diagnostics.
      }],
      /*returnType=*/  "::mlir::LogicalResult",
      /*methodName=*/  "isValidAtomicXchg",
      /*args=*/        (ins "::mlir::Type":$type,
                            "::mlir::ptr::AtomicOrdering":$successOrdering,
                            "::mlir::ptr::AtomicOrdering":$failureOrdering,
                            "::mlir::IntegerAttr":$alignment,
                            "::mlir::Operation*":$diagnosticOp),
      /*methodBody=*/  [{}],
      /*defaultImpl=*/ [{}]
    >,
    InterfaceMethod<
      /*desc=*/        [{
        This method checks if it's valid to perform an `addrspacecast` op
        in the memory space.
        Both types are expected to be vectors of rank 1, or scalars of `ptr`
        type.
        If `diagnosticOp` is non-null then the method might emit diagnostics.
      }],
      /*returnType=*/  "::mlir::LogicalResult",
      /*methodName=*/  "isValidAddrSpaceCast",
      /*args=*/        (ins "::mlir::Type":$tgt,
                            "::mlir::Type":$src,
                            "::mlir::Operation*":$diagnosticOp),
      /*methodBody=*/  [{}],
      /*defaultImpl=*/ [{}]
    >,
    InterfaceMethod<
      /*desc=*/        [{
        This method checks if it's valid to perform a `ptrtoint` or `inttoptr`
        op in the memory space. `CastValidity::InvalidSourceType` always refers
        to the 'ptr-like' type and `CastValidity::InvalidTargetType` always
        refers to the `int-like` type.
        The first type is expected to be integer-like, while the second must be a
        ptr-like type.
        If `diagnosticOp` is non-null then the method might emit diagnostics.
      }],
      /*returnType=*/  "::mlir::LogicalResult",
      /*methodName=*/  "isValidPtrIntCast",
      /*args=*/        (ins "::mlir::Type":$intLikeTy,
                            "::mlir::Type":$ptrLikeTy,
                            "::mlir::Operation*":$diagnosticOp),
      /*methodBody=*/  [{}],
      /*defaultImpl=*/ [{}]
    >,
  ];
}

def MemoryModelOpInterface : OpInterface<"MemoryModelOpInterface"> {
  let description = [{
    An interface for operations with a memory model.
  }];

  let cppNamespace = "::mlir::ptr";

  let methods = [
    InterfaceMethod<
      /*desc=*/        "Returns the memory model of the op.",
      /*returnType=*/  "MemoryModel",
      /*methodName=*/  "getMemoryModel",
      /*args=*/        (ins),
      /*methodBody=*/  [{}],
      /*defaultImpl=*/ [{}]
      >,
  ];
}
#endif // PTR_MEMORYSPACEINTERFACES
