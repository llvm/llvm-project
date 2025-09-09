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
    a memory space and the operations in the pointer dialect.

    Furthermore, this interface allows concepts such as read-only memory to be
    adequately modeled and enforced.
  }];
  let cppNamespace = "::mlir::ptr";
  let methods = [
    InterfaceMethod<
      /*desc=*/        [{
        This method checks if it's valid to load a value from the memory space
        with a specific type, alignment, and atomic ordering.
        If `emitError` is non-null then the method is allowed to emit errors.
        Furthermore, if `emitError` is non-null and the result is `false` an
        error must have been emitted.
      }],
      /*returnType=*/  "bool",
      /*methodName=*/  "isValidLoad",
      /*args=*/        (ins "::mlir::Type":$type,
                            "::mlir::ptr::AtomicOrdering":$ordering,
                            "std::optional<int64_t>":$alignment,
                            "::llvm::function_ref<::mlir::InFlightDiagnostic()>":$emitError)
    >,
    InterfaceMethod<
      /*desc=*/        [{
        This method checks if it's valid to store a value in the memory space
        with a specific type, alignment, and atomic ordering.
        If `emitError` is non-null then the method is allowed to emit errors.
        Furthermore, if `emitError` is non-null and the result is `false` an
        error must have been emitted.
      }],
      /*returnType=*/  "bool",
      /*methodName=*/  "isValidStore",
      /*args=*/        (ins "::mlir::Type":$type,
                            "::mlir::ptr::AtomicOrdering":$ordering,
                            "std::optional<int64_t>":$alignment,
                            "::llvm::function_ref<::mlir::InFlightDiagnostic()>":$emitError)
    >,
    InterfaceMethod<
      /*desc=*/        [{
        This method checks if it's valid to perform an atomic operation in the
        memory space with a specific type, alignment, and atomic ordering.
        If `emitError` is non-null then the method is allowed to emit errors.
        Furthermore, if `emitError` is non-null and the result is `false` an
        error must have been emitted.
      }],
      /*returnType=*/  "bool",
      /*methodName=*/  "isValidAtomicOp",
      /*args=*/        (ins "::mlir::ptr::AtomicBinOp":$op,
                            "::mlir::Type":$type,
                            "::mlir::ptr::AtomicOrdering":$ordering,
                            "std::optional<int64_t>":$alignment,
                            "::llvm::function_ref<::mlir::InFlightDiagnostic()>":$emitError)
    >,
    InterfaceMethod<
      /*desc=*/        [{
        This method checks if it's valid to perform an atomic exchange operation
        in the memory space with a specific type, alignment, and atomic
        orderings.
        If `emitError` is non-null then the method is allowed to emit errors.
        Furthermore, if `emitError` is non-null and the result is `false` an
        error must have been emitted.
      }],
      /*returnType=*/  "bool",
      /*methodName=*/  "isValidAtomicXchg",
      /*args=*/        (ins "::mlir::Type":$type,
                            "::mlir::ptr::AtomicOrdering":$successOrdering,
                            "::mlir::ptr::AtomicOrdering":$failureOrdering,
                            "std::optional<int64_t>":$alignment,
                            "::llvm::function_ref<::mlir::InFlightDiagnostic()>":$emitError)
    >,
    InterfaceMethod<
      /*desc=*/        [{
        This method checks if it's valid to perform an `addrspacecast` op
        in the memory space.
        If `emitError` is non-null then the method is allowed to emit errors.
        Furthermore, if `emitError` is non-null and the result is `false` an
        error must have been emitted.
      }],
      /*returnType=*/  "bool",
      /*methodName=*/  "isValidAddrSpaceCast",
      /*args=*/        (ins "::mlir::Type":$tgt,
                            "::mlir::Type":$src,
                            "::llvm::function_ref<::mlir::InFlightDiagnostic()>":$emitError)
    >,
    InterfaceMethod<
      /*desc=*/        [{
        This method checks if it's valid to perform a `ptrtoint` or `inttoptr`
        op in the memory space.
        The first type is expected to be integer-like, while the second must be
        a ptr-like type.
        If `emitError` is non-null then the method is allowed to emit errors.
        Furthermore, if `emitError` is non-null and the result is `false` an
        error must have been emitted.
      }],
      /*returnType=*/  "bool",
      /*methodName=*/  "isValidPtrIntCast",
      /*args=*/        (ins "::mlir::Type":$intLikeTy,
                            "::mlir::Type":$ptrLikeTy,
                            "::llvm::function_ref<::mlir::InFlightDiagnostic()>":$emitError)
    >,
  ];
}

#endif // PTR_MEMORYSPACEINTERFACES
