//===- MemOpInterfaces.td - Memory operation interfaces -----*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains interfaces for operations that interact with memory.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_INTERFACES_MEMOPINTERFACES_TD
#define MLIR_INTERFACES_MEMOPINTERFACES_TD

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def MemorySpaceCastConsumerOpInterface :
    OpInterface<"MemorySpaceCastConsumerOpInterface"> {
  let description = [{
    An interface for operations that can consume memory-space cast-like
    operations.
  }];
  let cppNamespace = "::mlir";
  let methods = [
    InterfaceMethod<[{
        Attempt to bubble-down the incoming cast-like operands. On success
        returns any new results, and whether the operation was modified in
        place, otherwise it returns failure.
        If new results are produced, these must be compatible with the original
        operation results.

        If the operation was not modified in place, then the interface
        guarantees it is valid to erase the original operation.
        If the operation was modified in place, then the interface must
        guarantee no operations were created by the method, and that no further
        IR modification is necessary.

        Any implementations of this method must not erase/replace the original
        operation, instead it is the caller responsibility to erase or replace
        the op with the results provided by the method.

        Finally, any implementations of this method have to guarantee that the
        IR remains valid at all times.
      }],
      "::llvm::FailureOr<std::pair<::llvm::SmallVector<::mlir::Value>, bool>>",
      "bubbleDownCasts",
      (ins "::mlir::OpBuilder &":$builder)
    >,
  ];
}

def MemorySpaceCastOpInterface : OpInterface<"MemorySpaceCastOpInterface"> {
  let description = [{
    An interface for operations that perform memory-space casts. This
    interface assumes that the cast operation is `pure`.

    These operations expect to have a well-defined ptr-like operand, and
    a well-defined target ptr-like result.
  }];
  let cppNamespace = "::mlir";
  let methods = [
    InterfaceMethod<[{
        Returns the source ptr-like value.
      }],
      "::mlir::TypedValue<::mlir::PtrLikeTypeInterface>",  "getSourcePtr"
    >,
    InterfaceMethod<[{
        Returns the target ptr-like value.
      }],
      "::mlir::TypedValue<::mlir::PtrLikeTypeInterface>", "getTargetPtr"
    >,
    InterfaceMethod<[{
        Returns whether the memory space cast specified by `tgt` and `src`
        is supported.
      }],
      "bool", "isValidMemorySpaceCast",
      (ins "::mlir::PtrLikeTypeInterface":$tgt,
           "::mlir::PtrLikeTypeInterface":$src)
    >,
    InterfaceMethod<[{
        Clones the memory space cast op with the given source and target type.
      }],
      "::mlir::MemorySpaceCastOpInterface", "cloneMemorySpaceCastOp",
      (ins "::mlir::OpBuilder &":$builder, "::mlir::PtrLikeTypeInterface":$tgt,
           "::mlir::TypedValue<::mlir::PtrLikeTypeInterface>":$src)
    >,
    InterfaceMethod<[{
        Returns whether the source pointer of the memory-space cast can be used
        by the `MemorySpaceCastConsumerOpInterface::bubbleDownCasts` method to
        promote the source pointer and bubble down the cast.
      }],
      "bool", "isSourcePromotable"
    >
  ];
  let verify = [{
    return ::mlir::detail::verifyMemorySpaceCastOpInterface($_op);
  }];
  let dependentTraits = [Pure];
  let extraClassDeclaration = [{
    /// Returns the underlying `MemorySpaceCastOpInterface` op if `value`
    /// is produced by a `MemorySpaceCastOpInterface` op, and
    /// `isSourcePromotable` returns true, otherwise it returns null.
    static ::mlir::MemorySpaceCastOpInterface
    getIfPromotableCast(::mlir::Value value) {
      auto op = ::llvm::dyn_cast_or_null<::mlir::MemorySpaceCastOpInterface>(
        value.getDefiningOp());
      if (!op || !op.isSourcePromotable())
        return nullptr;
      return op;
    }
  }];
}

#endif // MLIR_INTERFACES_MEMOPINTERFACES_TD
