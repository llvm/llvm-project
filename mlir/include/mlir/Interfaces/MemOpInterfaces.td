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

def FuseMemorySpaceCastConsumerOpInterface :
    OpInterface<"FuseMemorySpaceCastConsumerOpInterface"> {
  let description = [{
    An interface to fuse memory-space cast operands into a consumer operation.
    It is the responsibility of the interface to determine which casts can be
    fused into the operation.
  }];
  let cppNamespace = "::mlir";
  let methods = [
    InterfaceMethod<[{
        Attempt to fuse the incoming cast-like operands. Returns `success`
        and any new results on fusion success, otherwise it returns failure.
        If new results are produced, these must be compatible with the original
        operation results.

        The `modifiedInPlace` parameter indicates whether the operation was
        modified in place. If `false` and the fusion succeeded, then the
        interface guarantees it is valid to erase the original operation.
        If `true`, then the interface must guarantee no operations were created
        by the method, and that no further IR modification is necessary. It is
        considered an error if `modifiedInPlace` is true and the fusion failed.

        Any implementations of this method must not erase/replace the original
        operation, instead it is the caller responsibility to erase or replace
        the op with the results provided by the method.

        Finally, any implementations of this method have to guarantee that the
        IR remains valid at all times.
      }],
      "::llvm::FailureOr<::llvm::SmallVector<::mlir::Value>>", "fuseCastOperands",
      (ins "::mlir::OpBuilder &":$builder, "bool &":$modifiedInPlace)
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
      (ins "::mlir::OpBuilder &":$builder, "::mlir::Type":$tgt,
           "::mlir::Value":$src)
    >,
    InterfaceMethod<[{
        Returns whether the cast allows to be fused.
      }],
      "bool", "isFusableMemorySpaceCast"
    >
  ];
  let verify = [{
    return ::mlir::detail::verifyMemorySpaceCastOpInterface($_op);
  }];
  let dependentTraits = [Pure];
  let extraClassDeclaration = [{
    /// Returns the underlying `MemorySpaceCastOpInterface` op if `value`
    /// is produced by a `MemorySpaceCastOpInterface` op, and
    /// `isFusableMemorySpaceCast` returns true, otherwise it returns null.
    static ::mlir::MemorySpaceCastOpInterface
    getIfFusableCast(::mlir::Value value) {
      auto op = ::llvm::dyn_cast_or_null<::mlir::MemorySpaceCastOpInterface>(
        value.getDefiningOp());
      if (!op || !op.isFusableMemorySpaceCast())
        return nullptr;
      return op;
    }
  }];
}

#endif // MLIR_INTERFACES_MEMOPINTERFACES_TD
