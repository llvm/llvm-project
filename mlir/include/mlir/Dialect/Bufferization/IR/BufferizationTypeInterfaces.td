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

    `isTensorLike` refines whether a particular instance of the type is
    tensor-like; `isa<TensorLikeType>` requires both a registered model and
    this predicate.
  }];

  let methods = [
    InterfaceMethod<[{
        Returns whether a BufferLike type is compatible to this TensorLike type.
        The BufferLike type is assumed to be created by unknown type converter.
      }],
      /*retTy=*/"::mlir::LogicalResult",
      /*methodName=*/"verifyCompatibleBufferType",
      /*args=*/(ins
        "::mlir::bufferization::BufferLikeType":$bufferType,
        "::llvm::function_ref<::mlir::InFlightDiagnostic()>":$emitError)
    >,
    InterfaceMethod<[{
        Returns whether this instance of the type is a tensor type for
        bufferization purposes. A type overrides this when only some of its
        instances are, for example when that depends on its parameters.
      }],
      /*retTy=*/"bool",
      /*methodName=*/"isTensorLike",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/"return true;"
    >
  ];

  let extraClassOf = [{
    return $_type.isTensorLike();
  }];
}

def Bufferization_BufferLikeTypeInterface
    : TypeInterface<"BufferLikeType"> {
  let cppNamespace = "::mlir::bufferization";
  let description = [{
    Indicates that this type is a buffer type (similarly to a MLIR builtin
    memref) for bufferization purposes.
  }];

  let methods = [
    InterfaceMethod<[{
        Returns whether this instance of the type says how its values are
        deallocated and identified, by `genFree` and `genIdentity`, which may
        be called only where this holds.
      }],
      /*retTy=*/"bool",
      /*methodName=*/"supportsDeallocation",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/"return false;"
    >,
    InterfaceMethod<[{
        Builds the deallocation of `buffer`, a value of this type that a
        `bufferization.dealloc` deallocates. Which values of the type may be
        deallocated is the type's to say; for a builtin memref it is the
        originally allocated memref, as for `memref.dealloc`.
      }],
      /*retTy=*/"void",
      /*methodName=*/"genFree",
      /*args=*/(ins "::mlir::OpBuilder &":$builder, "::mlir::Location":$loc,
                    "::mlir::Value":$buffer),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        llvm_unreachable("genFree not implemented");
      }]
    >,
    InterfaceMethod<[{
        Builds an `index` that the lowering of `bufferization.dealloc`
        compares at runtime to decide whether two deallocated values, or a
        deallocated and a retained value, belong to the same allocation:
        equal identities are taken to be the same allocation, different ones
        different allocations. Identities are compared across buffer types. A
        builtin memref's identity is its aligned pointer, so two views of one
        allocation whose aligned pointers differ have different identities.
      }],
      /*retTy=*/"::mlir::Value",
      /*methodName=*/"genIdentity",
      /*args=*/(ins "::mlir::OpBuilder &":$builder, "::mlir::Location":$loc,
                    "::mlir::Value":$buffer),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        llvm_unreachable("genIdentity not implemented");
      }]
    >
  ];
}

#endif // BUFFERIZATION_TYPE_INTERFACES
