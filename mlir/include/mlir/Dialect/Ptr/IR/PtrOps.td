//===- PtrOps.td - Pointer dialect ops ---------------------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef PTR_OPS
#define PTR_OPS

include "mlir/Dialect/Ptr/IR/PtrDialect.td"
include "mlir/Dialect/Ptr/IR/PtrAttrDefs.td"
include "mlir/Dialect/Ptr/IR/PtrEnums.td"
include "mlir/Dialect/Ptr/IR/MemorySpaceInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/ViewLikeInterface.td"
include "mlir/IR/OpAsmInterface.td"

//===----------------------------------------------------------------------===//
// FromPtrOp
//===----------------------------------------------------------------------===//

def Ptr_FromPtrOp : Pointer_Op<"from_ptr", [
    Pure, OptionalTypesMatchWith<"metadata type", "result", "metadata",
            "PtrMetadataType::get(cast<PtrLikeTypeInterface>($_self))">
  ]> {
  let summary = "Casts a `!ptr.ptr` value to a ptr-like value.";
  let description = [{
    The `from_ptr` operation casts a `ptr` value to a ptr-like object. It's
    important to note that:
    - The ptr-like object cannot be a `!ptr.ptr`.
    - The memory-space of both the `ptr` and ptr-like object must match.
    - The cast is Pure (no UB and side-effect free).

    The optional `metadata` operand exists to provide any ptr-like metadata
    that might be required to perform the cast.

    Example:

    ```mlir
    %typed_ptr = ptr.from_ptr %ptr : !ptr.ptr<#ptr.generic_space> -> !my.ptr<f32, #ptr.generic_space>
    %memref = ptr.from_ptr %ptr metadata %md : !ptr.ptr<#ptr.generic_space> -> memref<f32, #ptr.generic_space>
  
    // Cast the `%ptr` to a memref without utilizing metadata.
    %memref = ptr.from_ptr %ptr : !ptr.ptr<#ptr.generic_space> -> memref<f32, #ptr.generic_space>
    ```
  }];

  let arguments = (ins Ptr_PtrType:$ptr, Optional<Ptr_PtrMetadata>:$metadata);
  let results = (outs PtrLikeTypeInterface:$result);
  let assemblyFormat = [{
    $ptr (`metadata` $metadata^)? attr-dict `:` type($ptr) `->` type($result)
  }];
  let hasFolder = 1;
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// GetMetadataOp
//===----------------------------------------------------------------------===//

def Ptr_GetMetadataOp : Pointer_Op<"get_metadata", [
    Pure, TypesMatchWith<"metadata type", "ptr", "result",
            "PtrMetadataType::get(cast<PtrLikeTypeInterface>($_self))">
  ]> {
  let summary = "SSA value representing pointer metadata.";
  let description = [{
    The `get_metadata` operation produces an opaque value that encodes the
    metadata of the ptr-like type.

    Example:

    ```mlir
    %metadata = ptr.get_metadata %memref : memref<?x?xf32>
    ```
  }];

  let arguments = (ins PtrLikeTypeInterface:$ptr);
  let results = (outs Ptr_PtrMetadata:$result);
  let assemblyFormat = [{
    $ptr attr-dict `:` type($ptr)
  }];
}

//===----------------------------------------------------------------------===//
// PtrAddOp
//===----------------------------------------------------------------------===//

def Ptr_PtrAddOp : Pointer_Op<"ptr_add", [
    Pure, AllTypesMatch<["base", "result"]>, ViewLikeOpInterface
  ]> {
  let summary = "Pointer add operation";
  let description = [{
    The `ptr_add` operation adds an integer offset to a pointer to produce a new
    pointer. The input and output pointer types are always the same.

    Example:

    ```mlir
    %x_off  = ptr.ptr_add %x, %off : !ptr.ptr<#ptr.generic_space>, i32
    %x_off0 = ptr.ptr_add nusw %x, %off : !ptr.ptr<#ptr.generic_space>, i32
    ```
  }];

  let arguments = (ins
    Ptr_PtrType:$base,
    AnySignlessIntegerOrIndex:$offset,
    DefaultValuedProp<EnumProp<Ptr_PtrAddFlags>, "PtrAddFlags::none">:$flags);
  let results = (outs Ptr_PtrType:$result);
  let assemblyFormat = [{
    ($flags^)? $base `,` $offset attr-dict `:` type($base) `,` type($offset)
  }];
  let hasFolder = 1;
  let extraClassDeclaration = [{
    /// `ViewLikeOp::getViewSource` method. 
    Value getViewSource() { return getBase(); }
  }];
}

//===----------------------------------------------------------------------===//
// ToPtrOp
//===----------------------------------------------------------------------===//

def Ptr_ToPtrOp : Pointer_Op<"to_ptr", [Pure]> {
  let summary = "Casts a ptr-like value to a `!ptr.ptr` value.";
  let description = [{
    The `to_ptr` operation casts a ptr-like object to a `!ptr.ptr`. It's
    important to note that:
    - The ptr-like object cannot be a `!ptr.ptr`.
    - The memory-space of both the `ptr` and ptr-like object must match.
    - The cast is side-effect free.

    Example:

    ```mlir
    %ptr0 = ptr.to_ptr %my_ptr : !my.ptr<f32, #ptr.generic_space> -> !ptr.ptr<#ptr.generic_space>
    %ptr1 = ptr.to_ptr %memref : memref<f32, #ptr.generic_space> -> !ptr.ptr<#ptr.generic_space>
    ```
  }];

  let arguments = (ins PtrLikeTypeInterface:$ptr);
  let results = (outs Ptr_PtrType:$result);
  let assemblyFormat = [{
    $ptr attr-dict `:` type($ptr) `->` type($result)
  }];
  let hasFolder = 1;
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// TypeOffsetOp
//===----------------------------------------------------------------------===//

def Ptr_TypeOffsetOp : Pointer_Op<"type_offset", [Pure]> {
  let summary = "Type offset operation";
  let description = [{
    The `type_offset` operation produces an int or index-typed SSA value
    equal to a target-specific constant representing the offset of a single
    element of the given type.

    Example:

    ```mlir
    // Return the offset between two f32 stored in memory
    %0 = ptr.type_offset f32 : index
    // Return the offset between two memref descriptors stored in memory
    %1 = ptr.type_offset memref<12 x f64> : i32
    ```
  }];

  let arguments = (ins TypeAttr:$elementType);
  let results = (outs AnySignlessIntegerOrIndex:$result);
  let assemblyFormat = [{
    $elementType attr-dict `:` type($result)
  }];
  let extraClassDeclaration = [{
    /// Returns the type offset according to `layout`. If `layout` is `nullopt`
    /// the nearest layout the op will be used for the computation.
    llvm::TypeSize getTypeSize(std::optional<DataLayout> layout = std::nullopt);
  }];
}

#endif // PTR_OPS
