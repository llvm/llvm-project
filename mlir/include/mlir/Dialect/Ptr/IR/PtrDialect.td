//===- PtrDialect.td - Pointer dialect ---------------------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef PTR_DIALECT
#define PTR_DIALECT

include "mlir/Interfaces/DataLayoutInterfaces.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinTypeInterfaces.td"
include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// Pointer dialect definition.
//===----------------------------------------------------------------------===//

def Ptr_Dialect : Dialect {
  let name = "ptr";
  let summary = "Pointer dialect";
  let description = [{
    The pointer dialect provides types and operations for representing and
    interacting with pointer values in MLIR, such as loading and storing values
    from/to memory addresses.

    The dialect's main type is an opaque pointer (`ptr`) that can be
    parameterized by a memory space. This type represents a handle to an object
    in memory, or target-dependent values like `nullptr`. Further, the dialect
    assumes that the minimum addressable unit by a pointer is a byte. However,
    the dialect does not make assumptions about the size of a byte, which is
    considered a target-specific property.
  }];
  let cppNamespace = "::mlir::ptr";
  let useDefaultTypePrinterParser = 1;
  let useDefaultAttributePrinterParser = 1;
}

//===----------------------------------------------------------------------===//
// Pointer type definitions
//===----------------------------------------------------------------------===//

class Ptr_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<Ptr_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

def Ptr_PtrType : Ptr_Type<"Ptr", "ptr", [
    MemRefElementTypeInterface,
    PtrLikeTypeInterface,
    VectorElementTypeInterface,
    DeclareTypeInterfaceMethods<DataLayoutTypeInterface, [
      "areCompatible", "getIndexBitwidth", "verifyEntries",
      "getPreferredAlignment"]>
  ]> {
  let summary = "pointer type";
  let description = [{
    The `ptr` type is an opaque pointer type. This type typically represents a
    handle to an object in memory or target-dependent values like `nullptr`.
    Pointers are parameterized by a memory space.

    Syntax:

    ```mlir
    pointer ::= `ptr` (`<` memory-space `>`)?
    memory-space ::= attribute-value
    ```
  }];
  let parameters = (ins "MemorySpaceAttrInterface":$memorySpace);
  let assemblyFormat = "`<` $memorySpace `>`";
  let builders = [
    TypeBuilderWithInferredContext<(ins
      "MemorySpaceAttrInterface":$memorySpace), [{
      return $_get(memorySpace.getContext(), memorySpace);
    }]>
  ];
  let extraClassDeclaration = [{
    // `PtrLikeTypeInterface` interface methods.
    /// Returns `Type()` as this pointer type is opaque.
    Type getElementType() const {
      return Type();
    }
    /// Clones the pointer with specified memory space or returns failure
    /// if an `elementType` was specified or if the memory space doesn't
    /// implement `MemorySpaceAttrInterface`.
    FailureOr<PtrLikeTypeInterface> clonePtrWith(Attribute memorySpace,
      std::optional<Type> elementType) const {
      if (elementType)
        return failure();
      if (auto ms = dyn_cast<MemorySpaceAttrInterface>(memorySpace))
        return cast<PtrLikeTypeInterface>(get(ms));
      return failure();
    }
    /// `!ptr.ptr` types are seen as ptr-like objects with no metadata.
    bool hasPtrMetadata() const {
      return false;
    }
  }];
}

def Ptr_PtrMetadata : Ptr_Type<"PtrMetadata", "ptr_metadata"> {
  let summary = "Pointer metadata type";
  let description = [{
    The `ptr_metadata` type represents an opaque-view of the metadata associated
    with a `ptr-like` object type.

    Note: It's a verification error to construct a `ptr_metadata` type using a
    `ptr-like` type with no metadata.

    Example:

    ```mlir
    // The metadata associated with a `memref` type.
    !ptr.ptr_metadata<memref<f32>>
    ```
  }];
  let parameters = (ins "PtrLikeTypeInterface":$type);
  let assemblyFormat = "`<` $type `>`";
  let builders = [
    TypeBuilderWithInferredContext<(ins
      "PtrLikeTypeInterface":$ptrLike), [{
      return $_get(ptrLike.getContext(), ptrLike);
    }]>
  ];
  let genVerifyDecl = 1;
}

//===----------------------------------------------------------------------===//
// Base address operation definition.
//===----------------------------------------------------------------------===//

class Pointer_Op<string mnemonic, list<Trait> traits = []> :
        Op<Ptr_Dialect, mnemonic, traits>;

#endif // PTR_DIALECT
