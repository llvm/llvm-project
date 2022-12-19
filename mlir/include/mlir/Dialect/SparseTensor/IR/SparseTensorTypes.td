//===- SparseTensorTypes.td - Sparse tensor dialect types ------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef SPARSETENSOR_TYPES
#define SPARSETENSOR_TYPES

include "mlir/Dialect/SparseTensor/IR/SparseTensorAttrDefs.td"
include "mlir/Dialect/SparseTensor/IR/SparseTensorBase.td"

//===----------------------------------------------------------------------===//
// Base class.
//===----------------------------------------------------------------------===//

// Base class for Builtin dialect types.
class SparseTensor_Type<string name, list<Trait> traits = [],
                   string baseCppClass = "::mlir::Type">
    : TypeDef<SparseTensor_Dialect, name, traits, baseCppClass> {}

//===----------------------------------------------------------------------===//
// Sparse Tensor Dialect Types.
//===----------------------------------------------------------------------===//

def SparseTensor_StorageSpecifier : SparseTensor_Type<"StorageSpecifier"> {
  let mnemonic = "storage_specifier";
  let summary = "Structured metadata for sparse tensor low-level storage scheme";

  let description = [{
    Syntax:

    ```
    storage_specifier-type ::= `!storage_specifier` `<` encoding `>`
    encoding ::= attribute-value
    ```

    Values with storage_specifier types represent aggregated storage scheme metadata
    for the given sparse tensor encoding.
    It currently holds a set of values for sizes of sparse tensor dimension, index array,
    pointer array and value array.
    Note that the type is not yet stable and subject to change in the near future.

    Examples:

    ```mlir
    // A storage specifier that can be used to store storage scheme metadata from CSR matrix.
    !storage_specifier<#CSR>
    ```
  }];

  let parameters = (ins SparseTensorEncodingAttr : $encoding);
  let builders = [
    TypeBuilderWithInferredContext<(ins "SparseTensorEncodingAttr":$encoding), [{
      assert(encoding && "sparse tensor encoding should not be null");
      return $_get(encoding.getContext(), encoding);
    }]>,
    TypeBuilderWithInferredContext<(ins "Type":$type), [{
      return get(getSparseTensorEncoding(type));
    }]>,
    TypeBuilderWithInferredContext<(ins "Value":$tensor), [{
      return get(tensor.getType());
    }]>
  ];

  let extraClassDeclaration = [{
    // Get the integer type used to store memory and dimension sizes.
    IntegerType getSizesType() const;
    Type getFieldType(StorageSpecifierKind kind, std::optional<unsigned> dim) const;
    Type getFieldType(StorageSpecifierKind kind, std::optional<APInt> dim) const;
  }];

  let assemblyFormat="`<` qualified($encoding) `>`";
}

def IsSparseTensorStorageSpecifierTypePred
    : CPred<"$_self.isa<::mlir::sparse_tensor::StorageSpecifierType>()">;

def SparseTensorStorageSpecifier
    : Type<CPred<"$_self.isa<::mlir::sparse_tensor::StorageSpecifierType>()">, "metadata",
          "::mlir::sparse_tensor::StorageSpecifierType">;

#endif // SPARSETENSOR_TYPES
