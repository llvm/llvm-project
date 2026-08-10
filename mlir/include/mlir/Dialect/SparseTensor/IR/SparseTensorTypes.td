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
    Values with storage_specifier types represent aggregated storage scheme
    metadata for the given sparse tensor encoding.  It currently holds
    a set of values for level-sizes, coordinate arrays, position arrays,
    and value array.  Note that the type is not yet stable and subject to
    change in the near future.

    Examples:

    ```mlir
    // A storage specifier that can be used to store storage scheme metadata from CSR matrix.
    !storage_specifier<#CSR>
    ```
  }];

  let parameters = (ins SparseTensorEncodingAttr : $encoding);
  let builders = [
    TypeBuilder<(ins "SparseTensorEncodingAttr":$encoding)>,
    TypeBuilderWithInferredContext<(ins "SparseTensorEncodingAttr":$encoding), [{
      return get(encoding.getContext(), encoding);
    }]>,
    TypeBuilderWithInferredContext<(ins "Type":$type), [{
      return get(getSparseTensorEncoding(type));
    }]>,
    TypeBuilderWithInferredContext<(ins "Value":$tensor), [{
      return get(tensor.getType());
    }]>
  ];

  // We skipped the default builder that simply takes the input sparse tensor encoding
  // attribute since we need to normalize the dimension level type and remove unrelated
  // fields that are irrelavant to sparse tensor storage scheme.
  let skipDefaultBuilders = 1;
  let assemblyFormat="`<` qualified($encoding) `>`";
}

def IsSparseTensorStorageSpecifierTypePred
    : CPred<"::llvm::isa<::mlir::sparse_tensor::StorageSpecifierType>($_self)">;

def SparseTensorStorageSpecifier
    : Type<CPred<"::llvm::isa<::mlir::sparse_tensor::StorageSpecifierType>($_self)">, "metadata",
          "::mlir::sparse_tensor::StorageSpecifierType">;

//===----------------------------------------------------------------------===//
// Sparse Tensor Iteration Types.
//===----------------------------------------------------------------------===//

def SparseTensor_IterSpace : SparseTensor_Type<"IterSpace"> {
  let mnemonic = "iter_space";

  let description = [{
    A sparse iteration space that represents an abstract N-D (sparse) iteration space
    extracted from a sparse tensor, i.e., a set of (crd_0, crd_1, ..., crd_N) for
    every stored element (usually nonzeros) in a sparse tensor between the specified
    [$loLvl, $hiLvl) levels.

    Examples:

    ```mlir
    // An iteration space extracted from a CSR tensor between levels [0, 2).
    !iter_space<#CSR, lvls = 0 to 2>
    ```
  }];

  let parameters = (ins
     SparseTensorEncodingAttr : $encoding,
     "Level" : $loLvl,
     "Level" : $hiLvl
  );

  let extraClassDeclaration = [{
     /// The the dimension of the iteration space.
     unsigned getSpaceDim() const {
       return getHiLvl() - getLoLvl();
     }

     /// Get the level types for the iteration space.
     ArrayRef<LevelType> getLvlTypes() const {
       return getEncoding().getLvlTypes().slice(getLoLvl(), getSpaceDim());
     }

     /// Whether the iteration space is unique (i.e., no duplicated coordinate).
     bool isUnique() {
       return !getLvlTypes().back().isa<LevelPropNonDefault::Nonunique>();
     }

     /// Get the corresponding iterator type.
     ::mlir::sparse_tensor::IteratorType getIteratorType() const;
  }];

  let assemblyFormat="`<` $encoding `,` `lvls` `=` custom<LevelRange>($loLvl, $hiLvl) `>`";
}

def SparseTensor_Iterator : SparseTensor_Type<"Iterator"> {
  let mnemonic = "iterator";

  let description = [{
    An iterator that points to the current element in the corresponding iteration space.

    Examples:

    ```mlir
    // An iterator that iterates over a iteration space of type `!iter_space<#CSR, lvls = 0 to 2>`
    !iterator<#CSR, lvls = 0 to 2>
    ```
  }];

  let parameters = (ins
     SparseTensorEncodingAttr : $encoding,
     "Level" : $loLvl,
     "Level" : $hiLvl
  );

  let extraClassDeclaration = [{
     /// Get the corresponding iteration space type.
     ::mlir::sparse_tensor::IterSpaceType getIterSpaceType() const;

     unsigned getSpaceDim() const { return getIterSpaceType().getSpaceDim(); }
     ArrayRef<LevelType> getLvlTypes() const { return getIterSpaceType().getLvlTypes(); }
     bool isUnique() { return getIterSpaceType().isUnique(); }
  }];

  let assemblyFormat="`<` $encoding `,` `lvls` `=` custom<LevelRange>($loLvl, $hiLvl) `>`";
}

def IsSparseSparseIterSpaceTypePred
    : CPred<"::llvm::isa<::mlir::sparse_tensor::IterSpaceType>($_self)">;

def IsSparseSparseIteratorTypePred
    : CPred<"::llvm::isa<::mlir::sparse_tensor::IteratorType>($_self)">;

def AnySparseIterSpace
    : Type<IsSparseSparseIterSpaceTypePred, "sparse iteration space",
          "::mlir::sparse_tensor::IterSpaceType">;

def AnySparseIterator
    : Type<IsSparseSparseIteratorTypePred, "sparse iterator",
          "::mlir::sparse_tensor::IteratorType">;


#endif // SPARSETENSOR_TYPES
