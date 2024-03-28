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

def LevelTypeArrayParameter : ArrayRefParameter<"::mlir::sparse_tensor::LevelType", "level-types"> {
  let printer = [{
    auto lvlStrings = llvm::map_range($_self, [](auto lt){ return lt.toMLIRString(); });
    $_printer << "[" << llvm::join(lvlStrings, ",") << "]";
  }];

  let parser = [{ [&]() -> FailureOr<SmallVector<::mlir::sparse_tensor::LevelType>> {
    SmallVector<::mlir::sparse_tensor::LevelType> ret;

    const auto res = $_parser.parseCommaSeparatedList(
      mlir::OpAsmParser::Delimiter::Square,
      [&]() -> ParseResult {
        ::mlir::sparse_tensor::ir_detail::LvlTypeParser lParser;
        auto lvlTpOrFail = lParser.parseLvlType($_parser);
        if (failed(lvlTpOrFail))
          return failure();
        ret.emplace_back(*lvlTpOrFail);
        return success();
      }, " in level-type list");

    if (failed(res))
      return failure();
    return ret;
  }() }];
}

def SparseTensor_Iterator : SparseTensor_Type<"Iterator"> {
  let mnemonic = "iterator";

  let parameters = (ins
     LevelTypeArrayParameter: $lvlTypes
  );

  let extraClassDeclaration = [{
     unsigned getSpaceDim() {
       return getLvlTypes().size();
     }
  }];


  let assemblyFormat="`<` $lvlTypes `>`";
}


def SparseTensor_IterSpace : SparseTensor_Type<"IterSpace"> {
  let mnemonic = "iter_space";

  let parameters = (ins
     LevelTypeArrayParameter: $lvlTypes
  );

  let extraClassDeclaration = [{
     unsigned getSpaceDim() {
       return getLvlTypes().size();
     }

     ::mlir::sparse_tensor::IteratorType getIteratorType() const {
        return IteratorType::get(getContext(), getLvlTypes());
     }
  }];

  let assemblyFormat="`<` $lvlTypes `>`";
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
