//===- TypeDetail.h - MLIR Type storage details -----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This holds implementation details of Type.
//
//===----------------------------------------------------------------------===//
#ifndef TYPEDETAIL_H_
#define TYPEDETAIL_H_

#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/TypeRange.h"
#include "llvm/ADT/bit.h"
#include "llvm/Support/TrailingObjects.h"

namespace mlir {

namespace detail {

/// Integer Type Storage and Uniquing.
struct IntegerTypeStorage : public TypeStorage {
  IntegerTypeStorage(unsigned width,
                     IntegerType::SignednessSemantics signedness)
      : width(width), signedness(signedness) {}

  /// The hash key used for uniquing.
  using KeyTy = std::tuple<unsigned, IntegerType::SignednessSemantics>;

  static llvm::hash_code hashKey(const KeyTy &key) {
    return llvm::hash_value(key);
  }

  bool operator==(const KeyTy &key) const {
    return KeyTy(width, signedness) == key;
  }

  static IntegerTypeStorage *construct(TypeStorageAllocator &allocator,
                                       KeyTy key) {
    return new (allocator.allocate<IntegerTypeStorage>())
        IntegerTypeStorage(std::get<0>(key), std::get<1>(key));
  }

  KeyTy getAsKey() const { return KeyTy(width, signedness); }

  unsigned width : 30;
  IntegerType::SignednessSemantics signedness : 2;
};

/// Function Type Storage and Uniquing.
struct FunctionTypeStorage : public TypeStorage {
  FunctionTypeStorage(unsigned numInputs, unsigned numResults,
                      Type const *inputsAndResults)
      : numInputs(numInputs), numResults(numResults),
        inputsAndResults(inputsAndResults) {}

  /// The hash key used for uniquing.
  using KeyTy = std::tuple<TypeRange, TypeRange>;
  bool operator==(const KeyTy &key) const {
    if (std::get<0>(key) == getInputs())
      return std::get<1>(key) == getResults();
    return false;
  }

  /// Construction.
  static FunctionTypeStorage *construct(TypeStorageAllocator &allocator,
                                        const KeyTy &key) {
    auto [inputs, results] = key;

    // Copy the inputs and results into the bump pointer.
    SmallVector<Type, 16> types;
    types.reserve(inputs.size() + results.size());
    types.append(inputs.begin(), inputs.end());
    types.append(results.begin(), results.end());
    auto typesList = allocator.copyInto(ArrayRef<Type>(types));

    // Initialize the memory using placement new.
    return new (allocator.allocate<FunctionTypeStorage>())
        FunctionTypeStorage(inputs.size(), results.size(), typesList.data());
  }

  ArrayRef<Type> getInputs() const {
    return ArrayRef<Type>(inputsAndResults, numInputs);
  }
  ArrayRef<Type> getResults() const {
    return ArrayRef<Type>(inputsAndResults + numInputs, numResults);
  }

  KeyTy getAsKey() const { return KeyTy(getInputs(), getResults()); }

  unsigned numInputs;
  unsigned numResults;
  Type const *inputsAndResults;
};

/// A type representing a collection of other types.
struct TupleTypeStorage final
    : public TypeStorage,
      private llvm::TrailingObjects<TupleTypeStorage, Type> {
  friend llvm::TrailingObjects<TupleTypeStorage, Type>;
  using KeyTy = TypeRange;

  TupleTypeStorage(unsigned numTypes) : numElements(numTypes) {}

  /// Construction.
  static TupleTypeStorage *construct(TypeStorageAllocator &allocator,
                                     TypeRange key) {
    // Allocate a new storage instance.
    auto byteSize = TupleTypeStorage::totalSizeToAlloc<Type>(key.size());
    auto *rawMem = allocator.allocate(byteSize, alignof(TupleTypeStorage));
    auto *result = ::new (rawMem) TupleTypeStorage(key.size());

    // Copy in the element types into the trailing storage.
    llvm::uninitialized_copy(key, result->getTrailingObjects());
    return result;
  }

  bool operator==(const KeyTy &key) const { return key == getTypes(); }

  /// Return the number of held types.
  unsigned size() const { return numElements; }

  /// Return the held types.
  ArrayRef<Type> getTypes() const { return getTrailingObjects(size()); }

  KeyTy getAsKey() const { return getTypes(); }

  /// The number of tuple elements.
  unsigned numElements;
};

/// Checks if the memorySpace has supported Attribute type.
bool isSupportedMemorySpace(Attribute memorySpace);

/// Wraps deprecated integer memory space to the new Attribute form.
Attribute wrapIntegerMemorySpace(unsigned memorySpace, MLIRContext *ctx);

/// Replaces default memorySpace (integer == `0`) with empty Attribute.
Attribute skipDefaultMemorySpace(Attribute memorySpace);

/// [deprecated] Returns the memory space in old raw integer representation.
/// New `Attribute getMemorySpace()` method should be used instead.
unsigned getMemorySpaceAsInt(Attribute memorySpace);

/// Quantile Type Storage and Uniquing.
struct QuantileTypeStorage : public TypeStorage {
  QuantileTypeStorage(Type storageType, Type quantileType,
                      ArrayRef<double> quantiles)
      : storageType(storageType), quantileType(quantileType),
        quantilesData(quantiles.data()), numQuantiles(quantiles.size()) {}

  /// The hash key used for uniquing.
  using KeyTy = std::tuple<Type, Type, ArrayRef<double>>;

  static llvm::hash_code hashKey(const KeyTy &key) {
    auto quantiles = std::get<2>(key);
    // Bit-cast doubles to int64_t for hashing since LLVM hashing
    // does not natively support double.
    auto *quantilesBits = llvm::bit_cast<const int64_t *>(quantiles.data());
    ArrayRef<int64_t> quantilesAsInts(quantilesBits, quantiles.size());
    return llvm::hash_combine(std::get<0>(key), std::get<1>(key),
                              llvm::hash_combine_range(quantilesAsInts.begin(),
                                                       quantilesAsInts.end()));
  }

  bool operator==(const KeyTy &key) const {
    return storageType == std::get<0>(key) &&
           quantileType == std::get<1>(key) &&
           getQuantiles() == std::get<2>(key);
  }

  static QuantileTypeStorage *construct(TypeStorageAllocator &allocator,
                                        const KeyTy &key) {
    ArrayRef<double> quantiles = allocator.copyInto(std::get<2>(key));
    return new (allocator.allocate<QuantileTypeStorage>())
        QuantileTypeStorage(std::get<0>(key), std::get<1>(key), quantiles);
  }

  Type getStorageType() const { return storageType; }
  Type getQuantileType() const { return quantileType; }
  ArrayRef<double> getQuantiles() const {
    return ArrayRef<double>(quantilesData, numQuantiles);
  }

  Type storageType;
  Type quantileType;
  const double *quantilesData;
  unsigned numQuantiles;
};

} // namespace detail
} // namespace mlir

#endif // TYPEDETAIL_H_
