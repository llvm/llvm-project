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
#include "mlir/IR/Identifier.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/StandardTypes.h"
#include "llvm/ADT/bit.h"
#include "llvm/Support/TrailingObjects.h"

namespace mlir {

class MLIRContext;

namespace detail {

/// Opaque Type Storage and Uniquing.
struct OpaqueTypeStorage : public TypeStorage {
  OpaqueTypeStorage(Identifier dialectNamespace, StringRef typeData)
      : dialectNamespace(dialectNamespace), typeData(typeData) {}

  /// The hash key used for uniquing.
  using KeyTy = std::pair<Identifier, StringRef>;
  bool operator==(const KeyTy &key) const {
    return key == KeyTy(dialectNamespace, typeData);
  }

  static OpaqueTypeStorage *construct(TypeStorageAllocator &allocator,
                                      const KeyTy &key) {
    StringRef tyData = allocator.copyInto(key.second);
    return new (allocator.allocate<OpaqueTypeStorage>())
        OpaqueTypeStorage(key.first, tyData);
  }

  // The dialect namespace.
  Identifier dialectNamespace;

  // The parser type data for this opaque type.
  StringRef typeData;
};

/// Integer Type Storage and Uniquing.
struct IntegerTypeStorage : public TypeStorage {
  IntegerTypeStorage(unsigned width,
                     IntegerType::SignednessSemantics signedness)
      : TypeStorage(packKeyBits(width, signedness)) {}

  /// The hash key used for uniquing.
  using KeyTy = std::pair<unsigned, IntegerType::SignednessSemantics>;

  static llvm::hash_code hashKey(const KeyTy &key) {
    return llvm::hash_value(packKeyBits(key.first, key.second));
  }

  bool operator==(const KeyTy &key) const {
    return getSubclassData() == packKeyBits(key.first, key.second);
  }

  static IntegerTypeStorage *construct(TypeStorageAllocator &allocator,
                                       KeyTy key) {
    return new (allocator.allocate<IntegerTypeStorage>())
        IntegerTypeStorage(key.first, key.second);
  }

  struct KeyBits {
    unsigned width : 30;
    unsigned signedness : 2;
  };

  /// Pack the given `width` and `signedness` as a key.
  static unsigned packKeyBits(unsigned width,
                              IntegerType::SignednessSemantics signedness) {
    KeyBits bits{width, static_cast<unsigned>(signedness)};
    return llvm::bit_cast<unsigned>(bits);
  }

  static KeyBits unpackKeyBits(unsigned bits) {
    return llvm::bit_cast<KeyBits>(bits);
  }

  unsigned getWidth() { return unpackKeyBits(getSubclassData()).width; }

  IntegerType::SignednessSemantics getSignedness() {
    return static_cast<IntegerType::SignednessSemantics>(
        unpackKeyBits(getSubclassData()).signedness);
  }
};

/// Function Type Storage and Uniquing.
struct FunctionTypeStorage : public TypeStorage {
  FunctionTypeStorage(unsigned numInputs, unsigned numResults,
                      Type const *inputsAndResults)
      : TypeStorage(numInputs), numResults(numResults),
        inputsAndResults(inputsAndResults) {}

  /// The hash key used for uniquing.
  using KeyTy = std::pair<ArrayRef<Type>, ArrayRef<Type>>;
  bool operator==(const KeyTy &key) const {
    return key == KeyTy(getInputs(), getResults());
  }

  /// Construction.
  static FunctionTypeStorage *construct(TypeStorageAllocator &allocator,
                                        const KeyTy &key) {
    ArrayRef<Type> inputs = key.first, results = key.second;

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
    return ArrayRef<Type>(inputsAndResults, getSubclassData());
  }
  ArrayRef<Type> getResults() const {
    return ArrayRef<Type>(inputsAndResults + getSubclassData(), numResults);
  }

  unsigned numResults;
  Type const *inputsAndResults;
};

/// Shaped Type Storage.
struct ShapedTypeStorage : public TypeStorage {
  ShapedTypeStorage(Type elementTy, unsigned subclassData = 0)
      : TypeStorage(subclassData), elementType(elementTy) {}

  /// The hash key used for uniquing.
  using KeyTy = Type;
  bool operator==(const KeyTy &key) const { return key == elementType; }

  Type elementType;
};

/// Vector Type Storage and Uniquing.
struct VectorTypeStorage : public ShapedTypeStorage {
  VectorTypeStorage(unsigned shapeSize, Type elementTy,
                    const int64_t *shapeElements)
      : ShapedTypeStorage(elementTy, shapeSize), shapeElements(shapeElements) {}

  /// The hash key used for uniquing.
  using KeyTy = std::pair<ArrayRef<int64_t>, Type>;
  bool operator==(const KeyTy &key) const {
    return key == KeyTy(getShape(), elementType);
  }

  /// Construction.
  static VectorTypeStorage *construct(TypeStorageAllocator &allocator,
                                      const KeyTy &key) {
    // Copy the shape into the bump pointer.
    ArrayRef<int64_t> shape = allocator.copyInto(key.first);

    // Initialize the memory using placement new.
    return new (allocator.allocate<VectorTypeStorage>())
        VectorTypeStorage(shape.size(), key.second, shape.data());
  }

  ArrayRef<int64_t> getShape() const {
    return ArrayRef<int64_t>(shapeElements, getSubclassData());
  }

  const int64_t *shapeElements;
};

struct RankedTensorTypeStorage : public ShapedTypeStorage {
  RankedTensorTypeStorage(unsigned shapeSize, Type elementTy,
                          const int64_t *shapeElements)
      : ShapedTypeStorage(elementTy, shapeSize), shapeElements(shapeElements) {}

  /// The hash key used for uniquing.
  using KeyTy = std::pair<ArrayRef<int64_t>, Type>;
  bool operator==(const KeyTy &key) const {
    return key == KeyTy(getShape(), elementType);
  }

  /// Construction.
  static RankedTensorTypeStorage *construct(TypeStorageAllocator &allocator,
                                            const KeyTy &key) {
    // Copy the shape into the bump pointer.
    ArrayRef<int64_t> shape = allocator.copyInto(key.first);

    // Initialize the memory using placement new.
    return new (allocator.allocate<RankedTensorTypeStorage>())
        RankedTensorTypeStorage(shape.size(), key.second, shape.data());
  }

  ArrayRef<int64_t> getShape() const {
    return ArrayRef<int64_t>(shapeElements, getSubclassData());
  }

  const int64_t *shapeElements;
};

struct UnrankedTensorTypeStorage : public ShapedTypeStorage {
  using ShapedTypeStorage::KeyTy;
  using ShapedTypeStorage::ShapedTypeStorage;

  /// Construction.
  static UnrankedTensorTypeStorage *construct(TypeStorageAllocator &allocator,
                                              Type elementTy) {
    return new (allocator.allocate<UnrankedTensorTypeStorage>())
        UnrankedTensorTypeStorage(elementTy);
  }
};

struct MemRefTypeStorage : public ShapedTypeStorage {
  MemRefTypeStorage(unsigned shapeSize, Type elementType,
                    const int64_t *shapeElements, const unsigned numAffineMaps,
                    AffineMap const *affineMapList, const unsigned memorySpace)
      : ShapedTypeStorage(elementType, shapeSize), shapeElements(shapeElements),
        numAffineMaps(numAffineMaps), affineMapList(affineMapList),
        memorySpace(memorySpace) {}

  /// The hash key used for uniquing.
  // MemRefs are uniqued based on their shape, element type, affine map
  // composition, and memory space.
  using KeyTy =
      std::tuple<ArrayRef<int64_t>, Type, ArrayRef<AffineMap>, unsigned>;
  bool operator==(const KeyTy &key) const {
    return key == KeyTy(getShape(), elementType, getAffineMaps(), memorySpace);
  }

  /// Construction.
  static MemRefTypeStorage *construct(TypeStorageAllocator &allocator,
                                      const KeyTy &key) {
    // Copy the shape into the bump pointer.
    ArrayRef<int64_t> shape = allocator.copyInto(std::get<0>(key));

    // Copy the affine map composition into the bump pointer.
    ArrayRef<AffineMap> affineMapComposition =
        allocator.copyInto(std::get<2>(key));

    // Initialize the memory using placement new.
    return new (allocator.allocate<MemRefTypeStorage>())
        MemRefTypeStorage(shape.size(), std::get<1>(key), shape.data(),
                          affineMapComposition.size(),
                          affineMapComposition.data(), std::get<3>(key));
  }

  ArrayRef<int64_t> getShape() const {
    return ArrayRef<int64_t>(shapeElements, getSubclassData());
  }

  ArrayRef<AffineMap> getAffineMaps() const {
    return ArrayRef<AffineMap>(affineMapList, numAffineMaps);
  }

  /// An array of integers which stores the shape dimension sizes.
  const int64_t *shapeElements;
  /// The number of affine maps in the 'affineMapList' array.
  const unsigned numAffineMaps;
  /// List of affine maps in the memref's layout/index map composition.
  AffineMap const *affineMapList;
  /// Memory space in which data referenced by memref resides.
  const unsigned memorySpace;
};

/// Unranked MemRef is a MemRef with unknown rank.
/// Only element type and memory space are known
struct UnrankedMemRefTypeStorage : public ShapedTypeStorage {

  UnrankedMemRefTypeStorage(Type elementTy, const unsigned memorySpace)
      : ShapedTypeStorage(elementTy), memorySpace(memorySpace) {}

  /// The hash key used for uniquing.
  using KeyTy = std::tuple<Type, unsigned>;
  bool operator==(const KeyTy &key) const {
    return key == KeyTy(elementType, memorySpace);
  }

  /// Construction.
  static UnrankedMemRefTypeStorage *construct(TypeStorageAllocator &allocator,
                                              const KeyTy &key) {

    // Initialize the memory using placement new.
    return new (allocator.allocate<UnrankedMemRefTypeStorage>())
        UnrankedMemRefTypeStorage(std::get<0>(key), std::get<1>(key));
  }
  /// Memory space in which data referenced by memref resides.
  const unsigned memorySpace;
};

/// Complex Type Storage.
struct ComplexTypeStorage : public TypeStorage {
  ComplexTypeStorage(Type elementType) : elementType(elementType) {}

  /// The hash key used for uniquing.
  using KeyTy = Type;
  bool operator==(const KeyTy &key) const { return key == elementType; }

  /// Construction.
  static ComplexTypeStorage *construct(TypeStorageAllocator &allocator,
                                       Type elementType) {
    return new (allocator.allocate<ComplexTypeStorage>())
        ComplexTypeStorage(elementType);
  }

  Type elementType;
};

/// A type representing a collection of other types.
struct TupleTypeStorage final
    : public TypeStorage,
      public llvm::TrailingObjects<TupleTypeStorage, Type> {
  using KeyTy = ArrayRef<Type>;

  TupleTypeStorage(unsigned numTypes) : TypeStorage(numTypes) {}

  /// Construction.
  static TupleTypeStorage *construct(TypeStorageAllocator &allocator,
                                     ArrayRef<Type> key) {
    // Allocate a new storage instance.
    auto byteSize = TupleTypeStorage::totalSizeToAlloc<Type>(key.size());
    auto rawMem = allocator.allocate(byteSize, alignof(TupleTypeStorage));
    auto result = ::new (rawMem) TupleTypeStorage(key.size());

    // Copy in the element types into the trailing storage.
    std::uninitialized_copy(key.begin(), key.end(),
                            result->getTrailingObjects<Type>());
    return result;
  }

  bool operator==(const KeyTy &key) const { return key == getTypes(); }

  /// Return the number of held types.
  unsigned size() const { return getSubclassData(); }

  /// Return the held types.
  ArrayRef<Type> getTypes() const {
    return {getTrailingObjects<Type>(), size()};
  }
};

} // namespace detail
} // namespace mlir

#endif // TYPEDETAIL_H_
