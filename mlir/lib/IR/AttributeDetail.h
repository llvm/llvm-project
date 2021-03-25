//===- AttributeDetail.h - MLIR Affine Map details Class --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This holds implementation details of Attribute.
//
//===----------------------------------------------------------------------===//

#ifndef ATTRIBUTEDETAIL_H_
#define ATTRIBUTEDETAIL_H_

#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Identifier.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Support/StorageUniquer.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/PointerIntPair.h"
#include "llvm/Support/TrailingObjects.h"

namespace mlir {
namespace detail {

//===----------------------------------------------------------------------===//
// Elements Attributes
//===----------------------------------------------------------------------===//

/// Return the bit width which DenseElementsAttr should use for this type.
inline size_t getDenseElementBitWidth(Type eltType) {
  // Align the width for complex to 8 to make storage and interpretation easier.
  if (ComplexType comp = eltType.dyn_cast<ComplexType>())
    return llvm::alignTo<8>(getDenseElementBitWidth(comp.getElementType())) * 2;
  if (eltType.isIndex())
    return IndexType::kInternalStorageBitWidth;
  return eltType.getIntOrFloatBitWidth();
}

/// An attribute representing a reference to a dense vector or tensor object.
struct DenseElementsAttributeStorage : public AttributeStorage {
public:
  DenseElementsAttributeStorage(ShapedType ty, bool isSplat)
      : AttributeStorage(ty), isSplat(isSplat) {}

  bool isSplat;
};

/// An attribute representing a reference to a dense vector or tensor object.
struct DenseIntOrFPElementsAttrStorage : public DenseElementsAttributeStorage {
  DenseIntOrFPElementsAttrStorage(ShapedType ty, ArrayRef<char> data,
                                  bool isSplat = false)
      : DenseElementsAttributeStorage(ty, isSplat), data(data) {}

  struct KeyTy {
    KeyTy(ShapedType type, ArrayRef<char> data, llvm::hash_code hashCode,
          bool isSplat = false)
        : type(type), data(data), hashCode(hashCode), isSplat(isSplat) {}

    /// The type of the dense elements.
    ShapedType type;

    /// The raw buffer for the data storage.
    ArrayRef<char> data;

    /// The computed hash code for the storage data.
    llvm::hash_code hashCode;

    /// A boolean that indicates if this data is a splat or not.
    bool isSplat;
  };

  /// Compare this storage instance with the provided key.
  bool operator==(const KeyTy &key) const {
    if (key.type != getType())
      return false;

    // For boolean splats we need to explicitly check that the first bit is the
    // same. Boolean values are packed at the bit level, and even though a splat
    // is detected the rest of the bits in the first byte may differ from the
    // splat value.
    if (key.type.getElementType().isInteger(1)) {
      if (key.isSplat != isSplat)
        return false;
      if (isSplat)
        return (key.data.front() & 1) == data.front();
    }

    // Otherwise, we can default to just checking the data.
    return key.data == data;
  }

  /// Construct a key from a shaped type, raw data buffer, and a flag that
  /// signals if the data is already known to be a splat. Callers to this
  /// function are expected to tag preknown splat values when possible, e.g. one
  /// element shapes.
  static KeyTy getKey(ShapedType ty, ArrayRef<char> data, bool isKnownSplat) {
    // Handle an empty storage instance.
    if (data.empty())
      return KeyTy(ty, data, 0);

    // If the data is already known to be a splat, the key hash value is
    // directly the data buffer.
    if (isKnownSplat)
      return KeyTy(ty, data, llvm::hash_value(data), isKnownSplat);

    // Otherwise, we need to check if the data corresponds to a splat or not.

    // Handle the simple case of only one element.
    size_t numElements = ty.getNumElements();
    assert(numElements != 1 && "splat of 1 element should already be detected");

    // Handle boolean values directly as they are packed to 1-bit.
    if (ty.getElementType().isInteger(1) == 1)
      return getKeyForBoolData(ty, data, numElements);

    size_t elementWidth = getDenseElementBitWidth(ty.getElementType());
    // Non 1-bit dense elements are padded to 8-bits.
    size_t storageSize = llvm::divideCeil(elementWidth, CHAR_BIT);
    assert(((data.size() / storageSize) == numElements) &&
           "data does not hold expected number of elements");

    // Create the initial hash value with just the first element.
    auto firstElt = data.take_front(storageSize);
    auto hashVal = llvm::hash_value(firstElt);

    // Check to see if this storage represents a splat. If it doesn't then
    // combine the hash for the data starting with the first non splat element.
    for (size_t i = storageSize, e = data.size(); i != e; i += storageSize)
      if (memcmp(data.data(), &data[i], storageSize))
        return KeyTy(ty, data, llvm::hash_combine(hashVal, data.drop_front(i)));

    // Otherwise, this is a splat so just return the hash of the first element.
    return KeyTy(ty, firstElt, hashVal, /*isSplat=*/true);
  }

  /// Construct a key with a set of boolean data.
  static KeyTy getKeyForBoolData(ShapedType ty, ArrayRef<char> data,
                                 size_t numElements) {
    ArrayRef<char> splatData = data;
    bool splatValue = splatData.front() & 1;

    // Helper functor to generate a KeyTy for a boolean splat value.
    auto generateSplatKey = [=] {
      return KeyTy(ty, data.take_front(1),
                   llvm::hash_value(ArrayRef<char>(splatValue ? 1 : 0)),
                   /*isSplat=*/true);
    };

    // Handle the case where the potential splat value is 1 and the number of
    // elements is non 8-bit aligned.
    size_t numOddElements = numElements % CHAR_BIT;
    if (splatValue && numOddElements != 0) {
      // Check that all bits are set in the last value.
      char lastElt = splatData.back();
      if (lastElt != llvm::maskTrailingOnes<unsigned char>(numOddElements))
        return KeyTy(ty, data, llvm::hash_value(data));

      // If this is the only element, the data is known to be a splat.
      if (splatData.size() == 1)
        return generateSplatKey();
      splatData = splatData.drop_back();
    }

    // Check that the data buffer corresponds to a splat of the proper mask.
    char mask = splatValue ? ~0 : 0;
    return llvm::all_of(splatData, [mask](char c) { return c == mask; })
               ? generateSplatKey()
               : KeyTy(ty, data, llvm::hash_value(data));
  }

  /// Hash the key for the storage.
  static llvm::hash_code hashKey(const KeyTy &key) {
    return llvm::hash_combine(key.type, key.hashCode);
  }

  /// Construct a new storage instance.
  static DenseIntOrFPElementsAttrStorage *
  construct(AttributeStorageAllocator &allocator, KeyTy key) {
    // If the data buffer is non-empty, we copy it into the allocator with a
    // 64-bit alignment.
    ArrayRef<char> copy, data = key.data;
    if (!data.empty()) {
      char *rawData = reinterpret_cast<char *>(
          allocator.allocate(data.size(), alignof(uint64_t)));
      std::memcpy(rawData, data.data(), data.size());

      // If this is a boolean splat, make sure only the first bit is used.
      if (key.isSplat && key.type.getElementType().isInteger(1))
        rawData[0] &= 1;
      copy = ArrayRef<char>(rawData, data.size());
    }

    return new (allocator.allocate<DenseIntOrFPElementsAttrStorage>())
        DenseIntOrFPElementsAttrStorage(key.type, copy, key.isSplat);
  }

  ArrayRef<char> data;
};

/// An attribute representing a reference to a dense vector or tensor object
/// containing strings.
struct DenseStringElementsAttrStorage : public DenseElementsAttributeStorage {
  DenseStringElementsAttrStorage(ShapedType ty, ArrayRef<StringRef> data,
                                 bool isSplat = false)
      : DenseElementsAttributeStorage(ty, isSplat), data(data) {}

  struct KeyTy {
    KeyTy(ShapedType type, ArrayRef<StringRef> data, llvm::hash_code hashCode,
          bool isSplat = false)
        : type(type), data(data), hashCode(hashCode), isSplat(isSplat) {}

    /// The type of the dense elements.
    ShapedType type;

    /// The raw buffer for the data storage.
    ArrayRef<StringRef> data;

    /// The computed hash code for the storage data.
    llvm::hash_code hashCode;

    /// A boolean that indicates if this data is a splat or not.
    bool isSplat;
  };

  /// Compare this storage instance with the provided key.
  bool operator==(const KeyTy &key) const {
    if (key.type != getType())
      return false;

    // Otherwise, we can default to just checking the data. StringRefs compare
    // by contents.
    return key.data == data;
  }

  /// Construct a key from a shaped type, StringRef data buffer, and a flag that
  /// signals if the data is already known to be a splat. Callers to this
  /// function are expected to tag preknown splat values when possible, e.g. one
  /// element shapes.
  static KeyTy getKey(ShapedType ty, ArrayRef<StringRef> data,
                      bool isKnownSplat) {
    // Handle an empty storage instance.
    if (data.empty())
      return KeyTy(ty, data, 0);

    // If the data is already known to be a splat, the key hash value is
    // directly the data buffer.
    if (isKnownSplat)
      return KeyTy(ty, data, llvm::hash_value(data.front()), isKnownSplat);

    // Handle the simple case of only one element.
    assert(ty.getNumElements() != 1 &&
           "splat of 1 element should already be detected");

    // Create the initial hash value with just the first element.
    const auto &firstElt = data.front();
    auto hashVal = llvm::hash_value(firstElt);

    // Check to see if this storage represents a splat. If it doesn't then
    // combine the hash for the data starting with the first non splat element.
    for (size_t i = 1, e = data.size(); i != e; i++)
      if (!firstElt.equals(data[i]))
        return KeyTy(ty, data, llvm::hash_combine(hashVal, data.drop_front(i)));

    // Otherwise, this is a splat so just return the hash of the first element.
    return KeyTy(ty, data.take_front(), hashVal, /*isSplat=*/true);
  }

  /// Hash the key for the storage.
  static llvm::hash_code hashKey(const KeyTy &key) {
    return llvm::hash_combine(key.type, key.hashCode);
  }

  /// Construct a new storage instance.
  static DenseStringElementsAttrStorage *
  construct(AttributeStorageAllocator &allocator, KeyTy key) {
    // If the data buffer is non-empty, we copy it into the allocator with a
    // 64-bit alignment.
    ArrayRef<StringRef> copy, data = key.data;
    if (data.empty()) {
      return new (allocator.allocate<DenseStringElementsAttrStorage>())
          DenseStringElementsAttrStorage(key.type, copy, key.isSplat);
    }

    int numEntries = key.isSplat ? 1 : data.size();

    // Compute the amount data needed to store the ArrayRef and StringRef
    // contents.
    size_t dataSize = sizeof(StringRef) * numEntries;
    for (int i = 0; i < numEntries; i++)
      dataSize += data[i].size();

    char *rawData = reinterpret_cast<char *>(
        allocator.allocate(dataSize, alignof(uint64_t)));

    // Setup a mutable array ref of our string refs so that we can update their
    // contents.
    auto mutableCopy = MutableArrayRef<StringRef>(
        reinterpret_cast<StringRef *>(rawData), numEntries);
    auto stringData = rawData + numEntries * sizeof(StringRef);

    for (int i = 0; i < numEntries; i++) {
      memcpy(stringData, data[i].data(), data[i].size());
      mutableCopy[i] = StringRef(stringData, data[i].size());
      stringData += data[i].size();
    }

    copy =
        ArrayRef<StringRef>(reinterpret_cast<StringRef *>(rawData), numEntries);

    return new (allocator.allocate<DenseStringElementsAttrStorage>())
        DenseStringElementsAttrStorage(key.type, copy, key.isSplat);
  }

  ArrayRef<StringRef> data;
};

} // namespace detail
} // namespace mlir

#endif // ATTRIBUTEDETAIL_H_
