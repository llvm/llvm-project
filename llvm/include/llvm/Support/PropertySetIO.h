//==-- PropertySetIO.h -- models a sequence of property sets and their I/O -==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Models a sequence of property sets and their input and output operations.
// PropertyValue set format:
//   '['<PropertyValue set name>']'
//   <property name>=<property type>'|'<property value>
//   <property name>=<property type>'|'<property value>
//   ...
//   '['<PropertyValue set name>']'
//   <property name>=<property type>'|'<property value>
// where
//   <PropertyValue set name>, <property name> are strings
//   <property type> - string representation of the property type
//   <property value> - string representation of the property value.
//
// For example:
// [Staff/Ages]
// person1=1|20
// person2=1|25
// [Staff/Experience]
// person1=1|1
// person2=1|2
// person3=1|12
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_PROPERTYSETIO_H
#define LLVM_SUPPORT_PROPERTYSETIO_H

#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/xxhash.h"
#include <variant>

namespace llvm {
namespace util {

// Represents a property value. PropertyValue name is stored in the encompassing
// container.
class PropertyValue {
public:
  // Type of the size of the value. Value size gets serialized along with the
  // value data in some cases for later reading at runtime, so size_t is not
  // suitable as its size varies.
  using SizeTy = uint64_t;

  // Defines supported property types
  enum Type { first = 0, NONE = first, UINT32, BYTE_ARRAY, last = BYTE_ARRAY };

  // Translates C++ type to the corresponding type tag.
  template <typename T> static Type getTypeTag();

  // Casts from int value to a type tag.
  static Expected<Type> getTypeTag(int T) {
    if (T < first || T > last)
      return createStringError(std::error_code(), "bad property type ", T);
    return static_cast<Type>(T);
  }

  ~PropertyValue() {}

  PropertyValue() = default;
  PropertyValue(Type T) : Ty(T) {}

  PropertyValue(uint32_t Val) : Ty(UINT32), Val({Val}) {}
  PropertyValue(const std::byte *Data, SizeTy DataBitSize);
  template <typename C, typename T = typename C::value_type>
  PropertyValue(const C &Data)
      : PropertyValue(reinterpret_cast<const std::byte *>(Data.data()),
                      Data.size() * sizeof(T) * CHAR_BIT) {}
  PropertyValue(const llvm::StringRef &Str)
      : PropertyValue(reinterpret_cast<const std::byte *>(Str.data()),
                      Str.size() * sizeof(char) * /* bits in one byte */ 8) {}
  PropertyValue(const PropertyValue &P);
  PropertyValue(PropertyValue &&P);

  PropertyValue &operator=(PropertyValue &&P);

  PropertyValue &operator=(const PropertyValue &P);

  // Get property value as unsigned 32-bit integer
  uint32_t asUint32() const {
    if (Ty != UINT32)
      llvm_unreachable("must be UINT32 value");
    return std::get<uint32_t>(Val);
  }

  // Get raw data size in bits.
  SizeTy getByteArraySizeInBits() const {
    if (Ty != BYTE_ARRAY)
      llvm_unreachable("must be BYTE_ARRAY value");
    SizeTy Res = 0;

    for (size_t I = 0; I < sizeof(SizeTy); ++I) {
      auto ByteArrayVal = std::get<std::byte *>(Val);
      Res |= (SizeTy)ByteArrayVal[I] << (8 * I);
    }
    return Res;
  }

  // Get byte array data size in bytes.
  SizeTy getByteArraySize() const {
    SizeTy SizeInBits = getByteArraySizeInBits();
    constexpr unsigned int MASK = 0x7;
    return ((SizeInBits + MASK) & ~MASK) / 8;
  }

  // Get byte array data size in bytes, including the leading bytes encoding the
  // size.
  SizeTy getRawByteArraySize() const {
    return getByteArraySize() + sizeof(SizeTy);
  }

  // Get byte array data including the leading bytes encoding the size.
  const std::byte *asRawByteArray() const {
    if (Ty != BYTE_ARRAY)
      llvm_unreachable("must be BYTE_ARRAY value");
    auto *ByteArrayVal = std::get<std::byte *>(Val);
    return ByteArrayVal;
  }

  // Get byte array data excluding the leading bytes encoding the size.
  const std::byte *asByteArray() const {
    if (Ty != BYTE_ARRAY)
      llvm_unreachable("must be BYTE_ARRAY value");

    auto ByteArrayVal = std::get<std::byte *>(Val);
    return ByteArrayVal + sizeof(SizeTy);
  }

  bool isValid() const { return getType() != NONE; }

  // Set property value; the 'T' type must be convertible to a property type tag
  void set(uint32_t V) {
    if (Ty != UINT32)
      llvm_unreachable("invalid type tag for this operation");
    Val = V;
  }

  // Set property value; the 'T' type must be convertible to a property type tag
  void set(std::byte *V, int DataSize) {
    if (Ty != BYTE_ARRAY)
      llvm_unreachable("invalid type tag for this operation");
    size_t DataBitSize = DataSize * CHAR_BIT;
    constexpr size_t SizeFieldSize = sizeof(SizeTy);
    // Allocate space for size and data.
    Val = new std::byte[SizeFieldSize + DataSize];

    // Write the size into first bytes.
    for (size_t I = 0; I < SizeFieldSize; ++I) {
      auto ByteArrayVal = std::get<std::byte *>(Val);
      ByteArrayVal[I] = (std::byte)DataBitSize;
      DataBitSize >>= CHAR_BIT;
    }
    // Append data.
    auto ByteArrayVal = std::get<std::byte *>(Val);
    std::memcpy(ByteArrayVal + SizeFieldSize, V, DataSize);
  }

  Type getType() const { return Ty; }

  SizeTy size() const {
    switch (Ty) {
    case UINT32:
      return sizeof(uint32_t);
    case BYTE_ARRAY:
      return getRawByteArraySize();
    default:
      llvm_unreachable_internal("unsupported property type");
    }
  }

private:
  void copy(const PropertyValue &P);

  Type Ty = NONE;
  std::variant<uint32_t, std::byte *> Val;
};

/// Structure for specialization of DenseMap in PropertySetRegistry.
struct PropertySetKeyInfo {
  static unsigned getHashValue(const SmallString<16> &K) { return xxHash64(K); }

  static SmallString<16> getEmptyKey() { return SmallString<16>(""); }

  static SmallString<16> getTombstoneKey() { return SmallString<16>("_"); }

  static bool isEqual(StringRef L, StringRef R) { return L == R; }
};

using PropertyMapTy = DenseMap<SmallString<16>, unsigned, PropertySetKeyInfo>;
/// A property set. Preserves insertion order when iterating elements.
using PropertySet = MapVector<SmallString<16>, PropertyValue, PropertyMapTy>;

/// A registry of property sets. Maps a property set name to its
/// content.
///
/// The order of keys is preserved and corresponds to the order of insertion.
class PropertySetRegistry {
public:
  using MapTy = MapVector<SmallString<16>, PropertySet, PropertyMapTy>;

  // Specific property category names used by tools.
  static constexpr char SYCL_SPECIALIZATION_CONSTANTS[] =
      "SYCL/specialization constants";
  static constexpr char SYCL_SPEC_CONSTANTS_DEFAULT_VALUES[] =
      "SYCL/specialization constants default values";
  static constexpr char SYCL_DEVICELIB_REQ_MASK[] = "SYCL/devicelib req mask";
  static constexpr char SYCL_KERNEL_PARAM_OPT_INFO[] = "SYCL/kernel param opt";
  static constexpr char SYCL_PROGRAM_METADATA[] = "SYCL/program metadata";
  static constexpr char SYCL_MISC_PROP[] = "SYCL/misc properties";
  static constexpr char SYCL_ASSERT_USED[] = "SYCL/assert used";
  static constexpr char SYCL_EXPORTED_SYMBOLS[] = "SYCL/exported symbols";
  static constexpr char SYCL_IMPORTED_SYMBOLS[] = "SYCL/imported symbols";
  static constexpr char SYCL_DEVICE_GLOBALS[] = "SYCL/device globals";
  static constexpr char SYCL_DEVICE_REQUIREMENTS[] = "SYCL/device requirements";
  static constexpr char SYCL_HOST_PIPES[] = "SYCL/host pipes";
  static constexpr char SYCL_VIRTUAL_FUNCTIONS[] = "SYCL/virtual functions";

  /// Function for bulk addition of an entire property set in the given
  /// \p Category .
  template <typename MapTy> void add(StringRef Category, const MapTy &Props) {
    assert(PropSetMap.find(Category) == PropSetMap.end() &&
           "category already added");
    auto &PropSet = PropSetMap[Category];

    for (const auto &[PropName, PropVal] : Props)
      PropSet.insert_or_assign(PropName, PropertyValue(PropVal));
  }

  /// Adds the given \p PropVal with the given \p PropName into the given \p
  /// Category .
  template <typename T>
  void add(StringRef Category, StringRef PropName, const T &PropVal) {
    auto &PropSet = PropSetMap[Category];
    PropSet.insert({PropName, PropertyValue(PropVal)});
  }

  /// Parses from the given \p Buf a property set registry.
  static Expected<std::unique_ptr<PropertySetRegistry>>
  read(const MemoryBuffer *Buf);

  /// Dumps the property set registry to the given \p Out stream.
  void write(raw_ostream &Out) const;

  MapTy::const_iterator begin() const { return PropSetMap.begin(); }
  MapTy::const_iterator end() const { return PropSetMap.end(); }

  /// Retrieves a property set with given \p Name .
  PropertySet &operator[](StringRef Name) { return PropSetMap[Name]; }
  /// Constant access to the underlying map.
  const MapTy &getPropSets() const { return PropSetMap; }

private:
  MapTy PropSetMap;
};

} // namespace util

} // namespace llvm

#endif // #define LLVM_SUPPORT_PROPERTYSETIO_H
