//=- SYCLPropertySetIO.h -- models a sequence of property sets and their I/O =//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Models a sequence of property sets and their input and output operations.
// SYCLPropertyValue set format:
//   '['<SYCLPropertyValue set name>']'
//   <property name>=<property type>'|'<property value>
//   <property name>=<property type>'|'<property value>
//   ...
//   '['<SYCLPropertyValue set name>']'
//   <property name>=<property type>'|'<property value>
// where
//   <SYCLPropertyValue set name>, <property name> are strings
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

#ifndef LLVM_SUPPORT_SYCLPROPERTYSETIO_H
#define LLVM_SUPPORT_SYCLPROPERTYSETIO_H

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

typedef struct Deleter {
  void operator()(std::byte *ptr) const { delete[] ptr; }
} Deleter;

// Represents a SYCL property value. SYCLPropertyValue name is stored in the
// encompassing container.
class SYCLPropertyValue {
public:
  // Type of the size of the value. Value size gets serialized along with the
  // value data in some cases for later reading at runtime, so size_t is not
  // suitable as its size varies.
  using SizeTy = uint64_t;

  // Defines supported property types
  enum Type { first = 0, None = first, UInt32, ByteArray, last = ByteArray };

  // Translates C++ type to the corresponding type tag.
  template <typename T> static Type getTypeTag();

  // Casts from int value to a type tag.
  static Expected<Type> getTypeTag(int T) {
    if (T < first || T > last)
      return createStringError(std::error_code(), "bad property type ", T);
    return static_cast<Type>(T);
  }

  ~SYCLPropertyValue() {}
  SYCLPropertyValue() = default;

  SYCLPropertyValue(Type Ty) {
    if (Ty == UInt32)
      Val = (uint32_t)0;
    else if (Ty == ByteArray)
      Val = std::unique_ptr<std::byte, Deleter>(new std::byte[1], Deleter{});
    else
      llvm_unreachable_internal("unsupported SYCL property type");
  }
  SYCLPropertyValue(uint32_t Val) : Val({Val}) {}
  SYCLPropertyValue(const std::byte *Data, SizeTy DataBitSize);
  template <typename C, typename T = typename C::value_type>
  SYCLPropertyValue(const C &Data)
      : SYCLPropertyValue(reinterpret_cast<const std::byte *>(Data.data()),
                          Data.size() * sizeof(T) * CHAR_BIT) {}
  SYCLPropertyValue(const llvm::StringRef &Str)
      : SYCLPropertyValue(reinterpret_cast<const std::byte *>(Str.data()),
                          Str.size() * sizeof(char) * CHAR_BIT) {}
  SYCLPropertyValue(const SYCLPropertyValue &P);
  SYCLPropertyValue(SYCLPropertyValue &&P);

  SYCLPropertyValue &operator=(SYCLPropertyValue &&P);

  SYCLPropertyValue &operator=(const SYCLPropertyValue &P);

  // Get property value as unsigned 32-bit integer
  uint32_t asUint32() const {
    assert(std::holds_alternative<uint32_t>(Val) && "must be a uint32_t value");
    return std::get<uint32_t>(Val);
  }

  // Get raw data size in bits.
  SizeTy getByteArraySizeInBits() const {
    SizeTy Res = 0;
    if (auto ByteArrayVal =
            std::get_if<std::unique_ptr<std::byte, Deleter>>(&Val)) {
      for (size_t I = 0; I < sizeof(SizeTy); ++I)
        Res |= (SizeTy)(*ByteArrayVal).get()[I] << (8 * I);
    } else
      llvm_unreachable("must be a byte array value");
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
    if (auto ByteArrayVal =
            std::get_if<std::unique_ptr<std::byte, Deleter>>(&Val))
      return (*ByteArrayVal).get();
    else
      llvm_unreachable("must be a byte array value");
  }

  // Get byte array data excluding the leading bytes encoding the size.
  const std::byte *asByteArray() const {
    if (auto ByteArrayVal =
            std::get_if<std::unique_ptr<std::byte, Deleter>>(&Val))
      return (*ByteArrayVal).get() + sizeof(SizeTy);
    else
      llvm_unreachable("must be a byte array value");
  }

  bool isValid() const { return getType() != None; }

  // Set property value when data type is uint32_t
  void set(uint32_t V) {
    assert(std::holds_alternative<uint32_t>(Val) && "must be a uint32_t value");
    Val = V;
  }

  // Set property value when data type is 'std::byte *'
  void set(std::byte *V, int DataSize) {
    size_t DataBitSize = DataSize * CHAR_BIT;
    constexpr size_t SizeFieldSize = sizeof(SizeTy);
    // Allocate space for size and data.
    Val = std::unique_ptr<std::byte, Deleter>(
        new std::byte[SizeFieldSize + DataSize], Deleter{});

    // Write the size into first bytes.
    if (auto ByteArrayVal =
            std::get_if<std::unique_ptr<std::byte, Deleter>>(&Val)) {
      for (size_t I = 0; I < SizeFieldSize; ++I) {
        (*ByteArrayVal).get()[I] = (std::byte)DataBitSize;
        DataBitSize >>= CHAR_BIT;
      }
      // Append data.
      std::memcpy((*ByteArrayVal).get() + SizeFieldSize, V, DataSize);
    } else
      llvm_unreachable("must be a byte array value");
  }

  Type getType() const {
    if (std::holds_alternative<uint32_t>(Val))
      return UInt32;
    if (std::holds_alternative<std::unique_ptr<std::byte, Deleter>>(Val))
      return ByteArray;
    return None;
  }

  SizeTy size() const {
    if (std::holds_alternative<uint32_t>(Val))
      return sizeof(uint32_t);
    if (std::holds_alternative<std::unique_ptr<std::byte, Deleter>>(Val))
      return getRawByteArraySize();
    llvm_unreachable_internal("unsupported SYCL property type");
  }

private:
  void copy(const SYCLPropertyValue &P);

  std::variant<uint32_t, std::unique_ptr<std::byte, Deleter>> Val;
};

/// Structure for specialization of DenseMap in SYCLPropertySetRegistry.
struct SYCLPropertySetKeyInfo {
  static unsigned getHashValue(const SmallString<16> &K) { return xxHash64(K); }

  static SmallString<16> getEmptyKey() { return SmallString<16>(""); }

  static SmallString<16> getTombstoneKey() { return SmallString<16>("_"); }

  static bool isEqual(StringRef L, StringRef R) { return L == R; }
};

using SYCLPropertyMapTy =
    DenseMap<SmallString<16>, unsigned, SYCLPropertySetKeyInfo>;
/// A property set. Preserves insertion order when iterating elements.
using SYCLPropertySet =
    MapVector<SmallString<16>, SYCLPropertyValue, SYCLPropertyMapTy>;

/// A registry of property sets. Maps a property set name to its
/// content.
///
/// The order of keys is preserved and corresponds to the order of insertion.
class SYCLPropertySetRegistry {
public:
  using MapTy = MapVector<SmallString<16>, SYCLPropertySet, SYCLPropertyMapTy>;

  // Specific property category names used by tools.
  static constexpr char SYCLSpecializationConstants[] =
      "SYCL/specialization constants";
  static constexpr char SYCLSpecConstantsDefaultValues[] =
      "SYCL/specialization constants default values";
  static constexpr char SYCLDeviceLibReqMask[] = "SYCL/devicelib req mask";
  static constexpr char SYCLKernelParamOptInfo[] = "SYCL/kernel param opt";
  static constexpr char SYCLProgramMetadata[] = "SYCL/program metadata";
  static constexpr char SYCLMiscProp[] = "SYCL/misc properties";
  static constexpr char SYCLAssertUsed[] = "SYCL/assert used";
  static constexpr char SYCLExportedSymbols[] = "SYCL/exported symbols";
  static constexpr char SYCLImportedSymbols[] = "SYCL/imported symbols";
  static constexpr char SYCLDeviceGlobals[] = "SYCL/device globals";
  static constexpr char SYCLDeviceRequirements[] = "SYCL/device requirements";
  static constexpr char SYCLHostPipes[] = "SYCL/host pipes";
  static constexpr char SYCLVirtualFunctions[] = "SYCL/virtual functions";

  /// Function for bulk addition of an entire property set in the given
  /// \p Category .
  template <typename MapTy> void add(StringRef Category, const MapTy &Props) {
    assert(PropSetMap.find(Category) == PropSetMap.end() &&
           "category already added");
    auto &PropSet = PropSetMap[Category];

    for (const auto &[PropName, PropVal] : Props)
      PropSet.insert_or_assign(PropName, SYCLPropertyValue(PropVal));
  }

  /// Adds the given \p PropVal with the given \p PropName into the given \p
  /// Category .
  template <typename T>
  void add(StringRef Category, StringRef PropName, const T &PropVal) {
    auto &PropSet = PropSetMap[Category];
    PropSet.insert({PropName, SYCLPropertyValue(PropVal)});
  }

  /// Parses from the given \p Buf a property set registry.
  static Expected<std::unique_ptr<SYCLPropertySetRegistry>>
  read(const MemoryBuffer *Buf);

  /// Dumps the property set registry to the given \p Out stream.
  void write(raw_ostream &Out) const;

  MapTy::const_iterator begin() const { return PropSetMap.begin(); }
  MapTy::const_iterator end() const { return PropSetMap.end(); }

  /// Retrieves a property set with given \p Name .
  SYCLPropertySet &operator[](StringRef Name) { return PropSetMap[Name]; }
  /// Constant access to the underlying map.
  const MapTy &getPropSets() const { return PropSetMap; }

private:
  MapTy PropSetMap;
};

} // namespace util

} // namespace llvm

#endif // #define LLVM_SUPPORT_SYCLPROPERTYSETIO_H
