//===-----------------------------------------------------------*- C++ -*--===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Compact table to map enumeration to strings, generated at compile-time. Basic
// usage:
//
//     // Don't reference this elsewhere, this should _not_ go into the final
//     // binary!
//     constexpr EnumStringDef<EnumTy> Defs[] = {
//         {{"abc"}, 1},
//         {{"def"}, 2},
//     };
//     static constexpr auto EnumStrs = BUILD_ENUM_STRINGS(Defs);
//     // ...
//     StringRef Str = EnumStrings(EnumStrs).toString(2); // "def"
//
// The table supports multiple strings per enumeration entry for alternative
// representations, this is e.g. used by llvm-readobj for LLVM and GNU names.
//
// Internally, "EnumStrs" above holds the array of enum definitions (EnumString)
// and the actual strings in one data structure. This permits the EnumString to
// reference the string with a small 16-bit offset from its location in memory.
// This design allows for a small size and doesn't require any relocations.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_ENUM_H
#define LLVM_SUPPORT_ENUM_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/DataTypes.h"

namespace llvm {

/// Compile-time data representation of enum entries. Only use for constexpr
/// variables passed to BUILD_ENUM_STRINGS, do NOT access the created variable
/// directly in code! The idea is that this is only used at compile-time to
/// build a more compact and relocation-free representation in the binary.
template <typename T, unsigned NumStrs = 1> struct EnumStringDef {
  std::array<std::string_view, NumStrs> Names;
  T Value;
};

template <typename T, unsigned NumStrs = 1> class EnumString {
  template <typename, unsigned, size_t, size_t>
  friend struct EnumStringsStorage;

  uint16_t NameOff[NumStrs] = {}; ///< Name offsets relative to this pointer.
  uint8_t NameSize[NumStrs] = {}; ///< Name string lengths.
  T Value{};                      ///< Enumeration value.

  constexpr EnumString() {}
  // Because name strings are stored relative to the address of this EnumString
  // in (read-only) memory, EnumString is neither movable nor copyable.
  EnumString(const EnumString &) = delete;
  EnumString &operator=(const EnumString &) = delete;

public:
  constexpr StringRef name(unsigned Idx = 0) const {
    assert(Idx < NumStrs);
    return {reinterpret_cast<const char *>(this) + NameOff[Idx], NameSize[Idx]};
  }
  constexpr T value() const { return Value; }
};

namespace detail {
template <typename T, unsigned NumStrs, size_t N>
constexpr unsigned
enumStringsStorageSize(const EnumStringDef<T, NumStrs> (&Entries)[N]) {
  unsigned Len = 0;
  for (unsigned i = 0; i != N; i++)
    for (unsigned j = 0; j != NumStrs; j++)
      Len += Entries[i].Names[j].size();
  return Len;
}
} // namespace detail

template <typename T, unsigned NumStrs, size_t N, size_t StrLen>
struct EnumStringsStorage {
  EnumString<T, NumStrs> Data[N];
  char Strs[StrLen];

  constexpr EnumStringsStorage(const EnumStringDef<T, NumStrs> (&Entries)[N])
      : Data{}, Strs{} {
    unsigned StrIdx = 0;
    for (unsigned i = 0; i < N; i++) {
      Data[i].Value = Entries[i].Value;
      for (unsigned j = 0; j < NumStrs; j++) {
        unsigned StrOff = offsetof(EnumStringsStorage, Strs) + StrIdx;
        unsigned DataOff = offsetof(EnumStringsStorage, Data) +
                           i * sizeof(EnumString<T, NumStrs>);
        assert(StrOff - DataOff <= UINT16_MAX && "enum string table too large");
        std::string_view Name = Entries[i].Names[j];
        assert(Name.size() <= UINT8_MAX && "enum name too long");
        Data[i].NameSize[j] = Name.size();
        Data[i].NameOff[j] = uint16_t(StrOff - DataOff);
        for (char C : Name)
          Strs[StrIdx++] = C;
      }
    };
  }

  constexpr size_t size() const { return N; }
  const EnumString<T, NumStrs> &operator[](size_t Idx) const {
    assert(Idx < N);
    return Data[Idx];
  }
  const EnumString<T, NumStrs> *begin() const { return std::begin(Data); }
  const EnumString<T, NumStrs> *end() const { return std::end(Data); }
};

#define BUILD_ENUM_STRINGS(Tab)                                                \
  (::llvm::EnumStringsStorage<decltype(Tab[0].Value), Tab[0].Names.size(),     \
                              sizeof(Tab) / sizeof(Tab[0]),                    \
                              ::llvm::detail::enumStringsStorageSize(Tab)>{    \
      Tab})

template <typename T, unsigned NumStrs = 1> class EnumStrings {
public:
  using EnumString = ::llvm::EnumString<T, NumStrs>;

  template <size_t N, size_t StrLen>
  EnumStrings(const EnumStringsStorage<T, NumStrs, N, StrLen> &Table)
      : EnumValues(Table.Data, N) {}

  template <typename TValue>
  StringRef toString(TValue Value, unsigned StrIdx = 0) const {
    // TODO: optimize with binary search?
    for (const auto &EnumItem : EnumValues)
      if (EnumItem.value() == Value)
        return EnumItem.name(StrIdx);
    return "";
  }

  template <typename TValue>
  std::string toStringOrHex(TValue Value, unsigned StrIdx = 0) const {
    if (StringRef Str = toString(Value, StrIdx); !Str.empty())
      return Str.str();
    return utohexstr(Value, true);
  }

  size_t size() const { return EnumValues.size(); }
  const EnumString &operator[](size_t Idx) const { return EnumValues[Idx]; }
  const EnumString *begin() const { return EnumValues.begin(); }
  const EnumString *end() const { return EnumValues.end(); }

private:
  ArrayRef<EnumString> EnumValues;
};

template <typename T, unsigned NumStrs, size_t N, size_t StrLen>
EnumStrings(const EnumStringsStorage<T, NumStrs, N, StrLen> &)
    -> EnumStrings<T, NumStrs>;

} // namespace llvm

#endif
