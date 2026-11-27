//===- PackIndex.h ----------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The keys that identify the records of a pack whatever their type index
// numbering, and the index that stores the keys of a pack next to it.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_DEBUGINFO_CVPACK_PACKINDEX_H
#define LLVM_DEBUGINFO_CVPACK_PACKINDEX_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/DebugInfo/CodeView/CVPacks.h"
#include "llvm/DebugInfo/CodeView/TypeHashing.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Error.h"
#include <array>
#include <cstdint>
#include <vector>

namespace llvm {
namespace codeview {
namespace cvpack {

using RecordKey = std::array<uint8_t, 16>;

/// Compute a stable key after replacing non-simple type indices with their
/// referenced record's global hash. Type and ID references use separate hash
/// arrays, indexed from TypeIndex::FirstNonSimpleIndex.
LLVM_ABI Expected<RecordKey>
getTypeRecordKey(ArrayRef<uint8_t> Record,
                 ArrayRef<GloballyHashedType> TypeHashes,
                 ArrayRef<GloballyHashedType> IdHashes);
LLVM_ABI Expected<RecordKey>
getSymbolRecordKey(ArrayRef<uint8_t> Record,
                   ArrayRef<GloballyHashedType> TypeHashes,
                   ArrayRef<GloballyHashedType> IdHashes);

/// A pack's key index, stored as <content id>.cvpx next to the pack: the record
/// keys of the pack, so that a producer can pack objects against it without
/// computing them. Entries are sorted by key, with one entry per key.
inline constexpr StringLiteral PackIndexFileExtension = ".cvpx";

struct PackIndexEntry {
  RecordKey Key{};
  /// The zero-based pack index of a type record, or the ordinal of a symbol or
  /// function.
  uint32_t Index = 0;
};

struct PackIndex {
  PackId PackContentId{};
  uint64_t PackFileSize = 0;
  uint32_t TypeCount = 0;
  uint32_t SymbolCount = 0;
  uint32_t FunctionCount = 0;
  /// Type records held at the first index of their key, every record they
  /// reference being held the same way: those an object can refer to.
  std::vector<PackIndexEntry> Types;
  std::vector<PackIndexEntry> Symbols;
  std::vector<PackIndexEntry> Functions;
};

/// Encode an index. The entries are sorted, and must have distinct keys.
LLVM_ABI Expected<std::vector<uint8_t>> writePackIndex(PackIndex Index);

/// A non-owning view of an encoded index, searched in place.
class PackIndexView {
public:
  LLVM_ABI static Expected<PackIndexView> create(ArrayRef<uint8_t> Data);
  const PackId &packContentId() const { return ContentId; }
  uint64_t packFileSize() const { return FileSize; }
  uint32_t typeCount() const { return TypeCount; }
  uint32_t symbolCount() const { return SymbolCount; }
  uint32_t functionCount() const { return FunctionCount; }
  /// Look up many keys at once, in one pass over the table: Keys must be
  /// sorted. Found is called with the position in Keys and the index of each
  /// key the table holds.
  enum class Table { Types, Symbols, Functions };
  LLVM_ABI void findSorted(Table T, ArrayRef<RecordKey> Keys,
                           function_ref<void(size_t, uint32_t)> Found) const;

private:
  PackId ContentId{};
  uint64_t FileSize = 0;
  uint32_t TypeCount = 0, SymbolCount = 0, FunctionCount = 0;
  ArrayRef<uint8_t> Types, Symbols, Functions;
};

} // namespace cvpack
} // namespace codeview
} // namespace llvm

#endif // LLVM_DEBUGINFO_CVPACK_PACKINDEX_H
