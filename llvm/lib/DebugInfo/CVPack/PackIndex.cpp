//===- PackIndex.cpp ------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Record keys, and the key index of a pack (<content id>.cvpx).
//
//===----------------------------------------------------------------------===//

#include "llvm/DebugInfo/CVPack/PackIndex.h"
#include "CVPackUtils.h"
#include "llvm/DebugInfo/CodeView/CVPacks.h"
#include "llvm/DebugInfo/CodeView/CVRecord.h"
#include "llvm/DebugInfo/CodeView/TypeIndex.h"
#include "llvm/Support/BLAKE3.h"
#include "llvm/Support/Endian.h"
#include <algorithm>
#include <cstring>

using namespace llvm;
using namespace llvm::codeview;
using namespace llvm::codeview::cvpack;
using namespace llvm::support::endian;

static Expected<RecordKey> getRecordKey(ArrayRef<uint8_t> Record, bool IsSymbol,
                                        ArrayRef<GloballyHashedType> TypeHashes,
                                        ArrayRef<GloballyHashedType> IdHashes) {
  if (Record.size() < sizeof(RecordPrefix) ||
      read16le(Record.data()) + sizeof(uint16_t) != Record.size())
    return malformedRecord(
        "invalid CodeView record size for CodeView pack key");

  std::vector<uint8_t> Normalized;
  Normalized.reserve(Record.size() + 32);
  size_t Cursor = 0;
  if (Error E = forEachTypeIndex(
          Record, IsSymbol, [&](uint32_t Offset, TiRefKind Kind) -> Error {
            if (Offset < Cursor)
              return malformedRecord(
                  "invalid type-index offset in CodeView record");
            Normalized.insert(Normalized.end(), Record.begin() + Cursor,
                              Record.begin() + Offset);
            Cursor = Offset + sizeof(uint32_t);
            uint32_t Index = read32le(Record.data() + Offset);
            if (Index < TypeIndex::FirstNonSimpleIndex) {
              Normalized.insert(Normalized.end(), Record.begin() + Offset,
                                Record.begin() + Cursor);
              return Error::success();
            }
            ArrayRef<GloballyHashedType> Hashes =
                Kind == TiRefKind::IndexRef ? IdHashes : TypeHashes;
            uint32_t ArrayIndex = (Index & ~TypeIndex::DecoratedItemIdMask) -
                                  TypeIndex::FirstNonSimpleIndex;
            if (ArrayIndex >= Hashes.size() || Hashes[ArrayIndex].empty())
              return malformedRecord(
                  "unresolved type index in CodeView record");
            Normalized.insert(Normalized.end(), Hashes[ArrayIndex].Hash.begin(),
                              Hashes[ArrayIndex].Hash.end());
            return Error::success();
          }))
    return std::move(E);
  Normalized.insert(Normalized.end(), Record.begin() + Cursor, Record.end());
  return BLAKE3::hash<16>(Normalized);
}

Expected<RecordKey> llvm::codeview::cvpack::getTypeRecordKey(
    ArrayRef<uint8_t> Record, ArrayRef<GloballyHashedType> TypeHashes,
    ArrayRef<GloballyHashedType> IdHashes) {
  return getRecordKey(Record, false, TypeHashes, IdHashes);
}

Expected<RecordKey> llvm::codeview::cvpack::getSymbolRecordKey(
    ArrayRef<uint8_t> Record, ArrayRef<GloballyHashedType> TypeHashes,
    ArrayRef<GloballyHashedType> IdHashes) {
  return getRecordKey(Record, true, TypeHashes, IdHashes);
}

// Pack index layout: a header, then the type, symbol and function entries,
// each a 16-byte key and a 4-byte index, sorted by key.
constexpr char PackIndexMagic[4] = {'C', 'V', 'P', 'X'};
constexpr uint16_t PackIndexVersion = 1;
constexpr size_t PackIndexHeaderSize = 4 + 2 + 2 + 32 + 8 + 4 * 3 + 4 * 3;
constexpr size_t PackIndexEntrySize = 16 + 4;

static Error sortIndexEntries(std::vector<PackIndexEntry> &Entries) {
  llvm::sort(Entries, [](const PackIndexEntry &A, const PackIndexEntry &B) {
    return A.Key < B.Key;
  });
  for (size_t I = 1; I < Entries.size(); ++I)
    if (Entries[I - 1].Key == Entries[I].Key)
      return malformedRecord("duplicate key in CodeView pack index");
  return Error::success();
}

Expected<std::vector<uint8_t>>
llvm::codeview::cvpack::writePackIndex(PackIndex Index) {
  for (std::vector<PackIndexEntry> *Entries :
       {&Index.Types, &Index.Symbols, &Index.Functions})
    if (Error E = sortIndexEntries(*Entries))
      return std::move(E);
  std::vector<uint8_t> Data;
  Data.reserve(PackIndexHeaderSize +
               PackIndexEntrySize * (Index.Types.size() + Index.Symbols.size() +
                                     Index.Functions.size()));
  Data.insert(Data.end(), std::begin(PackIndexMagic), std::end(PackIndexMagic));
  appendLE<uint32_t>(Data, PackIndexVersion); // version, then zero flags
  appendBytes(Data, Index.PackContentId);
  appendLE<uint64_t>(Data, Index.PackFileSize);
  appendLE<uint32_t>(Data, Index.TypeCount);
  appendLE<uint32_t>(Data, Index.SymbolCount);
  appendLE<uint32_t>(Data, Index.FunctionCount);
  appendLE<uint32_t>(Data, static_cast<uint32_t>(Index.Types.size()));
  appendLE<uint32_t>(Data, static_cast<uint32_t>(Index.Symbols.size()));
  appendLE<uint32_t>(Data, static_cast<uint32_t>(Index.Functions.size()));
  for (const std::vector<PackIndexEntry> *Entries :
       {&Index.Types, &Index.Symbols, &Index.Functions})
    for (const PackIndexEntry &Entry : *Entries) {
      appendBytes(Data, Entry.Key);
      appendLE<uint32_t>(Data, Entry.Index);
    }
  return Data;
}

Expected<PackIndexView>
llvm::codeview::cvpack::PackIndexView::create(ArrayRef<uint8_t> Data) {
  if (Data.size() < PackIndexHeaderSize ||
      memcmp(Data.data(), PackIndexMagic, 4) != 0)
    return malformedRecord("not a CodeView pack index");
  if (read16le(Data.data() + 4) != PackIndexVersion ||
      read16le(Data.data() + 6) != 0)
    return malformedRecord("unsupported CodeView pack index version");
  PackIndexView View;
  std::copy_n(Data.data() + 8, 32, View.ContentId.begin());
  View.FileSize = uint64_t(read32le(Data.data() + 40)) |
                  (uint64_t(read32le(Data.data() + 44)) << 32);
  View.TypeCount = read32le(Data.data() + 48);
  View.SymbolCount = read32le(Data.data() + 52);
  View.FunctionCount = read32le(Data.data() + 56);
  uint64_t Counts[3] = {read32le(Data.data() + 60), read32le(Data.data() + 64),
                        read32le(Data.data() + 68)};
  if (Counts[0] > View.TypeCount || Counts[1] > View.SymbolCount ||
      Counts[2] > View.FunctionCount ||
      Data.size() !=
          PackIndexHeaderSize +
              PackIndexEntrySize * (Counts[0] + Counts[1] + Counts[2]))
    return malformedRecord("truncated CodeView pack index");
  ArrayRef<uint8_t> Rest = Data.drop_front(PackIndexHeaderSize);
  View.Types = Rest.take_front(Counts[0] * PackIndexEntrySize);
  Rest = Rest.drop_front(Counts[0] * PackIndexEntrySize);
  View.Symbols = Rest.take_front(Counts[1] * PackIndexEntrySize);
  View.Functions = Rest.drop_front(Counts[1] * PackIndexEntrySize);
  // Entries are not checked here, so that opening an index does not read it:
  // a lookup checks the index it returns, and a producer verifies every
  // object it packs by restoring it.
  return View;
}

void llvm::codeview::cvpack::PackIndexView::findSorted(
    Table T, ArrayRef<RecordKey> Keys,
    function_ref<void(size_t, uint32_t)> Found) const {
  ArrayRef<uint8_t> Entries = T == Table::Types     ? Types
                              : T == Table::Symbols ? Symbols
                                                    : Functions;
  uint32_t Limit = T == Table::Types     ? TypeCount
                   : T == Table::Symbols ? SymbolCount
                                         : FunctionCount;
  size_t Count = Entries.size() / PackIndexEntrySize;
  size_t E = 0;
  for (size_t K = 0; K != Keys.size() && E != Count; ++K) {
    // Skip ahead in growing steps, then back to the first entry not below the
    // key: both lists are sorted.
    size_t Step = 1;
    while (E + Step < Count &&
           memcmp(Entries.data() + (E + Step) * PackIndexEntrySize,
                  Keys[K].data(), 16) < 0)
      E += Step, Step *= 2;
    while (E != Count && memcmp(Entries.data() + E * PackIndexEntrySize,
                                Keys[K].data(), 16) < 0)
      ++E;
    if (E != Count && memcmp(Entries.data() + E * PackIndexEntrySize,
                             Keys[K].data(), 16) == 0) {
      uint32_t Index = read32le(Entries.data() + E * PackIndexEntrySize + 16);
      if (Index < Limit)
        Found(K, Index);
    }
  }
}
