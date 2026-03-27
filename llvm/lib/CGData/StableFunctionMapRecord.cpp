//===-- StableFunctionMapRecord.cpp ---------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This implements the functionality for the StableFunctionMapRecord class,
// including methods for serialization and deserialization of stable function
// maps to and from raw and YAML streams. It also includes utilities for
// managing function entries and their metadata.
//
//===----------------------------------------------------------------------===//

#include "llvm/CGData/StableFunctionMapRecord.h"
#include "llvm/Support/EndianStream.h"

#define DEBUG_TYPE "stable-function-map-record"

using namespace llvm;
using namespace llvm::support;

LLVM_YAML_IS_SEQUENCE_VECTOR(IndexPairHash)
LLVM_YAML_IS_SEQUENCE_VECTOR(StableFunction)

namespace llvm {
namespace yaml {

template <> struct MappingTraits<IndexPairHash> {
  static void mapping(IO &IO, IndexPairHash &Key) {
    IO.mapRequired("InstIndex", Key.first.first);
    IO.mapRequired("OpndIndex", Key.first.second);
    IO.mapRequired("OpndHash", Key.second);
  }
};

template <> struct MappingTraits<StableFunction> {
  static void mapping(IO &IO, StableFunction &Func) {
    IO.mapRequired("Hash", Func.Hash);
    IO.mapRequired("FunctionName", Func.FunctionName);
    IO.mapRequired("ModuleName", Func.ModuleName);
    IO.mapRequired("InstCount", Func.InstCount);
    IO.mapRequired("IndexOperandHashes", Func.IndexOperandHashes);
  }
};

} // namespace yaml
} // namespace llvm

// Get a sorted vector of StableFunctionEntry pointers.
static SmallVector<const StableFunctionMap::StableFunctionEntry *>
getStableFunctionEntries(const StableFunctionMap &SFM) {
  SmallVector<const StableFunctionMap::StableFunctionEntry *> FuncEntries;
  for (const auto &P : SFM.getFunctionMap())
    for (auto &Func : P.second.Entries)
      FuncEntries.emplace_back(Func.get());

  llvm::stable_sort(
      FuncEntries, [&](auto &A, auto &B) {
        return std::tuple(A->Hash, SFM.getNameForId(A->ModuleNameId),
                          SFM.getNameForId(A->FunctionNameId)) <
               std::tuple(B->Hash, SFM.getNameForId(B->ModuleNameId),
                          SFM.getNameForId(B->FunctionNameId));
      });
  return FuncEntries;
}

// Get a sorted vector of IndexOperandHashes.
static IndexOperandHashVecType getStableIndexOperandHashes(
    const StableFunctionMap::StableFunctionEntry *FuncEntry) {
  IndexOperandHashVecType IndexOperandHashes;
  for (auto &[Indices, OpndHash] : *FuncEntry->IndexOperandHashMap)
    IndexOperandHashes.emplace_back(Indices, OpndHash);
  // The indices are unique, so we can just sort by the first.
  llvm::sort(IndexOperandHashes);
  return IndexOperandHashes;
}

void StableFunctionMapRecord::serialize(
    raw_ostream &OS, std::vector<CGDataPatchItem> &PatchItems) const {
  serialize(OS, FunctionMap.get(), PatchItems);
}

void StableFunctionMapRecord::serialize(
    raw_ostream &OS, const StableFunctionMap *FunctionMap,
    std::vector<CGDataPatchItem> &PatchItems) {
  support::endian::Writer Writer(OS, endianness::little);

  // Write Names.
  ArrayRef<std::string> Names = FunctionMap->getNames();
  Writer.write<uint32_t>(Names.size());
  // Remember the position, write back the total size of Names, so we can skip
  // reading them if needed.
  const uint64_t NamesByteSizeOffset = Writer.OS.tell();
  Writer.write<uint64_t>(0);
  for (auto &Name : Names)
    Writer.OS << Name << '\0';
  // Align current position to 4 bytes.
  uint32_t Padding = offsetToAlignment(Writer.OS.tell(), Align(4));
  for (uint32_t I = 0; I < Padding; ++I)
    Writer.OS << '\0';
  const auto NamesByteSize =
      Writer.OS.tell() - NamesByteSizeOffset - sizeof(NamesByteSizeOffset);
  PatchItems.emplace_back(NamesByteSizeOffset, &NamesByteSize, 1);

  // Write StableFunctionEntries whose pointers are sorted.
  auto FuncEntries = getStableFunctionEntries(*FunctionMap);
  Writer.write<uint32_t>(FuncEntries.size());
  for (const auto *FuncRef : FuncEntries)
    Writer.write<stable_hash>(FuncRef->Hash);
  std::vector<uint64_t> IndexOperandHashesOffsets;
  IndexOperandHashesOffsets.reserve(FuncEntries.size());
  for (const auto *FuncRef : FuncEntries) {
    Writer.write<uint32_t>(FuncRef->FunctionNameId);
    Writer.write<uint32_t>(FuncRef->ModuleNameId);
    Writer.write<uint32_t>(FuncRef->InstCount);
    const uint64_t Offset = Writer.OS.tell();
    IndexOperandHashesOffsets.push_back(Offset);
    Writer.write<uint64_t>(0);
  }
  const uint64_t IndexOperandHashesByteSizeOffset = Writer.OS.tell();
  Writer.write<uint64_t>(0);
  for (size_t I = 0; I < FuncEntries.size(); ++I) {
    const uint64_t Offset = Writer.OS.tell() - IndexOperandHashesOffsets[I];
    PatchItems.emplace_back(IndexOperandHashesOffsets[I], &Offset, 1);
    // Emit IndexOperandHashes sorted from IndexOperandHashMap.
    const auto *FuncRef = FuncEntries[I];
    IndexOperandHashVecType IndexOperandHashes =
        getStableIndexOperandHashes(FuncRef);
    Writer.write<uint32_t>(IndexOperandHashes.size());
    for (auto &IndexOperandHash : IndexOperandHashes) {
      Writer.write<uint32_t>(IndexOperandHash.first.first);
      Writer.write<uint32_t>(IndexOperandHash.first.second);
      Writer.write<stable_hash>(IndexOperandHash.second);
    }
  }
  // Write the total size of IndexOperandHashes.
  const uint64_t IndexOperandHashesByteSize =
      Writer.OS.tell() - IndexOperandHashesByteSizeOffset - sizeof(uint64_t);
  PatchItems.emplace_back(IndexOperandHashesByteSizeOffset,
                          &IndexOperandHashesByteSize, 1);
}

void StableFunctionMapRecord::deserializeEntry(const unsigned char *Ptr,
                                               stable_hash Hash,
                                               StableFunctionMap *FunctionMap) {
  auto FunctionNameId =
      endian::readNext<uint32_t, endianness::little, unaligned>(Ptr);
  if (FunctionMap->ReadStableFunctionMapNames)
    assert(FunctionMap->getNameForId(FunctionNameId) &&
           "FunctionNameId out of range");
  auto ModuleNameId =
      endian::readNext<uint32_t, endianness::little, unaligned>(Ptr);
  if (FunctionMap->ReadStableFunctionMapNames)
    assert(FunctionMap->getNameForId(ModuleNameId) &&
           "ModuleNameId out of range");
  auto InstCount =
      endian::readNext<uint32_t, endianness::little, unaligned>(Ptr);

  // Read IndexOperandHashes to build IndexOperandHashMap
  auto CurrentPosition = reinterpret_cast<uintptr_t>(Ptr);
  auto IndexOperandHashesOffset =
      endian::readNext<uint64_t, endianness::little, unaligned>(Ptr);
  auto *IndexOperandHashesPtr = reinterpret_cast<const unsigned char *>(
      CurrentPosition + IndexOperandHashesOffset);
  auto NumIndexOperandHashes =
      endian::readNext<uint32_t, endianness::little, unaligned>(
          IndexOperandHashesPtr);
  auto IndexOperandHashMap = std::make_unique<IndexOperandHashMapType>();
  for (unsigned J = 0; J < NumIndexOperandHashes; ++J) {
    auto InstIndex = endian::readNext<uint32_t, endianness::little, unaligned>(
        IndexOperandHashesPtr);
    auto OpndIndex = endian::readNext<uint32_t, endianness::little, unaligned>(
        IndexOperandHashesPtr);
    auto OpndHash =
        endian::readNext<stable_hash, endianness::little, unaligned>(
            IndexOperandHashesPtr);
    assert(InstIndex < InstCount && "InstIndex out of range");

    IndexOperandHashMap->try_emplace({InstIndex, OpndIndex}, OpndHash);
  }

  // Insert a new StableFunctionEntry into the map.
  auto FuncEntry = std::make_unique<StableFunctionMap::StableFunctionEntry>(
      Hash, FunctionNameId, ModuleNameId, InstCount,
      std::move(IndexOperandHashMap));

  FunctionMap->insert(std::move(FuncEntry));
}

void StableFunctionMapRecord::deserialize(const unsigned char *&Ptr,
                                          bool Lazy) {
  // Assert that Ptr is 4-byte aligned
  assert(((uintptr_t)Ptr % 4) == 0);
  // Read Names.
  auto NumNames =
      endian::readNext<uint32_t, endianness::little, unaligned>(Ptr);
  // Early exit if there is no name.
  if (NumNames == 0)
    return;
  const auto NamesByteSize =
      endian::readNext<uint64_t, endianness::little, unaligned>(Ptr);
  const auto NamesOffset = reinterpret_cast<uintptr_t>(Ptr);
  if (FunctionMap->ReadStableFunctionMapNames) {
    for (unsigned I = 0; I < NumNames; ++I) {
      StringRef Name(reinterpret_cast<const char *>(Ptr));
      Ptr += Name.size() + 1;
      FunctionMap->getIdOrCreateForName(Name);
    }
    // Align Ptr to 4 bytes.
    Ptr = reinterpret_cast<const uint8_t *>(alignAddr(Ptr, Align(4)));
    assert(reinterpret_cast<uintptr_t>(Ptr) - NamesOffset == NamesByteSize &&
           "NamesByteSize does not match the actual size of names");
  } else {
    // skip reading Names by advancing the pointer.
    Ptr = reinterpret_cast<const uint8_t *>(NamesOffset + NamesByteSize);
  }

  // Read StableFunctionEntries.
  auto NumFuncs =
      endian::readNext<uint32_t, endianness::little, unaligned>(Ptr);
  auto FixedSizeFieldsOffset =
      reinterpret_cast<uintptr_t>(Ptr) + NumFuncs * sizeof(stable_hash);
  constexpr uint32_t FixedSizeFieldsSizePerEntry =
      // FunctionNameId
      sizeof(uint32_t) +
      // ModuleNameId
      sizeof(uint32_t) +
      // InstCount
      sizeof(uint32_t) +
      // Relative offset to IndexOperandHashes
      sizeof(uint64_t);
  for (unsigned I = 0; I < NumFuncs; ++I) {
    auto Hash =
        endian::readNext<stable_hash, endianness::little, unaligned>(Ptr);
    if (Lazy) {
      auto It = FunctionMap->HashToFuncs.try_emplace(Hash).first;
      StableFunctionMap::EntryStorage &Storage = It->second;
      Storage.Offsets.push_back(FixedSizeFieldsOffset);
    } else {
      deserializeEntry(
          reinterpret_cast<const unsigned char *>(FixedSizeFieldsOffset), Hash,
          FunctionMap.get());
    }
    FixedSizeFieldsOffset += FixedSizeFieldsSizePerEntry;
  }

  // Update Ptr to the end of the serialized map to meet the expectation of
  // CodeGenDataReader.
  Ptr = reinterpret_cast<const unsigned char *>(FixedSizeFieldsOffset);
  auto IndexOperandHashesByteSize =
      endian::readNext<uint64_t, endianness::little, unaligned>(Ptr);
  Ptr = reinterpret_cast<const unsigned char *>(
      reinterpret_cast<uintptr_t>(Ptr) + IndexOperandHashesByteSize);
}

void StableFunctionMapRecord::deserialize(const unsigned char *&Ptr) {
  deserialize(Ptr, /*Lazy=*/false);
}

void StableFunctionMapRecord::lazyDeserialize(
    std::shared_ptr<MemoryBuffer> Buffer, uint64_t Offset) {
  const auto *Ptr = reinterpret_cast<const unsigned char *>(
      reinterpret_cast<uintptr_t>(Buffer->getBufferStart()) + Offset);
  deserialize(Ptr, /*Lazy=*/true);
  FunctionMap->Buffer = std::move(Buffer);
}

void StableFunctionMapRecord::serializeYAML(yaml::Output &YOS) const {
  auto FuncEntries = getStableFunctionEntries(*FunctionMap);
  SmallVector<StableFunction> Functions;
  for (const auto *FuncEntry : FuncEntries) {
    auto IndexOperandHashes = getStableIndexOperandHashes(FuncEntry);
    Functions.emplace_back(
        FuncEntry->Hash, *FunctionMap->getNameForId(FuncEntry->FunctionNameId),
        *FunctionMap->getNameForId(FuncEntry->ModuleNameId),
        FuncEntry->InstCount, std::move(IndexOperandHashes));
  }

  YOS << Functions;
}

void StableFunctionMapRecord::deserializeYAML(yaml::Input &YIS) {
  std::vector<StableFunction> Funcs;
  YIS >> Funcs;
  for (auto &Func : Funcs)
    FunctionMap->insert(Func);
  YIS.nextDocument();
}
