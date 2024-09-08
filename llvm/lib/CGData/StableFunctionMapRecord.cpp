//===-- StableFunctionMapRecord.cpp ---------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// TODO
//
//===----------------------------------------------------------------------===//

#include "llvm/CGData/StableFunctionMapRecord.h"
#include "llvm/ObjectYAML/YAML.h"
#include "llvm/Support/Endian.h"
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
static SmallVector<const StableFunctionEntry *>
getStableFunctionEntries(StableFunctionMap &SFM) {
  SmallVector<const StableFunctionEntry *> FuncEntries;
  for (const auto &P : SFM.getFunctionMap())
    for (auto &Func : P.second)
      FuncEntries.emplace_back(Func.get());

  std::stable_sort(
      FuncEntries.begin(), FuncEntries.end(), [&](auto &A, auto &B) {
        return std::tuple(A->Hash, SFM.getNameForId(A->ModuleNameId),
                          SFM.getNameForId(A->FunctionNameId)) <
               std::tuple(B->Hash, SFM.getNameForId(B->ModuleNameId),
                          SFM.getNameForId(B->FunctionNameId));
      });
  return FuncEntries;
}

// Get a sorted vector of IndexOperandHashes.
static IndexOperandHashVecType
getStableIndexOperandHashes(const StableFunctionEntry *FuncEntry) {
  IndexOperandHashVecType IndexOperandHashes;
  for (auto &[Indices, OpndHash] : *FuncEntry->IndexOperandHashMap)
    IndexOperandHashes.emplace_back(Indices, OpndHash);
  std::sort(IndexOperandHashes.begin(), IndexOperandHashes.end(),
            [](auto &A, auto &B) { return A.first < B.first; });
  return IndexOperandHashes;
}

void StableFunctionMapRecord::serialize(raw_ostream &OS) const {
  support::endian::Writer Writer(OS, endianness::little);

  // Write Names.
  auto &Names = FunctionMap->getNames();
  Writer.write<uint32_t>(Names.size());
  for (auto &Name : Names)
    Writer.OS << Name << '\0';

  // Write StableFunctionEntries whose pointers are sorted.
  auto FuncEntries = getStableFunctionEntries(*FunctionMap);
  Writer.write<uint32_t>(FuncEntries.size());

  for (const auto *FuncRef : FuncEntries) {
    Writer.write<stable_hash>(FuncRef->Hash);
    Writer.write<uint32_t>(FuncRef->FunctionNameId);
    Writer.write<uint32_t>(FuncRef->ModuleNameId);
    Writer.write<uint32_t>(FuncRef->InstCount);

    // Emit IndexOperandHashes sorted from IndexOperandHashMap.
    IndexOperandHashVecType IndexOperandHashes =
        getStableIndexOperandHashes(FuncRef);
    Writer.write<uint32_t>(IndexOperandHashes.size());
    for (auto &IndexOperandHash : IndexOperandHashes) {
      Writer.write<uint32_t>(IndexOperandHash.first.first);
      Writer.write<uint32_t>(IndexOperandHash.first.second);
      Writer.write<stable_hash>(IndexOperandHash.second);
    }
  }
}

void StableFunctionMapRecord::deserialize(const unsigned char *&Ptr) {

  // Read Names.
  auto NumNames =
      endian::readNext<uint32_t, endianness::little, unaligned>(Ptr);
  for (unsigned I = 0; I < NumNames; ++I) {
    std::string Name(reinterpret_cast<const char *>(Ptr));
    Ptr += Name.size() + 1;
    FunctionMap->getIdOrCreateForName(Name);
  }

  // Read StableFunctionEntries.
  auto NumFuncs =
      endian::readNext<uint32_t, endianness::little, unaligned>(Ptr);
  for (unsigned I = 0; I < NumFuncs; ++I) {
    auto Hash =
        endian::readNext<stable_hash, endianness::little, unaligned>(Ptr);
    auto FunctionNameId =
        endian::readNext<uint32_t, endianness::little, unaligned>(Ptr);
    assert(FunctionNameId < IdToName.size() && "FunctionNameId out of range");
    auto ModuleNameId =
        endian::readNext<uint32_t, endianness::little, unaligned>(Ptr);
    assert(ModuleNameId < IdToName.size() && "ModuleNameId out of range");
    auto InstCount =
        endian::readNext<uint32_t, endianness::little, unaligned>(Ptr);

    // Read IndexOperandHashes to build IndexOperandHashMap
    auto NumIndexOperandHashes =
        endian::readNext<uint32_t, endianness::little, unaligned>(Ptr);
    auto IndexOperandHashMap = std::make_unique<IndexOperandHashMapType>();
    for (unsigned J = 0; J < NumIndexOperandHashes; ++J) {
      auto InstIndex =
          endian::readNext<uint32_t, endianness::little, unaligned>(Ptr);
      auto OpndIndex =
          endian::readNext<uint32_t, endianness::little, unaligned>(Ptr);
      auto OpndHash =
          endian::readNext<stable_hash, endianness::little, unaligned>(Ptr);
      assert(InstIndex < InstCount && "InstIndex out of range");

      auto Indices = std::make_pair(InstIndex, OpndIndex);
      (*IndexOperandHashMap)[Indices] = OpndHash;
    }

    // Insert a new StableFunctionEntry into the map.
    auto FuncEntry = std::make_unique<StableFunctionEntry>(
        Hash, FunctionNameId, ModuleNameId, InstCount,
        std::move(IndexOperandHashMap));

    FunctionMap->insert(std::move(FuncEntry));
  }
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
