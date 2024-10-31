//===-- StableFunctionMap.cpp ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This implements the functionality for the StableFunctionMap class, which
// manages the mapping of stable function hashes to their metadata. It includes
// methods for inserting, merging, and finalizing function entries, as well as
// utilities for handling function names and IDs.
//
//===----------------------------------------------------------------------===//

#include "llvm/CGData/StableFunctionMap.h"

#define DEBUG_TYPE "stable-function-map"

using namespace llvm;

unsigned StableFunctionMap::getIdOrCreateForName(StringRef Name) {
  auto It = NameToId.find(Name);
  if (It != NameToId.end())
    return It->second;
  unsigned Id = IdToName.size();
  assert(Id == NameToId.size() && "ID collision");
  IdToName.emplace_back(Name.str());
  NameToId[IdToName.back()] = Id;
  return Id;
}

std::optional<std::string> StableFunctionMap::getNameForId(unsigned Id) const {
  if (Id >= IdToName.size())
    return std::nullopt;
  return IdToName[Id];
}

void StableFunctionMap::insert(const StableFunction &Func) {
  assert(!Finalized && "Cannot insert after finalization");
  auto FuncNameId = getIdOrCreateForName(Func.FunctionName);
  auto ModuleNameId = getIdOrCreateForName(Func.ModuleName);
  auto IndexOperandHashMap = std::make_unique<IndexOperandHashMapType>();
  for (auto &[Index, Hash] : Func.IndexOperandHashes)
    (*IndexOperandHashMap)[Index] = Hash;
  auto FuncEntry = std::make_unique<StableFunctionEntry>(
      Func.Hash, FuncNameId, ModuleNameId, Func.InstCount,
      std::move(IndexOperandHashMap));
  insert(std::move(FuncEntry));
}

void StableFunctionMap::merge(const StableFunctionMap &OtherMap) {
  assert(!Finalized && "Cannot merge after finalization");
  for (auto &[Hash, Funcs] : OtherMap.HashToFuncs) {
    auto &ThisFuncs = HashToFuncs[Hash];
    for (auto &Func : Funcs) {
      auto FuncNameId =
          getIdOrCreateForName(*OtherMap.getNameForId(Func->FunctionNameId));
      auto ModuleNameId =
          getIdOrCreateForName(*OtherMap.getNameForId(Func->ModuleNameId));
      auto ClonedIndexOperandHashMap =
          std::make_unique<IndexOperandHashMapType>(*Func->IndexOperandHashMap);
      ThisFuncs.emplace_back(std::make_unique<StableFunctionEntry>(
          Func->Hash, FuncNameId, ModuleNameId, Func->InstCount,
          std::move(ClonedIndexOperandHashMap)));
    }
  }
}

size_t StableFunctionMap::size(SizeType Type) const {
  switch (Type) {
  case UniqueHashCount:
    return HashToFuncs.size();
  case TotalFunctionCount: {
    size_t Count = 0;
    for (auto &Funcs : HashToFuncs)
      Count += Funcs.second.size();
    return Count;
  }
  case MergeableFunctionCount: {
    size_t Count = 0;
    for (auto &[Hash, Funcs] : HashToFuncs)
      if (Funcs.size() >= 2)
        Count += Funcs.size();
    return Count;
  }
  }
  llvm_unreachable("Unhandled size type");
}

using ParamLocs = SmallVector<IndexPair>;
static void removeIdenticalIndexPair(
    SmallVector<std::unique_ptr<StableFunctionMap::StableFunctionEntry>> &SFS) {
  auto &RSF = SFS[0];
  unsigned StableFunctionCount = SFS.size();

  SmallVector<IndexPair> ToDelete;
  for (auto &[Pair, Hash] : *(RSF->IndexOperandHashMap)) {
    bool Identical = true;
    for (unsigned J = 1; J < StableFunctionCount; ++J) {
      auto &SF = SFS[J];
      const auto &SHash = SF->IndexOperandHashMap->at(Pair);
      if (Hash != SHash) {
        Identical = false;
        break;
      }
    }

    // No need to parameterize them if the hashes are identical across stable
    // functions.
    if (Identical)
      ToDelete.emplace_back(Pair);
  }

  for (auto &Pair : ToDelete)
    for (auto &SF : SFS)
      SF->IndexOperandHashMap->erase(Pair);
}

void StableFunctionMap::finalize() {
  for (auto It = HashToFuncs.begin(); It != HashToFuncs.end(); ++It) {
    auto &[StableHash, SFS] = *It;

    // Group stable functions by ModuleIdentifier.
    std::stable_sort(SFS.begin(), SFS.end(),
                     [&](const std::unique_ptr<StableFunctionEntry> &L,
                         const std::unique_ptr<StableFunctionEntry> &R) {
                       return *getNameForId(L->ModuleNameId) <
                              *getNameForId(R->ModuleNameId);
                     });

    // Consider the first function as the root function.
    auto &RSF = SFS[0];

    bool Invalid = false;
    unsigned StableFunctionCount = SFS.size();
    for (unsigned I = 1; I < StableFunctionCount; ++I) {
      auto &SF = SFS[I];
      assert(RSF->Hash == SF->Hash);
      if (RSF->InstCount != SF->InstCount) {
        Invalid = true;
        break;
      }
      if (RSF->IndexOperandHashMap->size() != SF->IndexOperandHashMap->size()) {
        Invalid = true;
        break;
      }
      for (auto &P : *RSF->IndexOperandHashMap) {
        auto &InstOpndIndex = P.first;
        if (!SF->IndexOperandHashMap->count(InstOpndIndex)) {
          Invalid = true;
          break;
        }
      }
    }
    if (Invalid) {
      HashToFuncs.erase(It);
      continue;
    }

    // Trim the index pair that has the same operand hash across
    // stable functions.
    removeIdenticalIndexPair(SFS);
  }

  Finalized = true;
}
