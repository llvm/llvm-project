//===-- StableFunctionMap.cpp ---------------------------------------------===//
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

#include "llvm/CGData/StableFunctionMap.h"

#define DEBUG_TYPE "stable-function-map"

using namespace llvm;

unsigned StableFunctionMap::getIdOrCreateForName(StringRef Name) {
  auto It = NameToId.find(Name);
  if (It != NameToId.end()) {
    return It->second;
  } else {
    unsigned Id = IdToName.size();
    assert(Id == NameToId.size() && "ID collision");
    IdToName.emplace_back(Name.str());
    NameToId[IdToName.back()] = Id;
    return Id;
  }
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
  return 0;
}

using ParamLocs = SmallVector<IndexPair>;
static void removeIdenticalIndexPair(
    SmallVector<std::unique_ptr<StableFunctionEntry>> &SFS) {
  auto &RSF = SFS[0];
  unsigned StableFunctionCount = SFS.size();

  SmallVector<IndexPair> ToDelete;
  for (auto &[Pair, Hash] : *(RSF->IndexOperandHashMap)) {
    bool Identical = true;
    for (unsigned J = 1; J < StableFunctionCount; ++J) {
      auto &SF = SFS[J];
      assert(SF->IndexOperandHashMap->count(Pair));
      auto SHash = (*SF->IndexOperandHashMap)[Pair];
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

bool StableFunctionMap::finalize() {
  // TODO: Add an option for finalization.
  return false;

  bool Changed = false;

  for (auto It = HashToFuncs.begin(); It != HashToFuncs.end(); ++It) {
    auto &[StableHash, SFS] = *It;
    // No interest if there is no common stable function globally.
    if (SFS.size() < 2) {
      HashToFuncs.erase(It);
      Changed = true;
      continue;
    }

    // Group stable functions by ModuleIdentifier.
    std::stable_sort(SFS.begin(), SFS.end(),
                     [&](const std::unique_ptr<StableFunctionEntry> &L,
                         const std::unique_ptr<StableFunctionEntry> &R) {
                       return *getNameForId(L->ModuleNameId) <
                              *getNameForId(R->ModuleNameId);
                     });

    // Consider the first function as the root function.
    auto &RSF = SFS[0];

    bool IsValid = true;
    unsigned StableFunctionCount = SFS.size();
    for (unsigned I = 1; I < StableFunctionCount; ++I) {
      auto &SF = SFS[I];
      assert(RSF->Hash == SF->Hash);
      if (RSF->InstCount != SF->InstCount) {
        IsValid = false;
        break;
      }
      if (RSF->IndexOperandHashMap->size() != SF->IndexOperandHashMap->size()) {
        IsValid = false;
        break;
      }
      for (auto &P : *RSF->IndexOperandHashMap) {
        auto &InstOpndIndex = P.first;
        if (!SF->IndexOperandHashMap->count(InstOpndIndex)) {
          IsValid = false;
          break;
        }
      }
    }
    if (!IsValid) {
      HashToFuncs.erase(It);
      Changed = true;
      continue;
    }

    // Trim the index pair that has the same operand hash across
    // stable functions.
    removeIdenticalIndexPair(SFS);
  }
  Finalized = true;

  return Changed;
}
