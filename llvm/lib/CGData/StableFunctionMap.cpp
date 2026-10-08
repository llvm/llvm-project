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
#include "CGDataOptions.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/CGData/StableFunctionMapRecord.h"
#include "llvm/Support/Debug.h"

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
  deserializeLazyLoadingEntries();
  for (auto &[Hash, Funcs] : OtherMap.HashToFuncs) {
    auto &ThisFuncs = HashToFuncs[Hash].Entries;
    for (auto &Func : Funcs.Entries) {
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
    deserializeLazyLoadingEntries();
    size_t Count = 0;
    for (auto &Funcs : HashToFuncs)
      Count += Funcs.second.Entries.size();
    return Count;
  }
  case MergeableFunctionCount: {
    deserializeLazyLoadingEntries();
    size_t Count = 0;
    for (auto &[Hash, Funcs] : HashToFuncs)
      if (Funcs.Entries.size() >= 2)
        Count += Funcs.Entries.size();
    return Count;
  }
  }
  llvm_unreachable("Unhandled size type");
}

const StableFunctionMap::StableFunctionEntries &
StableFunctionMap::at(HashFuncsMapType::key_type FunctionHash) const {
  auto It = HashToFuncs.find(FunctionHash);
  assert(It != HashToFuncs.end() && "FunctionHash not found!");
  if (isLazilyLoaded())
    deserializeLazyLoadingEntry(It);
  return It->second.Entries;
}

void StableFunctionMap::deserializeLazyLoadingEntry(
    HashFuncsMapType::iterator It) const {
  assert(isLazilyLoaded() && "Cannot deserialize non-lazily-loaded map");
  auto &[Hash, Storage] = *It;
  std::call_once(Storage.LazyLoadFlag,
                 [this, HashArg = Hash, &StorageArg = Storage]() {
                   for (auto Offset : StorageArg.Offsets)
                     StableFunctionMapRecord::deserializeEntry(
                         reinterpret_cast<const unsigned char *>(Offset),
                         HashArg, const_cast<StableFunctionMap *>(this));
                 });
}

void StableFunctionMap::deserializeLazyLoadingEntries() const {
  if (!isLazilyLoaded())
    return;
  for (auto It = HashToFuncs.begin(); It != HashToFuncs.end(); ++It)
    deserializeLazyLoadingEntry(It);
}

const StableFunctionMap::HashFuncsMapType &
StableFunctionMap::getFunctionMap() const {
  // Ensure all entries are deserialized before returning the raw map.
  if (isLazilyLoaded())
    deserializeLazyLoadingEntries();
  return HashToFuncs;
}

using ParamLocs = SmallVector<IndexPair>;
static void
removeIdenticalIndexPair(StableFunctionMap::StableFunctionEntries &SFS) {
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

static bool isProfitable(const StableFunctionMap::StableFunctionEntries &SFS) {
  const CGDataOptions &Opts = CGDataOptions::Global;
  unsigned StableFunctionCount = SFS.size();
  if (StableFunctionCount < Opts.global_merging_min_merges)
    return false;

  unsigned InstCount = SFS[0]->InstCount;
  if (InstCount < Opts.global_merging_min_instrs)
    return false;

  double Cost = 0.0;
  SmallSet<stable_hash, 8> UniqueHashVals;
  for (auto &SF : SFS) {
    UniqueHashVals.clear();
    for (auto &[IndexPair, Hash] : *SF->IndexOperandHashMap)
      UniqueHashVals.insert(Hash);
    unsigned ParamCount = UniqueHashVals.size();
    if (ParamCount > Opts.global_merging_max_params)
      return false;
    // Theoretically, if ParamCount is 0, it results in identical code folding
    // (ICF), which we can skip merging here since the linker already handles
    // ICF. This pass would otherwise introduce unnecessary thunks that are
    // merely direct jumps. However, enabling this could be beneficial depending
    // on downstream passes, so we provide an option for it.
    if (Opts.global_merging_skip_no_params && ParamCount == 0)
      return false;
    Cost += ParamCount * Opts.global_merging_param_overhead +
            Opts.global_merging_call_overhead;
  }
  Cost += Opts.global_merging_extra_threshold;

  double Benefit =
      InstCount * (StableFunctionCount - 1) * Opts.global_merging_inst_overhead;
  bool Result = Benefit > Cost;
  LLVM_DEBUG(dbgs() << "isProfitable: Hash = " << SFS[0]->Hash << ", "
                    << "StableFunctionCount = " << StableFunctionCount
                    << ", InstCount = " << InstCount
                    << ", Benefit = " << Benefit << ", Cost = " << Cost
                    << ", Result = " << (Result ? "true" : "false") << "\n");
  return Result;
}

void StableFunctionMap::finalize(bool SkipTrim) {
  deserializeLazyLoadingEntries();
  SmallVector<HashFuncsMapType::iterator> ToDelete;
  for (auto It = HashToFuncs.begin(); It != HashToFuncs.end(); ++It) {
    auto &[StableHash, Storage] = *It;
    auto &SFS = Storage.Entries;

    // Group stable functions by ModuleIdentifier.
    llvm::stable_sort(SFS, [&](const std::unique_ptr<StableFunctionEntry> &L,
                               const std::unique_ptr<StableFunctionEntry> &R) {
      return *getNameForId(L->ModuleNameId) < *getNameForId(R->ModuleNameId);
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
      ToDelete.push_back(It);
      continue;
    }

    if (SkipTrim)
      continue;

    // Trim the index pair that has the same operand hash across
    // stable functions.
    removeIdenticalIndexPair(SFS);

    if (!isProfitable(SFS))
      ToDelete.push_back(It);
  }
  for (auto It : ToDelete)
    HashToFuncs.erase(It);

  Finalized = true;
}
