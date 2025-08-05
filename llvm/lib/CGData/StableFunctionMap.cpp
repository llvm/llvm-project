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
#include "llvm/ADT/SmallSet.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "stable-function-map"

using namespace llvm;

static cl::opt<unsigned>
    GlobalMergingMinMerges("global-merging-min-merges",
                           cl::desc("Minimum number of similar functions with "
                                    "the same hash required for merging."),
                           cl::init(2), cl::Hidden);
static cl::opt<unsigned> GlobalMergingMinInstrs(
    "global-merging-min-instrs",
    cl::desc("The minimum instruction count required when merging functions."),
    cl::init(1), cl::Hidden);
static cl::opt<unsigned> GlobalMergingMaxParams(
    "global-merging-max-params",
    cl::desc(
        "The maximum number of parameters allowed when merging functions."),
    cl::init(std::numeric_limits<unsigned>::max()), cl::Hidden);
static cl::opt<bool> GlobalMergingSkipNoParams(
    "global-merging-skip-no-params",
    cl::desc("Skip merging functions with no parameters."), cl::init(true),
    cl::Hidden);
static cl::opt<double> GlobalMergingInstOverhead(
    "global-merging-inst-overhead",
    cl::desc("The overhead cost associated with each instruction when lowering "
             "to machine instruction."),
    cl::init(1.2), cl::Hidden);
static cl::opt<double> GlobalMergingParamOverhead(
    "global-merging-param-overhead",
    cl::desc("The overhead cost associated with each parameter when merging "
             "functions."),
    cl::init(2.0), cl::Hidden);
static cl::opt<double>
    GlobalMergingCallOverhead("global-merging-call-overhead",
                              cl::desc("The overhead cost associated with each "
                                       "function call when merging functions."),
                              cl::init(1.0), cl::Hidden);
static cl::opt<double> GlobalMergingExtraThreshold(
    "global-merging-extra-threshold",
    cl::desc("An additional cost threshold that must be exceeded for merging "
             "to be considered beneficial."),
    cl::init(0.0), cl::Hidden);

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

static bool isProfitable(
    const SmallVector<std::unique_ptr<StableFunctionMap::StableFunctionEntry>>
        &SFS) {
  unsigned StableFunctionCount = SFS.size();
  if (StableFunctionCount < GlobalMergingMinMerges)
    return false;

  unsigned InstCount = SFS[0]->InstCount;
  if (InstCount < GlobalMergingMinInstrs)
    return false;

  double Cost = 0.0;
  SmallSet<stable_hash, 8> UniqueHashVals;
  for (auto &SF : SFS) {
    UniqueHashVals.clear();
    for (auto &[IndexPair, Hash] : *SF->IndexOperandHashMap)
      UniqueHashVals.insert(Hash);
    unsigned ParamCount = UniqueHashVals.size();
    if (ParamCount > GlobalMergingMaxParams)
      return false;
    // Theoretically, if ParamCount is 0, it results in identical code folding
    // (ICF), which we can skip merging here since the linker already handles
    // ICF. This pass would otherwise introduce unnecessary thunks that are
    // merely direct jumps. However, enabling this could be beneficial depending
    // on downstream passes, so we provide an option for it.
    if (GlobalMergingSkipNoParams && ParamCount == 0)
      return false;
    Cost += ParamCount * GlobalMergingParamOverhead + GlobalMergingCallOverhead;
  }
  Cost += GlobalMergingExtraThreshold;

  double Benefit =
      InstCount * (StableFunctionCount - 1) * GlobalMergingInstOverhead;
  bool Result = Benefit > Cost;
  LLVM_DEBUG(dbgs() << "isProfitable: Hash = " << SFS[0]->Hash << ", "
                    << "StableFunctionCount = " << StableFunctionCount
                    << ", InstCount = " << InstCount
                    << ", Benefit = " << Benefit << ", Cost = " << Cost
                    << ", Result = " << (Result ? "true" : "false") << "\n");
  return Result;
}

void StableFunctionMap::finalize(bool SkipTrim) {
  for (auto It = HashToFuncs.begin(); It != HashToFuncs.end(); ++It) {
    auto &[StableHash, SFS] = *It;

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
      HashToFuncs.erase(It);
      continue;
    }

    if (SkipTrim)
      continue;

    // Trim the index pair that has the same operand hash across
    // stable functions.
    removeIdenticalIndexPair(SFS);

    if (!isProfitable(SFS))
      HashToFuncs.erase(It);
  }

  Finalized = true;
}
