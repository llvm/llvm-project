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
