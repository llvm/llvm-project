//===-- JITLinkRedirectableSymbolManager.cpp - JITLink redirection in Orc -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ExecutionEngine/Orc/JITLinkRedirectableSymbolManager.h"
#include "llvm/ExecutionEngine/Orc/Core.h"

#define DEBUG_TYPE "orc"

using namespace llvm;
using namespace llvm::orc;

void JITLinkRedirectableSymbolManager::emitRedirectableSymbols(
    std::unique_ptr<MaterializationResponsibility> R,
    const SymbolAddrMap &InitialDests) {
  auto &ES = ObjLinkingLayer.getExecutionSession();
  std::unique_lock<std::mutex> Lock(Mutex);
  if (GetNumAvailableStubs() < InitialDests.size())
    if (auto Err = grow(InitialDests.size() - GetNumAvailableStubs())) {
      ES.reportError(std::move(Err));
      R->failMaterialization();
      return;
    }

  JITDylib &TargetJD = R->getTargetJITDylib();
  SymbolMap NewSymbolDefs;
  std::vector<SymbolStringPtr> Symbols;
  for (auto &[K, V] : InitialDests) {
    StubHandle StubID = AvailableStubs.back();
    if (SymbolToStubs[&TargetJD].count(K)) {
      ES.reportError(make_error<StringError>(
          "Tried to create duplicate redirectable symbols",
          inconvertibleErrorCode()));
      R->failMaterialization();
      return;
    }
    SymbolToStubs[&TargetJD][K] = StubID;
    NewSymbolDefs[K] = JumpStubs[StubID];
    NewSymbolDefs[K].setFlags(V.getFlags());
    Symbols.push_back(K);
    AvailableStubs.pop_back();
  }

  // FIXME: when this fails we can return stubs to the pool
  if (auto Err = redirectInner(TargetJD, InitialDests)) {
    ES.reportError(std::move(Err));
    R->failMaterialization();
    return;
  }

  // FIXME: return stubs to the pool here too.
  if (auto Err = R->replace(absoluteSymbols(NewSymbolDefs))) {
    ES.reportError(std::move(Err));
    R->failMaterialization();
    return;
  }

  // FIXME: return stubs to the pool here too.
  if (auto Err = R->withResourceKeyDo([&](ResourceKey Key) {
        TrackedResources[Key].insert(TrackedResources[Key].end(),
                                     Symbols.begin(), Symbols.end());
      })) {
    ES.reportError(std::move(Err));
    R->failMaterialization();
    return;
  }
}

Error JITLinkRedirectableSymbolManager::redirect(
    JITDylib &TargetJD, const SymbolAddrMap &NewDests) {
  std::unique_lock<std::mutex> Lock(Mutex);
  return redirectInner(TargetJD, NewDests);
}

Error JITLinkRedirectableSymbolManager::redirectInner(
    JITDylib &TargetJD, const SymbolAddrMap &NewDests) {
  std::vector<tpctypes::PointerWrite> PtrWrites;
  for (auto &[K, V] : NewDests) {
    if (!SymbolToStubs[&TargetJD].count(K))
      return make_error<StringError>(
          "Tried to redirect non-existent redirectalbe symbol",
          inconvertibleErrorCode());
    StubHandle StubID = SymbolToStubs[&TargetJD].at(K);
    PtrWrites.push_back({StubPointers[StubID].getAddress(), V.getAddress()});
  }
  return ObjLinkingLayer.getExecutionSession()
      .getExecutorProcessControl()
      .getMemoryAccess()
      .writePointers(PtrWrites);
}

Error JITLinkRedirectableSymbolManager::grow(unsigned Need) {
  unsigned OldSize = JumpStubs.size();
  unsigned NumNewStubs = alignTo(Need, StubBlockSize);
  unsigned NewSize = OldSize + NumNewStubs;

  JumpStubs.resize(NewSize);
  StubPointers.resize(NewSize);
  AvailableStubs.reserve(NewSize);

  SymbolLookupSet LookupSymbols;
  DenseMap<SymbolStringPtr, ExecutorSymbolDef *> NewDefsMap;

  auto &ES = ObjLinkingLayer.getExecutionSession();
  Triple TT = ES.getTargetTriple();
  auto G = std::make_unique<jitlink::LinkGraph>(
      "<INDIRECT STUBS>", TT, TT.isArch64Bit() ? 8 : 4,
      TT.isLittleEndian() ? endianness::little : endianness::big,
      jitlink::getGenericEdgeKindName);
  auto &PointerSection =
      G->createSection(StubPtrTableName, MemProt::Write | MemProt::Read);
  auto &StubsSection =
      G->createSection(JumpStubTableName, MemProt::Exec | MemProt::Read);

  // FIXME: We can batch the stubs into one block and use address to access them
  for (size_t I = OldSize; I < NewSize; I++) {
    auto &Pointer = AnonymousPtrCreator(*G, PointerSection, nullptr, 0);

    StringRef PtrSymName = StubPtrSymbolName(I);
    Pointer.setName(PtrSymName);
    Pointer.setScope(jitlink::Scope::Default);
    LookupSymbols.add(ES.intern(PtrSymName));
    NewDefsMap[ES.intern(PtrSymName)] = &StubPointers[I];

    auto &Stub = PtrJumpStubCreator(*G, StubsSection, Pointer);

    StringRef JumpStubSymName = JumpStubSymbolName(I);
    Stub.setName(JumpStubSymName);
    Stub.setScope(jitlink::Scope::Default);
    LookupSymbols.add(ES.intern(JumpStubSymName));
    NewDefsMap[ES.intern(JumpStubSymName)] = &JumpStubs[I];
  }

  if (auto Err = ObjLinkingLayer.add(JD, std::move(G)))
    return Err;

  auto LookupResult = ES.lookup(makeJITDylibSearchOrder(&JD), LookupSymbols);
  if (auto Err = LookupResult.takeError())
    return Err;

  for (auto &[K, V] : *LookupResult)
    *NewDefsMap.at(K) = V;

  for (size_t I = OldSize; I < NewSize; I++)
    AvailableStubs.push_back(I);

  return Error::success();
}

Error JITLinkRedirectableSymbolManager::handleRemoveResources(
    JITDylib &TargetJD, ResourceKey K) {
  std::unique_lock<std::mutex> Lock(Mutex);
  for (auto &Symbol : TrackedResources[K]) {
    if (!SymbolToStubs[&TargetJD].count(Symbol))
      return make_error<StringError>(
          "Tried to remove non-existent redirectable symbol",
          inconvertibleErrorCode());
    AvailableStubs.push_back(SymbolToStubs[&TargetJD].at(Symbol));
    SymbolToStubs[&TargetJD].erase(Symbol);
    if (SymbolToStubs[&TargetJD].empty())
      SymbolToStubs.erase(&TargetJD);
  }
  TrackedResources.erase(K);

  return Error::success();
}

void JITLinkRedirectableSymbolManager::handleTransferResources(
    JITDylib &TargetJD, ResourceKey DstK, ResourceKey SrcK) {
  std::unique_lock<std::mutex> Lock(Mutex);
  TrackedResources[DstK].insert(TrackedResources[DstK].end(),
                                TrackedResources[SrcK].begin(),
                                TrackedResources[SrcK].end());
  TrackedResources.erase(SrcK);
}
