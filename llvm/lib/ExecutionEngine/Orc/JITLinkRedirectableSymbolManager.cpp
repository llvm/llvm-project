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

namespace {
constexpr StringRef JumpStubSectionName = "__orc_stubs";
constexpr StringRef StubPtrSectionName = "__orc_stub_ptrs";
constexpr StringRef StubSuffix = "$__stub_ptr";
} // namespace

void JITLinkRedirectableSymbolManager::emitRedirectableSymbols(
    std::unique_ptr<MaterializationResponsibility> R, SymbolMap InitialDests) {

  auto &ES = ObjLinkingLayer.getExecutionSession();
  Triple TT = ES.getTargetTriple();

  auto G = std::make_unique<jitlink::LinkGraph>(
      ("<indirect stubs graph #" + Twine(++StubGraphIdx) + ">").str(),
      ES.getSymbolStringPool(), TT, TT.isArch64Bit() ? 8 : 4,
      TT.isLittleEndian() ? endianness::little : endianness::big,
      jitlink::getGenericEdgeKindName);
  auto &PointerSection =
      G->createSection(StubPtrSectionName, MemProt::Write | MemProt::Read);
  auto &StubsSection =
      G->createSection(JumpStubSectionName, MemProt::Exec | MemProt::Read);

  SymbolFlagsMap NewSymbols;
  for (auto &[Name, Def] : InitialDests) {
    jitlink::Symbol *TargetSym = nullptr;
    if (Def.getAddress())
      TargetSym = &G->addAbsoluteSymbol(
          G->allocateName(*Name + "$__init_tgt"), Def.getAddress(), 0,
          jitlink::Linkage::Strong, jitlink::Scope::Local, false);

    auto PtrName = ES.intern((*Name + StubSuffix).str());
    auto &Ptr = AnonymousPtrCreator(*G, PointerSection, TargetSym, 0);
    Ptr.setName(PtrName);
    Ptr.setScope(jitlink::Scope::Hidden);
    auto &Stub = PtrJumpStubCreator(*G, StubsSection, Ptr);
    Stub.setName(Name);
    Stub.setScope(jitlink::Scope::Default);
    NewSymbols[std::move(PtrName)] = JITSymbolFlags();
  }

  // Try to claim responsibility for the new stub symbols.
  if (auto Err = R->defineMaterializing(std::move(NewSymbols))) {
    ES.reportError(std::move(Err));
    return R->failMaterialization();
  }

  ObjLinkingLayer.emit(std::move(R), std::move(G));
}

Error JITLinkRedirectableSymbolManager::redirect(JITDylib &JD,
                                                 const SymbolMap &NewDests) {
  auto &ES = ObjLinkingLayer.getExecutionSession();
  SymbolLookupSet LS;
  DenseMap<NonOwningSymbolStringPtr, SymbolStringPtr> PtrToStub;
  for (auto &[StubName, Sym] : NewDests) {
    auto PtrName = ES.intern((*StubName + StubSuffix).str());
    PtrToStub[NonOwningSymbolStringPtr(PtrName)] = StubName;
    LS.add(std::move(PtrName));
  }
  auto PtrSyms =
      ES.lookup({{&JD, JITDylibLookupFlags::MatchAllSymbols}}, std::move(LS));
  if (!PtrSyms)
    return PtrSyms.takeError();

  std::vector<tpctypes::PointerWrite> PtrWrites;
  for (auto &[PtrName, PtrSym] : *PtrSyms) {
    auto DestSymI = NewDests.find(PtrToStub[NonOwningSymbolStringPtr(PtrName)]);
    assert(DestSymI != NewDests.end() && "Bad ptr -> stub mapping");
    auto &DestSym = DestSymI->second;
    PtrWrites.push_back({PtrSym.getAddress(), DestSym.getAddress()});
  }

  return ObjLinkingLayer.getExecutionSession()
      .getExecutorProcessControl()
      .getMemoryAccess()
      .writePointers(PtrWrites);
}
