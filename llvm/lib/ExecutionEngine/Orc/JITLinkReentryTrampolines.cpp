//===----- JITLinkReentryTrampolines.cpp -- JITLink-based trampoline- -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ExecutionEngine/Orc/JITLinkReentryTrampolines.h"

#include "llvm/ExecutionEngine/JITLink/aarch64.h"
#include "llvm/ExecutionEngine/JITLink/x86_64.h"
#include "llvm/ExecutionEngine/Orc/ObjectLinkingLayer.h"

#include <memory>

#define DEBUG_TYPE "orc"

using namespace llvm;
using namespace llvm::jitlink;

namespace {
constexpr StringRef ReentryFnName = "__orc_rt_reenter";
constexpr StringRef ReentrySectionName = "__orc_stubs";
} // namespace

namespace llvm::orc {

class JITLinkReentryTrampolines::TrampolineAddrScraperPlugin
    : public ObjectLinkingLayer::Plugin {
public:
  void modifyPassConfig(MaterializationResponsibility &MR,
                        jitlink::LinkGraph &G,
                        jitlink::PassConfiguration &Config) override {
    Config.PreFixupPasses.push_back(
        [this](LinkGraph &G) { return recordTrampolineAddrs(G); });
  }

  Error notifyFailed(MaterializationResponsibility &MR) override {
    return Error::success();
  }

  Error notifyRemovingResources(JITDylib &JD, ResourceKey K) override {
    return Error::success();
  }

  void notifyTransferringResources(JITDylib &JD, ResourceKey DstKey,
                                   ResourceKey SrcKey) override {}

  void registerGraph(LinkGraph &G,
                     std::shared_ptr<std::vector<ExecutorSymbolDef>> Addrs) {
    std::lock_guard<std::mutex> Lock(M);
    assert(!PendingAddrs.count(&G) && "Duplicate registration");
    PendingAddrs[&G] = std::move(Addrs);
  }

  Error recordTrampolineAddrs(LinkGraph &G) {
    std::shared_ptr<std::vector<ExecutorSymbolDef>> Addrs;
    {
      std::lock_guard<std::mutex> Lock(M);
      auto I = PendingAddrs.find(&G);
      if (I == PendingAddrs.end())
        return Error::success();
      Addrs = std::move(I->second);
      PendingAddrs.erase(I);
    }

    auto *Sec = G.findSectionByName(ReentrySectionName);
    assert(Sec && "Reentry graph missing reentry section");
    assert(!Sec->empty() && "Reentry graph is empty");

    for (auto *Sym : Sec->symbols())
      if (!Sym->hasName())
        Addrs->push_back({Sym->getAddress(), JITSymbolFlags()});

    return Error::success();
  }

private:
  std::mutex M;
  DenseMap<LinkGraph *, std::shared_ptr<std::vector<ExecutorSymbolDef>>>
      PendingAddrs;
};

Expected<std::unique_ptr<JITLinkReentryTrampolines>>
JITLinkReentryTrampolines::Create(ObjectLinkingLayer &ObjLinkingLayer) {

  EmitTrampolineFn EmitTrampoline;

  const auto &TT = ObjLinkingLayer.getExecutionSession().getTargetTriple();
  switch (TT.getArch()) {
  case Triple::aarch64:
    EmitTrampoline = aarch64::createAnonymousReentryTrampoline;
    break;
  case Triple::x86_64:
    EmitTrampoline = x86_64::createAnonymousReentryTrampoline;
    break;
  default:
    return make_error<StringError>("JITLinkReentryTrampolines: architecture " +
				   TT.getArchName() + " not supported",
                                   inconvertibleErrorCode());
  }

  return std::make_unique<JITLinkReentryTrampolines>(ObjLinkingLayer,
                                                     std::move(EmitTrampoline));
}

JITLinkReentryTrampolines::JITLinkReentryTrampolines(
    ObjectLinkingLayer &ObjLinkingLayer, EmitTrampolineFn EmitTrampoline)
    : ObjLinkingLayer(ObjLinkingLayer),
      EmitTrampoline(std::move(EmitTrampoline)) {
  auto TAS = std::make_shared<TrampolineAddrScraperPlugin>();
  TrampolineAddrScraper = TAS.get();
  ObjLinkingLayer.addPlugin(std::move(TAS));
}

void JITLinkReentryTrampolines::emit(ResourceTrackerSP RT,
                                     size_t NumTrampolines,
                                     OnTrampolinesReadyFn OnTrampolinesReady) {

  if (NumTrampolines == 0)
    return OnTrampolinesReady(std::vector<ExecutorSymbolDef>());

  JITDylibSP JD(&RT->getJITDylib());
  auto &ES = ObjLinkingLayer.getExecutionSession();
  Triple TT = ES.getTargetTriple();

  auto ReentryGraphSym =
      ES.intern(("__orc_reentry_graph_#" + Twine(++ReentryGraphIdx)).str());

  auto G = std::make_unique<jitlink::LinkGraph>(
      (*ReentryGraphSym).str(), ES.getSymbolStringPool(), TT,
      TT.isArch64Bit() ? 8 : 4,
      TT.isLittleEndian() ? endianness::little : endianness::big,
      jitlink::getGenericEdgeKindName);

  auto &ReentryFnSym = G->addExternalSymbol(ReentryFnName, 0, false);

  auto &ReentrySection =
      G->createSection(ReentrySectionName, MemProt::Exec | MemProt::Read);

  for (size_t I = 0; I != NumTrampolines; ++I)
    EmitTrampoline(*G, ReentrySection, ReentryFnSym).setLive(true);

  auto &FirstBlock = **ReentrySection.blocks().begin();
  G->addDefinedSymbol(FirstBlock, 0, *ReentryGraphSym, FirstBlock.getSize(),
                      Linkage::Strong, Scope::SideEffectsOnly, true, true);

  auto TrampolineAddrs = std::make_shared<std::vector<ExecutorSymbolDef>>();
  TrampolineAddrScraper->registerGraph(*G, TrampolineAddrs);

  // Add Graph via object linking layer.
  if (auto Err = ObjLinkingLayer.add(std::move(RT), std::move(G)))
    return OnTrampolinesReady(std::move(Err));

  // Trigger graph emission.
  ES.lookup(
      LookupKind::Static, {{JD.get(), JITDylibLookupFlags::MatchAllSymbols}},
      SymbolLookupSet(ReentryGraphSym,
                      SymbolLookupFlags::WeaklyReferencedSymbol),
      SymbolState::Ready,
      [OnTrampolinesReady = std::move(OnTrampolinesReady),
       TrampolineAddrs =
           std::move(TrampolineAddrs)](Expected<SymbolMap> Result) mutable {
        if (Result)
          OnTrampolinesReady(std::move(*TrampolineAddrs));
        else
          OnTrampolinesReady(Result.takeError());
      },
      NoDependenciesToRegister);
}

Expected<std::unique_ptr<LazyReexportsManager>>
createJITLinkLazyReexportsManager(ObjectLinkingLayer &ObjLinkingLayer,
                                  RedirectableSymbolManager &RSMgr,
                                  JITDylib &PlatformJD) {
  auto JLT = JITLinkReentryTrampolines::Create(ObjLinkingLayer);
  if (!JLT)
    return JLT.takeError();

  return LazyReexportsManager::Create(
      [JLT = std::move(*JLT)](ResourceTrackerSP RT, size_t NumTrampolines,
                              LazyReexportsManager::OnTrampolinesReadyFn
                                  OnTrampolinesReady) mutable {
        JLT->emit(std::move(RT), NumTrampolines, std::move(OnTrampolinesReady));
      },
      RSMgr, PlatformJD);
}

} // namespace llvm::orc
