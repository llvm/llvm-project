//===---------- LazyReexports.cpp - Utilities for lazy reexports ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ExecutionEngine/Orc/LazyObjectLinkingLayer.h"

#include "llvm/ExecutionEngine/Orc/LazyReexports.h"
#include "llvm/ExecutionEngine/Orc/ObjectLinkingLayer.h"
#include "llvm/ExecutionEngine/Orc/RedirectionManager.h"

using namespace llvm;
using namespace llvm::jitlink;

namespace {

constexpr StringRef FnBodySuffix = "$orc_fnbody";

} // anonymous namespace

namespace llvm::orc {

class LazyObjectLinkingLayer::RenamerPlugin
    : public ObjectLinkingLayer::Plugin {
public:
  void modifyPassConfig(MaterializationResponsibility &MR,
                        jitlink::LinkGraph &LG,
                        jitlink::PassConfiguration &Config) override {
    // We need to insert this before the mark-live pass to ensure that we don't
    // delete the bodies (their names won't match the responsibility set until
    // after this pass completes.
    Config.PrePrunePasses.insert(
        Config.PrePrunePasses.begin(),
        [&MR](LinkGraph &G) { return renameFunctionBodies(G, MR); });
  }

  Error notifyFailed(MaterializationResponsibility &MR) override {
    return Error::success();
  }

  Error notifyRemovingResources(JITDylib &JD, ResourceKey K) override {
    return Error::success();
  }

  void notifyTransferringResources(JITDylib &JD, ResourceKey DstKey,
                                   ResourceKey SrcKey) override {}

private:
  static Error renameFunctionBodies(LinkGraph &G,
                                    MaterializationResponsibility &MR) {
    DenseMap<StringRef, NonOwningSymbolStringPtr> SymsToRename;
    for (auto &[Name, Flags] : MR.getSymbols())
      if ((*Name).ends_with(FnBodySuffix))
        SymsToRename[(*Name).drop_back(FnBodySuffix.size())] =
            NonOwningSymbolStringPtr(Name);

    for (auto *Sym : G.defined_symbols()) {
      if (!Sym->hasName())
        continue;
      auto I = SymsToRename.find(*Sym->getName());
      if (I == SymsToRename.end())
        continue;
      Sym->setName(G.intern(G.allocateName(*I->second)));
    }

    return Error::success();
  }
};

LazyObjectLinkingLayer::LazyObjectLinkingLayer(ObjectLinkingLayer &BaseLayer,
                                               LazyReexportsManager &LRMgr)
    : ObjectLayer(BaseLayer.getExecutionSession()), BaseLayer(BaseLayer),
      LRMgr(LRMgr) {
  BaseLayer.addPlugin(std::make_unique<RenamerPlugin>());
}

Error LazyObjectLinkingLayer::add(ResourceTrackerSP RT,
                                  std::unique_ptr<MemoryBuffer> O,
                                  MaterializationUnit::Interface I) {

  // Object files with initializer symbols can't be lazy.
  if (I.InitSymbol)
    return BaseLayer.add(std::move(RT), std::move(O), std::move(I));

  auto &ES = getExecutionSession();
  SymbolAliasMap LazySymbols;
  for (auto &[Name, Flags] : I.SymbolFlags)
    if (Flags.isCallable())
      LazySymbols[Name] = {ES.intern((*Name + FnBodySuffix).str()), Flags};

  for (auto &[Name, AI] : LazySymbols) {
    I.SymbolFlags.erase(Name);
    I.SymbolFlags[AI.Aliasee] = AI.AliasFlags;
  }

  if (auto Err = BaseLayer.add(RT, std::move(O), std::move(I)))
    return Err;

  auto &JD = RT->getJITDylib();
  return JD.define(lazyReexports(LRMgr, std::move(LazySymbols)), std::move(RT));
}

void LazyObjectLinkingLayer::emit(
    std::unique_ptr<MaterializationResponsibility> MR,
    std::unique_ptr<MemoryBuffer> Obj) {
  return BaseLayer.emit(std::move(MR), std::move(Obj));
}

} // namespace llvm::orc
