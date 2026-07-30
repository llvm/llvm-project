//===----- CompileOnDemandLayer.cpp - Lazily emit IR on first call --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ExecutionEngine/Orc/CompileOnDemandLayer.h"
#include "llvm/ExecutionEngine/Orc/Layer.h"
#include "llvm/IR/Module.h"

using namespace llvm;
using namespace llvm::orc;

CompileOnDemandLayer::CompileOnDemandLayer(
    ExecutionSession &ES, IRLayer &BaseLayer, LazyCallThroughManager &LCTMgr,
    IndirectStubsManagerBuilder BuildIndirectStubsManager)
    : IRLayer(ES, BaseLayer.getManglingOptions()), BaseLayer(BaseLayer),
      LCTMgr(LCTMgr),
      BuildIndirectStubsManager(std::move(BuildIndirectStubsManager)) {}

void CompileOnDemandLayer::setImplMap(ImplSymbolMap *Imp) {
  this->AliaseeImpls = Imp;
}

void CompileOnDemandLayer::emit(
    std::unique_ptr<MaterializationResponsibility> R, ThreadSafeModule TSM) {
  assert(TSM && "Null module");

  auto &ES = getExecutionSession();

  // Sort the callables and non-callables, build re-exports and lodge the
  // actual module with the implementation dylib.
  auto &PDR = getPerDylibResources(R->getTargetJITDylib());

  SymbolAliasMap NonCallables;
  SymbolAliasMap Callables;

  for (auto &KV : R->getSymbols()) {
    auto &Name = KV.first;
    auto &Flags = KV.second;
    if (Flags.isCallable())
      Callables[Name] = SymbolAliasMapEntry(Name, Flags);
    else
      NonCallables[Name] = SymbolAliasMapEntry(Name, Flags);
  }

  // Lodge symbols with the implementation dylib.
  if (auto Err = PDR.getImplDylib().define(
          std::make_unique<BasicIRLayerMaterializationUnit>(
              BaseLayer, *getManglingOptions(), std::move(TSM)))) {
    ES.reportError(std::move(Err));
    R->failMaterialization();
    return;
  }

  if (!NonCallables.empty())
    if (auto Err =
            R->replace(reexports(PDR.getImplDylib(), std::move(NonCallables),
                                 JITDylibLookupFlags::MatchAllSymbols))) {
      getExecutionSession().reportError(std::move(Err));
      R->failMaterialization();
      return;
    }
  if (!Callables.empty()) {
    if (auto Err = R->replace(
            lazyReexports(LCTMgr, PDR.getISManager(), PDR.getImplDylib(),
                          std::move(Callables), AliaseeImpls))) {
      getExecutionSession().reportError(std::move(Err));
      R->failMaterialization();
      return;
    }
  }
}

CompileOnDemandLayer::PerDylibResources &
CompileOnDemandLayer::getPerDylibResources(JITDylib &TargetD) {
  std::lock_guard<std::mutex> Lock(CODLayerMutex);

  auto I = DylibResources.find(&TargetD);
  if (I == DylibResources.end()) {
    auto &ImplD =
        getExecutionSession().createBareJITDylib(TargetD.getName() + ".impl");
    JITDylibSearchOrder NewLinkOrder;
    TargetD.withLinkOrderDo([&](const JITDylibSearchOrder &TargetLinkOrder) {
      NewLinkOrder = TargetLinkOrder;
    });

    assert(!NewLinkOrder.empty() && NewLinkOrder.front().first == &TargetD &&
           NewLinkOrder.front().second ==
               JITDylibLookupFlags::MatchAllSymbols &&
           "TargetD must be at the front of its own search order and match "
           "non-exported symbol");
    NewLinkOrder.insert(std::next(NewLinkOrder.begin()),
                        {&ImplD, JITDylibLookupFlags::MatchAllSymbols});
    ImplD.setLinkOrder(NewLinkOrder, false);
    TargetD.setLinkOrder(std::move(NewLinkOrder), false);

    PerDylibResources PDR(ImplD, BuildIndirectStubsManager());
    I = DylibResources.insert(std::make_pair(&TargetD, std::move(PDR))).first;
  }

  return I->second;
}
