//===----- LinkGraphLinkingLayer.cpp - JITLink backed ORC ObjectLayer -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ExecutionEngine/Orc/LinkGraphLinkingLayer.h"
#include "llvm/ExecutionEngine/JITLink/EHFrameSupport.h"
#include "llvm/ExecutionEngine/JITLink/aarch32.h"
#include "llvm/ExecutionEngine/Orc/DebugUtils.h"
#include "llvm/ExecutionEngine/Orc/Shared/ObjectFormats.h"
#include "llvm/Support/MemoryBuffer.h"

#define DEBUG_TYPE "orc"

using namespace llvm;
using namespace llvm::jitlink;
using namespace llvm::orc;

namespace {

ExecutorAddr getJITSymbolPtrForSymbol(Symbol &Sym, const Triple &TT) {
  switch (TT.getArch()) {
  case Triple::arm:
  case Triple::armeb:
  case Triple::thumb:
  case Triple::thumbeb:
    if (hasTargetFlags(Sym, aarch32::ThumbSymbol)) {
      // Set LSB to indicate thumb target
      assert(Sym.isCallable() && "Only callable symbols can have thumb flag");
      assert((Sym.getAddress().getValue() & 0x01) == 0 && "LSB is clear");
      return Sym.getAddress() + 0x01;
    }
    return Sym.getAddress();
  default:
    return Sym.getAddress();
  }
}

} // end anonymous namespace

namespace llvm {
namespace orc {

class LinkGraphLinkingLayer::JITLinkCtx final : public JITLinkContext {
public:
  JITLinkCtx(LinkGraphLinkingLayer &Layer,
             std::unique_ptr<MaterializationResponsibility> MR,
             std::unique_ptr<MemoryBuffer> ObjBuffer)
      : JITLinkContext(&MR->getTargetJITDylib()), Layer(Layer),
        MR(std::move(MR)), ObjBuffer(std::move(ObjBuffer)) {
    std::lock_guard<std::mutex> Lock(Layer.LayerMutex);
    Plugins = Layer.Plugins;
  }

  ~JITLinkCtx() {
    // If there is an object buffer return function then use it to
    // return ownership of the buffer.
    if (Layer.ReturnObjectBuffer && ObjBuffer)
      Layer.ReturnObjectBuffer(std::move(ObjBuffer));
  }

  JITLinkMemoryManager &getMemoryManager() override { return Layer.MemMgr; }

  void notifyMaterializing(LinkGraph &G) {
    for (auto &P : Plugins)
      P->notifyMaterializing(*MR, G, *this,
                             ObjBuffer ? ObjBuffer->getMemBufferRef()
                                       : MemoryBufferRef());
  }

  void notifyFailed(Error Err) override {
    for (auto &P : Plugins)
      Err = joinErrors(std::move(Err), P->notifyFailed(*MR));
    Layer.getExecutionSession().reportError(std::move(Err));
    MR->failMaterialization();
  }

  void lookup(const LookupMap &Symbols,
              std::unique_ptr<JITLinkAsyncLookupContinuation> LC) override {

    JITDylibSearchOrder LinkOrder;
    MR->getTargetJITDylib().withLinkOrderDo(
        [&](const JITDylibSearchOrder &LO) { LinkOrder = LO; });

    auto &ES = Layer.getExecutionSession();

    SymbolLookupSet LookupSet;
    for (auto &KV : Symbols) {
      orc::SymbolLookupFlags LookupFlags;
      switch (KV.second) {
      case jitlink::SymbolLookupFlags::RequiredSymbol:
        LookupFlags = orc::SymbolLookupFlags::RequiredSymbol;
        break;
      case jitlink::SymbolLookupFlags::WeaklyReferencedSymbol:
        LookupFlags = orc::SymbolLookupFlags::WeaklyReferencedSymbol;
        break;
      }
      LookupSet.add(KV.first, LookupFlags);
    }

    // OnResolve -- De-intern the symbols and pass the result to the linker.
    auto OnResolve = [LookupContinuation =
                          std::move(LC)](Expected<SymbolMap> Result) mutable {
      if (!Result)
        LookupContinuation->run(Result.takeError());
      else {
        AsyncLookupResult LR;
        for (auto &KV : *Result)
          LR[KV.first] = KV.second;
        LookupContinuation->run(std::move(LR));
      }
    };

    ES.lookup(LookupKind::Static, LinkOrder, std::move(LookupSet),
              SymbolState::Resolved, std::move(OnResolve),
              [this](const SymbolDependenceMap &Deps) {
                // Translate LookupDeps map to SymbolSourceJD.
                for (auto &[DepJD, Deps] : Deps)
                  for (auto &DepSym : Deps)
                    SymbolSourceJDs[NonOwningSymbolStringPtr(DepSym)] = DepJD;
              });
  }

  Error notifyResolved(LinkGraph &G) override {

    SymbolFlagsMap ExtraSymbolsToClaim;
    bool AutoClaim = Layer.AutoClaimObjectSymbols;

    SymbolMap InternedResult;
    for (auto *Sym : G.defined_symbols())
      if (Sym->getScope() < Scope::SideEffectsOnly) {
        auto Ptr = getJITSymbolPtrForSymbol(*Sym, G.getTargetTriple());
        auto Flags = getJITSymbolFlagsForSymbol(*Sym);
        InternedResult[Sym->getName()] = {Ptr, Flags};
        if (AutoClaim && !MR->getSymbols().count(Sym->getName())) {
          assert(!ExtraSymbolsToClaim.count(Sym->getName()) &&
                 "Duplicate symbol to claim?");
          ExtraSymbolsToClaim[Sym->getName()] = Flags;
        }
      }

    for (auto *Sym : G.absolute_symbols())
      if (Sym->getScope() < Scope::SideEffectsOnly) {
        auto Ptr = getJITSymbolPtrForSymbol(*Sym, G.getTargetTriple());
        auto Flags = getJITSymbolFlagsForSymbol(*Sym);
        InternedResult[Sym->getName()] = {Ptr, Flags};
        if (AutoClaim && !MR->getSymbols().count(Sym->getName())) {
          assert(!ExtraSymbolsToClaim.count(Sym->getName()) &&
                 "Duplicate symbol to claim?");
          ExtraSymbolsToClaim[Sym->getName()] = Flags;
        }
      }

    if (!ExtraSymbolsToClaim.empty())
      if (auto Err = MR->defineMaterializing(ExtraSymbolsToClaim))
        return Err;

    {

      // Check that InternedResult matches up with MR->getSymbols(), overriding
      // flags if requested.
      // This guards against faulty transformations / compilers / object caches.

      // First check that there aren't any missing symbols.
      size_t NumMaterializationSideEffectsOnlySymbols = 0;
      SymbolNameVector MissingSymbols;
      for (auto &[Sym, Flags] : MR->getSymbols()) {

        auto I = InternedResult.find(Sym);

        // If this is a materialization-side-effects only symbol then bump
        // the counter and remove in from the result, otherwise make sure that
        // it's defined.
        if (Flags.hasMaterializationSideEffectsOnly())
          ++NumMaterializationSideEffectsOnlySymbols;
        else if (I == InternedResult.end())
          MissingSymbols.push_back(Sym);
        else if (Layer.OverrideObjectFlags)
          I->second.setFlags(Flags);
      }

      // If there were missing symbols then report the error.
      if (!MissingSymbols.empty())
        return make_error<MissingSymbolDefinitions>(
            Layer.getExecutionSession().getSymbolStringPool(), G.getName(),
            std::move(MissingSymbols));

      // If there are more definitions than expected, add them to the
      // ExtraSymbols vector.
      SymbolNameVector ExtraSymbols;
      if (InternedResult.size() >
          MR->getSymbols().size() - NumMaterializationSideEffectsOnlySymbols) {
        for (auto &KV : InternedResult)
          if (!MR->getSymbols().count(KV.first))
            ExtraSymbols.push_back(KV.first);
      }

      // If there were extra definitions then report the error.
      if (!ExtraSymbols.empty())
        return make_error<UnexpectedSymbolDefinitions>(
            Layer.getExecutionSession().getSymbolStringPool(), G.getName(),
            std::move(ExtraSymbols));
    }

    if (auto Err = MR->notifyResolved(InternedResult))
      return Err;

    notifyLoaded();
    return Error::success();
  }

  void notifyFinalized(JITLinkMemoryManager::FinalizedAlloc A) override {
    if (auto Err = notifyEmitted(std::move(A))) {
      Layer.getExecutionSession().reportError(std::move(Err));
      MR->failMaterialization();
      return;
    }

    if (auto Err = MR->notifyEmitted(SymbolDepGroups)) {
      Layer.getExecutionSession().reportError(std::move(Err));
      MR->failMaterialization();
    }
  }

  LinkGraphPassFunction getMarkLivePass(const Triple &TT) const override {
    return [this](LinkGraph &G) { return markResponsibilitySymbolsLive(G); };
  }

  Error modifyPassConfig(LinkGraph &LG, PassConfiguration &Config) override {
    // Add passes to mark duplicate defs as should-discard, and to walk the
    // link graph to build the symbol dependence graph.
    Config.PrePrunePasses.push_back([this](LinkGraph &G) {
      return claimOrExternalizeWeakAndCommonSymbols(G);
    });

    for (auto &P : Plugins)
      P->modifyPassConfig(*MR, LG, Config);

    Config.PreFixupPasses.push_back(
        [this](LinkGraph &G) { return registerDependencies(G); });

    return Error::success();
  }

  void notifyLoaded() {
    for (auto &P : Plugins)
      P->notifyLoaded(*MR);
  }

  Error notifyEmitted(jitlink::JITLinkMemoryManager::FinalizedAlloc FA) {
    Error Err = Error::success();
    for (auto &P : Plugins)
      Err = joinErrors(std::move(Err), P->notifyEmitted(*MR));

    if (Err) {
      if (FA)
        Err =
            joinErrors(std::move(Err), Layer.MemMgr.deallocate(std::move(FA)));
      return Err;
    }

    if (FA)
      return Layer.recordFinalizedAlloc(*MR, std::move(FA));

    return Error::success();
  }

private:
  Error claimOrExternalizeWeakAndCommonSymbols(LinkGraph &G) {
    SymbolFlagsMap NewSymbolsToClaim;
    std::vector<std::pair<SymbolStringPtr, Symbol *>> NameToSym;

    auto ProcessSymbol = [&](Symbol *Sym) {
      if (Sym->hasName() && Sym->getLinkage() == Linkage::Weak &&
          Sym->getScope() != Scope::Local) {
        if (!MR->getSymbols().count(Sym->getName())) {
          NewSymbolsToClaim[Sym->getName()] =
              getJITSymbolFlagsForSymbol(*Sym) | JITSymbolFlags::Weak;
          NameToSym.push_back(std::make_pair(Sym->getName(), Sym));
        }
      }
    };

    for (auto *Sym : G.defined_symbols())
      ProcessSymbol(Sym);
    for (auto *Sym : G.absolute_symbols())
      ProcessSymbol(Sym);

    // Attempt to claim all weak defs that we're not already responsible for.
    // This may fail if the resource tracker has become defunct, but should
    // always succeed otherwise.
    if (auto Err = MR->defineMaterializing(std::move(NewSymbolsToClaim)))
      return Err;

    // Walk the list of symbols that we just tried to claim. Symbols that we're
    // responsible for are marked live. Symbols that we're not responsible for
    // are turned into external references.
    for (auto &KV : NameToSym) {
      if (MR->getSymbols().count(KV.first))
        KV.second->setLive(true);
      else
        G.makeExternal(*KV.second);
    }

    return Error::success();
  }

  Error markResponsibilitySymbolsLive(LinkGraph &G) const {
    for (auto *Sym : G.defined_symbols())
      if (Sym->hasName() && MR->getSymbols().count(Sym->getName()))
        Sym->setLive(true);
    return Error::success();
  }

  Error registerDependencies(LinkGraph &G) {

    struct BlockInfo {
      bool InWorklist = false;
      DenseSet<Symbol *> Defs;
      DenseSet<Symbol *> SymbolDeps;
      DenseSet<Block *> AnonEdges, AnonBackEdges;
    };

    DenseMap<Block *, BlockInfo> BlockInfos;

    // Reserve space so that BlockInfos doesn't need to resize. This is
    // essential to avoid invalidating pointers to entries below.
    {
      size_t NumBlocks = 0;
      for (auto &Sec : G.sections())
        NumBlocks += Sec.blocks_size();
      BlockInfos.reserve(NumBlocks);
    }

    // Identify non-locally-scoped symbols defined by each block.
    for (auto *Sym : G.defined_symbols()) {
      if (Sym->getScope() != Scope::Local)
        BlockInfos[&Sym->getBlock()].Defs.insert(Sym);
    }

    // Identify the symbolic and anonymous-block dependencies for each block.
    for (auto *B : G.blocks()) {
      auto &BI = BlockInfos[B];

      for (auto &E : B->edges()) {

        // External symbols are trivially depended on.
        if (E.getTarget().isExternal()) {
          BI.SymbolDeps.insert(&E.getTarget());
          continue;
        }

        // Anonymous symbols aren't depended on at all (they're assumed to be
        // already available).
        if (E.getTarget().isAbsolute())
          continue;

        // If we get here then we depend on a symbol defined by some other
        // block.
        auto &TgtBI = BlockInfos[&E.getTarget().getBlock()];

        // If that block has any definitions then use the first one as the
        // "effective" dependence here (all symbols in TgtBI will become
        // ready at the same time, and chosing a single symbol to represent
        // the block keeps the SymbolDepGroup size small).
        if (!TgtBI.Defs.empty()) {
          BI.SymbolDeps.insert(*TgtBI.Defs.begin());
          continue;
        }

        // Otherwise we've got a dependence on an anonymous block. Record it
        // here for back-propagating symbol dependencies below.
        BI.AnonEdges.insert(&E.getTarget().getBlock());
        TgtBI.AnonBackEdges.insert(B);
      }
    }

    // Prune anonymous blocks.
    {
      std::vector<Block *> BlocksToRemove;
      for (auto &[B, BI] : BlockInfos) {
        // Skip blocks with defs. We only care about anonyous blocks.
        if (!BI.Defs.empty())
          continue;

        BlocksToRemove.push_back(B);

        for (auto *FB : BI.AnonEdges)
          BlockInfos[FB].AnonBackEdges.erase(B);

        for (auto *BB : BI.AnonBackEdges)
          BlockInfos[BB].AnonEdges.erase(B);

        for (auto *FB : BI.AnonEdges) {
          auto &FBI = BlockInfos[FB];
          for (auto *BB : BI.AnonBackEdges)
            FBI.AnonBackEdges.insert(BB);
        }

        for (auto *BB : BI.AnonBackEdges) {
          auto &BBI = BlockInfos[BB];
          for (auto *SD : BI.SymbolDeps)
            BBI.SymbolDeps.insert(SD);
          for (auto *FB : BI.AnonEdges)
            BBI.AnonEdges.insert(FB);
        }
      }

      for (auto *B : BlocksToRemove)
        BlockInfos.erase(B);
    }

    // Build the initial dependence propagation worklist.
    std::deque<Block *> Worklist;
    for (auto &[B, BI] : BlockInfos) {
      if (!BI.SymbolDeps.empty() && !BI.AnonBackEdges.empty()) {
        Worklist.push_back(B);
        BI.InWorklist = true;
      }
    }

    // Propagate symbol deps through the graph.
    while (!Worklist.empty()) {
      auto *B = Worklist.front();
      Worklist.pop_front();

      auto &BI = BlockInfos[B];
      BI.InWorklist = false;

      for (auto *DB : BI.AnonBackEdges) {
        auto &DBI = BlockInfos[DB];
        for (auto *Sym : BI.SymbolDeps) {
          if (DBI.SymbolDeps.insert(Sym).second && !DBI.InWorklist) {
            Worklist.push_back(DB);
            DBI.InWorklist = true;
          }
        }
      }
    }

    // Transform our local dependence information into a list of
    // SymbolDependenceGroups (in the SymbolDepGroups member), ready for use in
    // the upcoming notifyFinalized call.
    auto &TargetJD = MR->getTargetJITDylib();

    for (auto &[B, BI] : BlockInfos) {
      if (!BI.Defs.empty()) {
        SymbolDepGroups.push_back(SymbolDependenceGroup());
        auto &SDG = SymbolDepGroups.back();

        for (auto *Def : BI.Defs)
          SDG.Symbols.insert(Def->getName());

        for (auto *Dep : BI.SymbolDeps) {
          auto DepName = Dep->getName();
          if (Dep->isDefined())
            SDG.Dependencies[&TargetJD].insert(std::move(DepName));
          else {
            auto SourceJDItr =
                SymbolSourceJDs.find(NonOwningSymbolStringPtr(DepName));
            if (SourceJDItr != SymbolSourceJDs.end())
              SDG.Dependencies[SourceJDItr->second].insert(std::move(DepName));
          }
        }
      }
    }

    return Error::success();
  }

  LinkGraphLinkingLayer &Layer;
  std::vector<std::shared_ptr<LinkGraphLinkingLayer::Plugin>> Plugins;
  std::unique_ptr<MaterializationResponsibility> MR;
  std::unique_ptr<MemoryBuffer> ObjBuffer;
  DenseMap<NonOwningSymbolStringPtr, JITDylib *> SymbolSourceJDs;
  std::vector<SymbolDependenceGroup> SymbolDepGroups;
};

LinkGraphLinkingLayer::Plugin::~Plugin() = default;

LinkGraphLinkingLayer::LinkGraphLinkingLayer(ExecutionSession &ES)
    : LinkGraphLayer(ES), MemMgr(ES.getExecutorProcessControl().getMemMgr()) {
  ES.registerResourceManager(*this);
}

LinkGraphLinkingLayer::LinkGraphLinkingLayer(ExecutionSession &ES,
                                             JITLinkMemoryManager &MemMgr)
    : LinkGraphLayer(ES), MemMgr(MemMgr) {
  ES.registerResourceManager(*this);
}

LinkGraphLinkingLayer::LinkGraphLinkingLayer(
    ExecutionSession &ES, std::unique_ptr<JITLinkMemoryManager> MemMgr)
    : LinkGraphLayer(ES), MemMgr(*MemMgr), MemMgrOwnership(std::move(MemMgr)) {
  ES.registerResourceManager(*this);
}

LinkGraphLinkingLayer::~LinkGraphLinkingLayer() {
  assert(Allocs.empty() && "Layer destroyed with resources still attached");
  getExecutionSession().deregisterResourceManager(*this);
}

void LinkGraphLinkingLayer::emit(
    std::unique_ptr<MaterializationResponsibility> R,
    std::unique_ptr<LinkGraph> G) {
  assert(R && "R must not be null");
  assert(G && "G must not be null");
  auto Ctx = std::make_unique<JITLinkCtx>(*this, std::move(R), nullptr);
  Ctx->notifyMaterializing(*G);
  link(std::move(G), std::move(Ctx));
}

void LinkGraphLinkingLayer::emit(
    std::unique_ptr<MaterializationResponsibility> R,
    std::unique_ptr<LinkGraph> G, std::unique_ptr<MemoryBuffer> ObjBuf) {
  assert(R && "R must not be null");
  assert(G && "G must not be null");
  assert(ObjBuf && "Object must not be null");
  auto Ctx =
      std::make_unique<JITLinkCtx>(*this, std::move(R), std::move(ObjBuf));
  Ctx->notifyMaterializing(*G);
  link(std::move(G), std::move(Ctx));
}

Error LinkGraphLinkingLayer::recordFinalizedAlloc(
    MaterializationResponsibility &MR, FinalizedAlloc FA) {
  auto Err = MR.withResourceKeyDo(
      [&](ResourceKey K) { Allocs[K].push_back(std::move(FA)); });

  if (Err)
    Err = joinErrors(std::move(Err), MemMgr.deallocate(std::move(FA)));

  return Err;
}

Error LinkGraphLinkingLayer::handleRemoveResources(JITDylib &JD,
                                                   ResourceKey K) {

  {
    Error Err = Error::success();
    for (auto &P : Plugins)
      Err = joinErrors(std::move(Err), P->notifyRemovingResources(JD, K));
    if (Err)
      return Err;
  }

  std::vector<FinalizedAlloc> AllocsToRemove;
  getExecutionSession().runSessionLocked([&] {
    auto I = Allocs.find(K);
    if (I != Allocs.end()) {
      std::swap(AllocsToRemove, I->second);
      Allocs.erase(I);
    }
  });

  if (AllocsToRemove.empty())
    return Error::success();

  return MemMgr.deallocate(std::move(AllocsToRemove));
}

void LinkGraphLinkingLayer::handleTransferResources(JITDylib &JD,
                                                    ResourceKey DstKey,
                                                    ResourceKey SrcKey) {
  if (Allocs.contains(SrcKey)) {
    // DstKey may not be in the DenseMap yet, so the following line may resize
    // the container and invalidate iterators and value references.
    auto &DstAllocs = Allocs[DstKey];
    auto &SrcAllocs = Allocs[SrcKey];
    DstAllocs.reserve(DstAllocs.size() + SrcAllocs.size());
    for (auto &Alloc : SrcAllocs)
      DstAllocs.push_back(std::move(Alloc));

    Allocs.erase(SrcKey);
  }

  for (auto &P : Plugins)
    P->notifyTransferringResources(JD, DstKey, SrcKey);
}

EHFrameRegistrationPlugin::EHFrameRegistrationPlugin(
    ExecutionSession &ES, std::unique_ptr<EHFrameRegistrar> Registrar)
    : ES(ES), Registrar(std::move(Registrar)) {}

void EHFrameRegistrationPlugin::modifyPassConfig(
    MaterializationResponsibility &MR, LinkGraph &G,
    PassConfiguration &PassConfig) {

  PassConfig.PostFixupPasses.push_back(createEHFrameRecorderPass(
      G.getTargetTriple(), [this, &MR](ExecutorAddr Addr, size_t Size) {
        if (Addr) {
          std::lock_guard<std::mutex> Lock(EHFramePluginMutex);
          assert(!InProcessLinks.count(&MR) &&
                 "Link for MR already being tracked?");
          InProcessLinks[&MR] = {Addr, Size};
        }
      }));
}

Error EHFrameRegistrationPlugin::notifyEmitted(
    MaterializationResponsibility &MR) {

  ExecutorAddrRange EmittedRange;
  {
    std::lock_guard<std::mutex> Lock(EHFramePluginMutex);

    auto EHFrameRangeItr = InProcessLinks.find(&MR);
    if (EHFrameRangeItr == InProcessLinks.end())
      return Error::success();

    EmittedRange = EHFrameRangeItr->second;
    assert(EmittedRange.Start && "eh-frame addr to register can not be null");
    InProcessLinks.erase(EHFrameRangeItr);
  }

  if (auto Err = MR.withResourceKeyDo(
          [&](ResourceKey K) { EHFrameRanges[K].push_back(EmittedRange); }))
    return Err;

  return Registrar->registerEHFrames(EmittedRange);
}

Error EHFrameRegistrationPlugin::notifyFailed(
    MaterializationResponsibility &MR) {
  std::lock_guard<std::mutex> Lock(EHFramePluginMutex);
  InProcessLinks.erase(&MR);
  return Error::success();
}

Error EHFrameRegistrationPlugin::notifyRemovingResources(JITDylib &JD,
                                                         ResourceKey K) {
  std::vector<ExecutorAddrRange> RangesToRemove;

  ES.runSessionLocked([&] {
    auto I = EHFrameRanges.find(K);
    if (I != EHFrameRanges.end()) {
      RangesToRemove = std::move(I->second);
      EHFrameRanges.erase(I);
    }
  });

  Error Err = Error::success();
  while (!RangesToRemove.empty()) {
    auto RangeToRemove = RangesToRemove.back();
    RangesToRemove.pop_back();
    assert(RangeToRemove.Start && "Untracked eh-frame range must not be null");
    Err = joinErrors(std::move(Err),
                     Registrar->deregisterEHFrames(RangeToRemove));
  }

  return Err;
}

void EHFrameRegistrationPlugin::notifyTransferringResources(
    JITDylib &JD, ResourceKey DstKey, ResourceKey SrcKey) {
  auto SI = EHFrameRanges.find(SrcKey);
  if (SI == EHFrameRanges.end())
    return;

  auto DI = EHFrameRanges.find(DstKey);
  if (DI != EHFrameRanges.end()) {
    auto &SrcRanges = SI->second;
    auto &DstRanges = DI->second;
    DstRanges.reserve(DstRanges.size() + SrcRanges.size());
    for (auto &SrcRange : SrcRanges)
      DstRanges.push_back(std::move(SrcRange));
    EHFrameRanges.erase(SI);
  } else {
    // We need to move SrcKey's ranges over without invalidating the SI
    // iterator.
    auto Tmp = std::move(SI->second);
    EHFrameRanges.erase(SI);
    EHFrameRanges[DstKey] = std::move(Tmp);
  }
}

} // End namespace orc.
} // End namespace llvm.
