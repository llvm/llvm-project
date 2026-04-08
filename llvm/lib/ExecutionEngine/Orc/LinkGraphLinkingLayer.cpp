//===------ LinkGraphLinkingLayer.cpp - Link LinkGraphs with JITLink ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ExecutionEngine/Orc/LinkGraphLinkingLayer.h"

#include "llvm/ADT/SCCIterator.h"
#include "llvm/ExecutionEngine/JITLink/EHFrameSupport.h"
#include "llvm/ExecutionEngine/JITLink/aarch32.h"
#include "llvm/ExecutionEngine/Orc/DebugUtils.h"
#include "llvm/ExecutionEngine/Orc/Shared/ObjectFormats.h"
#include "llvm/Support/MemoryBuffer.h"

#define DEBUG_TYPE "orc"

using namespace llvm;
using namespace llvm::jitlink;
using namespace llvm::orc;

namespace llvm {

struct BlockDepInfo;

using BlockDepInfoMap = DenseMap<jitlink::Block *, BlockDepInfo>;

struct BlockDepInfo {
  using SymbolDefList = SmallVector<jitlink::Symbol *>;
  using SymbolDepSet = DenseSet<jitlink::Symbol *>;
  using AnonBlockDepSet = DenseSet<jitlink::Block *>;

  BlockDepInfoMap *Graph = nullptr;
  SymbolDefList SymbolDefs;
  SymbolDepSet SymbolDeps;
  AnonBlockDepSet AnonBlockDeps;
  BlockDepInfo *SCCRoot = nullptr;
  std::optional<size_t> DepGroupIndex;
};

template <> struct GraphTraits<BlockDepInfo *> {
  using NodeRef = BlockDepInfo *;

  class ChildIteratorType {
    using impl_iterator = BlockDepInfo::AnonBlockDepSet::iterator;

  public:
    ChildIteratorType(NodeRef Parent, impl_iterator I)
        : Parent(Parent), I(std::move(I)) {}

    friend bool operator==(const ChildIteratorType &LHS,
                           const ChildIteratorType &RHS) {
      return LHS.I == RHS.I;
    }
    friend bool operator!=(const ChildIteratorType &LHS,
                           const ChildIteratorType &RHS) {
      return LHS.I != RHS.I;
    }

    ChildIteratorType &operator++() {
      ++I;
      return *this;
    }
    ChildIteratorType operator++(int) {
      auto Tmp = *this;
      ++I;
      return Tmp;
    }
    NodeRef operator*() {
      assert(Parent->Graph && "No pointer to BlockDepInfoMap");
      return &(*Parent->Graph)[*I];
    }

  private:
    NodeRef Parent;
    BlockDepInfo::AnonBlockDepSet::iterator I;
  };

  static NodeRef getEntryNode(NodeRef N) { return N; }

  static ChildIteratorType child_begin(NodeRef N) {
    return ChildIteratorType(N, N->AnonBlockDeps.begin());
  }
  static ChildIteratorType child_end(NodeRef N) {
    return ChildIteratorType(N, N->AnonBlockDeps.end());
  }
};

} // namespace llvm

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

  ~JITLinkCtx() override {
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
        LR.insert_range(*Result);
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
    auto &TargetJD = MR->getTargetJITDylib();
    for (auto &[Defs, Deps] : calculateDepGroups(G)) {
      SymbolDepGroups.push_back(SymbolDependenceGroup());
      auto &SDG = SymbolDepGroups.back();
      for (auto *Def : Defs)
        SDG.Symbols.insert(Def->getName());
      for (auto *Dep : Deps) {
        if (Dep->isDefined())
          SDG.Dependencies[&TargetJD].insert(Dep->getName());
        else {
          auto I =
              SymbolSourceJDs.find(NonOwningSymbolStringPtr(Dep->getName()));
          if (I != SymbolSourceJDs.end()) {
            auto &SymJD = *I->second;
            SDG.Dependencies[&SymJD].insert(Dep->getName());
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
  assert(Allocs.empty() &&
         "Layer destroyed with resources still attached "
         "(ExecutionSession::endSession() must be called prior to "
         "destruction)");
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

SmallVector<LinkGraphLinkingLayer::SymbolDepGroup>
LinkGraphLinkingLayer::calculateDepGroups(LinkGraph &G) {

  // Step 1.
  // Build initial map entries and symbol def lists.
  BlockDepInfoMap BlockDepInfos;
  for (auto *Sym : G.defined_symbols())
    if (Sym->getScope() != Scope::Local)
      BlockDepInfos[&Sym->getBlock()].SymbolDefs.push_back(Sym);

  // Step 2.
  // Complete the BlockDepInfos "graph" by adding symbol and block dependencies
  // for each block.
  {
    SmallVector<Block *> Worklist;
    Worklist.reserve(BlockDepInfos.size());

    // Build worklist, link each BlockDepInfo "node" back to the BlockInfos map
    // "graph" for our GraphTraits specialization above. This will allow us to
    // walk the SCCs of the anonymous-block-dependence graph.
    for (auto &[B, BDInfo] : BlockDepInfos) {
      BDInfo.Graph = &BlockDepInfos;
      Worklist.push_back(B);
    }

    // Calculate the relevant symbol and block dependencies for each block:
    // 1. Absolute symbols are ignored.
    // 2. External symbols are included in a block's symbol dep set.
    // 3. Blocks that do not define any symbols are included in the anonymous
    //    block dependence sets.
    // 4. For blocks that do define symbols we add only the first defined
    //    symbol to the symbol dep set (since all symbols for the block will
    //    have the same dependencies).
    while (!Worklist.empty()) {
      auto *B = Worklist.pop_back_val();
      BlockDepInfo *BDInfo = nullptr; // Populated lazily.

      for (auto &E : B->edges()) {
        if (E.getTarget().isAbsolute()) // skip: absolutes are assumed ready
          continue;

        if (!BDInfo) // Populate -- we'll need it below.
          BDInfo = &BlockDepInfos[B];

        if (E.getTarget().isExternal()) { // include and continue
          BDInfo->SymbolDeps.insert(&E.getTarget());
          continue;
        }

        // Target must be defined.
        auto *TgtB = &E.getTarget().getBlock();
        auto I = BlockDepInfos.find(TgtB);

        if (I != BlockDepInfos.end()) {
          // TgtB is in BlockInfos. Record a symbol dependence (if it defines
          // any symbols) or anonymous block dependence.
          auto &TgtBInfo = I->second;
          if (!TgtBInfo.SymbolDefs.empty())
            BDInfo->SymbolDeps.insert(TgtBInfo.SymbolDefs.front());
          else
            BDInfo->AnonBlockDeps.insert(TgtB);
        } else {
          // TgtB not in BlockInfos. It must be anonymous. We need to:
          // 1. Record the dependence.
          // 2. Add BlockInfos and Worklist entries for TgtB.
          // 3. Reset BInfo, since step (2) may have invalidated the pointer.
          BDInfo->AnonBlockDeps.insert(TgtB);
          Worklist.push_back(TgtB);
          BlockDepInfos[TgtB].Graph = &BlockDepInfos;
          BDInfo = nullptr;
          continue;
        }
      }
    }
  }

  // Step 3.
  // Convert block deps to SCC deps.
  SmallVector<SymbolDepGroup> DGs;
  for (auto &[B, BDInfo] : BlockDepInfos) {
    for (auto &SCC : make_range(scc_begin(&BDInfo), scc_end(&BDInfo))) {

      auto &SCCRootInfo = *SCC.front();

      // Continue if already visited. The loop over the SCC elements below
      // deletes the SCCs below as it goes, so this early continue just saves
      // us looking at a bunch of empty sets below that.
      if (SCCRootInfo.SCCRoot)
        continue;
      SCCRootInfo.SCCRoot = &SCCRootInfo;

      // Collect all symbol defs, deps, and anonymous block deps, and remove
      // the links to already visited SCCs.
      auto SCCSymbolDefs = std::move(SCCRootInfo.SymbolDefs);
      auto SCCSymbolDeps = std::move(SCCRootInfo.SymbolDeps);
      auto SCCAnonBlockDeps = std::move(SCCRootInfo.AnonBlockDeps);
      for (auto *SCCBInfo : make_range(std::next(SCC.begin()), SCC.end())) {
        SCCBInfo->SCCRoot = &SCCRootInfo;
        SCCSymbolDefs.append(SCCBInfo->SymbolDefs);
        SCCBInfo->SymbolDefs.clear();
        SCCSymbolDeps.insert(SCCBInfo->SymbolDeps.begin(),
                             SCCBInfo->SymbolDeps.end());
        SCCBInfo->SymbolDeps.clear();
        SCCAnonBlockDeps.insert(SCCBInfo->AnonBlockDeps.begin(),
                                SCCBInfo->AnonBlockDeps.end());
        SCCBInfo->AnonBlockDeps.clear();
      }

      // Identify DepGroups emitted for previously visited SCCs that this
      // SCC depends on.
      DenseSet<size_t> SrcDepGroups;
      for (auto *DepB : SCCAnonBlockDeps) {
        assert(BlockDepInfos.count(DepB) && "Unrecognized block");
        auto &DepBRootInfo = *BlockDepInfos[DepB].SCCRoot;
        if (DepBRootInfo.DepGroupIndex)
          SrcDepGroups.insert(*DepBRootInfo.DepGroupIndex);
      }

      // If this SCC doesn't depend on any existing dep groups then check
      // whether it has direct symbol deps of its own.
      if (SrcDepGroups.empty()) {

        // If this SCC has its own symbol deps then add a dep-group and
        // continue.
        if (!SCCSymbolDeps.empty()) {
          SCCRootInfo.DepGroupIndex = DGs.size();
          DGs.push_back({});
          DGs.back().Defs = std::move(SCCSymbolDefs);
          DGs.back().Deps = std::move(SCCSymbolDeps);
        }
        // Otherwise just continue.
        continue;
      }

      // Special case: If we only depend on one dep group and this SCC
      // doesn't have any symbol deps of its own then just merge this SCC's
      // defs into the existing dep group and continue.
      if (SrcDepGroups.size() == 1 && SCCSymbolDeps.empty()) {
        SCCRootInfo.DepGroupIndex = *SrcDepGroups.begin();
        DGs[*SCCRootInfo.DepGroupIndex].Defs.append(SCCSymbolDefs);
        continue;
      }

      // General case: This SCC depends on multiple dep groups, and/or has
      // its own symbol deps. Build a new dep group for it.
      SCCRootInfo.DepGroupIndex = DGs.size();
      DGs.push_back({});
      auto &DG = DGs.back();
      DG.Defs = std::move(SCCSymbolDefs);
      for (auto &DGIndex : SrcDepGroups)
        DG.Deps.insert(DGs[DGIndex].Deps.begin(), DGs[DGIndex].Deps.end());
      DG.Deps.insert(SCCSymbolDeps.begin(), SCCSymbolDeps.end());
    }
  }

  // Remove self-reference from each dep group, and filter out any dep groups
  // whose resulting deps or defs are empty.
  for (size_t I = 0; I != DGs.size();) {
    auto &DG = DGs[I];

    // Remove self-deps.
    for (auto &Def : DG.Defs)
      DG.Deps.erase(Def);

    // Remove groups with empty defs or deps.
    if (DG.Defs.empty() || DG.Deps.empty()) {
      std::swap(DG, DGs.back());
      DGs.pop_back();
    } else
      ++I;
  }

  return DGs;
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

} // End namespace orc.
} // End namespace llvm.
