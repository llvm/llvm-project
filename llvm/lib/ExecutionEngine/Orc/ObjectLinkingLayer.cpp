//===------- ObjectLinkingLayer.cpp - JITLink backed ORC ObjectLayer ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ExecutionEngine/Orc/ObjectLinkingLayer.h"

#include "llvm/ADT/Optional.h"
#include "llvm/ExecutionEngine/JITLink/EHFrameSupport.h"

#include <vector>

#define DEBUG_TYPE "orc"

using namespace llvm;
using namespace llvm::jitlink;
using namespace llvm::orc;

namespace llvm {
namespace orc {

class ObjectLinkingLayerJITLinkContext final : public JITLinkContext {
public:
  ObjectLinkingLayerJITLinkContext(ObjectLinkingLayer &Layer,
                                   MaterializationResponsibility MR,
                                   std::unique_ptr<MemoryBuffer> ObjBuffer)
      : Layer(Layer), MR(std::move(MR)), ObjBuffer(std::move(ObjBuffer)) {}

  ~ObjectLinkingLayerJITLinkContext() {
    // If there is an object buffer return function then use it to
    // return ownership of the buffer.
    if (Layer.ReturnObjectBuffer)
      Layer.ReturnObjectBuffer(std::move(ObjBuffer));
  }

  JITLinkMemoryManager &getMemoryManager() override { return *Layer.MemMgr; }

  MemoryBufferRef getObjectBuffer() const override {
    return ObjBuffer->getMemBufferRef();
  }

  void notifyFailed(Error Err) override {
    Layer.getExecutionSession().reportError(std::move(Err));
    MR.failMaterialization();
  }

  void lookup(const LookupMap &Symbols,
              std::unique_ptr<JITLinkAsyncLookupContinuation> LC) override {

    JITDylibSearchOrder SearchOrder;
    MR.getTargetJITDylib().withSearchOrderDo(
        [&](const JITDylibSearchOrder &O) { SearchOrder = O; });

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
      LookupSet.add(ES.intern(KV.first), LookupFlags);
    }

    // OnResolve -- De-intern the symbols and pass the result to the linker.
    auto OnResolve = [this, LookupContinuation = std::move(LC)](
                         Expected<SymbolMap> Result) mutable {
      auto Main = Layer.getExecutionSession().intern("_main");
      if (!Result)
        LookupContinuation->run(Result.takeError());
      else {
        AsyncLookupResult LR;
        for (auto &KV : *Result)
          LR[*KV.first] = KV.second;
        LookupContinuation->run(std::move(LR));
      }
    };

    ES.lookup(LookupKind::Static, SearchOrder, std::move(LookupSet),
              SymbolState::Resolved, std::move(OnResolve),
              [this](const SymbolDependenceMap &Deps) {
                registerDependencies(Deps);
              });
  }

  void notifyResolved(LinkGraph &G) override {
    auto &ES = Layer.getExecutionSession();

    SymbolFlagsMap ExtraSymbolsToClaim;
    bool AutoClaim = Layer.AutoClaimObjectSymbols;

    SymbolMap InternedResult;
    for (auto *Sym : G.defined_symbols())
      if (Sym->hasName() && Sym->getScope() != Scope::Local) {
        auto InternedName = ES.intern(Sym->getName());
        JITSymbolFlags Flags;

        if (Sym->isCallable())
          Flags |= JITSymbolFlags::Callable;
        if (Sym->getScope() == Scope::Default)
          Flags |= JITSymbolFlags::Exported;

        InternedResult[InternedName] =
            JITEvaluatedSymbol(Sym->getAddress(), Flags);
        if (AutoClaim && !MR.getSymbols().count(InternedName)) {
          assert(!ExtraSymbolsToClaim.count(InternedName) &&
                 "Duplicate symbol to claim?");
          ExtraSymbolsToClaim[InternedName] = Flags;
        }
      }

    for (auto *Sym : G.absolute_symbols())
      if (Sym->hasName()) {
        auto InternedName = ES.intern(Sym->getName());
        JITSymbolFlags Flags;
        Flags |= JITSymbolFlags::Absolute;
        if (Sym->isCallable())
          Flags |= JITSymbolFlags::Callable;
        if (Sym->getLinkage() == Linkage::Weak)
          Flags |= JITSymbolFlags::Weak;
        InternedResult[InternedName] =
            JITEvaluatedSymbol(Sym->getAddress(), Flags);
        if (AutoClaim && !MR.getSymbols().count(InternedName)) {
          assert(!ExtraSymbolsToClaim.count(InternedName) &&
                 "Duplicate symbol to claim?");
          ExtraSymbolsToClaim[InternedName] = Flags;
        }
      }

    if (!ExtraSymbolsToClaim.empty())
      if (auto Err = MR.defineMaterializing(ExtraSymbolsToClaim))
        return notifyFailed(std::move(Err));
    if (auto Err = MR.notifyResolved(InternedResult)) {
      Layer.getExecutionSession().reportError(std::move(Err));
      MR.failMaterialization();
      return;
    }
    Layer.notifyLoaded(MR);
  }

  void notifyFinalized(
      std::unique_ptr<JITLinkMemoryManager::Allocation> A) override {
    if (auto Err = Layer.notifyEmitted(MR, std::move(A))) {
      Layer.getExecutionSession().reportError(std::move(Err));
      MR.failMaterialization();
      return;
    }
    if (auto Err = MR.notifyEmitted()) {
      Layer.getExecutionSession().reportError(std::move(Err));
      MR.failMaterialization();
    }
  }

  LinkGraphPassFunction getMarkLivePass(const Triple &TT) const override {
    return [this](LinkGraph &G) { return markResponsibilitySymbolsLive(G); };
  }

  Error modifyPassConfig(const Triple &TT, PassConfiguration &Config) override {
    // Add passes to mark duplicate defs as should-discard, and to walk the
    // link graph to build the symbol dependence graph.
    Config.PrePrunePasses.push_back(
        [this](LinkGraph &G) { return externalizeWeakAndCommonSymbols(G); });
    Config.PostPrunePasses.push_back(
        [this](LinkGraph &G) { return computeNamedSymbolDependencies(G); });

    Layer.modifyPassConfig(MR, TT, Config);

    return Error::success();
  }

private:
  using AnonToNamedDependenciesMap = DenseMap<const Symbol *, SymbolNameSet>;

  Error externalizeWeakAndCommonSymbols(LinkGraph &G) {
    auto &ES = Layer.getExecutionSession();
    for (auto *Sym : G.defined_symbols())
      if (Sym->hasName() && Sym->getLinkage() == Linkage::Weak) {
        if (!MR.getSymbols().count(ES.intern(Sym->getName())))
          G.makeExternal(*Sym);
      }

    for (auto *Sym : G.absolute_symbols())
      if (Sym->hasName() && Sym->getLinkage() == Linkage::Weak) {
        if (!MR.getSymbols().count(ES.intern(Sym->getName())))
          G.makeExternal(*Sym);
      }

    return Error::success();
  }

  Error markResponsibilitySymbolsLive(LinkGraph &G) const {
    auto &ES = Layer.getExecutionSession();
    for (auto *Sym : G.defined_symbols())
      if (Sym->hasName() && MR.getSymbols().count(ES.intern(Sym->getName())))
        Sym->setLive(true);
    return Error::success();
  }

  Error computeNamedSymbolDependencies(LinkGraph &G) {
    auto &ES = MR.getTargetJITDylib().getExecutionSession();
    auto AnonDeps = computeAnonDeps(G);

    for (auto *Sym : G.defined_symbols()) {

      // Skip anonymous and non-global atoms: we do not need dependencies for
      // these.
      if (Sym->getScope() == Scope::Local)
        continue;

      auto SymName = ES.intern(Sym->getName());
      SymbolNameSet &SymDeps = NamedSymbolDeps[SymName];

      for (auto &E : Sym->getBlock().edges()) {
        auto &TargetSym = E.getTarget();

        if (TargetSym.getScope() != Scope::Local)
          SymDeps.insert(ES.intern(TargetSym.getName()));
        else {
          assert(TargetSym.isDefined() &&
                 "Anonymous/local symbols must be defined");
          auto I = AnonDeps.find(&TargetSym);
          if (I != AnonDeps.end())
            for (auto &S : I->second)
              SymDeps.insert(S);
        }
      }
    }

    return Error::success();
  }

  AnonToNamedDependenciesMap computeAnonDeps(LinkGraph &G) {

    auto &ES = MR.getTargetJITDylib().getExecutionSession();
    AnonToNamedDependenciesMap DepMap;

    // For all anonymous symbols:
    // (1) Add their named dependencies.
    // (2) Add them to the worklist for further iteration if they have any
    //     depend on any other anonymous symbols.
    struct WorklistEntry {
      WorklistEntry(Symbol *Sym, DenseSet<Symbol *> SymAnonDeps)
          : Sym(Sym), SymAnonDeps(std::move(SymAnonDeps)) {}

      Symbol *Sym = nullptr;
      DenseSet<Symbol *> SymAnonDeps;
    };
    std::vector<WorklistEntry> Worklist;
    for (auto *Sym : G.defined_symbols())
      if (!Sym->hasName()) {
        auto &SymNamedDeps = DepMap[Sym];
        DenseSet<Symbol *> SymAnonDeps;

        for (auto &E : Sym->getBlock().edges()) {
          auto &TargetSym = E.getTarget();
          if (TargetSym.hasName())
            SymNamedDeps.insert(ES.intern(TargetSym.getName()));
          else {
            assert(TargetSym.isDefined() &&
                   "Anonymous symbols must be defined");
            SymAnonDeps.insert(&TargetSym);
          }
        }

        if (!SymAnonDeps.empty())
          Worklist.push_back(WorklistEntry(Sym, std::move(SymAnonDeps)));
      }

    // Loop over all anonymous symbols with anonymous dependencies, propagating
    // their respective *named* dependencies. Iterate until we hit a stable
    // state.
    bool Changed;
    do {
      Changed = false;
      for (auto &WLEntry : Worklist) {
        auto *Sym = WLEntry.Sym;
        auto &SymNamedDeps = DepMap[Sym];
        auto &SymAnonDeps = WLEntry.SymAnonDeps;

        for (auto *TargetSym : SymAnonDeps) {
          auto I = DepMap.find(TargetSym);
          if (I != DepMap.end())
            for (const auto &S : I->second)
              Changed |= SymNamedDeps.insert(S).second;
        }
      }
    } while (Changed);

    return DepMap;
  }

  void registerDependencies(const SymbolDependenceMap &QueryDeps) {
    for (auto &NamedDepsEntry : NamedSymbolDeps) {
      auto &Name = NamedDepsEntry.first;
      auto &NameDeps = NamedDepsEntry.second;
      SymbolDependenceMap SymbolDeps;

      for (const auto &QueryDepsEntry : QueryDeps) {
        JITDylib &SourceJD = *QueryDepsEntry.first;
        const SymbolNameSet &Symbols = QueryDepsEntry.second;
        auto &DepsForJD = SymbolDeps[&SourceJD];

        for (const auto &S : Symbols)
          if (NameDeps.count(S))
            DepsForJD.insert(S);

        if (DepsForJD.empty())
          SymbolDeps.erase(&SourceJD);
      }

      MR.addDependencies(Name, SymbolDeps);
    }
  }

  ObjectLinkingLayer &Layer;
  MaterializationResponsibility MR;
  std::unique_ptr<MemoryBuffer> ObjBuffer;
  DenseMap<SymbolStringPtr, SymbolNameSet> NamedSymbolDeps;
};

ObjectLinkingLayer::Plugin::~Plugin() {}

ObjectLinkingLayer::ObjectLinkingLayer(
    ExecutionSession &ES, std::unique_ptr<JITLinkMemoryManager> MemMgr)
    : ObjectLayer(ES), MemMgr(std::move(MemMgr)) {}

ObjectLinkingLayer::~ObjectLinkingLayer() {
  if (auto Err = removeAllModules())
    getExecutionSession().reportError(std::move(Err));
}

void ObjectLinkingLayer::emit(MaterializationResponsibility R,
                              std::unique_ptr<MemoryBuffer> O) {
  assert(O && "Object must not be null");
  jitLink(std::make_unique<ObjectLinkingLayerJITLinkContext>(
      *this, std::move(R), std::move(O)));
}

void ObjectLinkingLayer::modifyPassConfig(MaterializationResponsibility &MR,
                                          const Triple &TT,
                                          PassConfiguration &PassConfig) {
  for (auto &P : Plugins)
    P->modifyPassConfig(MR, TT, PassConfig);
}

void ObjectLinkingLayer::notifyLoaded(MaterializationResponsibility &MR) {
  for (auto &P : Plugins)
    P->notifyLoaded(MR);
}

Error ObjectLinkingLayer::notifyEmitted(MaterializationResponsibility &MR,
                                        AllocPtr Alloc) {
  Error Err = Error::success();
  for (auto &P : Plugins)
    Err = joinErrors(std::move(Err), P->notifyEmitted(MR));

  if (Err)
    return Err;

  {
    std::lock_guard<std::mutex> Lock(LayerMutex);
    UntrackedAllocs.push_back(std::move(Alloc));
  }

  return Error::success();
}

Error ObjectLinkingLayer::removeModule(VModuleKey K) {
  Error Err = Error::success();

  for (auto &P : Plugins)
    Err = joinErrors(std::move(Err), P->notifyRemovingModule(K));

  AllocPtr Alloc;

  {
    std::lock_guard<std::mutex> Lock(LayerMutex);
    auto AllocItr = TrackedAllocs.find(K);
    Alloc = std::move(AllocItr->second);
    TrackedAllocs.erase(AllocItr);
  }

  assert(Alloc && "No allocation for key K");

  return joinErrors(std::move(Err), Alloc->deallocate());
}

Error ObjectLinkingLayer::removeAllModules() {

  Error Err = Error::success();

  for (auto &P : Plugins)
    Err = joinErrors(std::move(Err), P->notifyRemovingAllModules());

  std::vector<AllocPtr> Allocs;
  {
    std::lock_guard<std::mutex> Lock(LayerMutex);
    Allocs = std::move(UntrackedAllocs);

    for (auto &KV : TrackedAllocs)
      Allocs.push_back(std::move(KV.second));

    TrackedAllocs.clear();
  }

  while (!Allocs.empty()) {
    Err = joinErrors(std::move(Err), Allocs.back()->deallocate());
    Allocs.pop_back();
  }

  return Err;
}

EHFrameRegistrationPlugin::EHFrameRegistrationPlugin(
    EHFrameRegistrar &Registrar)
    : Registrar(Registrar) {}

void EHFrameRegistrationPlugin::modifyPassConfig(
    MaterializationResponsibility &MR, const Triple &TT,
    PassConfiguration &PassConfig) {
  assert(!InProcessLinks.count(&MR) && "Link for MR already being tracked?");

  PassConfig.PostFixupPasses.push_back(
      createEHFrameRecorderPass(TT, [this, &MR](JITTargetAddress Addr,
                                                size_t Size) {
        if (Addr)
          InProcessLinks[&MR] = { Addr, Size };
      }));
}

Error EHFrameRegistrationPlugin::notifyEmitted(
    MaterializationResponsibility &MR) {

  auto EHFrameRangeItr = InProcessLinks.find(&MR);
  if (EHFrameRangeItr == InProcessLinks.end())
    return Error::success();

  auto EHFrameRange = EHFrameRangeItr->second;
  assert(EHFrameRange.Addr &&
         "eh-frame addr to register can not be null");

  InProcessLinks.erase(EHFrameRangeItr);
  if (auto Key = MR.getVModuleKey())
    TrackedEHFrameRanges[Key] = EHFrameRange;
  else
    UntrackedEHFrameRanges.push_back(EHFrameRange);

  return Registrar.registerEHFrames(EHFrameRange.Addr, EHFrameRange.Size);
}

Error EHFrameRegistrationPlugin::notifyRemovingModule(VModuleKey K) {
  auto EHFrameRangeItr = TrackedEHFrameRanges.find(K);
  if (EHFrameRangeItr == TrackedEHFrameRanges.end())
    return Error::success();

  auto EHFrameRange = EHFrameRangeItr->second;
  assert(EHFrameRange.Addr && "Tracked eh-frame range must not be null");

  TrackedEHFrameRanges.erase(EHFrameRangeItr);

  return Registrar.deregisterEHFrames(EHFrameRange.Addr, EHFrameRange.Size);
}

Error EHFrameRegistrationPlugin::notifyRemovingAllModules() {

  std::vector<EHFrameRange> EHFrameRanges =
    std::move(UntrackedEHFrameRanges);
  EHFrameRanges.reserve(EHFrameRanges.size() + TrackedEHFrameRanges.size());

  for (auto &KV : TrackedEHFrameRanges)
    EHFrameRanges.push_back(KV.second);

  TrackedEHFrameRanges.clear();

  Error Err = Error::success();

  while (!EHFrameRanges.empty()) {
    auto EHFrameRange = EHFrameRanges.back();
    assert(EHFrameRange.Addr && "Untracked eh-frame range must not be null");
    EHFrameRanges.pop_back();
    Err = joinErrors(std::move(Err),
                     Registrar.deregisterEHFrames(EHFrameRange.Addr,
                                                  EHFrameRange.Size));
  }

  return Err;
}

} // End namespace orc.
} // End namespace llvm.
