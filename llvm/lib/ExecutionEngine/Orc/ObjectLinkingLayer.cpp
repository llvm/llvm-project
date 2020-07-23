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

  JITLinkMemoryManager &getMemoryManager() override { return Layer.MemMgr; }

  MemoryBufferRef getObjectBuffer() const override {
    return ObjBuffer->getMemBufferRef();
  }

  void notifyFailed(Error Err) override {
    Layer.getExecutionSession().reportError(std::move(Err));
    MR.failMaterialization();
  }

  void lookup(const LookupMap &Symbols,
              std::unique_ptr<JITLinkAsyncLookupContinuation> LC) override {

    JITDylibSearchOrder LinkOrder;
    MR.getTargetJITDylib().withLinkOrderDo(
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

    for (auto &KV : InternalNamedSymbolDeps) {
      SymbolDependenceMap InternalDeps;
      InternalDeps[&MR.getTargetJITDylib()] = std::move(KV.second);
      MR.addDependencies(KV.first, InternalDeps);
    }

    ES.lookup(LookupKind::Static, LinkOrder, std::move(LookupSet),
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

    {

      // Check that InternedResult matches up with MR.getSymbols().
      // This guards against faulty transformations / compilers / object caches.

      // First check that there aren't any missing symbols.
      size_t NumMaterializationSideEffectsOnlySymbols = 0;
      SymbolNameVector ExtraSymbols;
      SymbolNameVector MissingSymbols;
      for (auto &KV : MR.getSymbols()) {

        // If this is a materialization-side-effects only symbol then bump
        // the counter and make sure it's *not* defined, otherwise make
        // sure that it is defined.
        if (KV.second.hasMaterializationSideEffectsOnly()) {
          ++NumMaterializationSideEffectsOnlySymbols;
          if (InternedResult.count(KV.first))
            ExtraSymbols.push_back(KV.first);
          continue;
        } else if (!InternedResult.count(KV.first))
          MissingSymbols.push_back(KV.first);
      }

      // If there were missing symbols then report the error.
      if (!MissingSymbols.empty()) {
        ES.reportError(make_error<MissingSymbolDefinitions>(
            G.getName(), std::move(MissingSymbols)));
        MR.failMaterialization();
        return;
      }

      // If there are more definitions than expected, add them to the
      // ExtraSymbols vector.
      if (InternedResult.size() >
          MR.getSymbols().size() - NumMaterializationSideEffectsOnlySymbols) {
        for (auto &KV : InternedResult)
          if (!MR.getSymbols().count(KV.first))
            ExtraSymbols.push_back(KV.first);
      }

      // If there were extra definitions then report the error.
      if (!ExtraSymbols.empty()) {
        ES.reportError(make_error<UnexpectedSymbolDefinitions>(
            G.getName(), std::move(ExtraSymbols)));
        MR.failMaterialization();
        return;
      }
    }

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

    Layer.modifyPassConfig(MR, TT, Config);

    Config.PostPrunePasses.push_back(
        [this](LinkGraph &G) { return computeNamedSymbolDependencies(G); });

    return Error::success();
  }

private:
  struct LocalSymbolNamedDependencies {
    SymbolNameSet Internal, External;
  };

  using LocalSymbolNamedDependenciesMap =
      DenseMap<const Symbol *, LocalSymbolNamedDependencies>;

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
    auto LocalDeps = computeLocalDeps(G);

    // Compute dependencies for symbols defined in the JITLink graph.
    for (auto *Sym : G.defined_symbols()) {

      // Skip local symbols: we do not track dependencies for these.
      if (Sym->getScope() == Scope::Local)
        continue;
      assert(Sym->hasName() &&
             "Defined non-local jitlink::Symbol should have a name");

      SymbolNameSet ExternalSymDeps, InternalSymDeps;

      // Find internal and external named symbol dependencies.
      for (auto &E : Sym->getBlock().edges()) {
        auto &TargetSym = E.getTarget();

        if (TargetSym.getScope() != Scope::Local) {
          if (TargetSym.isExternal())
            ExternalSymDeps.insert(ES.intern(TargetSym.getName()));
          else if (&TargetSym != Sym)
            InternalSymDeps.insert(ES.intern(TargetSym.getName()));
        } else {
          assert(TargetSym.isDefined() &&
                 "local symbols must be defined");
          auto I = LocalDeps.find(&TargetSym);
          if (I != LocalDeps.end()) {
            for (auto &S : I->second.External)
              ExternalSymDeps.insert(S);
            for (auto &S : I->second.Internal)
              InternalSymDeps.insert(S);
          }
        }
      }

      if (ExternalSymDeps.empty() && InternalSymDeps.empty())
        continue;

      auto SymName = ES.intern(Sym->getName());
      if (!ExternalSymDeps.empty())
        ExternalNamedSymbolDeps[SymName] = std::move(ExternalSymDeps);
      if (!InternalSymDeps.empty())
        InternalNamedSymbolDeps[SymName] = std::move(InternalSymDeps);
    }

    for (auto &P : Layer.Plugins) {
      auto SyntheticLocalDeps = P->getSyntheticSymbolLocalDependencies(MR);
      if (SyntheticLocalDeps.empty())
        continue;

      for (auto &KV : SyntheticLocalDeps) {
        auto &Name = KV.first;
        auto &LocalDepsForName = KV.second;
        for (auto *Local : LocalDepsForName) {
          assert(Local->getScope() == Scope::Local &&
                 "Dependence on non-local symbol");
          auto LocalNamedDepsItr = LocalDeps.find(Local);
          if (LocalNamedDepsItr == LocalDeps.end())
            continue;
          for (auto &S : LocalNamedDepsItr->second.Internal)
            InternalNamedSymbolDeps[Name].insert(S);
          for (auto &S : LocalNamedDepsItr->second.External)
            ExternalNamedSymbolDeps[Name].insert(S);
        }
      }
    }

    return Error::success();
  }

  LocalSymbolNamedDependenciesMap computeLocalDeps(LinkGraph &G) {
    DenseMap<jitlink::Symbol *, DenseSet<jitlink::Symbol *>> DepMap;

    // For all local symbols:
    // (1) Add their named dependencies.
    // (2) Add them to the worklist for further iteration if they have any
    //     depend on any other local symbols.
    struct WorklistEntry {
      WorklistEntry(Symbol *Sym, DenseSet<Symbol *> LocalDeps)
          : Sym(Sym), LocalDeps(std::move(LocalDeps)) {}

      Symbol *Sym = nullptr;
      DenseSet<Symbol *> LocalDeps;
    };
    std::vector<WorklistEntry> Worklist;
    for (auto *Sym : G.defined_symbols())
      if (Sym->getScope() == Scope::Local) {
        auto &SymNamedDeps = DepMap[Sym];
        DenseSet<Symbol *> LocalDeps;

        for (auto &E : Sym->getBlock().edges()) {
          auto &TargetSym = E.getTarget();
          if (TargetSym.getScope() != Scope::Local)
            SymNamedDeps.insert(&TargetSym);
          else {
            assert(TargetSym.isDefined() &&
                   "local symbols must be defined");
            LocalDeps.insert(&TargetSym);
          }
        }

        if (!LocalDeps.empty())
          Worklist.push_back(WorklistEntry(Sym, std::move(LocalDeps)));
      }

    // Loop over all local symbols with local dependencies, propagating
    // their respective non-local dependencies. Iterate until we hit a stable
    // state.
    bool Changed;
    do {
      Changed = false;
      for (auto &WLEntry : Worklist) {
        auto *Sym = WLEntry.Sym;
        auto &NamedDeps = DepMap[Sym];
        auto &LocalDeps = WLEntry.LocalDeps;

        for (auto *TargetSym : LocalDeps) {
          auto I = DepMap.find(TargetSym);
          if (I != DepMap.end())
            for (const auto &S : I->second)
              Changed |= NamedDeps.insert(S).second;
        }
      }
    } while (Changed);

    // Intern the results to produce a mapping of jitlink::Symbol* to internal
    // and external symbol names.
    auto &ES = Layer.getExecutionSession();
    LocalSymbolNamedDependenciesMap Result;
    for (auto &KV : DepMap) {
      auto *Local = KV.first;
      assert(Local->getScope() == Scope::Local &&
             "DepMap keys should all be local symbols");
      auto &LocalNamedDeps = Result[Local];
      for (auto *Named : KV.second) {
        assert(Named->getScope() != Scope::Local &&
               "DepMap values should all be non-local symbol sets");
        if (Named->isExternal())
          LocalNamedDeps.External.insert(ES.intern(Named->getName()));
        else
          LocalNamedDeps.Internal.insert(ES.intern(Named->getName()));
      }
    }

    return Result;
  }

  void registerDependencies(const SymbolDependenceMap &QueryDeps) {
    for (auto &NamedDepsEntry : ExternalNamedSymbolDeps) {
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
  DenseMap<SymbolStringPtr, SymbolNameSet> ExternalNamedSymbolDeps;
  DenseMap<SymbolStringPtr, SymbolNameSet> InternalNamedSymbolDeps;
};

ObjectLinkingLayer::Plugin::~Plugin() {}

ObjectLinkingLayer::ObjectLinkingLayer(ExecutionSession &ES,
                                       JITLinkMemoryManager &MemMgr)
    : ObjectLayer(ES), MemMgr(MemMgr) {}

ObjectLinkingLayer::ObjectLinkingLayer(
    ExecutionSession &ES, std::unique_ptr<JITLinkMemoryManager> MemMgr)
    : ObjectLayer(ES), MemMgr(*MemMgr), MemMgrOwnership(std::move(MemMgr)) {}

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

  PassConfig.PostFixupPasses.push_back(createEHFrameRecorderPass(
      TT, [this, &MR](JITTargetAddress Addr, size_t Size) {
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
  std::lock_guard<std::mutex> Lock(EHFramePluginMutex);

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
  std::lock_guard<std::mutex> Lock(EHFramePluginMutex);

  auto EHFrameRangeItr = TrackedEHFrameRanges.find(K);
  if (EHFrameRangeItr == TrackedEHFrameRanges.end())
    return Error::success();

  auto EHFrameRange = EHFrameRangeItr->second;
  assert(EHFrameRange.Addr && "Tracked eh-frame range must not be null");

  TrackedEHFrameRanges.erase(EHFrameRangeItr);

  return Registrar.deregisterEHFrames(EHFrameRange.Addr, EHFrameRange.Size);
}

Error EHFrameRegistrationPlugin::notifyRemovingAllModules() {
  std::lock_guard<std::mutex> Lock(EHFramePluginMutex);

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
