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

  JITLinkMemoryManager &getMemoryManager() override { return Layer.MemMgr; }

  MemoryBufferRef getObjectBuffer() const override {
    return ObjBuffer->getMemBufferRef();
  }

  void notifyFailed(Error Err) override {
    Layer.getExecutionSession().reportError(std::move(Err));
    MR.failMaterialization();
  }

  void lookup(const DenseSet<StringRef> &Symbols,
              JITLinkAsyncLookupContinuation LookupContinuation) override {

    JITDylibSearchList SearchOrder;
    MR.getTargetJITDylib().withSearchOrderDo(
        [&](const JITDylibSearchList &JDs) { SearchOrder = JDs; });

    auto &ES = Layer.getExecutionSession();

    SymbolNameSet InternedSymbols;
    for (auto &S : Symbols)
      InternedSymbols.insert(ES.intern(S));

    // OnResolve -- De-intern the symbols and pass the result to the linker.
    // FIXME: Capture LookupContinuation by move once we have c++14.
    auto SharedLookupContinuation =
        std::make_shared<JITLinkAsyncLookupContinuation>(
            std::move(LookupContinuation));
    auto OnResolve = [SharedLookupContinuation](Expected<SymbolMap> Result) {
      if (!Result)
        (*SharedLookupContinuation)(Result.takeError());
      else {
        AsyncLookupResult LR;
        for (auto &KV : *Result)
          LR[*KV.first] = KV.second;
        (*SharedLookupContinuation)(std::move(LR));
      }
    };

    ES.lookup(SearchOrder, std::move(InternedSymbols), SymbolState::Resolved,
              std::move(OnResolve), [this](const SymbolDependenceMap &Deps) {
                registerDependencies(Deps);
              });
  }

  void notifyResolved(AtomGraph &G) override {
    auto &ES = Layer.getExecutionSession();

    SymbolFlagsMap ExtraSymbolsToClaim;
    bool AutoClaim = Layer.AutoClaimObjectSymbols;

    SymbolMap InternedResult;
    for (auto *DA : G.defined_atoms())
      if (DA->hasName() && DA->isGlobal()) {
        auto InternedName = ES.intern(DA->getName());
        JITSymbolFlags Flags;

        if (DA->isExported())
          Flags |= JITSymbolFlags::Exported;
        if (DA->isWeak())
          Flags |= JITSymbolFlags::Weak;
        if (DA->isCallable())
          Flags |= JITSymbolFlags::Callable;
        if (DA->isCommon())
          Flags |= JITSymbolFlags::Common;

        InternedResult[InternedName] =
            JITEvaluatedSymbol(DA->getAddress(), Flags);
        if (AutoClaim && !MR.getSymbols().count(InternedName)) {
          assert(!ExtraSymbolsToClaim.count(InternedName) &&
                 "Duplicate symbol to claim?");
          ExtraSymbolsToClaim[InternedName] = Flags;
        }
      }

    for (auto *A : G.absolute_atoms())
      if (A->hasName()) {
        auto InternedName = ES.intern(A->getName());
        JITSymbolFlags Flags;
        Flags |= JITSymbolFlags::Absolute;
        if (A->isWeak())
          Flags |= JITSymbolFlags::Weak;
        if (A->isCallable())
          Flags |= JITSymbolFlags::Callable;
        InternedResult[InternedName] =
            JITEvaluatedSymbol(A->getAddress(), Flags);
        if (AutoClaim && !MR.getSymbols().count(InternedName)) {
          assert(!ExtraSymbolsToClaim.count(InternedName) &&
                 "Duplicate symbol to claim?");
          ExtraSymbolsToClaim[InternedName] = Flags;
        }
      }

    if (!ExtraSymbolsToClaim.empty())
      if (auto Err = MR.defineMaterializing(ExtraSymbolsToClaim))
        return notifyFailed(std::move(Err));

    MR.notifyResolved(InternedResult);

    Layer.notifyLoaded(MR);
  }

  void notifyFinalized(
      std::unique_ptr<JITLinkMemoryManager::Allocation> A) override {

    if (auto Err = Layer.notifyEmitted(MR, std::move(A))) {
      Layer.getExecutionSession().reportError(std::move(Err));
      MR.failMaterialization();

      return;
    }
    MR.notifyEmitted();
  }

  AtomGraphPassFunction getMarkLivePass(const Triple &TT) const override {
    return [this](AtomGraph &G) { return markResponsibilitySymbolsLive(G); };
  }

  Error modifyPassConfig(const Triple &TT, PassConfiguration &Config) override {
    // Add passes to mark duplicate defs as should-discard, and to walk the
    // atom graph to build the symbol dependence graph.
    Config.PrePrunePasses.push_back(
        [this](AtomGraph &G) { return markSymbolsToDiscard(G); });
    Config.PostPrunePasses.push_back(
        [this](AtomGraph &G) { return computeNamedSymbolDependencies(G); });

    Layer.modifyPassConfig(MR, TT, Config);

    return Error::success();
  }

private:
  using AnonAtomNamedDependenciesMap =
      DenseMap<const DefinedAtom *, SymbolNameSet>;

  Error markSymbolsToDiscard(AtomGraph &G) {
    auto &ES = Layer.getExecutionSession();
    for (auto *DA : G.defined_atoms())
      if (DA->isWeak() && DA->hasName()) {
        auto S = ES.intern(DA->getName());
        auto I = MR.getSymbols().find(S);
        if (I == MR.getSymbols().end())
          DA->setShouldDiscard(true);
      }

    for (auto *A : G.absolute_atoms())
      if (A->isWeak() && A->hasName()) {
        auto S = ES.intern(A->getName());
        auto I = MR.getSymbols().find(S);
        if (I == MR.getSymbols().end())
          A->setShouldDiscard(true);
      }

    return Error::success();
  }

  Error markResponsibilitySymbolsLive(AtomGraph &G) const {
    auto &ES = Layer.getExecutionSession();
    for (auto *DA : G.defined_atoms())
      if (DA->hasName() &&
          MR.getSymbols().count(ES.intern(DA->getName())))
        DA->setLive(true);
    return Error::success();
  }

  Error computeNamedSymbolDependencies(AtomGraph &G) {
    auto &ES = MR.getTargetJITDylib().getExecutionSession();
    auto AnonDeps = computeAnonDeps(G);

    for (auto *DA : G.defined_atoms()) {

      // Skip anonymous and non-global atoms: we do not need dependencies for
      // these.
      if (!DA->hasName() || !DA->isGlobal())
        continue;

      auto DAName = ES.intern(DA->getName());
      SymbolNameSet &DADeps = NamedSymbolDeps[DAName];

      for (auto &E : DA->edges()) {
        auto &TA = E.getTarget();

        if (TA.hasName())
          DADeps.insert(ES.intern(TA.getName()));
        else {
          assert(TA.isDefined() && "Anonymous atoms must be defined");
          auto &DTA = static_cast<DefinedAtom &>(TA);
          auto I = AnonDeps.find(&DTA);
          if (I != AnonDeps.end())
            for (auto &S : I->second)
              DADeps.insert(S);
        }
      }
    }

    return Error::success();
  }

  AnonAtomNamedDependenciesMap computeAnonDeps(AtomGraph &G) {

    auto &ES = MR.getTargetJITDylib().getExecutionSession();
    AnonAtomNamedDependenciesMap DepMap;

    // For all anonymous atoms:
    // (1) Add their named dependencies.
    // (2) Add them to the worklist for further iteration if they have any
    //     depend on any other anonymous atoms.
    struct WorklistEntry {
      WorklistEntry(DefinedAtom *DA, DenseSet<DefinedAtom *> DAAnonDeps)
          : DA(DA), DAAnonDeps(std::move(DAAnonDeps)) {}

      DefinedAtom *DA = nullptr;
      DenseSet<DefinedAtom *> DAAnonDeps;
    };
    std::vector<WorklistEntry> Worklist;
    for (auto *DA : G.defined_atoms())
      if (!DA->hasName()) {
        auto &DANamedDeps = DepMap[DA];
        DenseSet<DefinedAtom *> DAAnonDeps;

        for (auto &E : DA->edges()) {
          auto &TA = E.getTarget();
          if (TA.hasName())
            DANamedDeps.insert(ES.intern(TA.getName()));
          else {
            assert(TA.isDefined() && "Anonymous atoms must be defined");
            DAAnonDeps.insert(static_cast<DefinedAtom *>(&TA));
          }
        }

        if (!DAAnonDeps.empty())
          Worklist.push_back(WorklistEntry(DA, std::move(DAAnonDeps)));
      }

    // Loop over all anonymous atoms with anonymous dependencies, propagating
    // their respective *named* dependencies. Iterate until we hit a stable
    // state.
    bool Changed;
    do {
      Changed = false;
      for (auto &WLEntry : Worklist) {
        auto *DA = WLEntry.DA;
        auto &DANamedDeps = DepMap[DA];
        auto &DAAnonDeps = WLEntry.DAAnonDeps;

        for (auto *TA : DAAnonDeps) {
          auto I = DepMap.find(TA);
          if (I != DepMap.end())
            for (const auto &S : I->second)
              Changed |= DANamedDeps.insert(S).second;
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

ObjectLinkingLayer::ObjectLinkingLayer(ExecutionSession &ES,
                                       JITLinkMemoryManager &MemMgr)
    : ObjectLayer(ES), MemMgr(MemMgr) {}

ObjectLinkingLayer::~ObjectLinkingLayer() {
  if (auto Err = removeAllModules())
    getExecutionSession().reportError(std::move(Err));
}

void ObjectLinkingLayer::emit(MaterializationResponsibility R,
                              std::unique_ptr<MemoryBuffer> O) {
  assert(O && "Object must not be null");
  jitLink(llvm::make_unique<ObjectLinkingLayerJITLinkContext>(
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
    jitlink::EHFrameRegistrar &Registrar)
    : Registrar(Registrar) {}

void EHFrameRegistrationPlugin::modifyPassConfig(
    MaterializationResponsibility &MR, const Triple &TT,
    PassConfiguration &PassConfig) {
  assert(!InProcessLinks.count(&MR) && "Link for MR already being tracked?");

  PassConfig.PostFixupPasses.push_back(
      createEHFrameRecorderPass(TT, [this, &MR](JITTargetAddress Addr) {
        if (Addr)
          InProcessLinks[&MR] = Addr;
      }));
}

Error EHFrameRegistrationPlugin::notifyEmitted(
    MaterializationResponsibility &MR) {

  auto EHFrameAddrItr = InProcessLinks.find(&MR);
  if (EHFrameAddrItr == InProcessLinks.end())
    return Error::success();

  auto EHFrameAddr = EHFrameAddrItr->second;
  assert(EHFrameAddr && "eh-frame addr to register can not be null");

  InProcessLinks.erase(EHFrameAddrItr);
  if (auto Key = MR.getVModuleKey())
    TrackedEHFrameAddrs[Key] = EHFrameAddr;
  else
    UntrackedEHFrameAddrs.push_back(EHFrameAddr);

  return Registrar.registerEHFrames(EHFrameAddr);
}

Error EHFrameRegistrationPlugin::notifyRemovingModule(VModuleKey K) {
  auto EHFrameAddrItr = TrackedEHFrameAddrs.find(K);
  if (EHFrameAddrItr == TrackedEHFrameAddrs.end())
    return Error::success();

  auto EHFrameAddr = EHFrameAddrItr->second;
  assert(EHFrameAddr && "Tracked eh-frame addr must not be null");

  TrackedEHFrameAddrs.erase(EHFrameAddrItr);

  return Registrar.deregisterEHFrames(EHFrameAddr);
}

Error EHFrameRegistrationPlugin::notifyRemovingAllModules() {

  std::vector<JITTargetAddress> EHFrameAddrs = std::move(UntrackedEHFrameAddrs);
  EHFrameAddrs.reserve(EHFrameAddrs.size() + TrackedEHFrameAddrs.size());

  for (auto &KV : TrackedEHFrameAddrs)
    EHFrameAddrs.push_back(KV.second);

  TrackedEHFrameAddrs.clear();

  Error Err = Error::success();

  while (!EHFrameAddrs.empty()) {
    auto EHFrameAddr = EHFrameAddrs.back();
    assert(EHFrameAddr && "Untracked eh-frame addr must not be null");
    EHFrameAddrs.pop_back();
    Err = joinErrors(std::move(Err), Registrar.deregisterEHFrames(EHFrameAddr));
  }

  return Err;
}

} // End namespace orc.
} // End namespace llvm.
