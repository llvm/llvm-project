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
#include "llvm/ExecutionEngine/Orc/DebugObjectManagerPlugin.h"
#include "llvm/Support/MemoryBuffer.h"

#include <vector>

#define DEBUG_TYPE "orc"

using namespace llvm;
using namespace llvm::jitlink;
using namespace llvm::orc;

namespace {

class LinkGraphMaterializationUnit : public MaterializationUnit {
private:
  struct LinkGraphInterface {
    SymbolFlagsMap SymbolFlags;
    SymbolStringPtr InitSymbol;
  };

public:
  static std::unique_ptr<LinkGraphMaterializationUnit>
  Create(ObjectLinkingLayer &ObjLinkingLayer, std::unique_ptr<LinkGraph> G) {
    auto LGI = scanLinkGraph(ObjLinkingLayer.getExecutionSession(), *G);
    return std::unique_ptr<LinkGraphMaterializationUnit>(
        new LinkGraphMaterializationUnit(ObjLinkingLayer, std::move(G),
                                         std::move(LGI)));
  }

  StringRef getName() const override { return G->getName(); }
  void materialize(std::unique_ptr<MaterializationResponsibility> MR) override {
    ObjLinkingLayer.emit(std::move(MR), std::move(G));
  }

private:
  static LinkGraphInterface scanLinkGraph(ExecutionSession &ES, LinkGraph &G) {

    LinkGraphInterface LGI;

    for (auto *Sym : G.defined_symbols()) {
      // Skip local symbols.
      if (Sym->getScope() == Scope::Local)
        continue;
      assert(Sym->hasName() && "Anonymous non-local symbol?");

      JITSymbolFlags Flags;
      if (Sym->getScope() == Scope::Default)
        Flags |= JITSymbolFlags::Exported;

      if (Sym->isCallable())
        Flags |= JITSymbolFlags::Callable;

      LGI.SymbolFlags[ES.intern(Sym->getName())] = Flags;
    }

    if (G.getTargetTriple().isOSBinFormatMachO())
      if (hasMachOInitSection(G))
        LGI.InitSymbol = makeInitSymbol(ES, G);

    return LGI;
  }

  static bool hasMachOInitSection(LinkGraph &G) {
    for (auto &Sec : G.sections())
      if (Sec.getName() == "__DATA,__obj_selrefs" ||
          Sec.getName() == "__DATA,__objc_classlist" ||
          Sec.getName() == "__TEXT,__swift5_protos" ||
          Sec.getName() == "__TEXT,__swift5_proto" ||
          Sec.getName() == "__DATA,__mod_init_func")
        return true;
    return false;
  }

  static SymbolStringPtr makeInitSymbol(ExecutionSession &ES, LinkGraph &G) {
    std::string InitSymString;
    raw_string_ostream(InitSymString)
        << "$." << G.getName() << ".__inits" << Counter++;
    return ES.intern(InitSymString);
  }

  LinkGraphMaterializationUnit(ObjectLinkingLayer &ObjLinkingLayer,
                               std::unique_ptr<LinkGraph> G,
                               LinkGraphInterface LGI)
      : MaterializationUnit(std::move(LGI.SymbolFlags),
                            std::move(LGI.InitSymbol)),
        ObjLinkingLayer(ObjLinkingLayer), G(std::move(G)) {}

  void discard(const JITDylib &JD, const SymbolStringPtr &Name) override {
    for (auto *Sym : G->defined_symbols())
      if (Sym->getName() == *Name) {
        assert(Sym->getLinkage() == Linkage::Weak &&
               "Discarding non-weak definition");
        G->makeExternal(*Sym);
        break;
      }
  }

  ObjectLinkingLayer &ObjLinkingLayer;
  std::unique_ptr<LinkGraph> G;
  static std::atomic<uint64_t> Counter;
};

std::atomic<uint64_t> LinkGraphMaterializationUnit::Counter{0};

} // end anonymous namespace

namespace llvm {
namespace orc {

class ObjectLinkingLayerJITLinkContext final : public JITLinkContext {
public:
  ObjectLinkingLayerJITLinkContext(
      ObjectLinkingLayer &Layer,
      std::unique_ptr<MaterializationResponsibility> MR,
      std::unique_ptr<MemoryBuffer> ObjBuffer)
      : JITLinkContext(&MR->getTargetJITDylib()), Layer(Layer),
        MR(std::move(MR)), ObjBuffer(std::move(ObjBuffer)) {}

  ~ObjectLinkingLayerJITLinkContext() {
    // If there is an object buffer return function then use it to
    // return ownership of the buffer.
    if (Layer.ReturnObjectBuffer && ObjBuffer)
      Layer.ReturnObjectBuffer(std::move(ObjBuffer));
  }

  JITLinkMemoryManager &getMemoryManager() override { return Layer.MemMgr; }

  void notifyMaterializing(LinkGraph &G) {
    for (auto &P : Layer.Plugins)
      P->notifyMaterializing(*MR, G, *this,
                             ObjBuffer ? ObjBuffer->getMemBufferRef()
                             : MemoryBufferRef());
  }

  void notifyFailed(Error Err) override {
    for (auto &P : Layer.Plugins)
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
      LookupSet.add(ES.intern(KV.first), LookupFlags);
    }

    // OnResolve -- De-intern the symbols and pass the result to the linker.
    auto OnResolve = [LookupContinuation =
                          std::move(LC)](Expected<SymbolMap> Result) mutable {
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
      InternalDeps[&MR->getTargetJITDylib()] = std::move(KV.second);
      MR->addDependencies(KV.first, InternalDeps);
    }

    ES.lookup(LookupKind::Static, LinkOrder, std::move(LookupSet),
              SymbolState::Resolved, std::move(OnResolve),
              [this](const SymbolDependenceMap &Deps) {
                registerDependencies(Deps);
              });
  }

  Error notifyResolved(LinkGraph &G) override {
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
        if (AutoClaim && !MR->getSymbols().count(InternedName)) {
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
        if (AutoClaim && !MR->getSymbols().count(InternedName)) {
          assert(!ExtraSymbolsToClaim.count(InternedName) &&
                 "Duplicate symbol to claim?");
          ExtraSymbolsToClaim[InternedName] = Flags;
        }
      }

    if (!ExtraSymbolsToClaim.empty())
      if (auto Err = MR->defineMaterializing(ExtraSymbolsToClaim))
        return Err;

    {

      // Check that InternedResult matches up with MR->getSymbols().
      // This guards against faulty transformations / compilers / object caches.

      // First check that there aren't any missing symbols.
      size_t NumMaterializationSideEffectsOnlySymbols = 0;
      SymbolNameVector ExtraSymbols;
      SymbolNameVector MissingSymbols;
      for (auto &KV : MR->getSymbols()) {

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
      if (!MissingSymbols.empty())
        return make_error<MissingSymbolDefinitions>(G.getName(),
                                                    std::move(MissingSymbols));

      // If there are more definitions than expected, add them to the
      // ExtraSymbols vector.
      if (InternedResult.size() >
          MR->getSymbols().size() - NumMaterializationSideEffectsOnlySymbols) {
        for (auto &KV : InternedResult)
          if (!MR->getSymbols().count(KV.first))
            ExtraSymbols.push_back(KV.first);
      }

      // If there were extra definitions then report the error.
      if (!ExtraSymbols.empty())
        return make_error<UnexpectedSymbolDefinitions>(G.getName(),
                                                       std::move(ExtraSymbols));
    }

    if (auto Err = MR->notifyResolved(InternedResult))
      return Err;

    Layer.notifyLoaded(*MR);
    return Error::success();
  }

  void notifyFinalized(
      std::unique_ptr<JITLinkMemoryManager::Allocation> A) override {
    if (auto Err = Layer.notifyEmitted(*MR, std::move(A))) {
      Layer.getExecutionSession().reportError(std::move(Err));
      MR->failMaterialization();
      return;
    }
    if (auto Err = MR->notifyEmitted()) {
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

    Layer.modifyPassConfig(*MR, LG, Config);

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

  Error claimOrExternalizeWeakAndCommonSymbols(LinkGraph &G) {
    auto &ES = Layer.getExecutionSession();

    SymbolFlagsMap NewSymbolsToClaim;
    std::vector<std::pair<SymbolStringPtr, Symbol *>> NameToSym;

    auto ProcessSymbol = [&](Symbol *Sym) {
      if (Sym->hasName() && Sym->getLinkage() == Linkage::Weak) {
        auto Name = ES.intern(Sym->getName());
        if (!MR->getSymbols().count(ES.intern(Sym->getName()))) {
          JITSymbolFlags SF = JITSymbolFlags::Weak;
          if (Sym->getScope() == Scope::Default)
            SF |= JITSymbolFlags::Exported;
          NewSymbolsToClaim[Name] = SF;
          NameToSym.push_back(std::make_pair(std::move(Name), Sym));
        }
      }
    };

    for (auto *Sym : G.defined_symbols())
      ProcessSymbol(Sym);
    for (auto *Sym : G.absolute_symbols())
      ProcessSymbol(Sym);

    // Attempt to claim all weak defs that we're not already responsible for.
    // This cannot fail -- any clashes will just result in rejection of our
    // claim, at which point we'll externalize that symbol.
    cantFail(MR->defineMaterializing(std::move(NewSymbolsToClaim)));

    for (auto &KV : NameToSym)
      if (!MR->getSymbols().count(KV.first))
        G.makeExternal(*KV.second);

    return Error::success();
  }

  Error markResponsibilitySymbolsLive(LinkGraph &G) const {
    auto &ES = Layer.getExecutionSession();
    for (auto *Sym : G.defined_symbols())
      if (Sym->hasName() && MR->getSymbols().count(ES.intern(Sym->getName())))
        Sym->setLive(true);
    return Error::success();
  }

  Error computeNamedSymbolDependencies(LinkGraph &G) {
    auto &ES = MR->getTargetJITDylib().getExecutionSession();
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
      auto SyntheticLocalDeps = P->getSyntheticSymbolLocalDependencies(*MR);
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

      MR->addDependencies(Name, SymbolDeps);
    }
  }

  ObjectLinkingLayer &Layer;
  std::unique_ptr<MaterializationResponsibility> MR;
  std::unique_ptr<MemoryBuffer> ObjBuffer;
  DenseMap<SymbolStringPtr, SymbolNameSet> ExternalNamedSymbolDeps;
  DenseMap<SymbolStringPtr, SymbolNameSet> InternalNamedSymbolDeps;
};

ObjectLinkingLayer::Plugin::~Plugin() {}

char ObjectLinkingLayer::ID;

using BaseT = RTTIExtends<ObjectLinkingLayer, ObjectLayer>;

ObjectLinkingLayer::ObjectLinkingLayer(ExecutionSession &ES,
                                       JITLinkMemoryManager &MemMgr)
    : BaseT(ES), MemMgr(MemMgr) {
  ES.registerResourceManager(*this);
}

ObjectLinkingLayer::ObjectLinkingLayer(
    ExecutionSession &ES, std::unique_ptr<JITLinkMemoryManager> MemMgr)
    : BaseT(ES), MemMgr(*MemMgr), MemMgrOwnership(std::move(MemMgr)) {
  ES.registerResourceManager(*this);
}

ObjectLinkingLayer::~ObjectLinkingLayer() {
  assert(Allocs.empty() && "Layer destroyed with resources still attached");
  getExecutionSession().deregisterResourceManager(*this);
}

Error ObjectLinkingLayer::add(ResourceTrackerSP RT,
                              std::unique_ptr<LinkGraph> G) {
  auto &JD = RT->getJITDylib();
  return JD.define(LinkGraphMaterializationUnit::Create(*this, std::move(G)),
                   std::move(RT));
}

void ObjectLinkingLayer::emit(std::unique_ptr<MaterializationResponsibility> R,
                              std::unique_ptr<MemoryBuffer> O) {
  assert(O && "Object must not be null");
  MemoryBufferRef ObjBuffer = O->getMemBufferRef();

  auto Ctx = std::make_unique<ObjectLinkingLayerJITLinkContext>(
      *this, std::move(R), std::move(O));
  if (auto G = createLinkGraphFromObject(ObjBuffer)) {
    Ctx->notifyMaterializing(**G);
    link(std::move(*G), std::move(Ctx));
  } else {
    Ctx->notifyFailed(G.takeError());
  }
}

void ObjectLinkingLayer::emit(std::unique_ptr<MaterializationResponsibility> R,
                              std::unique_ptr<LinkGraph> G) {
  auto Ctx = std::make_unique<ObjectLinkingLayerJITLinkContext>(
      *this, std::move(R), nullptr);
  Ctx->notifyMaterializing(*G);
  link(std::move(G), std::move(Ctx));
}

void ObjectLinkingLayer::modifyPassConfig(MaterializationResponsibility &MR,
                                          LinkGraph &G,
                                          PassConfiguration &PassConfig) {
  for (auto &P : Plugins)
    P->modifyPassConfig(MR, G, PassConfig);
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

  return MR.withResourceKeyDo(
      [&](ResourceKey K) { Allocs[K].push_back(std::move(Alloc)); });
}

Error ObjectLinkingLayer::handleRemoveResources(ResourceKey K) {

  Error Err = Error::success();

  for (auto &P : Plugins)
    Err = joinErrors(std::move(Err), P->notifyRemovingResources(K));

  std::vector<AllocPtr> AllocsToRemove;
  getExecutionSession().runSessionLocked([&] {
    auto I = Allocs.find(K);
    if (I != Allocs.end()) {
      std::swap(AllocsToRemove, I->second);
      Allocs.erase(I);
    }
  });

  while (!AllocsToRemove.empty()) {
    Err = joinErrors(std::move(Err), AllocsToRemove.back()->deallocate());
    AllocsToRemove.pop_back();
  }

  return Err;
}

void ObjectLinkingLayer::handleTransferResources(ResourceKey DstKey,
                                                 ResourceKey SrcKey) {
  auto I = Allocs.find(SrcKey);
  if (I != Allocs.end()) {
    auto &SrcAllocs = I->second;
    auto &DstAllocs = Allocs[DstKey];
    DstAllocs.reserve(DstAllocs.size() + SrcAllocs.size());
    for (auto &Alloc : SrcAllocs)
      DstAllocs.push_back(std::move(Alloc));

    // Erase SrcKey entry using value rather than iterator I: I may have been
    // invalidated when we looked up DstKey.
    Allocs.erase(SrcKey);
  }

  for (auto &P : Plugins)
    P->notifyTransferringResources(DstKey, SrcKey);
}

EHFrameRegistrationPlugin::EHFrameRegistrationPlugin(
    ExecutionSession &ES, std::unique_ptr<EHFrameRegistrar> Registrar)
    : ES(ES), Registrar(std::move(Registrar)) {}

void EHFrameRegistrationPlugin::modifyPassConfig(
    MaterializationResponsibility &MR, LinkGraph &G,
    PassConfiguration &PassConfig) {

  PassConfig.PostFixupPasses.push_back(createEHFrameRecorderPass(
      G.getTargetTriple(), [this, &MR](JITTargetAddress Addr, size_t Size) {
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

  EHFrameRange EmittedRange;
  {
    std::lock_guard<std::mutex> Lock(EHFramePluginMutex);

    auto EHFrameRangeItr = InProcessLinks.find(&MR);
    if (EHFrameRangeItr == InProcessLinks.end())
      return Error::success();

    EmittedRange = EHFrameRangeItr->second;
    assert(EmittedRange.Addr && "eh-frame addr to register can not be null");
    InProcessLinks.erase(EHFrameRangeItr);
  }

  if (auto Err = MR.withResourceKeyDo(
          [&](ResourceKey K) { EHFrameRanges[K].push_back(EmittedRange); }))
    return Err;

  return Registrar->registerEHFrames(EmittedRange.Addr, EmittedRange.Size);
}

Error EHFrameRegistrationPlugin::notifyFailed(
    MaterializationResponsibility &MR) {
  std::lock_guard<std::mutex> Lock(EHFramePluginMutex);
  InProcessLinks.erase(&MR);
  return Error::success();
}

Error EHFrameRegistrationPlugin::notifyRemovingResources(ResourceKey K) {
  std::vector<EHFrameRange> RangesToRemove;

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
    assert(RangeToRemove.Addr && "Untracked eh-frame range must not be null");
    Err = joinErrors(
        std::move(Err),
        Registrar->deregisterEHFrames(RangeToRemove.Addr, RangeToRemove.Size));
  }

  return Err;
}

void EHFrameRegistrationPlugin::notifyTransferringResources(
    ResourceKey DstKey, ResourceKey SrcKey) {
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
