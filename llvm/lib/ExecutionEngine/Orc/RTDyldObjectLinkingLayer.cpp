//===-- RTDyldObjectLinkingLayer.cpp - RuntimeDyld backed ORC ObjectLayer -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ExecutionEngine/Orc/RTDyldObjectLinkingLayer.h"
#include "llvm/Object/COFF.h"

namespace {

using namespace llvm;
using namespace llvm::orc;

class JITDylibSearchOrderResolver : public JITSymbolResolver {
public:
  JITDylibSearchOrderResolver(MaterializationResponsibility &MR) : MR(MR) {}

  void lookup(const LookupSet &Symbols, OnResolvedFunction OnResolved) {
    auto &ES = MR.getTargetJITDylib().getExecutionSession();
    SymbolLookupSet InternedSymbols;

    // Intern the requested symbols: lookup takes interned strings.
    for (auto &S : Symbols)
      InternedSymbols.add(ES.intern(S));

    // Build an OnResolve callback to unwrap the interned strings and pass them
    // to the OnResolved callback.
    auto OnResolvedWithUnwrap =
        [OnResolved = std::move(OnResolved)](
            Expected<SymbolMap> InternedResult) mutable {
          if (!InternedResult) {
            OnResolved(InternedResult.takeError());
            return;
          }

          LookupResult Result;
          for (auto &KV : *InternedResult)
            Result[*KV.first] = std::move(KV.second);
          OnResolved(Result);
        };

    // Register dependencies for all symbols contained in this set.
    auto RegisterDependencies = [&](const SymbolDependenceMap &Deps) {
      MR.addDependenciesForAll(Deps);
    };

    JITDylibSearchOrder SearchOrder;
    MR.getTargetJITDylib().withSearchOrderDo(
        [&](const JITDylibSearchOrder &JDs) { SearchOrder = JDs; });
    ES.lookup(LookupKind::Static, SearchOrder, InternedSymbols,
              SymbolState::Resolved, std::move(OnResolvedWithUnwrap),
              RegisterDependencies);
  }

  Expected<LookupSet> getResponsibilitySet(const LookupSet &Symbols) {
    LookupSet Result;

    for (auto &KV : MR.getSymbols()) {
      if (Symbols.count(*KV.first))
        Result.insert(*KV.first);
    }

    return Result;
  }

private:
  MaterializationResponsibility &MR;
};

} // end anonymous namespace

namespace llvm {
namespace orc {

RTDyldObjectLinkingLayer::RTDyldObjectLinkingLayer(
    ExecutionSession &ES, GetMemoryManagerFunction GetMemoryManager)
    : ObjectLayer(ES), GetMemoryManager(GetMemoryManager) {}

RTDyldObjectLinkingLayer::~RTDyldObjectLinkingLayer() {
  std::lock_guard<std::mutex> Lock(RTDyldLayerMutex);
  for (auto &MemMgr : MemMgrs) {
    for (auto *L : EventListeners)
      L->notifyFreeingObject(
          static_cast<uint64_t>(reinterpret_cast<uintptr_t>(MemMgr.get())));
    MemMgr->deregisterEHFrames();
  }
}

void RTDyldObjectLinkingLayer::emit(MaterializationResponsibility R,
                                    std::unique_ptr<MemoryBuffer> O) {
  assert(O && "Object must not be null");

  // This method launches an asynchronous link step that will fulfill our
  // materialization responsibility. We need to switch R to be heap
  // allocated before that happens so it can live as long as the asynchronous
  // link needs it to (i.e. it must be able to outlive this method).
  auto SharedR = std::make_shared<MaterializationResponsibility>(std::move(R));

  auto &ES = getExecutionSession();

  auto Obj = object::ObjectFile::createObjectFile(*O);

  if (!Obj) {
    getExecutionSession().reportError(Obj.takeError());
    SharedR->failMaterialization();
    return;
  }

  // Collect the internal symbols from the object file: We will need to
  // filter these later.
  auto InternalSymbols = std::make_shared<std::set<StringRef>>();
  {
    for (auto &Sym : (*Obj)->symbols()) {

      // Skip file symbols.
      if (auto SymType = Sym.getType()) {
        if (*SymType == object::SymbolRef::ST_File)
          continue;
      } else {
        ES.reportError(SymType.takeError());
        R.failMaterialization();
        return;
      }

      // Don't include symbols that aren't global.
      if (!(Sym.getFlags() & object::BasicSymbolRef::SF_Global)) {
        if (auto SymName = Sym.getName())
          InternalSymbols->insert(*SymName);
        else {
          ES.reportError(SymName.takeError());
          R.failMaterialization();
          return;
        }
      }
    }
  }

  auto K = R.getVModuleKey();
  RuntimeDyld::MemoryManager *MemMgr = nullptr;

  // Create a record a memory manager for this object.
  {
    auto Tmp = GetMemoryManager();
    std::lock_guard<std::mutex> Lock(RTDyldLayerMutex);
    MemMgrs.push_back(std::move(Tmp));
    MemMgr = MemMgrs.back().get();
  }

  JITDylibSearchOrderResolver Resolver(*SharedR);

  jitLinkForORC(
      object::OwningBinary<object::ObjectFile>(std::move(*Obj), std::move(O)),
      *MemMgr, Resolver, ProcessAllSections,
      [this, K, SharedR, MemMgr, InternalSymbols](
          const object::ObjectFile &Obj,
          std::unique_ptr<RuntimeDyld::LoadedObjectInfo> LoadedObjInfo,
          std::map<StringRef, JITEvaluatedSymbol> ResolvedSymbols) {
        return onObjLoad(K, *SharedR, Obj, MemMgr, std::move(LoadedObjInfo),
                         ResolvedSymbols, *InternalSymbols);
      },
      [this, K, SharedR, MemMgr](object::OwningBinary<object::ObjectFile> Obj,
                                 Error Err) mutable {
        onObjEmit(K, *SharedR, std::move(Obj), MemMgr, std::move(Err));
      });
}

void RTDyldObjectLinkingLayer::registerJITEventListener(JITEventListener &L) {
  std::lock_guard<std::mutex> Lock(RTDyldLayerMutex);
  assert(llvm::none_of(EventListeners,
                       [&](JITEventListener *O) { return O == &L; }) &&
         "Listener has already been registered");
  EventListeners.push_back(&L);
}

void RTDyldObjectLinkingLayer::unregisterJITEventListener(JITEventListener &L) {
  std::lock_guard<std::mutex> Lock(RTDyldLayerMutex);
  auto I = llvm::find(EventListeners, &L);
  assert(I != EventListeners.end() && "Listener not registered");
  EventListeners.erase(I);
}

Error RTDyldObjectLinkingLayer::onObjLoad(
    VModuleKey K, MaterializationResponsibility &R,
    const object::ObjectFile &Obj, RuntimeDyld::MemoryManager *MemMgr,
    std::unique_ptr<RuntimeDyld::LoadedObjectInfo> LoadedObjInfo,
    std::map<StringRef, JITEvaluatedSymbol> Resolved,
    std::set<StringRef> &InternalSymbols) {
  SymbolFlagsMap ExtraSymbolsToClaim;
  SymbolMap Symbols;

  // Hack to support COFF constant pool comdats introduced during compilation:
  // (See http://llvm.org/PR40074)
  if (auto *COFFObj = dyn_cast<object::COFFObjectFile>(&Obj)) {
    auto &ES = getExecutionSession();

    // For all resolved symbols that are not already in the responsibilty set:
    // check whether the symbol is in a comdat section and if so mark it as
    // weak.
    for (auto &Sym : COFFObj->symbols()) {
      if (Sym.getFlags() & object::BasicSymbolRef::SF_Undefined)
        continue;
      auto Name = Sym.getName();
      if (!Name)
        return Name.takeError();
      auto I = Resolved.find(*Name);

      // Skip unresolved symbols, internal symbols, and symbols that are
      // already in the responsibility set.
      if (I == Resolved.end() || InternalSymbols.count(*Name) ||
          R.getSymbols().count(ES.intern(*Name)))
        continue;
      auto Sec = Sym.getSection();
      if (!Sec)
        return Sec.takeError();
      if (*Sec == COFFObj->section_end())
        continue;
      auto &COFFSec = *COFFObj->getCOFFSection(**Sec);
      if (COFFSec.Characteristics & COFF::IMAGE_SCN_LNK_COMDAT)
        I->second.setFlags(I->second.getFlags() | JITSymbolFlags::Weak);
    }
  }

  for (auto &KV : Resolved) {
    // Scan the symbols and add them to the Symbols map for resolution.

    // We never claim internal symbols.
    if (InternalSymbols.count(KV.first))
      continue;

    auto InternedName = getExecutionSession().intern(KV.first);
    auto Flags = KV.second.getFlags();

    // Override object flags and claim responsibility for symbols if
    // requested.
    if (OverrideObjectFlags || AutoClaimObjectSymbols) {
      auto I = R.getSymbols().find(InternedName);

      if (OverrideObjectFlags && I != R.getSymbols().end())
        Flags = I->second;
      else if (AutoClaimObjectSymbols && I == R.getSymbols().end())
        ExtraSymbolsToClaim[InternedName] = Flags;
    }

    Symbols[InternedName] = JITEvaluatedSymbol(KV.second.getAddress(), Flags);
  }

  if (!ExtraSymbolsToClaim.empty()) {
    if (auto Err = R.defineMaterializing(ExtraSymbolsToClaim))
      return Err;

    // If we claimed responsibility for any weak symbols but were rejected then
    // we need to remove them from the resolved set.
    for (auto &KV : ExtraSymbolsToClaim)
      if (KV.second.isWeak() && !R.getSymbols().count(KV.first))
        Symbols.erase(KV.first);
  }

  if (const auto &InitSym = R.getInitializerSymbol())
    Symbols[InitSym] = JITEvaluatedSymbol();

  if (auto Err = R.notifyResolved(Symbols)) {
    R.failMaterialization();
    return Err;
  }

  if (NotifyLoaded)
    NotifyLoaded(K, Obj, *LoadedObjInfo);

  std::lock_guard<std::mutex> Lock(RTDyldLayerMutex);
  assert(!LoadedObjInfos.count(MemMgr) && "Duplicate loaded info for MemMgr");
  LoadedObjInfos[MemMgr] = std::move(LoadedObjInfo);

  return Error::success();
}

void RTDyldObjectLinkingLayer::onObjEmit(
    VModuleKey K, MaterializationResponsibility &R,
    object::OwningBinary<object::ObjectFile> O,
    RuntimeDyld::MemoryManager *MemMgr, Error Err) {
  if (Err) {
    getExecutionSession().reportError(std::move(Err));
    R.failMaterialization();
    return;
  }

  if (auto Err = R.notifyEmitted()) {
    getExecutionSession().reportError(std::move(Err));
    R.failMaterialization();
    return;
  }

  std::unique_ptr<object::ObjectFile> Obj;
  std::unique_ptr<MemoryBuffer> ObjBuffer;
  std::tie(Obj, ObjBuffer) = O.takeBinary();

  // Run EventListener notifyLoaded callbacks.
  {
    std::lock_guard<std::mutex> Lock(RTDyldLayerMutex);
    auto LOIItr = LoadedObjInfos.find(MemMgr);
    assert(LOIItr != LoadedObjInfos.end() && "LoadedObjInfo missing");
    for (auto *L : EventListeners)
      L->notifyObjectLoaded(
          static_cast<uint64_t>(reinterpret_cast<uintptr_t>(MemMgr)), *Obj,
          *LOIItr->second);
    LoadedObjInfos.erase(MemMgr);
  }

  if (NotifyEmitted)
    NotifyEmitted(K, std::move(ObjBuffer));
}

LegacyRTDyldObjectLinkingLayer::LegacyRTDyldObjectLinkingLayer(
    ExecutionSession &ES, ResourcesGetter GetResources,
    NotifyLoadedFtor NotifyLoaded, NotifyFinalizedFtor NotifyFinalized,
    NotifyFreedFtor NotifyFreed)
    : ES(ES), GetResources(std::move(GetResources)),
      NotifyLoaded(std::move(NotifyLoaded)),
      NotifyFinalized(std::move(NotifyFinalized)),
      NotifyFreed(std::move(NotifyFreed)), ProcessAllSections(false) {}

} // End namespace orc.
} // End namespace llvm.
