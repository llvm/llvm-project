//===-- RTDyldObjectLinkingLayer.cpp - RuntimeDyld backed ORC ObjectLayer -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ExecutionEngine/Orc/RTDyldObjectLinkingLayer.h"

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
  for (auto &MemMgr : MemMgrs)
    MemMgr->deregisterEHFrames();
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

  // Create a MemoryBufferRef backed MemoryBuffer (i.e. shallow) copy of the
  // the underlying buffer to pass into RuntimeDyld. This allows us to hold
  // ownership of the real underlying buffer and return it to the user once
  // the object has been emitted.
  auto ObjBuffer = MemoryBuffer::getMemBuffer(O->getMemBufferRef(), false);

  auto Obj = object::ObjectFile::createObjectFile(*ObjBuffer);

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
      **Obj, std::move(O), *MemMgr, Resolver, ProcessAllSections,
      [this, K, SharedR, &Obj, InternalSymbols](
          std::unique_ptr<RuntimeDyld::LoadedObjectInfo> LoadedObjInfo,
          std::map<StringRef, JITEvaluatedSymbol> ResolvedSymbols) {
        return onObjLoad(K, *SharedR, **Obj, std::move(LoadedObjInfo),
                         ResolvedSymbols, *InternalSymbols);
      },
      [this, K, SharedR, O = std::move(O)](Error Err) mutable {
        onObjEmit(K, std::move(O), *SharedR, std::move(Err));
      });
}

Error RTDyldObjectLinkingLayer::onObjLoad(
    VModuleKey K, MaterializationResponsibility &R, object::ObjectFile &Obj,
    std::unique_ptr<RuntimeDyld::LoadedObjectInfo> LoadedObjInfo,
    std::map<StringRef, JITEvaluatedSymbol> Resolved,
    std::set<StringRef> &InternalSymbols) {
  SymbolFlagsMap ExtraSymbolsToClaim;
  SymbolMap Symbols;
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

  if (!ExtraSymbolsToClaim.empty())
    if (auto Err = R.defineMaterializing(ExtraSymbolsToClaim))
      return Err;

  if (auto Err = R.notifyResolved(Symbols)) {
    R.failMaterialization();
    return Err;
  }

  if (NotifyLoaded)
    NotifyLoaded(K, Obj, *LoadedObjInfo);

  return Error::success();
}

void RTDyldObjectLinkingLayer::onObjEmit(
    VModuleKey K, std::unique_ptr<MemoryBuffer> ObjBuffer,
    MaterializationResponsibility &R, Error Err) {
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
