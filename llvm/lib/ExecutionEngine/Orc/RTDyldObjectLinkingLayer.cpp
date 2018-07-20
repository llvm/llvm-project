//===-- RTDyldObjectLinkingLayer.cpp - RuntimeDyld backed ORC ObjectLayer -===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/ExecutionEngine/Orc/RTDyldObjectLinkingLayer.h"

namespace llvm {
namespace orc {

RTDyldObjectLinkingLayer2::RTDyldObjectLinkingLayer2(
    ExecutionSession &ES, ResourcesGetterFunction GetResources,
    NotifyLoadedFunction NotifyLoaded, NotifyFinalizedFunction NotifyFinalized)
    : ObjectLayer(ES), GetResources(std::move(GetResources)),
      NotifyLoaded(std::move(NotifyLoaded)),
      NotifyFinalized(std::move(NotifyFinalized)), ProcessAllSections(false) {}

void RTDyldObjectLinkingLayer2::emit(MaterializationResponsibility R,
                                     VModuleKey K,
                                     std::unique_ptr<MemoryBuffer> O) {
  assert(O && "Object must not be null");

  auto &ES = getExecutionSession();

  auto ObjFile = object::ObjectFile::createObjectFile(*O);
  if (!ObjFile) {
    getExecutionSession().reportError(ObjFile.takeError());
    R.failMaterialization();
  }

  auto Resources = GetResources(K);

  JITSymbolResolverAdapter ResolverAdapter(ES, *Resources.Resolver, &R);
  auto RTDyld =
      llvm::make_unique<RuntimeDyld>(*Resources.MemMgr, ResolverAdapter);
  RTDyld->setProcessAllSections(ProcessAllSections);

  {
    std::lock_guard<std::mutex> Lock(RTDyldLayerMutex);

    assert(!ActiveRTDylds.count(K) &&
           "An active RTDyld already exists for this key?");
    ActiveRTDylds[K] = RTDyld.get();

    assert(!MemMgrs.count(K) &&
           "A memory manager already exists for this key?");
    MemMgrs[K] = Resources.MemMgr;
  }

  auto Info = RTDyld->loadObject(**ObjFile);

  {
    std::set<StringRef> InternalSymbols;
    for (auto &Sym : (*ObjFile)->symbols()) {
      if (!(Sym.getFlags() & object::BasicSymbolRef::SF_Global)) {
        if (auto SymName = Sym.getName())
          InternalSymbols.insert(*SymName);
        else {
          ES.reportError(SymName.takeError());
          R.failMaterialization();
          return;
        }
      }
    }

    SymbolMap Symbols;
    for (auto &KV : RTDyld->getSymbolTable())
      if (!InternalSymbols.count(KV.first))
        Symbols[ES.getSymbolStringPool().intern(KV.first)] = KV.second;

    R.resolve(Symbols);
  }

  if (NotifyLoaded)
    NotifyLoaded(K, **ObjFile, *Info);

  RTDyld->finalizeWithMemoryManagerLocking();

  {
    std::lock_guard<std::mutex> Lock(RTDyldLayerMutex);
    ActiveRTDylds.erase(K);
  }

  if (RTDyld->hasError()) {
    ES.reportError(make_error<StringError>(RTDyld->getErrorString(),
                                           inconvertibleErrorCode()));
    R.failMaterialization();
    return;
  }

  R.finalize();

  if (NotifyFinalized)
    NotifyFinalized(K);
}

void RTDyldObjectLinkingLayer2::mapSectionAddress(
    VModuleKey K, const void *LocalAddress, JITTargetAddress TargetAddr) const {
  std::lock_guard<std::mutex> Lock(RTDyldLayerMutex);
  auto ActiveRTDyldItr = ActiveRTDylds.find(K);

  assert(ActiveRTDyldItr != ActiveRTDylds.end() &&
         "No active RTDyld instance found for key");
  ActiveRTDyldItr->second->mapSectionAddress(LocalAddress, TargetAddr);
}

} // End namespace orc.
} // End namespace llvm.
