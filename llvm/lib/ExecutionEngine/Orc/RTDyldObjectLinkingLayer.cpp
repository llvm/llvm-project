//===-- RTDyldObjectLinkingLayer.cpp - RuntimeDyld backed ORC ObjectLayer -===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/ExecutionEngine/Orc/RTDyldObjectLinkingLayer.h"

namespace {

using namespace llvm;
using namespace llvm::orc;

class JITDylibSearchOrderResolver : public JITSymbolResolver {
public:
  JITDylibSearchOrderResolver(MaterializationResponsibility &MR) : MR(MR) {}

  Expected<LookupResult> lookup(const LookupSet &Symbols) {
    auto &ES = MR.getTargetJITDylib().getExecutionSession();
    SymbolNameSet InternedSymbols;

    for (auto &S : Symbols)
      InternedSymbols.insert(ES.getSymbolStringPool().intern(S));

    auto RegisterDependencies = [&](const SymbolDependenceMap &Deps) {
      MR.addDependenciesForAll(Deps);
    };

    auto InternedResult =
        MR.getTargetJITDylib().withSearchOrderDo([&](const JITDylibList &JDs) {
          return ES.lookup(JDs, InternedSymbols, RegisterDependencies, false);
        });

    if (!InternedResult)
      return InternedResult.takeError();

    LookupResult Result;
    for (auto &KV : *InternedResult)
      Result[*KV.first] = std::move(KV.second);

    return Result;
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

RTDyldObjectLinkingLayer2::RTDyldObjectLinkingLayer2(
    ExecutionSession &ES, GetMemoryManagerFunction GetMemoryManager,
    NotifyLoadedFunction NotifyLoaded, NotifyEmittedFunction NotifyEmitted)
    : ObjectLayer(ES), GetMemoryManager(GetMemoryManager),
      NotifyLoaded(std::move(NotifyLoaded)),
      NotifyEmitted(std::move(NotifyEmitted)) {}

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

  auto MemoryManager = GetMemoryManager(K);

  JITDylibSearchOrderResolver Resolver(R);
  auto RTDyld = llvm::make_unique<RuntimeDyld>(*MemoryManager, Resolver);
  RTDyld->setProcessAllSections(ProcessAllSections);

  {
    std::lock_guard<std::mutex> Lock(RTDyldLayerMutex);

    assert(!MemMgrs.count(K) &&
           "A memory manager already exists for this key?");
    MemMgrs[K] = std::move(MemoryManager);
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

    SymbolFlagsMap ExtraSymbolsToClaim;
    SymbolMap Symbols;
    for (auto &KV : RTDyld->getSymbolTable()) {
      // Scan the symbols and add them to the Symbols map for resolution.

      // We never claim internal symbols.
      if (InternalSymbols.count(KV.first))
        continue;

      auto InternedName = ES.getSymbolStringPool().intern(KV.first);
      auto Flags = KV.second.getFlags();

      // Override object flags and claim responsibility for symbols if
      // requested.
      if (OverrideObjectFlags || AutoClaimObjectSymbols) {
        auto I = R.getSymbols().find(InternedName);

        if (OverrideObjectFlags && I != R.getSymbols().end())
          Flags = JITSymbolFlags::stripTransientFlags(I->second);
        else if (AutoClaimObjectSymbols && I == R.getSymbols().end())
          ExtraSymbolsToClaim[InternedName] = Flags;
      }

      Symbols[InternedName] = JITEvaluatedSymbol(KV.second.getAddress(), Flags);
    }

    if (!ExtraSymbolsToClaim.empty())
      if (auto Err = R.defineMaterializing(ExtraSymbolsToClaim)) {
        ES.reportError(std::move(Err));
        R.failMaterialization();
        return;
      }

    R.resolve(Symbols);
  }

  if (NotifyLoaded)
    NotifyLoaded(K, **ObjFile, *Info);

  RTDyld->finalizeWithMemoryManagerLocking();

  if (RTDyld->hasError()) {
    ES.reportError(make_error<StringError>(RTDyld->getErrorString(),
                                           inconvertibleErrorCode()));
    R.failMaterialization();
    return;
  }

  R.emit();

  if (NotifyEmitted)
    NotifyEmitted(K);
}

} // End namespace orc.
} // End namespace llvm.
