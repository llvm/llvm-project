//===- COFFAutoImportGenerator.cpp - COFF dllimport auto-import ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ExecutionEngine/Orc/COFFAutoImportGenerator.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Twine.h"
#include "llvm/ExecutionEngine/JITLink/JITLink.h"
#include "llvm/ExecutionEngine/Orc/Shared/ExecutorSymbolDef.h"

namespace llvm {
namespace orc {

Expected<std::unique_ptr<COFFAutoImportGenerator>>
COFFAutoImportGenerator::Load(ExecutionSession &ES, ObjectLinkingLayer &L,
                              DylibManager &DylibMgr, const char *LibraryPath) {
  Triple TT = ES.getTargetTriple();

  auto CreatePointer = jitlink::getAnonymousPointerCreator(TT);
  if (!CreatePointer)
    return make_error<StringError>(
        "COFFAutoImportGenerator: no pointer creator for " + TT.str(),
        inconvertibleErrorCode());

  auto CreateStub = jitlink::getPointerJumpStubCreator(TT);
  if (!CreateStub)
    return make_error<StringError>(
        "COFFAutoImportGenerator: no stub creator for " + TT.str(),
        inconvertibleErrorCode());

  auto LibHandle = DylibMgr.loadDylib(LibraryPath);
  if (!LibHandle)
    return LibHandle.takeError();

  return std::unique_ptr<COFFAutoImportGenerator>(new COFFAutoImportGenerator(
      ES, L, DylibMgr, *LibHandle, std::move(CreatePointer),
      std::move(CreateStub)));
}

Error COFFAutoImportGenerator::tryToGenerate(LookupState &LS, LookupKind K,
                                             JITDylib &JD,
                                             JITDylibLookupFlags JDLookupFlags,
                                             const SymbolLookupSet &Symbols) {
  if (Symbols.empty())
    return Error::success();

  // Weakly reference each symbol (minus any __imp_ prefix) so unexported names
  // are left unresolved; de-dup __imp_X and X into one lookup.
  SymbolLookupSet LookupSymbols;
  DenseSet<SymbolStringPtr> Seen;
  for (auto &KV : Symbols) {
    StringRef Base = *KV.first;
    if (Base.starts_with(getImpPrefix()))
      Base = Base.drop_front(getImpPrefix().size());
    SymbolStringPtr BaseName = ES.intern(Base);
    if (Seen.insert(BaseName).second)
      LookupSymbols.add(BaseName, SymbolLookupFlags::WeaklyReferencedSymbol);
  }

  DylibMgr.lookupSymbolsAsync(
      LibHandle, LookupSymbols,
      [this, &JD, LS = std::move(LS), LookupSymbols](auto Result) mutable {
        if (!Result)
          return LS.continueLookup(Result.takeError());

        // Keep the exported (non-null) results.
        SymbolMap Resolved;
        for (auto [Sym, Addr] : llvm::zip_equal(LookupSymbols, *Result))
          if (Addr && *Addr)
            Resolved[Sym.first] = {*Addr, JITSymbolFlags::Exported |
                                              JITSymbolFlags::Callable};

        if (Resolved.empty())
          return LS.continueLookup(Error::success());

        auto G = createStubsGraph(Resolved);
        if (!G)
          return LS.continueLookup(G.takeError());

        // One tracker owns all stubs so they can be reclaimed together.
        if (!ImportStubsRT || ImportStubsRT->isDefunct())
          ImportStubsRT = JD.createResourceTracker();
        LS.continueLookup(L.add(ImportStubsRT, std::move(*G)));
      });

  return Error::success();
}

// FIXME: Pull this into a helper shared with
// DLLImportDefinitionGenerator::createStubsGraph (ExecutionUtils.cpp), which
// builds the same __imp_X + thunk stubs. Until then, fixes here may need to
// be mirrored there too.
Expected<std::unique_ptr<jitlink::LinkGraph>>
COFFAutoImportGenerator::createStubsGraph(const SymbolMap &Resolved) {
  Triple TT = ES.getTargetTriple();

  auto G = std::make_unique<jitlink::LinkGraph>(
      "<AUTOIMPORT_STUBS>", ES.getSymbolStringPool(), TT, SubtargetFeatures(),
      jitlink::getGenericEdgeKindName);
  jitlink::Section &Sec =
      G->createSection(getSectionName(), MemProt::Read | MemProt::Exec);

  for (auto &KV : Resolved) {
    // X's address as a local absolute symbol, referenced only by __imp_ (so it
    // can't collide with the X thunk below).
    jitlink::Symbol &Target = G->addAbsoluteSymbol(
        *KV.first, KV.second.getAddress(), G->getPointerSize(),
        jitlink::Linkage::Strong, jitlink::Scope::Local, false);

    // __imp_X: pointer slot holding X's address.
    jitlink::Symbol &Ptr = CreatePointer(*G, Sec, &Target, 0);
    Ptr.setName(G->intern((Twine(getImpPrefix()) + *KV.first).str()));
    // Weak: a later real definition overrides this fallback (link.exe-style).
    Ptr.setLinkage(jitlink::Linkage::Weak);
    Ptr.setScope(jitlink::Scope::Default);

    // X: thunk "jmpq *__imp_X(%rip)" so direct calls work too.
    jitlink::Symbol &Stub = CreateStub(*G, Sec, Ptr);
    Stub.setName(G->intern(*KV.first));
    Stub.setLinkage(jitlink::Linkage::Weak);
    Stub.setScope(jitlink::Scope::Default);
  }

  return std::move(G);
}

} // end namespace orc
} // end namespace llvm
