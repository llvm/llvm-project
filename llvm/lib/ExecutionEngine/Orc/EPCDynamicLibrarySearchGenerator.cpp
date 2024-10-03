//===---------------- EPCDynamicLibrarySearchGenerator.cpp ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ExecutionEngine/Orc/EPCDynamicLibrarySearchGenerator.h"
#include "llvm/ExecutionEngine/Orc/DebugUtils.h"
#include "llvm/Support/Error.h"

#define DEBUG_TYPE "orc"

namespace llvm {
namespace orc {

Expected<std::unique_ptr<EPCDynamicLibrarySearchGenerator>>
EPCDynamicLibrarySearchGenerator::Load(
    ExecutionSession &ES, const char *LibraryPath, SymbolPredicate Allow,
    AddAbsoluteSymbolsFn AddAbsoluteSymbols) {
  auto Handle = ES.getExecutorProcessControl().loadDylib(LibraryPath);
  if (!Handle)
    return Handle.takeError();

  return std::make_unique<EPCDynamicLibrarySearchGenerator>(
      ES, *Handle, std::move(Allow), std::move(AddAbsoluteSymbols));
}

Error EPCDynamicLibrarySearchGenerator::tryToGenerate(
    LookupState &LS, LookupKind K, JITDylib &JD,
    JITDylibLookupFlags JDLookupFlags, const SymbolLookupSet &Symbols) {

  if (Symbols.empty())
    return Error::success();

  LLVM_DEBUG({
    dbgs() << "EPCDynamicLibrarySearchGenerator trying to generate " << Symbols
           << "\n";
  });

  SymbolLookupSet LookupSymbols;

  for (auto &KV : Symbols) {
    // Skip symbols that don't match the filter.
    if (Allow && !Allow(KV.first))
      continue;
    LookupSymbols.add(KV.first, SymbolLookupFlags::WeaklyReferencedSymbol);
  }

  ExecutorProcessControl::LookupRequest Request(H, LookupSymbols);
  // Copy-capture LookupSymbols, since LookupRequest keeps a reference.
  EPC.lookupSymbolsAsync(Request, [this, &JD, LS = std::move(LS),
                                   LookupSymbols](auto Result) mutable {
    if (!Result) {
      LLVM_DEBUG({
        dbgs() << "EPCDynamicLibrarySearchGenerator lookup failed due to error";
      });
      return LS.continueLookup(Result.takeError());
    }

    assert(Result->size() == 1 && "Results for more than one library returned");
    assert(Result->front().size() == LookupSymbols.size() &&
           "Result has incorrect number of elements");

    SymbolMap NewSymbols;
    auto ResultI = Result->front().begin();
    for (auto &KV : LookupSymbols) {
      if (ResultI->getAddress())
        NewSymbols[KV.first] = *ResultI;
      ++ResultI;
    }

    LLVM_DEBUG({
      dbgs() << "EPCDynamicLibrarySearchGenerator lookup returned "
             << NewSymbols << "\n";
    });

    // If there were no resolved symbols bail out.
    if (NewSymbols.empty())
      return LS.continueLookup(Error::success());

    // Define resolved symbols.
    Error Err = AddAbsoluteSymbols
                    ? AddAbsoluteSymbols(JD, std::move(NewSymbols))
                    : JD.define(absoluteSymbols(std::move(NewSymbols)));

    LS.continueLookup(std::move(Err));
  });

  return Error::success();
}

Error AutoLoadDynamicLibrarySearchGenerator::tryToGenerate(
    LookupState &LS, LookupKind K, JITDylib &JD,
    JITDylibLookupFlags JDLookupFlags, const SymbolLookupSet &Symbols) {

  if (Symbols.empty())
    return Error::success();

  LLVM_DEBUG({
    dbgs() << "AutoLoadDynamicLibrarySearchGenerator trying to generate "
           << Symbols << "\n";
  });

  SymbolNameSet CandidateSyms;
  for (auto &KV : Symbols) {
    if (GlobalFilter.IsInitialized() && !GlobalFilter.MayContain(*KV.first) &&
        !ExcludedSymbols.count(*KV.first))
      continue;

    CandidateSyms.insert(KV.first);
  }

  if (CandidateSyms.empty())
    return Error::success();

  auto Err = tryToResolve(CandidateSyms, [this, &JD, LS = std::move(LS),
                                          CandidateSyms](auto Result) mutable {
    auto &ResolveRes = Result->front();
    bool IsFilter = GlobalFilter.IsInitialized();
    if (!IsFilter && ResolveRes.Filter.has_value()) {
      GlobalFilter = std::move(ResolveRes.Filter.value());
    }

    if (!Result) {
      LLVM_DEBUG({
        dbgs() << "AutoLoadDynamicLibrarySearchGenerator resolve failed due to "
                  "error";
      });
      return LS.continueLookup(Result.takeError());
    }

    auto &Symbols = ResolveRes.SymbolDef;
    assert(Result->size() == 1 && "Results for more than one library returned");
    assert(Symbols.size() == CandidateSyms.size() &&
           "Result has incorrect number of elements");

    SymbolMap NewSymbols;
    auto ResultI = Symbols.begin();
    for (auto &S : CandidateSyms) {
      if (ResultI->getAddress())
        NewSymbols[S] = *ResultI;
      else if (IsFilter)
        ExcludedSymbols.insert(*S);
      ++ResultI;
    }

    LLVM_DEBUG({
      dbgs() << "AutoLoadDynamicLibrarySearchGenerator resolve returned "
             << NewSymbols << "\n";
    });

    // If there were no resolved symbols bail out.
    if (NewSymbols.empty())
      return LS.continueLookup(Error::success());

    // Define resolved symbols.
    Error Err = AddAbsoluteSymbols
                    ? AddAbsoluteSymbols(JD, std::move(NewSymbols))
                    : JD.define(absoluteSymbols(std::move(NewSymbols)));

    LS.continueLookup(std::move(Err));
  });

  return Err;
}

Error AutoLoadDynamicLibrarySearchGenerator::tryToResolve(
    SymbolNameSet CandidateSyms,
    ExecutorProcessControl::ResolveSymbolsCompleteFn OnCompleteFn) {

  LLVM_DEBUG({
    dbgs() << "AutoLoadDynamicLibrarySearchGenerator trying to resolve "
           << CandidateSyms << "\n";
  });

  SymbolLookupSet LookupSymbols;

  for (auto &S : CandidateSyms) {
    LookupSymbols.add(S, SymbolLookupFlags::WeaklyReferencedSymbol);
  }

  EPC.resolveSymbolsAsync(LookupSymbols, std::move(OnCompleteFn));

  return Error::success();
}

} // end namespace orc
} // end namespace llvm
