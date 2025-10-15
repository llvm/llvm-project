//===---------------- EPCDynamicLibrarySearchGenerator.cpp ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ExecutionEngine/Orc/EPCDynamicLibrarySearchGenerator.h"

#include "llvm/ExecutionEngine/Orc/AbsoluteSymbols.h"
#include "llvm/ExecutionEngine/Orc/DebugUtils.h"
#include "llvm/Support/Error.h"

#define DEBUG_TYPE "orc"

namespace llvm {
namespace orc {

Expected<std::unique_ptr<EPCDynamicLibrarySearchGenerator>>
EPCDynamicLibrarySearchGenerator::Load(
    ExecutionSession &ES, const char *LibraryPath, SymbolPredicate Allow,
    AddAbsoluteSymbolsFn AddAbsoluteSymbols) {
  auto Handle =
      ES.getExecutorProcessControl().getDylibMgr().loadDylib(LibraryPath);
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
      dbgs() << "EPCDynamicLibrarySearchGenerator trying to generate "
             << Symbols << "\n";
    });

  // If there's no handle then resolve all requested symbols to null.
  if (!H) {
    assert(Allow && "No handle or filter?");
    SymbolMap Nulls;
    for (auto &[Name, LookupFlags] : Symbols) {
      if (Allow(Name))
        Nulls[Name] = {};
    }
    return addAbsolutes(JD, std::move(Nulls));
  }

  // Otherwise proceed with lookup in the remote.
  SymbolLookupSet LookupSymbols;

  for (auto &KV : Symbols) {
    // Skip symbols that don't match the filter.
    if (Allow && !Allow(KV.first))
      continue;
    LookupSymbols.add(KV.first, SymbolLookupFlags::WeaklyReferencedSymbol);
  }

  DylibManager::LookupRequest Request(*H, LookupSymbols);
  // Copy-capture LookupSymbols, since LookupRequest keeps a reference.
  EPC.getDylibMgr().lookupSymbolsAsync(Request, [this, &JD, LS = std::move(LS),
                                                 LookupSymbols](
                                                    auto Result) mutable {
    if (!Result) {
      LLVM_DEBUG({
        dbgs() << "EPCDynamicLibrarySearchGenerator lookup failed due to error";
      });
      return LS.continueLookup(Result.takeError());
    }

    assert(Result->size() == 1 && "Results for more than one library returned");
    assert(Result->front().size() == LookupSymbols.size() &&
           "Result has incorrect number of elements");

    auto SymsIt = Result->front().begin();
    SymbolNameSet MissingSymbols;
    SymbolMap NewSymbols;
    for (auto &[Name, Flags] : LookupSymbols) {
      const auto &Sym = *SymsIt++;
      if (Sym && Sym->getAddress())
        NewSymbols[Name] = *Sym;
      else if (LLVM_UNLIKELY(!Sym &&
                             Flags == SymbolLookupFlags::RequiredSymbol))
        MissingSymbols.insert(Name);
    }

    LLVM_DEBUG({
      dbgs() << "EPCDynamicLibrarySearchGenerator lookup returned "
             << NewSymbols << "\n";
    });

    // If there were no resolved symbols bail out.
    if (NewSymbols.empty())
      return LS.continueLookup(Error::success());

    if (LLVM_UNLIKELY(!MissingSymbols.empty()))
      return LS.continueLookup(make_error<SymbolsNotFound>(
          this->EPC.getSymbolStringPool(), std::move(MissingSymbols)));

    // Define resolved symbols.
    Error Err = addAbsolutes(JD, std::move(NewSymbols));

    LS.continueLookup(std::move(Err));
  });

  return Error::success();
}

Error EPCDynamicLibrarySearchGenerator::addAbsolutes(JITDylib &JD,
                                                     SymbolMap Symbols) {
  return AddAbsoluteSymbols ? AddAbsoluteSymbols(JD, std::move(Symbols))
                            : JD.define(absoluteSymbols(std::move(Symbols)));
}

} // end namespace orc
} // end namespace llvm
