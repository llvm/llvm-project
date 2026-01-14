//===---- ExecutorProcessControl.cpp -- Executor process control APIs -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ExecutionEngine/Orc/ExecutorResolutionGenerator.h"

#include "llvm/ExecutionEngine/Orc/DebugUtils.h"
#include "llvm/ExecutionEngine/Orc/Shared/ExecutorSymbolDef.h"
#include "llvm/Support/Error.h"

#define DEBUG_TYPE "orc"

namespace llvm {
namespace orc {

Expected<std::unique_ptr<ExecutorResolutionGenerator>>
ExecutorResolutionGenerator::Load(ExecutionSession &ES, const char *LibraryPath,
                                  SymbolPredicate Allow,
                                  AbsoluteSymbolsFn AbsoluteSymbols) {
  auto H = ES.getExecutorProcessControl().getDylibMgr().loadDylib(LibraryPath);
  if (H)
    return H.takeError();
  return std::make_unique<ExecutorResolutionGenerator>(
      ES, *H, std::move(Allow), std::move(AbsoluteSymbols));
}

Error ExecutorResolutionGenerator::tryToGenerate(
    LookupState &LS, LookupKind K, JITDylib &JD,
    JITDylibLookupFlags JDLookupFlags, const SymbolLookupSet &LookupSet) {

  if (LookupSet.empty())
    return Error::success();

  LLVM_DEBUG({
    dbgs() << "ExecutorResolutionGenerator trying to generate " << LookupSet
           << "\n";
  });

  SymbolLookupSet LookupSymbols;
  for (auto &[Name, LookupFlag] : LookupSet) {
    if (Allow && !Allow(Name))
      continue;
    LookupSymbols.add(Name, LookupFlag);
  }

  DylibManager::LookupRequest LR(H, LookupSymbols);
  EPC.getDylibMgr().lookupSymbolsAsync(
      LR, [this, LS = std::move(LS), JD = JITDylibSP(&JD),
           LookupSymbols](auto Result) mutable {
        if (Result) {
          LLVM_DEBUG({
            dbgs() << "ExecutorResolutionGenerator lookup failed due to error";
          });
          return LS.continueLookup(Result.takeError());
        }
        assert(Result->size() == 1 &&
               "Results for more than one library returned");
        assert(Result->front().size() == LookupSymbols.size() &&
               "Result has incorrect number of elements");

        // const tpctypes::LookupResult &Syms = Result->front();
        // size_t SymIdx = 0;
        auto Syms = Result->front().begin();
        SymbolNameSet MissingSymbols;
        SymbolMap NewSyms;
        for (auto &[Name, Flags] : LookupSymbols) {
          const auto &Sym = *Syms++;
          if (Sym && Sym->getAddress())
            NewSyms[Name] = *Sym;
          else if (LLVM_UNLIKELY(!Sym &&
                                 Flags == SymbolLookupFlags::RequiredSymbol))
            MissingSymbols.insert(Name);
        }

        LLVM_DEBUG({
          dbgs() << "ExecutorResolutionGenerator lookup returned " << NewSyms
                 << "\n";
        });

        if (NewSyms.empty())
          return LS.continueLookup(Error::success());

        if (LLVM_UNLIKELY(!MissingSymbols.empty()))
          return LS.continueLookup(make_error<SymbolsNotFound>(
              this->EPC.getSymbolStringPool(), std::move(MissingSymbols)));

        LS.continueLookup(JD->define(AbsoluteSymbols(std::move(NewSyms))));
      });

  return Error::success();
}

} // end namespace orc
} // end namespace llvm
