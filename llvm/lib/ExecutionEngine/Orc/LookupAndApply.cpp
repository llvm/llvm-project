//===- LookupAndApply.cpp - Compose a lookup from handlers ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ExecutionEngine/Orc/LookupAndApply.h"

#include "llvm/Support/MSVCErrorWorkarounds.h"

#include <future>

namespace llvm::orc {

void lookupAndApply(unique_function<void(Error)> OnApplied,
                    ExecutionSession &ES, LookupKind K,
                    const JITDylibSearchOrder &SearchOrder,
                    ArrayRef<LookupPrepareFn> PrepareFns) {
  // Collect the symbols to look up. Each prepare function hands back the
  // applicator that will act on the result; the prepare functions themselves
  // are not needed beyond this point.
  SymbolLookupSet Symbols;
  std::vector<LookupApplyFn> Applies;
  Applies.reserve(PrepareFns.size());
  for (const auto &PF : PrepareFns)
    Applies.push_back(PF(Symbols, ES));

  // PrepareFns are independent, so two of them may legitimately ask for the
  // same symbol. ExecutionSession::lookup requires a duplicate-free set, and
  // the applicators read the result by name, so merging here is invisible to
  // them.
  Symbols.mergeEntries();

  ES.lookup(
      K, SearchOrder, std::move(Symbols), SymbolState::Ready,
      [Applies = std::move(Applies),
       OnApplied = std::move(OnApplied)](Expected<SymbolMap> Result) mutable {
        if (!Result)
          return OnApplied(Result.takeError());
        for (auto &Apply : Applies)
          Apply(*Result);
        OnApplied(Error::success());
      },
      NoDependenciesToRegister);
}

Error lookupAndApply(ExecutionSession &ES, LookupKind K,
                     const JITDylibSearchOrder &SearchOrder,
                     ArrayRef<LookupPrepareFn> PrepareFns) {
  std::promise<MSVCPError> ResultP;
  auto ResultF = ResultP.get_future();
  lookupAndApply([&](Error Err) { ResultP.set_value(std::move(Err)); }, ES, K,
                 SearchOrder, PrepareFns);
  return ResultF.get();
}

void lookupAndApply(unique_function<void(Error)> OnApplied, JITDylib &JD,
                    ArrayRef<LookupPrepareFn> PrepareFns) {
  lookupAndApply(std::move(OnApplied), JD.getExecutionSession(),
                 LookupKind::Static, makeJITDylibSearchOrder(&JD), PrepareFns);
}

Error lookupAndApply(JITDylib &JD, ArrayRef<LookupPrepareFn> PrepareFns) {
  return lookupAndApply(JD.getExecutionSession(), LookupKind::Static,
                        makeJITDylibSearchOrder(&JD), PrepareFns);
}

} // namespace llvm::orc
