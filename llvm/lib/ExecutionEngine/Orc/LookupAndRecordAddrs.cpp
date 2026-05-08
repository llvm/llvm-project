//===------- LookupAndRecordAddrs.h - Symbol lookup support utility -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ExecutionEngine/Orc/LookupAndRecordAddrs.h"

#include <future>

namespace llvm {
namespace orc {

void lookupAndRecordAddrs(
    unique_function<void(Error)> OnRecorded, ExecutionSession &ES, LookupKind K,
    const JITDylibSearchOrder &SearchOrder,
    std::vector<std::pair<SymbolStringPtr, ExecutorAddr *>> Pairs,
    SymbolLookupFlags LookupFlags) {

  SymbolLookupSet Symbols;
  for (auto &KV : Pairs)
    Symbols.add(KV.first, LookupFlags);

  ES.lookup(
      K, SearchOrder, std::move(Symbols), SymbolState::Ready,
      [Pairs = std::move(Pairs),
       OnRec = std::move(OnRecorded)](Expected<SymbolMap> Result) mutable {
        if (!Result)
          return OnRec(Result.takeError());
        for (auto &KV : Pairs) {
          auto I = Result->find(KV.first);
          *KV.second =
              I != Result->end() ? I->second.getAddress() : orc::ExecutorAddr();
        }
        OnRec(Error::success());
      },
      NoDependenciesToRegister);
}

Error lookupAndRecordAddrs(
    ExecutionSession &ES, LookupKind K, const JITDylibSearchOrder &SearchOrder,
    std::vector<std::pair<SymbolStringPtr, ExecutorAddr *>> Pairs,
    SymbolLookupFlags LookupFlags) {

  std::promise<MSVCPError> ResultP;
  auto ResultF = ResultP.get_future();
  lookupAndRecordAddrs([&](Error Err) { ResultP.set_value(std::move(Err)); },
                       ES, K, SearchOrder, std::move(Pairs), LookupFlags);
  return ResultF.get();
}

} // End namespace orc.
} // End namespace llvm.
