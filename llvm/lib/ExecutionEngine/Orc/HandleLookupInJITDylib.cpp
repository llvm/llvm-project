//===- HandleLookupInJITDylib.cpp - Lookups in JITDylibs ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ExecutionEngine/Orc/HandleLookupInJITDylib.h"
#include "llvm/ExecutionEngine/Orc/SymbolLookupSet.h"
#include "llvm/Support/FormatVariadic.h"

using namespace llvm;
using namespace llvm::orc;

HandleLookupInJITDylibFn
llvm::orc::createLookupInJITDylibHandler(ExecutionSession &ES,
                                         GetJITDylibForKeyFn GetJD) {
  return [&ES, GetJD = std::move(GetJD)](
             unique_function<void(Expected<SymbolLookupResult>)> SendResult,
             ExecutorAddr Key, JITDylibLookupFlags JDLookupFlags,
             const std::vector<std::pair<StringRef, bool>> &Symbols) mutable {
    JITDylibSP JD = GetJD(Key);
    if (!JD)
      return SendResult(make_error<StringError>(
          "No JITDylib associated with key " + formatv("{0:x}", Key),
          inconvertibleErrorCode()));

    // Intern the names now: they refer into the argument buffer, which doesn't
    // outlive this call. Names records the request order for the result.
    std::vector<SymbolStringPtr> Names;
    Names.reserve(Symbols.size());
    SymbolLookupSet LS;
    for (auto &[Name, Required] : Symbols) {
      Names.push_back(ES.intern(Name));
      LS.add(Names.back(), Required
                               ? SymbolLookupFlags::RequiredSymbol
                               : SymbolLookupFlags::WeaklyReferencedSymbol);
    }

    // The executor may name a symbol more than once, but the lookup needs a
    // proper set. The result is rebuilt from Names, so it still has one entry
    // per element of the request.
    LS.mergeEntries();

    auto &JDRef = *JD;
    ES.lookup(
        LookupKind::DLSym, {{&JDRef, JDLookupFlags}}, std::move(LS),
        SymbolState::Ready,
        [JD = std::move(JD), Names = std::move(Names),
         SendResult = std::move(SendResult)](Expected<SymbolMap> Syms) mutable {
          if (!Syms)
            return SendResult(Syms.takeError());

          // A missing required symbol fails the lookup, so any name without a
          // definition here was weakly referenced: report it as nullopt.
          SymbolLookupResult Result;
          Result.reserve(Names.size());
          for (auto &Name : Names) {
            auto I = Syms->find(Name);
            if (I != Syms->end())
              Result.push_back(I->second.getAddress());
            else
              Result.push_back(std::nullopt);
          }
          SendResult(std::move(Result));
        },
        NoDependenciesToRegister);
  };
}
