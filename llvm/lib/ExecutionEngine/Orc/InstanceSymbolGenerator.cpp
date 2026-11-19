//===- InstanceSymbolGenerator.cpp - Executor instance symbols ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ExecutionEngine/Orc/InstanceSymbolGenerator.h"

#include "llvm/ExecutionEngine/Orc/AbsoluteSymbols.h"
#include "llvm/ExecutionEngine/Orc/DebugUtils.h"

#define DEBUG_TYPE "orc"

namespace llvm::orc {

Error InstanceSymbolGenerator::tryToGenerate(LookupState &LS, LookupKind K,
                                             JITDylib &JD,
                                             JITDylibLookupFlags JDLookupFlags,
                                             const SymbolLookupSet &Symbols) {

  if (Symbols.empty())
    return Error::success();

  LLVM_DEBUG({
    dbgs() << "InstanceSymbolGenerator trying to generate " << Symbols << "\n";
  });

  // Look the symbols up as required, so that names that aren't instance
  // symbols come back as std::nullopt rather than as null addresses. This
  // doesn't fail the lookup: those names are just left undefined.
  SymbolLookupSet LookupSymbols;
  for (auto &[Name, Flags] : Symbols)
    LookupSymbols.add(Name, SymbolLookupFlags::RequiredSymbol);

  Lookup(
      [&JD, LS = std::move(LS),
       LookupSymbols](Expected<SymbolLookupResult> Result) mutable {
        if (!Result)
          return LS.continueLookup(Result.takeError());

        if (Result->size() != LookupSymbols.size())
          return LS.continueLookup(make_error<StringError>(
              "Instance symbol lookup returned " + Twine(Result->size()) +
                  " results for " + Twine(LookupSymbols.size()) + " symbols",
              inconvertibleErrorCode()));

        SymbolMap NewSymbols;
        auto ResultIt = Result->begin();
        for (auto &[Name, Flags] : LookupSymbols) {
          const auto &Addr = *ResultIt++;
          if (Addr)
            NewSymbols[Name] = {*Addr, JITSymbolFlags::Exported};
        }

        LLVM_DEBUG({
          dbgs() << "InstanceSymbolGenerator lookup returned " << NewSymbols
                 << "\n";
        });

        if (NewSymbols.empty())
          return LS.continueLookup(Error::success());

        LS.continueLookup(JD.define(absoluteSymbols(std::move(NewSymbols))));
      },
      JD.getExecutionSession(), LookupSymbols);

  return Error::success();
}

} // namespace llvm::orc
