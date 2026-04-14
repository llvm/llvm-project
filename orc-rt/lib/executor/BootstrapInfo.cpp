//===- BootstrapInfo.cpp --------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Contains the implementation of APIs in the orc-rt/BootstrapInfo.h header.
//
//===----------------------------------------------------------------------===//

#include "orc-rt/BootstrapInfo.h"

#include "orc-rt/ExecutorProcessInfo.h"
#include "orc-rt/Session.h"

namespace orc_rt {

BootstrapInfo::BootstrapInfo(Session &S, SimpleSymbolTable Symbols,
                             ValueMap Values)
    : S(S), Symbols(std::move(Symbols)), Values(std::move(Values)) {}

Expected<BootstrapInfo>
BootstrapInfo::CreateDefault(Session &S,
                             InitialSymbolsBuilder AddInitialSymbols,
                             InitialValuesBuilder AddInitialValues) {

  SimpleSymbolTable InitialSymbols;
  // Add session symbol.
  std::pair<const char *, const void *> SessionSymbol[] = {
      {"orc_rt_Session_Instance", static_cast<const void *>(&S)}};
  if (auto Err = InitialSymbols.addUnique(SessionSymbol))
    return std::move(Err);

  if (AddInitialSymbols)
    if (auto Err = AddInitialSymbols(InitialSymbols))
      return std::move(Err);

  ValueMap InitialValues;
  if (AddInitialValues)
    if (auto Err = AddInitialValues(InitialValues))
      return std::move(Err);

  return BootstrapInfo(S, std::move(InitialSymbols), std::move(InitialValues));
}

const ExecutorProcessInfo &BootstrapInfo::processInfo() const noexcept {
  return S.processInfo();
}

} // namespace orc_rt
