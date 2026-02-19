//===---- RedirectionManager.cpp - Redirection manager interface in Orc ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ExecutionEngine/Orc/RedirectionManager.h"

#define DEBUG_TYPE "orc"

using namespace llvm;
using namespace llvm::orc;

Error RedirectionManager::redirect(JITDylib &JD, SymbolMap NewDests) {
  std::promise<MSVCPError> P;
  redirect(JD, std::move(NewDests),
           [&P](Error E) { P.set_value(std::move(E)); });
  auto F = P.get_future();
  return F.get();
}

void RedirectionManager::anchor() {}

Error RedirectableSymbolManager::createRedirectableSymbols(
    ResourceTrackerSP RT, SymbolMap InitialDests) {
  auto &JD = RT->getJITDylib();
  return JD.define(std::make_unique<RedirectableMaterializationUnit>(
                       *this, std::move(InitialDests)),
                   RT);
}
