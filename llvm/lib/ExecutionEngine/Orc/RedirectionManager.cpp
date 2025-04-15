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

void RedirectionManager::anchor() {}

Error RedirectableSymbolManager::createRedirectableSymbols(
    ResourceTrackerSP RT, SymbolMap InitialDests) {
  auto &JD = RT->getJITDylib();
  return JD.define(std::make_unique<RedirectableMaterializationUnit>(
                       *this, std::move(InitialDests)),
                   RT);
}
