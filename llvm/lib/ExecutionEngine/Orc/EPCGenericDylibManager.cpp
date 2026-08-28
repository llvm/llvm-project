//===------- EPCGenericDylibManager.cpp -- Dylib management via EPC -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ExecutionEngine/Orc/EPCGenericDylibManager.h"

#include "llvm/ExecutionEngine/Orc/Core.h"

namespace llvm {
namespace orc {

Expected<tpctypes::DylibHandle> EPCGenericDylibManager::open(StringRef Path,
                                                             uint64_t Mode) {
  return B.Open(ES, B.Instance, Path, Mode);
}

void EPCGenericDylibManager::lookupAsync(tpctypes::DylibHandle H,
                                         const SymbolLookupSet &Lookup,
                                         SymbolLookupCompleteFn Complete) {
  B.Resolve(std::move(Complete), ES, B.Instance, H, Lookup);
}

Expected<tpctypes::DylibHandle>
EPCGenericDylibManager::loadDylib(const char *DylibPath) {
  return open(DylibPath, 0);
}

void EPCGenericDylibManager::lookupSymbolsAsync(
    tpctypes::DylibHandle H, const SymbolLookupSet &Symbols,
    DylibManager::SymbolLookupCompleteFn Complete) {
  lookupAsync(H, Symbols, std::move(Complete));
}

} // end namespace orc
} // end namespace llvm
