//===---------- NullResolver.cpp - Reject symbol lookup requests ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ExecutionEngine/Orc/NullResolver.h"

#include "llvm/Support/ErrorHandling.h"

namespace llvm {
namespace orc {

SymbolNameSet NullResolver::getResponsibilitySet(const SymbolNameSet &Symbols) {
  return Symbols;
}

SymbolNameSet
NullResolver::lookup(std::shared_ptr<AsynchronousSymbolQuery> Query,
                     SymbolNameSet Symbols) {
  assert(Symbols.empty() && "Null resolver: Symbols must be empty");
  return Symbols;
}

JITSymbol NullLegacyResolver::findSymbol(const std::string &Name) {
  llvm_unreachable("Unexpected cross-object symbol reference");
}

JITSymbol
NullLegacyResolver::findSymbolInLogicalDylib(const std::string &Name) {
  llvm_unreachable("Unexpected cross-object symbol reference");
}

} // End namespace orc.
} // End namespace llvm.
