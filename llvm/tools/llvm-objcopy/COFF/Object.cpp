//===- Object.cpp ---------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Object.h"
#include <algorithm>

namespace llvm {
namespace objcopy {
namespace coff {

using namespace object;

void Object::addSymbols(ArrayRef<Symbol> NewSymbols) {
  for (Symbol S : NewSymbols) {
    S.UniqueId = NextSymbolUniqueId++;
    Symbols.emplace_back(S);
  }
  updateSymbols();
}

void Object::updateSymbols() {
  SymbolMap = DenseMap<size_t, Symbol *>(Symbols.size());
  size_t RawSymIndex = 0;
  for (Symbol &Sym : Symbols) {
    SymbolMap[Sym.UniqueId] = &Sym;
    Sym.RawIndex = RawSymIndex;
    RawSymIndex += 1 + Sym.Sym.NumberOfAuxSymbols;
  }
}

const Symbol *Object::findSymbol(size_t UniqueId) const {
  auto It = SymbolMap.find(UniqueId);
  if (It == SymbolMap.end())
    return nullptr;
  return It->second;
}

void Object::removeSymbols(function_ref<bool(const Symbol &)> ToRemove) {
  Symbols.erase(
      std::remove_if(std::begin(Symbols), std::end(Symbols),
                     [ToRemove](const Symbol &Sym) { return ToRemove(Sym); }),
      std::end(Symbols));
  updateSymbols();
}

Error Object::markSymbols() {
  for (Symbol &Sym : Symbols)
    Sym.Referenced = false;
  for (const Section &Sec : Sections) {
    for (const Relocation &R : Sec.Relocs) {
      auto It = SymbolMap.find(R.Target);
      if (It == SymbolMap.end())
        return make_error<StringError>("Relocation target " + Twine(R.Target) +
                                           " not found",
                                       object_error::invalid_symbol_index);
      It->second->Referenced = true;
    }
  }
  return Error::success();
}

} // end namespace coff
} // end namespace objcopy
} // end namespace llvm
