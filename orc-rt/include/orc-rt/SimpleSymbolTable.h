//===------- SimpleSymbolTable.h -- Simple Symbol Table ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Simple symbol table.
//
//===----------------------------------------------------------------------===//

#ifndef ORC_RT_SIMPLESYMBOLTABLE_H
#define ORC_RT_SIMPLESYMBOLTABLE_H

#include "orc-rt/Error.h"
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#define ORC_RT_SYMTAB_PAIR(sym) {{#sym}, reinterpret_cast<const void *>(&sym)}

namespace orc_rt {

/// A simple string-to-pointer symbol table. Symbols are added via
/// addSymbolsUnique, which rejects duplicates with an error.
class SimpleSymbolTable {
public:
  using SymbolTable = std::unordered_map<std::string, const void *>;
  using iterator = SymbolTable::const_iterator;

  bool empty() const noexcept { return Symbols.empty(); }
  size_t size() const noexcept { return Symbols.size(); }
  iterator begin() const noexcept { return Symbols.begin(); }
  iterator end() const noexcept { return Symbols.end(); }

  template <typename KeyT> decltype(auto) count(KeyT &&K) const {
    return Symbols.count(std::forward<KeyT>(K));
  }

  template <typename KeyT> decltype(auto) at(KeyT &&K) const {
    return Symbols.at(std::forward<KeyT>(K));
  }

  /// Adds symbol/address pairs from NewSymbols, first checking that all
  /// symbols in NewSymbols are unique (i.e. not previously defined).
  ///
  /// NewSymbols must not contain any internal duplicates.
  template <typename SymbolRangeT> Error addUnique(SymbolRangeT &&NewSymbols) {

    // First check for incompatible duplicate definitions (duplicates are
    // only permitted if they resolve to the same address). Error out if any
    // incompatible defs are found.
    {
      std::vector<std::string_view> IncompatibleDefs;
      for (auto &[Name, Addr] : NewSymbols) {
        auto I = Symbols.find(Name);
        if (I == Symbols.end() || I->second == Addr)
          continue;
        if (Symbols.count(Name))
          IncompatibleDefs.push_back(Name);
      }
      if (!IncompatibleDefs.empty())
        return makeIncompatibleDefsError(std::move(IncompatibleDefs));
    }

    // No duplicates. Add entries.
    for (auto &P : NewSymbols) {
      [[maybe_unused]] auto [I, Added] = Symbols.insert(P);
      assert((Added || I->second == P.second) &&
             "NewSymbols contains incompatible definitions");
    }

    return Error::success();
  }

private:
  static Error
  makeIncompatibleDefsError(std::vector<std::string_view> IncompatibleDefs);

  SymbolTable Symbols;
};

} // namespace orc_rt

#endif // ORC_RT_SIMPLESYMBOLTABLE_H
