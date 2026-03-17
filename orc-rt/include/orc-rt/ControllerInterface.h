//===--- ControllerInterface.h -- Controller Interface Symtab ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Controller interface symbol table.
//
//===----------------------------------------------------------------------===//

#ifndef ORC_RT_CONTROLLERINTERFACE_H
#define ORC_RT_CONTROLLERINTERFACE_H

#include "orc-rt/Error.h"
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#define ORC_RT_SYMTAB_PAIR(sym) {{#sym}, reinterpret_cast<const void *>(&sym)}

namespace orc_rt {

/// A symbol table defining the interface exposed by the ORC runtime to the
/// controller. Symbols are added via addSymbolsUnique, which rejects
/// duplicates with an error.
class ControllerInterface {
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
  template <typename SymbolRangeT>
  Error addSymbolsUnique(SymbolRangeT &&NewSymbols) {

    // First check for duplicates, error out if any are found.
    {
      std::vector<std::string_view> Dups;
      for (auto &[Name, Addr] : NewSymbols)
        if (Symbols.count(Name))
          Dups.push_back(Name);
      if (!Dups.empty())
        return makeDuplicatesError(std::move(Dups));
    }

    // No duplicates. Add entries.
    for (auto &P : NewSymbols) {
      [[maybe_unused]] bool Added = Symbols.insert(P).second;
      assert(Added && "NewSymbols contains duplicate definitions");
    }

    return Error::success();
  }

private:
  static Error makeDuplicatesError(std::vector<std::string_view> Dups);

  SymbolTable Symbols;
};

} // namespace orc_rt

#endif // ORC_RT_CONTROLLERINTERFACE_H
