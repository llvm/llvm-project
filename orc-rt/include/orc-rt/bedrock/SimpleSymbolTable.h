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

#ifndef ORC_RT_BEDROCK_SIMPLESYMBOLTABLE_H
#define ORC_RT_BEDROCK_SIMPLESYMBOLTABLE_H

#include "orc-rt/support/Error.h"
#include "orc-rt/support/Mangling.h"
#include "orc-rt/support/move_only_function.h"

#include <algorithm>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

/// Builds a (name, address) pair for a symbol, taking its C-level name from the
/// identifier itself. For use in the interface arrays passed to addUnique.
#define ORC_RT_SYMTAB_C_PAIR(sym)                                              \
  {SymbolNameSpec::c(#sym), reinterpret_cast<const void *>(&sym)}

namespace orc_rt {

/// A simple symbol table mapping linker-level names to addresses.
///
/// Entries are added via addUnique. Keys are always linker-level names: names
/// given as SymbolNameSpecs are mangled on the way in, and count() and at()
/// mangle their argument the same way.
class SimpleSymbolTable {
public:
  using SymbolTable = std::unordered_map<std::string, const void *>;
  using iterator = SymbolTable::const_iterator;

  using MutatorFn = move_only_function<Error(SimpleSymbolTable &)>;

  bool empty() const noexcept { return Symbols.empty(); }
  size_t size() const noexcept { return Symbols.size(); }
  iterator begin() const noexcept { return Symbols.begin(); }
  iterator end() const noexcept { return Symbols.end(); }

  /// Returns 1 if NameSpec's mangled name is in the table, 0 otherwise.
  size_t count(const SymbolNameSpec &NameSpec) const {
    return Symbols.count(mangledCopy(NameSpec));
  }

  /// Returns the address registered for NameSpec's mangled name, which must be
  /// present in the table.
  const void *at(const SymbolNameSpec &NameSpec) const {
    auto MangledName = mangledCopy(NameSpec);
    assert(Symbols.count(MangledName) && "Name not present");
    return Symbols.at(MangledName);
  }

  /// Adds (name, address) pairs from NewSymbols, mangling each name according
  /// to its SymbolNameKind. Redundant definitions where the (name, address)
  /// pair matches an existing table entry are allowed. Duplicate symbol names
  /// with different addresses will return an error, and addUnique will leave
  /// the table unchanged.
  ///
  /// NewSymbols must not contain any internal duplicates.
  template <typename SymbolRangeT> Error addUnique(SymbolRangeT &&NewSymbols) {
    // Generate the mangled version of the NewSymbols map.
    std::vector<std::pair<std::string, const void *>> NewMangledSymbols;
    NewMangledSymbols.reserve(std::size(NewSymbols));
    for (auto &[NameSpec, Addr] : NewSymbols)
      NewMangledSymbols.emplace_back(mangledCopy(NameSpec), Addr);

    // Check for duplicate definitions whose addresses disagree.
    std::vector<std::string_view> IncompatibleDefs;
    for (auto &[MangledName, Addr] : NewMangledSymbols) {
      auto I = Symbols.find(MangledName);
      if (I != Symbols.end() && I->second != Addr)
        IncompatibleDefs.push_back(MangledName);
    }

    // If any incompatible definitions exist then return with an error.
    if (!IncompatibleDefs.empty())
      return makeIncompatibleDefsError(std::move(IncompatibleDefs));

    // Otherwise update the table.
    for (auto &[MangledName, Addr] : NewMangledSymbols) {
      [[maybe_unused]] auto [I, Added] =
          Symbols.insert({std::move(MangledName), Addr});
      assert((Added || I->second == Addr) &&
             "NewSymbols contains internal duplicates");
    }

    return Error::success();
  }

  /// Adds all entries from Other. Duplicate handling matches addUnique above:
  /// on error this table is left unchanged. Consumes Other.
  Error addUnique(SimpleSymbolTable &&Other) {
    std::vector<std::string_view> IncompatibleDefs;
    for (auto &[Name, Addr] : Other.Symbols) {
      auto I = Symbols.find(Name);
      if (I != Symbols.end() && I->second != Addr)
        IncompatibleDefs.push_back(Name);
    }

    if (!IncompatibleDefs.empty())
      return makeIncompatibleDefsError(std::move(IncompatibleDefs));

    // Splices nodes across, so no keys are copied.
    Symbols.merge(Other.Symbols);

    return Error::success();
  }

private:
  static Error
  makeIncompatibleDefsError(std::vector<std::string_view> IncompatibleDefs);

  SymbolTable Symbols;
};

} // namespace orc_rt

#endif // ORC_RT_BEDROCK_SIMPLESYMBOLTABLE_H
