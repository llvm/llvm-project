//===-- RemoteSymbolLookupSet.h -- Remote symbol lookup sets ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// RemoteSymbolLookupSet type.
//
//===----------------------------------------------------------------------===//

#ifndef ORC_RT_SUPPORT_REMOTESYMBOLLOOKUPSET_H
#define ORC_RT_SUPPORT_REMOTESYMBOLLOOKUPSET_H

#include "orc-rt/support/SymbolLookupFlags.h"

#include <cstddef>
#include <initializer_list>
#include <string>
#include <utility>
#include <vector>

namespace orc_rt {

/// An ordered sequence of symbol names to look up, each paired with a
/// SymbolLookupFlags value describing how the symbol is referenced.
///
/// Names are held as plain std::strings, the form in which lookup requests
/// arrive from the controller (cf. llvm::orc::RemoteSymbolLookupSet).
class RemoteSymbolLookupSet {
  using VectorType = std::vector<std::pair<std::string, SymbolLookupFlags>>;

public:
  using value_type = VectorType::value_type;
  using iterator = VectorType::iterator;
  using const_iterator = VectorType::const_iterator;

  RemoteSymbolLookupSet() = default;
  RemoteSymbolLookupSet(std::initializer_list<value_type> Symbols)
      : Symbols(Symbols) {}

  iterator begin() { return Symbols.begin(); }
  iterator end() { return Symbols.end(); }
  const_iterator begin() const { return Symbols.begin(); }
  const_iterator end() const { return Symbols.end(); }

  value_type &operator[](size_t I) { return Symbols[I]; }
  const value_type &operator[](size_t I) const { return Symbols[I]; }

  void push_back(value_type V) { Symbols.push_back(std::move(V)); }
  void reserve(size_t N) { Symbols.reserve(N); }

  bool empty() const { return Symbols.empty(); }
  size_t size() const { return Symbols.size(); }

private:
  VectorType Symbols;
};

} // namespace orc_rt

#endif // ORC_RT_SUPPORT_REMOTESYMBOLLOOKUPSET_H
