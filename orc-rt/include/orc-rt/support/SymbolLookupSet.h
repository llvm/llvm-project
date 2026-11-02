//===-- SymbolLookupSet.h -- Symbol lookup sets and results -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// SymbolLookupSet and SymbolLookupResult types.
//
//===----------------------------------------------------------------------===//

#ifndef ORC_RT_SUPPORT_SYMBOLLOOKUPSET_H
#define ORC_RT_SUPPORT_SYMBOLLOOKUPSET_H

#include "orc-rt/support/SymbolLookupFlags.h"

#include <cstddef>
#include <initializer_list>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace orc_rt {

/// An ordered sequence of symbol names to look up, each paired with a
/// SymbolLookupFlags value describing how the symbol is referenced.
class SymbolLookupSet {
  using VectorType = std::vector<std::pair<std::string, SymbolLookupFlags>>;

public:
  using value_type = VectorType::value_type;
  using iterator = VectorType::iterator;
  using const_iterator = VectorType::const_iterator;

  SymbolLookupSet() = default;
  SymbolLookupSet(std::initializer_list<value_type> Symbols)
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

/// The result of looking up a SymbolLookupSet: one entry per element of the
/// set, in the same order.
///
/// An empty optional indicates that a required symbol was not found. A missing
/// weakly referenced symbol is reported as a present optional holding a null
/// address.
class SymbolLookupResult {
  using VectorType = std::vector<std::optional<const void *>>;

public:
  using value_type = VectorType::value_type;
  using iterator = VectorType::iterator;
  using const_iterator = VectorType::const_iterator;

  iterator begin() { return Addrs.begin(); }
  iterator end() { return Addrs.end(); }
  const_iterator begin() const { return Addrs.begin(); }
  const_iterator end() const { return Addrs.end(); }

  value_type &operator[](size_t I) { return Addrs[I]; }
  const value_type &operator[](size_t I) const { return Addrs[I]; }

  void push_back(value_type V) { Addrs.push_back(std::move(V)); }
  void reserve(size_t N) { Addrs.reserve(N); }
  void resize(size_t N) { Addrs.resize(N); }

  bool empty() const { return Addrs.empty(); }
  size_t size() const { return Addrs.size(); }

private:
  VectorType Addrs;
};

} // namespace orc_rt

#endif // ORC_RT_SUPPORT_SYMBOLLOOKUPSET_H
