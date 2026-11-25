//===-- SymbolLookupSet.h -- Symbol lookup sets -----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// SymbolLookupSet type.
//
//===----------------------------------------------------------------------===//

#ifndef ORC_RT_SUPPORT_SYMBOLLOOKUPSET_H
#define ORC_RT_SUPPORT_SYMBOLLOOKUPSET_H

#include "orc-rt/support/StringPool.h"
#include "orc-rt/support/SymbolLookupFlags.h"

#include <cassert>
#include <cstddef>
#include <initializer_list>
#include <utility>
#include <vector>

namespace orc_rt {

/// An ordered sequence of interned symbol names to look up, each paired with
/// a SymbolLookupFlags value describing how the symbol is referenced.
///
/// Names are held as StringPool::Ptrs, so building a set does not copy the
/// name strings. Names must be non-null. StringPool::Ptrs compare by identity,
/// so a set is only meaningful to a table whose names were interned in the
/// same pool. RemoteSymbolLookupSet holds names as plain strings instead.
class SymbolLookupSet {
  using VectorType = std::vector<std::pair<StringPool::Ptr, SymbolLookupFlags>>;

public:
  using value_type = VectorType::value_type;
  using iterator = VectorType::iterator;
  using const_iterator = VectorType::const_iterator;

  SymbolLookupSet() = default;
  SymbolLookupSet(std::initializer_list<value_type> Symbols)
      : Symbols(Symbols) {
#ifndef NDEBUG
    for (auto &Sym : this->Symbols)
      assert(Sym.first && "Null name in SymbolLookupSet");
#endif
  }

  iterator begin() { return Symbols.begin(); }
  iterator end() { return Symbols.end(); }
  const_iterator begin() const { return Symbols.begin(); }
  const_iterator end() const { return Symbols.end(); }

  value_type &operator[](size_t I) { return Symbols[I]; }
  const value_type &operator[](size_t I) const { return Symbols[I]; }

  void push_back(value_type V) {
    assert(V.first && "Null name in SymbolLookupSet");
    Symbols.push_back(std::move(V));
  }
  void reserve(size_t N) { Symbols.reserve(N); }

  bool empty() const { return Symbols.empty(); }
  size_t size() const { return Symbols.size(); }

private:
  VectorType Symbols;
};

} // namespace orc_rt

#endif // ORC_RT_SUPPORT_SYMBOLLOOKUPSET_H
