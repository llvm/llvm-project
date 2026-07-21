//===------ SymbolLookupSet.h - Symbol set for ORC lookups ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// SymbolLookupSet class.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_EXECUTIONENGINE_ORC_SYMBOLLOOKUPSET_H
#define LLVM_EXECUTIONENGINE_ORC_SYMBOLLOOKUPSET_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ExecutionEngine/Orc/CoreContainers.h"
#include "llvm/Support/Error.h"

#include <initializer_list>
#include <type_traits>
#include <utility>
#include <vector>

namespace llvm::orc {

/// Lookup flags that apply to each symbol in a lookup.
///
/// If RequiredSymbol is used (the default) for a given symbol then that symbol
/// must be found during the lookup or the lookup will fail returning a
/// SymbolNotFound error. If WeaklyReferencedSymbol is used and the given
/// symbol is not found then the query will continue, and no result for the
/// missing symbol will be present in the result (assuming the rest of the
/// lookup succeeds).
enum class SymbolLookupFlags { RequiredSymbol, WeaklyReferencedSymbol };

/// A set of symbols to look up, each associated with a SymbolLookupFlags
/// value.
///
/// This class is backed by a vector and optimized for fast insertion,
/// deletion and iteration. It does not guarantee a stable order between
/// operations, and will not automatically detect duplicate elements (they
/// can be manually checked by calling the validate method).
class SymbolLookupSet {
public:
  using value_type = std::pair<SymbolStringPtr, SymbolLookupFlags>;
  using UnderlyingVector = std::vector<value_type>;
  using iterator = UnderlyingVector::iterator;
  using const_iterator = UnderlyingVector::const_iterator;

  SymbolLookupSet() = default;

  SymbolLookupSet(std::initializer_list<value_type> Elems) {
    for (auto &E : Elems)
      Symbols.push_back(std::move(E));
  }

  explicit SymbolLookupSet(
      SymbolStringPtr Name,
      SymbolLookupFlags Flags = SymbolLookupFlags::RequiredSymbol) {
    add(std::move(Name), Flags);
  }

  /// Construct a SymbolLookupSet from an initializer list of SymbolStringPtrs.
  explicit SymbolLookupSet(
      std::initializer_list<SymbolStringPtr> Names,
      SymbolLookupFlags Flags = SymbolLookupFlags::RequiredSymbol) {
    Symbols.reserve(Names.size());
    for (const auto &Name : Names)
      add(std::move(Name), Flags);
  }

  /// Construct a SymbolLookupSet from a SymbolNameSet with the given
  /// Flags used for each value.
  explicit SymbolLookupSet(
      const SymbolNameSet &Names,
      SymbolLookupFlags Flags = SymbolLookupFlags::RequiredSymbol) {
    Symbols.reserve(Names.size());
    for (const auto &Name : Names)
      add(Name, Flags);
  }

  /// Construct a SymbolLookupSet from a vector of symbols with the given Flags
  /// used for each value.
  /// If the ArrayRef contains duplicates it is up to the client to remove these
  /// before using this instance for lookup.
  explicit SymbolLookupSet(
      ArrayRef<SymbolStringPtr> Names,
      SymbolLookupFlags Flags = SymbolLookupFlags::RequiredSymbol) {
    Symbols.reserve(Names.size());
    for (const auto &Name : Names)
      add(Name, Flags);
  }

  /// Construct a SymbolLookupSet from DenseMap keys.
  template <typename ValT>
  static SymbolLookupSet
  fromMapKeys(const DenseMap<SymbolStringPtr, ValT> &M,
              SymbolLookupFlags Flags = SymbolLookupFlags::RequiredSymbol) {
    SymbolLookupSet Result;
    Result.Symbols.reserve(M.size());
    for (const auto &[Name, Val] : M)
      Result.add(Name, Flags);
    return Result;
  }

  /// Add an element to the set. The client is responsible for checking that
  /// duplicates are not added.
  SymbolLookupSet &
  add(SymbolStringPtr Name,
      SymbolLookupFlags Flags = SymbolLookupFlags::RequiredSymbol) {
    Symbols.push_back(std::make_pair(std::move(Name), Flags));
    return *this;
  }

  /// Quickly append one lookup set to another.
  SymbolLookupSet &append(SymbolLookupSet Other) {
    Symbols.reserve(Symbols.size() + Other.size());
    for (auto &KV : Other)
      Symbols.push_back(std::move(KV));
    return *this;
  }

  bool empty() const { return Symbols.empty(); }
  UnderlyingVector::size_type size() const { return Symbols.size(); }
  iterator begin() { return Symbols.begin(); }
  iterator end() { return Symbols.end(); }
  const_iterator begin() const { return Symbols.begin(); }
  const_iterator end() const { return Symbols.end(); }

  /// Removes the Ith element of the vector, replacing it with the last element.
  void remove(UnderlyingVector::size_type I) {
    std::swap(Symbols[I], Symbols.back());
    Symbols.pop_back();
  }

  /// Removes the element pointed to by the given iterator. This iterator and
  /// all subsequent ones (including end()) are invalidated.
  void remove(iterator I) { remove(I - begin()); }

  /// Removes all elements matching the given predicate, which must be callable
  /// as bool(const SymbolStringPtr &, SymbolLookupFlags Flags).
  template <typename PredFn> void remove_if(PredFn &&Pred) {
    UnderlyingVector::size_type I = 0;
    while (I != Symbols.size()) {
      const auto &Name = Symbols[I].first;
      auto Flags = Symbols[I].second;
      if (Pred(Name, Flags))
        remove(I);
      else
        ++I;
    }
  }

  /// Loop over the elements of this SymbolLookupSet, applying the Body function
  /// to each one. Body must be callable as
  /// bool(const SymbolStringPtr &, SymbolLookupFlags).
  /// If Body returns true then the element just passed in is removed from the
  /// set. If Body returns false then the element is retained.
  template <typename BodyFn>
  auto forEachWithRemoval(BodyFn &&Body) -> std::enable_if_t<
      std::is_same<decltype(Body(std::declval<const SymbolStringPtr &>(),
                                 std::declval<SymbolLookupFlags>())),
                   bool>::value> {
    UnderlyingVector::size_type I = 0;
    while (I != Symbols.size()) {
      const auto &Name = Symbols[I].first;
      auto Flags = Symbols[I].second;
      if (Body(Name, Flags))
        remove(I);
      else
        ++I;
    }
  }

  /// Loop over the elements of this SymbolLookupSet, applying the Body function
  /// to each one. Body must be callable as
  /// Expected<bool>(const SymbolStringPtr &, SymbolLookupFlags).
  /// If Body returns a failure value, the loop exits immediately. If Body
  /// returns true then the element just passed in is removed from the set. If
  /// Body returns false then the element is retained.
  template <typename BodyFn>
  auto forEachWithRemoval(BodyFn &&Body) -> std::enable_if_t<
      std::is_same<decltype(Body(std::declval<const SymbolStringPtr &>(),
                                 std::declval<SymbolLookupFlags>())),
                   Expected<bool>>::value,
      Error> {
    UnderlyingVector::size_type I = 0;
    while (I != Symbols.size()) {
      const auto &Name = Symbols[I].first;
      auto Flags = Symbols[I].second;
      auto Remove = Body(Name, Flags);
      if (!Remove)
        return Remove.takeError();
      if (*Remove)
        remove(I);
      else
        ++I;
    }
    return Error::success();
  }

  /// Construct a SymbolNameVector from this instance by dropping the Flags
  /// values.
  SymbolNameVector getSymbolNames() const {
    SymbolNameVector Names;
    Names.reserve(Symbols.size());
    for (const auto &KV : Symbols)
      Names.push_back(KV.first);
    return Names;
  }

  /// Sort the lookup set by pointer value. This sort is fast but sensitive to
  /// allocation order and so should not be used where a consistent order is
  /// required.
  void sortByAddress() { llvm::sort(Symbols, llvm::less_first()); }

  /// Sort the lookup set lexicographically. This sort is slow but the order
  /// is unaffected by allocation order.
  void sortByName() {
    llvm::sort(Symbols, [](const value_type &LHS, const value_type &RHS) {
      return *LHS.first < *RHS.first;
    });
  }

  /// Remove any duplicate elements. If a SymbolLookupSet is not duplicate-free
  /// by construction, this method can be used to turn it into a proper set.
  void removeDuplicates() {
    sortByAddress();
    auto LastI = llvm::unique(Symbols);
    Symbols.erase(LastI, Symbols.end());
  }

#ifndef NDEBUG
  /// Returns true if this set contains any duplicates. This should only be used
  /// in assertions.
  bool containsDuplicates() {
    if (Symbols.size() < 2)
      return false;
    sortByAddress();
    for (UnderlyingVector::size_type I = 1; I != Symbols.size(); ++I)
      if (Symbols[I].first == Symbols[I - 1].first)
        return true;
    return false;
  }
#endif

private:
  UnderlyingVector Symbols;
};

} // namespace llvm::orc

#endif // LLVM_EXECUTIONENGINE_ORC_SYMBOLLOOKUPSET_H
