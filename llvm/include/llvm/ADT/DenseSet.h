//===- llvm/ADT/DenseSet.h - Dense probed hash table ------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file defines the DenseSet and SmallDenseSet classes.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_ADT_DENSESET_H
#define LLVM_ADT_DENSESET_H

#include "llvm/ADT/ADL.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseMapInfo.h"
#include "llvm/ADT/STLForwardCompat.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/type_traits.h"
#include <cstddef>
#include <initializer_list>
#include <iterator>
#include <utility>

namespace llvm {

namespace detail {

struct DenseSetEmpty {};

// Use the empty base class trick so we can create a DenseMap where the buckets
// contain only a single item.
template <typename KeyT> class DenseSetPair : public DenseSetEmpty {
  KeyT key;

public:
  KeyT &getFirst() { return key; }
  const KeyT &getFirst() const { return key; }
  DenseSetEmpty &getSecond() { return *this; }
  const DenseSetEmpty &getSecond() const { return *this; }
};

/// Base class for DenseSet and DenseSmallSet.
///
/// MapTy should be either
///
///   DenseMap<ValueT, detail::DenseSetEmpty, ValueInfoT,
///            detail::DenseSetPair<ValueT>>
///
/// or the equivalent SmallDenseMap type.  ValueInfoT must implement the
/// DenseMapInfo "concept".
template <typename ValueT, typename MapTy, typename ValueInfoT>
class DenseSetImpl {
  static_assert(sizeof(typename MapTy::value_type) == sizeof(ValueT),
                "DenseMap buckets unexpectedly large!");
  MapTy TheMap;

  template <typename T>
  using const_arg_type_t = typename const_pointer_or_const_ref<T>::type;

public:
  using key_type = ValueT;
  using value_type = ValueT;
  using size_type = unsigned;

  explicit DenseSetImpl(unsigned InitialReserve = 0) : TheMap(InitialReserve) {}

  template <typename InputIt>
  DenseSetImpl(const InputIt &I, const InputIt &E)
      : DenseSetImpl(PowerOf2Ceil(std::distance(I, E))) {
    insert(I, E);
  }

  DenseSetImpl(std::initializer_list<ValueT> Elems)
      : DenseSetImpl(PowerOf2Ceil(Elems.size())) {
    insert(Elems.begin(), Elems.end());
  }

  template <typename Range>
  DenseSetImpl(llvm::from_range_t, Range &&R)
      : DenseSetImpl(adl_begin(R), adl_end(R)) {}

  bool empty() const { return TheMap.empty(); }
  size_type size() const { return TheMap.size(); }
  size_t getMemorySize() const { return TheMap.getMemorySize(); }

  /// Grow the DenseSet so that it has at least Size buckets. Will not shrink
  /// the Size of the set.
  void resize(size_t Size) { TheMap.resize(Size); }

  /// Grow the DenseSet so that it can contain at least \p NumEntries items
  /// before resizing again.
  void reserve(size_t Size) { TheMap.reserve(Size); }

  void clear() { TheMap.clear(); }

  bool erase(const ValueT &V) { return TheMap.erase(V); }

  void swap(DenseSetImpl &RHS) { TheMap.swap(RHS.TheMap); }

private:
  template <bool IsConst> class DenseSetIterator {
    friend class DenseSetImpl;

    using MapIteratorT =
        std::conditional_t<IsConst, typename MapTy::const_iterator,
                           typename MapTy::iterator>;

    MapIteratorT I;

  public:
    using difference_type = typename MapIteratorT::difference_type;
    using iterator_category = std::forward_iterator_tag;
    using value_type = ValueT;
    using pointer =
        std::conditional_t<IsConst, const value_type *, value_type *>;
    using reference =
        std::conditional_t<IsConst, const value_type &, value_type &>;

    DenseSetIterator() = default;
    DenseSetIterator(MapIteratorT I) : I(I) {}

    // Allow conversion from iterator to const_iterator.
    template <bool C = IsConst, typename = std::enable_if_t<C>>
    DenseSetIterator(const DenseSetIterator<false> &Other) : I(Other.I) {}

    reference operator*() const { return I->getFirst(); }
    pointer operator->() const { return &I->getFirst(); }

    DenseSetIterator &operator++() {
      ++I;
      return *this;
    }
    DenseSetIterator operator++(int) {
      auto T = *this;
      ++I;
      return T;
    }

    friend bool operator==(const DenseSetIterator &LHS,
                           const DenseSetIterator &RHS) {
      return LHS.I == RHS.I;
    }
    friend bool operator!=(const DenseSetIterator &LHS,
                           const DenseSetIterator &RHS) {
      return LHS.I != RHS.I;
    }
  };

public:
  using iterator = DenseSetIterator<false>;
  using const_iterator = DenseSetIterator<true>;

  iterator begin() { return iterator(TheMap.begin()); }
  iterator end() { return iterator(TheMap.end()); }

  const_iterator begin() const { return const_iterator(TheMap.begin()); }
  const_iterator end() const { return const_iterator(TheMap.end()); }

  iterator find(const_arg_type_t<ValueT> V) { return iterator(TheMap.find(V)); }
  const_iterator find(const_arg_type_t<ValueT> V) const {
    return const_iterator(TheMap.find(V));
  }

  /// Check if the set contains the given element.
  [[nodiscard]] bool contains(const_arg_type_t<ValueT> V) const {
    return TheMap.contains(V);
  }

  /// Return 1 if the specified key is in the set, 0 otherwise.
  [[nodiscard]] size_type count(const_arg_type_t<ValueT> V) const {
    return TheMap.count(V);
  }

  /// Alternative version of find() which allows a different, and possibly less
  /// expensive, key type.
  /// The DenseMapInfo is responsible for supplying methods
  /// getHashValue(LookupKeyT) and isEqual(LookupKeyT, KeyT) for each key type
  /// used.
  template <class LookupKeyT> iterator find_as(const LookupKeyT &Val) {
    return iterator(TheMap.find_as(Val));
  }
  template <class LookupKeyT>
  const_iterator find_as(const LookupKeyT &Val) const {
    return const_iterator(TheMap.find_as(Val));
  }

  void erase(iterator I) { return TheMap.erase(I.I); }
  void erase(const_iterator CI) { return TheMap.erase(CI.I); }

  std::pair<iterator, bool> insert(const ValueT &V) {
    detail::DenseSetEmpty Empty;
    return TheMap.try_emplace(V, Empty);
  }

  std::pair<iterator, bool> insert(ValueT &&V) {
    detail::DenseSetEmpty Empty;
    return TheMap.try_emplace(std::move(V), Empty);
  }

  /// Alternative version of insert that uses a different (and possibly less
  /// expensive) key type.
  template <typename LookupKeyT>
  std::pair<iterator, bool> insert_as(const ValueT &V,
                                      const LookupKeyT &LookupKey) {
    return TheMap.insert_as({V, detail::DenseSetEmpty()}, LookupKey);
  }
  template <typename LookupKeyT>
  std::pair<iterator, bool> insert_as(ValueT &&V, const LookupKeyT &LookupKey) {
    return TheMap.insert_as({std::move(V), detail::DenseSetEmpty()}, LookupKey);
  }

  // Range insertion of values.
  template <typename InputIt> void insert(InputIt I, InputIt E) {
    for (; I != E; ++I)
      insert(*I);
  }

  template <typename Range> void insert_range(Range &&R) {
    insert(adl_begin(R), adl_end(R));
  }
};

/// Equality comparison for DenseSet.
///
/// Iterates over elements of LHS confirming that each element is also a member
/// of RHS, and that RHS contains no additional values.
/// Equivalent to N calls to RHS.count. Amortized complexity is linear, worst
/// case is O(N^2) (if every hash collides).
template <typename ValueT, typename MapTy, typename ValueInfoT>
bool operator==(const DenseSetImpl<ValueT, MapTy, ValueInfoT> &LHS,
                const DenseSetImpl<ValueT, MapTy, ValueInfoT> &RHS) {
  if (LHS.size() != RHS.size())
    return false;

  for (auto &E : LHS)
    if (!RHS.count(E))
      return false;

  return true;
}

/// Inequality comparison for DenseSet.
///
/// Equivalent to !(LHS == RHS). See operator== for performance notes.
template <typename ValueT, typename MapTy, typename ValueInfoT>
bool operator!=(const DenseSetImpl<ValueT, MapTy, ValueInfoT> &LHS,
                const DenseSetImpl<ValueT, MapTy, ValueInfoT> &RHS) {
  return !(LHS == RHS);
}

} // end namespace detail

/// Implements a dense probed hash-table based set.
template <typename ValueT, typename ValueInfoT = DenseMapInfo<ValueT>>
class DenseSet : public detail::DenseSetImpl<
                     ValueT,
                     DenseMap<ValueT, detail::DenseSetEmpty, ValueInfoT,
                              detail::DenseSetPair<ValueT>>,
                     ValueInfoT> {
  using BaseT =
      detail::DenseSetImpl<ValueT,
                           DenseMap<ValueT, detail::DenseSetEmpty, ValueInfoT,
                                    detail::DenseSetPair<ValueT>>,
                           ValueInfoT>;

public:
  using BaseT::BaseT;
};

/// Implements a dense probed hash-table based set with some number of buckets
/// stored inline.
template <typename ValueT, unsigned InlineBuckets = 4,
          typename ValueInfoT = DenseMapInfo<ValueT>>
class SmallDenseSet
    : public detail::DenseSetImpl<
          ValueT,
          SmallDenseMap<ValueT, detail::DenseSetEmpty, InlineBuckets,
                        ValueInfoT, detail::DenseSetPair<ValueT>>,
          ValueInfoT> {
  using BaseT = detail::DenseSetImpl<
      ValueT,
      SmallDenseMap<ValueT, detail::DenseSetEmpty, InlineBuckets, ValueInfoT,
                    detail::DenseSetPair<ValueT>>,
      ValueInfoT>;

public:
  using BaseT::BaseT;
};

} // end namespace llvm

#endif // LLVM_ADT_DENSESET_H
