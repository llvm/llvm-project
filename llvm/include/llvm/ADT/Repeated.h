//===- llvm/ADT/Repeated.h - Repeated value range ---------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines the Repeated<T> class, a memory-efficient range representing N
// copies of the same value.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ADT_REPEATED_H
#define LLVM_ADT_REPEATED_H

#include "llvm/ADT/iterator.h"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <utility>

namespace llvm {

/// A random-access iterator that always dereferences to the same value.
template <typename T>
class RepeatedIterator
    : public iterator_facade_base<RepeatedIterator<T>,
                                  std::random_access_iterator_tag, T, ptrdiff_t,
                                  const T *, const T &> {
  const T *value = nullptr;
  ptrdiff_t index = 0;

public:
  RepeatedIterator() = default;
  RepeatedIterator(const T *value, ptrdiff_t index)
      : value(value), index(index) {}

  const T &operator*() const { return *value; }

  bool operator==(const RepeatedIterator &rhs) const {
    assert((!value || !rhs.value || value == rhs.value) &&
           "comparing iterators from different Repeated ranges");
    return index == rhs.index;
  }

  bool operator<(const RepeatedIterator &rhs) const {
    assert((!value || !rhs.value || value == rhs.value) &&
           "comparing iterators from different Repeated ranges");
    return index < rhs.index;
  }

  ptrdiff_t operator-(const RepeatedIterator &rhs) const {
    assert((!value || !rhs.value || value == rhs.value) &&
           "subtracting iterators from different Repeated ranges");
    return index - rhs.index;
  }

  RepeatedIterator &operator+=(ptrdiff_t n) {
    index += n;
    return *this;
  }

  RepeatedIterator &operator-=(ptrdiff_t n) {
    index -= n;
    return *this;
  }
};

/// A memory-efficient immutable range with a single value repeated N times.
/// The value is owned by the range.
///
/// `Repeated<T>` is also a proper random-access range: `begin()`/`end()`
/// return iterators that always dereference to the same stored value.
// At least 16-byte aligned so that Repeated<T>* has more low bits available
// than a plain pointer. The primary use case is pointer-like types (e.g. MLIR
// Type, Value) where Repeated<T>* appears in a PointerUnion alongside them.
template <typename T>
struct [[nodiscard]] alignas(std::max(size_t{16}, alignof(T))) Repeated {
  T storage;
  size_t count;

  /// Create a `value` repeated `count` times.
  /// Uses the same argument order like std container constructors.
  template <typename U>
  Repeated(size_t count, U &&value)
      : storage(std::forward<U>(value)), count(count) {}

  using iterator = RepeatedIterator<T>;
  using const_iterator = iterator;
  using reverse_iterator = std::reverse_iterator<iterator>;
  using const_reverse_iterator = reverse_iterator;
  using value_type = T;
  using size_type = size_t;

  iterator begin() const { return {&storage, 0}; }
  iterator end() const { return {&storage, static_cast<ptrdiff_t>(count)}; }
  reverse_iterator rbegin() const { return reverse_iterator(end()); }
  reverse_iterator rend() const { return reverse_iterator(begin()); }

  size_t size() const { return count; }
  bool empty() const { return count == 0; }

  const T &value() const { return storage; }
  const T &operator[](size_t idx) const {
    assert(idx < size() && "Out of bounds");
    (void)idx;
    return storage;
  }
};

template <typename U> Repeated(size_t, U &&) -> Repeated<std::decay_t<U>>;

} // namespace llvm

#endif // LLVM_ADT_REPEATED_H
