//===- Interval.h -----------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The Interval class is a generic interval of ordered objects that implement:
// - T * T::getPrevNode()
// - T * T::getNextNode()
// - bool T::comesBefore(const T *) const
//
// This is currently used for Instruction intervals.
// It provides an API for some basic operations on the interval, including some
// simple set operations, like union, interseciton and others.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_VECTORIZE_SANDBOXVECTORIZER_INSTRINTERVAL_H
#define LLVM_TRANSFORMS_VECTORIZE_SANDBOXVECTORIZER_INSTRINTERVAL_H

#include "llvm/ADT/ArrayRef.h"
#include <iterator>

namespace llvm::sandboxir {

/// A simple iterator for iterating the interval.
template <typename T, typename IntervalType> class IntervalIterator {
  T *I;
  IntervalType &R;

public:
  using difference_type = std::ptrdiff_t;
  using value_type = T;
  using pointer = value_type *;
  using reference = T &;
  using iterator_category = std::bidirectional_iterator_tag;

  IntervalIterator(T *I, IntervalType &R) : I(I), R(R) {}
  bool operator==(const IntervalIterator &Other) const {
    assert(&R == &Other.R && "Iterators belong to different regions!");
    return Other.I == I;
  }
  bool operator!=(const IntervalIterator &Other) const {
    return !(*this == Other);
  }
  IntervalIterator &operator++() {
    assert(I != nullptr && "already at end()!");
    I = I->getNextNode();
    return *this;
  }
  IntervalIterator operator++(int) {
    auto ItCopy = *this;
    ++*this;
    return ItCopy;
  }
  IntervalIterator &operator--() {
    // `I` is nullptr for end() when To is the BB terminator.
    I = I != nullptr ? I->getPrevNode() : R.To;
    return *this;
  }
  IntervalIterator operator--(int) {
    auto ItCopy = *this;
    --*this;
    return ItCopy;
  }
  template <typename HT = std::enable_if<std::is_same<T, T *&>::value>>
  T &operator*() {
    return *I;
  }
  T &operator*() const { return *I; }
};

template <typename T> class Interval {
  T *From;
  T *To;

public:
  Interval() : From(nullptr), To(nullptr) {}
  Interval(T *From, T *To) : From(From), To(To) {
    assert((From == To || From->comesBefore(To)) &&
           "From should come before From!");
  }
  Interval(ArrayRef<T *> Elems) {
    assert(!Elems.empty() && "Expected non-empty Elems!");
    From = Elems[0];
    To = Elems[0];
    for (auto *I : drop_begin(Elems)) {
      if (I->comesBefore(From))
        From = I;
      else if (To->comesBefore(I))
        To = I;
    }
  }
  bool empty() const {
    assert(((From == nullptr && To == nullptr) ||
            (From != nullptr && To != nullptr)) &&
           "Either none or both should be null");
    return From == nullptr;
  }
  bool contains(T *I) const {
    if (empty())
      return false;
    return (From == I || From->comesBefore(I)) &&
           (I == To || I->comesBefore(To));
  }
  T *top() const { return From; }
  T *bottom() const { return To; }

  using iterator = IntervalIterator<T, Interval>;
  using const_iterator = IntervalIterator<const T, const Interval>;
  iterator begin() { return iterator(From, *this); }
  iterator end() {
    return iterator(To != nullptr ? To->getNextNode() : nullptr, *this);
  }
  const_iterator begin() const { return const_iterator(From, *this); }
  const_iterator end() const {
    return const_iterator(To != nullptr ? To->getNextNode() : nullptr, *this);
  }
};

} // namespace llvm::sandboxir

#endif // LLVM_TRANSFORMS_VECTORIZE_SANDBOXVECTORIZER_INSTRINTERVAL_H
