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
// simple set operations, like union, intersection and others.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_VECTORIZE_SANDBOXVECTORIZER_INSTRINTERVAL_H
#define LLVM_TRANSFORMS_VECTORIZE_SANDBOXVECTORIZER_INSTRINTERVAL_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/SandboxIR/Instruction.h"
#include "llvm/Support/raw_ostream.h"
#include <iterator>
#include <type_traits>

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
    I = I != nullptr ? I->getPrevNode() : R.bottom();
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
  T *Top;
  T *Bottom;

public:
  Interval() : Top(nullptr), Bottom(nullptr) {}
  Interval(T *Top, T *Bottom) : Top(Top), Bottom(Bottom) {
    assert((Top == Bottom || Top->comesBefore(Bottom)) &&
           "Top should come before Bottom!");
  }
  Interval(ArrayRef<T *> Elems) {
    assert(!Elems.empty() && "Expected non-empty Elems!");
    Top = Elems[0];
    Bottom = Elems[0];
    for (auto *I : drop_begin(Elems)) {
      if (I->comesBefore(Top))
        Top = I;
      else if (Bottom->comesBefore(I))
        Bottom = I;
    }
  }
  bool empty() const {
    assert(((Top == nullptr && Bottom == nullptr) ||
            (Top != nullptr && Bottom != nullptr)) &&
           "Either none or both should be null");
    return Top == nullptr;
  }
  bool contains(T *I) const {
    if (empty())
      return false;
    return (Top == I || Top->comesBefore(I)) &&
           (I == Bottom || I->comesBefore(Bottom));
  }
  T *top() const { return Top; }
  T *bottom() const { return Bottom; }

  using iterator = IntervalIterator<T, Interval>;
  iterator begin() { return iterator(Top, *this); }
  iterator end() {
    return iterator(Bottom != nullptr ? Bottom->getNextNode() : nullptr, *this);
  }
  iterator begin() const {
    return iterator(Top, const_cast<Interval &>(*this));
  }
  iterator end() const {
    return iterator(Bottom != nullptr ? Bottom->getNextNode() : nullptr,
                    const_cast<Interval &>(*this));
  }
  /// Equality.
  bool operator==(const Interval &Other) const {
    return Top == Other.Top && Bottom == Other.Bottom;
  }
  /// Inequality.
  bool operator!=(const Interval &Other) const { return !(*this == Other); }
  /// \Returns true if this interval comes before \p Other in program order.
  /// This expects disjoint intervals.
  bool comesBefore(const Interval &Other) const {
    assert(disjoint(Other) && "Expect disjoint intervals!");
    return bottom()->comesBefore(Other.top());
  }
  /// \Returns true if this and \p Other have nothing in common.
  bool disjoint(const Interval &Other) const;
  /// \Returns the intersection between this and \p Other.
  // Example:
  // |----|   this
  //    |---| Other
  //    |-|   this->getIntersection(Other)
  Interval intersection(const Interval &Other) const {
    if (empty())
      return *this;
    if (Other.empty())
      return Interval();
    // 1. No overlap
    // A---B      this
    //       C--D Other
    if (Bottom->comesBefore(Other.Top) || Other.Bottom->comesBefore(Top))
      return Interval();
    // 2. Overlap.
    // A---B   this
    //   C--D  Other
    auto NewTopI = Top->comesBefore(Other.Top) ? Other.Top : Top;
    auto NewBottomI = Bottom->comesBefore(Other.Bottom) ? Bottom : Other.Bottom;
    return Interval(NewTopI, NewBottomI);
  }
  /// Difference operation. This returns up to two intervals.
  // Example:
  // |--------| this
  //    |-|     Other
  // |-|   |--| this - Other
  SmallVector<Interval, 2> operator-(const Interval &Other) {
    if (disjoint(Other))
      return {*this};
    if (Other.empty())
      return {*this};
    if (*this == Other)
      return {Interval()};
    Interval Intersection = intersection(Other);
    SmallVector<Interval, 2> Result;
    // Part 1, skip if empty.
    if (Top != Intersection.Top)
      Result.emplace_back(Top, Intersection.Top->getPrevNode());
    // Part 2, skip if empty.
    if (Intersection.Bottom != Bottom)
      Result.emplace_back(Intersection.Bottom->getNextNode(), Bottom);
    return Result;
  }
  /// \Returns the interval difference `this - Other`. This will crash in Debug
  /// if the result is not a single interval.
  Interval getSingleDiff(const Interval &Other) {
    auto Diff = *this - Other;
    assert(Diff.size() == 1 && "Expected a single interval!");
    return Diff[0];
  }
  /// \Returns a single interval that spans across both this and \p Other.
  // For example:
  // |---|        this
  //        |---| Other
  // |----------| this->getUnionInterval(Other)
  Interval getUnionInterval(const Interval &Other) {
    if (empty())
      return Other;
    if (Other.empty())
      return *this;
    auto *NewTop = Top->comesBefore(Other.Top) ? Top : Other.Top;
    auto *NewBottom = Bottom->comesBefore(Other.Bottom) ? Other.Bottom : Bottom;
    return {NewTop, NewBottom};
  }

  /// Update the interval when \p I is about to be moved before \p Before.
  // SFINAE disables this for non-Instructions.
  template <typename HelperT = T>
  std::enable_if_t<std::is_same<HelperT, Instruction>::value, void>
  notifyMoveInstr(HelperT *I, decltype(I->getIterator()) BeforeIt) {
    assert(contains(I) && "Expect `I` in interval!");
    assert(I->getIterator() != BeforeIt && "Can't move `I` before itself!");

    // Nothing to do if the instruction won't move.
    if (std::next(I->getIterator()) == BeforeIt)
      return;

    T *NewTop = Top->getIterator() == BeforeIt ? I
                : I == Top                     ? Top->getNextNode()
                                               : Top;
    T *NewBottom = std::next(Bottom->getIterator()) == BeforeIt ? I
                   : I == Bottom ? Bottom->getPrevNode()
                                 : Bottom;
    Top = NewTop;
    Bottom = NewBottom;
  }

#ifndef NDEBUG
  void print(raw_ostream &OS) const;
  LLVM_DUMP_METHOD void dump() const;
#endif
};

} // namespace llvm::sandboxir

#endif // LLVM_TRANSFORMS_VECTORIZE_SANDBOXVECTORIZER_INSTRINTERVAL_H
