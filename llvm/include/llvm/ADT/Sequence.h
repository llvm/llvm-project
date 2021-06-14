//===- Sequence.h - Utility for producing sequences of values ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
/// This routine provides some synthesis utilities to produce sequences of
/// values. The names are intentionally kept very short as they tend to occur
/// in common and widely used contexts.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_ADT_SEQUENCE_H
#define LLVM_ADT_SEQUENCE_H

#include <cstddef>  //std::ptrdiff_t
#include <iterator> //std::random_access_iterator_tag

namespace llvm {

namespace detail {

template <typename T, bool IsReversed> struct iota_range_iterator {
  using iterator_category = std::random_access_iterator_tag;
  using value_type = T;
  using difference_type = std::ptrdiff_t;
  using pointer = T *;
  using reference = T &;

private:
  struct Forward {
    static void increment(T &V) { ++V; }
    static void decrement(T &V) { --V; }
    static void offset(T &V, difference_type Offset) { V += Offset; }
    static T add(const T &V, difference_type Offset) { return V + Offset; }
    static difference_type difference(const T &A, const T &B) { return A - B; }
  };

  struct Reverse {
    static void increment(T &V) { --V; }
    static void decrement(T &V) { ++V; }
    static void offset(T &V, difference_type Offset) { V -= Offset; }
    static T add(const T &V, difference_type Offset) { return V - Offset; }
    static difference_type difference(const T &A, const T &B) { return B - A; }
  };

  using Op = std::conditional_t<!IsReversed, Forward, Reverse>;

public:
  // default-constructible
  iota_range_iterator() = default;
  // copy-constructible
  iota_range_iterator(const iota_range_iterator &) = default;
  // value constructor
  explicit iota_range_iterator(T Value) : Value(Value) {}
  // copy-assignable
  iota_range_iterator &operator=(const iota_range_iterator &) = default;
  // destructible
  ~iota_range_iterator() = default;

  // Can be compared for equivalence using the equality/inequality operators,
  bool operator!=(const iota_range_iterator &RHS) const {
    return Value != RHS.Value;
  }
  bool operator==(const iota_range_iterator &RHS) const {
    return Value == RHS.Value;
  }

  // Comparison
  bool operator<(const iota_range_iterator &Other) const {
    return Op::difference(Value, Other.Value) < 0;
  }
  bool operator<=(const iota_range_iterator &Other) const {
    return Op::difference(Value, Other.Value) <= 0;
  }
  bool operator>(const iota_range_iterator &Other) const {
    return Op::difference(Value, Other.Value) > 0;
  }
  bool operator>=(const iota_range_iterator &Other) const {
    return Op::difference(Value, Other.Value) >= 0;
  }

  // Dereference
  T operator*() const { return Value; }
  T operator[](difference_type Offset) const { return Op::add(Value, Offset); }

  // Arithmetic
  iota_range_iterator operator+(difference_type Offset) const {
    return {Op::add(Value, Offset)};
  }
  iota_range_iterator operator-(difference_type Offset) const {
    return {Op::add(Value, -Offset)};
  }

  // Iterator difference
  difference_type operator-(const iota_range_iterator &Other) const {
    return Op::difference(Value, Other.Value);
  }

  // Pre/Post Increment
  iota_range_iterator &operator++() {
    Op::increment(Value);
    return *this;
  }
  iota_range_iterator operator++(int) {
    iota_range_iterator Tmp = *this;
    Op::increment(Value);
    return Tmp;
  }

  // Pre/Post Decrement
  iota_range_iterator &operator--() {
    Op::decrement(Value);
    return *this;
  }
  iota_range_iterator operator--(int) {
    iota_range_iterator Tmp = *this;
    Op::decrement(Value);
    return Tmp;
  }

  // Compound assignment operators
  iota_range_iterator &operator+=(difference_type Offset) {
    Op::offset(Value, Offset);
    return *this;
  }
  iota_range_iterator &operator-=(difference_type Offset) {
    Op::offset(Value, -Offset);
    return *this;
  }

private:
  T Value;
};

} // namespace detail

template <typename ValueT> struct iota_range {
  static_assert(std::is_integral<ValueT>::value,
                "ValueT must be an integral type");

  using value_type = ValueT;
  using reference = ValueT &;
  using const_reference = const ValueT &;
  using iterator = detail::iota_range_iterator<value_type, false>;
  using const_iterator = iterator;
  using reverse_iterator = detail::iota_range_iterator<value_type, true>;
  using const_reverse_iterator = reverse_iterator;
  using difference_type = std::ptrdiff_t;
  using size_type = std::size_t;

  value_type Begin;
  value_type End;

  explicit iota_range(ValueT Begin, ValueT End) : Begin(Begin), End(End) {}

  size_t size() const { return End - Begin; }
  bool empty() const { return Begin == End; }

  auto begin() const { return const_iterator(Begin); }
  auto end() const { return const_iterator(End); }

  auto rbegin() const { return const_reverse_iterator(End - 1); }
  auto rend() const { return const_reverse_iterator(Begin - 1); }

private:
  static_assert(std::is_same<ValueT, std::remove_cv_t<ValueT>>::value,
                "ValueT must not be const nor volatile");
};

template <typename ValueT> auto seq(ValueT Begin, ValueT End) {
  return iota_range<ValueT>(Begin, End);
}

} // end namespace llvm

#endif // LLVM_ADT_SEQUENCE_H
