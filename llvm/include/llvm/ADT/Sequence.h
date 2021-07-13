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

#include <cassert>     // assert
#include <cstddef>     // std::ptrdiff_t
#include <iterator>    // std::random_access_iterator_tag
#include <limits>      // std::numeric_limits
#include <type_traits> // std::underlying_type, std::is_enum

namespace llvm {

namespace detail {

template <typename T, typename U, bool IsReversed> struct iota_range_iterator {
  using iterator_category = std::random_access_iterator_tag;
  using value_type = T;
  using difference_type = std::ptrdiff_t;
  using pointer = T *;
  using reference = T &;

  // default-constructible
  iota_range_iterator() = default;
  // copy-constructible
  iota_range_iterator(const iota_range_iterator &) = default;
  // value constructor
  explicit iota_range_iterator(U Value) : Value(Value) {}
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
  T operator*() const { return static_cast<T>(Value); }
  T operator[](difference_type Offset) const {
    return static_cast<T>(Op::add(Value, Offset));
  }

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
  struct Forward {
    static void increment(U &V) { ++V; }
    static void decrement(U &V) { --V; }
    static void offset(U &V, difference_type Offset) { V += Offset; }
    static U add(const U &V, difference_type Offset) { return V + Offset; }
    static difference_type difference(const U &A, const U &B) {
      return difference_type(A) - difference_type(B);
    }
  };

  struct Reverse {
    static void increment(U &V) { --V; }
    static void decrement(U &V) { ++V; }
    static void offset(U &V, difference_type Offset) { V -= Offset; }
    static U add(const U &V, difference_type Offset) { return V - Offset; }
    static difference_type difference(const U &A, const U &B) {
      return difference_type(B) - difference_type(A);
    }
  };

  using Op = std::conditional_t<!IsReversed, Forward, Reverse>;

  U Value;
};

// Providing std::type_identity for C++14.
template <class T> struct type_identity { using type = T; };

} // namespace detail

template <typename T> struct iota_range {
private:
  using underlying_type =
      typename std::conditional_t<std::is_enum<T>::value,
                                  std::underlying_type<T>,
                                  detail::type_identity<T>>::type;
  using numeric_type =
      typename std::conditional_t<std::is_signed<underlying_type>::value,
                                  intmax_t, uintmax_t>;

  static numeric_type compute_past_end(numeric_type End, bool Inclusive) {
    if (Inclusive) {
      // This assertion forbids overflow of `PastEndValue`.
      assert(End != std::numeric_limits<numeric_type>::max() &&
             "Forbidden End value for seq_inclusive.");
      return End + 1;
    }
    return End;
  }
  static numeric_type raw(T Value) { return static_cast<numeric_type>(Value); }

  numeric_type BeginValue;
  numeric_type PastEndValue;

public:
  using value_type = T;
  using reference = T &;
  using const_reference = const T &;
  using iterator = detail::iota_range_iterator<value_type, numeric_type, false>;
  using const_iterator = iterator;
  using reverse_iterator =
      detail::iota_range_iterator<value_type, numeric_type, true>;
  using const_reverse_iterator = reverse_iterator;
  using difference_type = std::ptrdiff_t;
  using size_type = std::size_t;

  explicit iota_range(T Begin, T End, bool Inclusive)
      : BeginValue(raw(Begin)),
        PastEndValue(compute_past_end(raw(End), Inclusive)) {
    assert(Begin <= End && "Begin must be less or equal to End.");
  }

  size_t size() const { return PastEndValue - BeginValue; }
  bool empty() const { return BeginValue == PastEndValue; }

  auto begin() const { return const_iterator(BeginValue); }
  auto end() const { return const_iterator(PastEndValue); }

  auto rbegin() const { return const_reverse_iterator(PastEndValue - 1); }
  auto rend() const {
    assert(std::is_unsigned<numeric_type>::value ||
           BeginValue != std::numeric_limits<numeric_type>::min() &&
               "Forbidden Begin value for reverse iteration");
    return const_reverse_iterator(BeginValue - 1);
  }

private:
  static_assert(std::is_integral<T>::value || std::is_enum<T>::value,
                "T must be an integral or enum type");
  static_assert(std::is_same<T, std::remove_cv_t<T>>::value,
                "T must not be const nor volatile");
  static_assert(std::is_integral<numeric_type>::value,
                "numeric_type must be an integral type");
};

/// Iterate over an integral/enum type from Begin up to - but not including -
/// End.
/// Note on enum iteration: `seq` will generate each consecutive value, even if
/// no enumerator with that value exists.
template <typename T> auto seq(T Begin, T End) {
  return iota_range<T>(Begin, End, false);
}

/// Iterate over an integral/enum type from Begin to End inclusive.
/// Note on enum iteration: `seq_inclusive` will generate each consecutive
/// value, even if no enumerator with that value exists.
/// To prevent overflow, `End` must be different from INTMAX_MAX if T is signed
/// (resp. UINTMAX_MAX if T is unsigned).
template <typename T> auto seq_inclusive(T Begin, T End) {
  return iota_range<T>(Begin, End, true);
}

} // end namespace llvm

#endif // LLVM_ADT_SEQUENCE_H
