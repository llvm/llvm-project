//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file defines the ValueOrSentinel class, which is a type akin to a
/// std::optional, but uses a sentinel rather than an additional "valid" flag.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_ADT_VALUEORSENTINEL_H
#define LLVM_ADT_VALUEORSENTINEL_H

#include <cassert>
#include <limits>
#include <optional>
#include <utility>

namespace llvm {

namespace detail {
/// An adjustment allows changing how the value is stored. An example use case
/// is to use zero as a sentinel value.
template <typename T> struct DefaultValueAdjustment {
  constexpr static T toRepresentation(T Value) { return Value; }
  constexpr static T fromRepresentation(T Value) { return Value; }
};
} // namespace detail

template <typename T, T Sentinel,
          typename Adjust = detail::DefaultValueAdjustment<T>>
class ValueOrSentinel {
public:
  constexpr ValueOrSentinel() = default;
  constexpr ValueOrSentinel(std::nullopt_t) {};
  constexpr ValueOrSentinel(T Value) : Value(Adjust::toRepresentation(Value)) {
    assert(this->Value != Sentinel &&
           "Value is sentinel (use default constructor)");
  };

  constexpr ValueOrSentinel &operator=(T NewValue) {
    Value = Adjust::toRepresentation(NewValue);
    assert(Value != Sentinel && "NewValue is sentinel (use .clear())");
    return *this;
  }

  constexpr bool operator==(ValueOrSentinel Other) const {
    return Value == Other.Value;
  }
  constexpr bool operator!=(ValueOrSentinel Other) const {
    return !(*this == Other);
  }

  T value() const {
    assert(has_value() && ".value() called on sentinel");
    return Adjust::fromRepresentation(Value);
  }
  T operator*() const { return value(); }

  explicit operator T() const { return value(); }
  explicit constexpr operator bool() const { return has_value(); }

  constexpr void clear() { Value = Sentinel; }
  constexpr bool has_value() const { return Value != Sentinel; }

  constexpr static ValueOrSentinel fromInternalRepresentation(T Value) {
    return {std::nullopt, Value};
  }
  constexpr T toInternalRepresentation() const { return Value; }

protected:
  ValueOrSentinel(std::nullopt_t, T Value) : Value(Value) {};

  T Value{Sentinel};
};

template <typename T>
using ValueOrSentinelIntMax = ValueOrSentinel<T, std::numeric_limits<T>::max()>;

} // namespace llvm

#endif
