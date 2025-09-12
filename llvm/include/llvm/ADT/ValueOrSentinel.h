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
#include <utility>

namespace llvm {

template <typename T, T Sentinel> class ValueOrSentinel {
public:
  ValueOrSentinel() = default;

  ValueOrSentinel(T Value) : Value(std::move(Value)) {
    assert(Value != Sentinel && "Value is sentinel (use default constructor)");
  };

  ValueOrSentinel &operator=(const T &NewValue) {
    assert(NewValue != Sentinel && "NewValue is sentinel (use .clear())");
    Value = NewValue;
    return *this;
  }

  bool operator==(const ValueOrSentinel &Other) const {
    return Value == Other.Value;
  }

  bool operator!=(const ValueOrSentinel &Other) const {
    return !(*this == Other);
  }

  T &value() {
    assert(has_value() && ".value() called on sentinel");
    return Value;
  }
  const T &value() const {
    return const_cast<ValueOrSentinel &>(*this).value();
  }

  T &operator*() { return value(); }
  const T &operator*() const { return value(); }

  bool has_value() const { return Value != Sentinel; }

  explicit operator bool() const { return has_value(); }
  explicit operator T() const { return value(); }

  void clear() { Value = Sentinel; }

private:
  T Value{Sentinel};
};

template <typename T>
using ValueOrSentinelIntMax = ValueOrSentinel<T, std::numeric_limits<T>::max()>;

} // namespace llvm

#endif
