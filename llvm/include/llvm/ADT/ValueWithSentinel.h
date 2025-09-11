//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file defines the ValueWithSentinel class, which is a type akin to a
/// std::optional, but uses a sentinel rather than an additional "valid" flag.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_ADT_VALUEWITHSENTINEL_H
#define LLVM_ADT_VALUEWITHSENTINEL_H

#include <cassert>
#include <limits>
#include <utility>

namespace llvm {

template <typename T, T Sentinel> class ValueWithSentinel {
public:
  ValueWithSentinel() = default;

  ValueWithSentinel(T Value) : Value(std::move(Value)) {
    assert(Value != Sentinel && "Value is sentinel (use default constructor)");
  };

  ValueWithSentinel &operator=(T const &NewValue) {
    assert(NewValue != Sentinel && "Assigned to sentinel (use .clear())");
    Value = NewValue;
    return *this;
  }

  bool operator==(ValueWithSentinel const &Other) const {
    return Value == Other.Value;
  }

  bool operator!=(ValueWithSentinel const &Other) const {
    return !(*this == Other);
  }

  T &operator*() {
    assert(has_value() && "Invalid value");
    return Value;
  }
  const T &operator*() const {
    return const_cast<ValueWithSentinel &>(*this).operator*();
  }

  T *operator->() { return &operator*(); }
  T const *operator->() const { return &operator*(); }

  T &value() { return operator*(); }
  T const &value() const { return operator*(); }

  bool has_value() const { return Value != Sentinel; }
  explicit operator bool() const { return has_value(); }

  void clear() { Value = Sentinel; }

private:
  T Value{Sentinel};
};

template <typename T>
using ValueWithSentinelNumericMax =
    ValueWithSentinel<T, std::numeric_limits<T>::max()>;

} // namespace llvm

#endif
