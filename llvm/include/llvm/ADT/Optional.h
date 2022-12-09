//===- Optional.h - Simple variant for passing optional values --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
///  This file provides Optional, a template class modeled in the spirit of
///  OCaml's 'opt' variant.  The idea is to strongly type whether or not
///  a value can be optional.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_ADT_OPTIONAL_H
#define LLVM_ADT_OPTIONAL_H

#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/None.h"
#include "llvm/ADT/STLForwardCompat.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/type_traits.h"
#include <cassert>
#include <new>
#include <utility>

namespace llvm {

class raw_ostream;

namespace optional_detail {

/// Storage for any type.
//
// The specialization condition intentionally uses
// llvm::is_trivially_{copy/move}_constructible instead of
// std::is_trivially_{copy/move}_constructible. GCC versions prior to 7.4 may
// instantiate the copy/move constructor of `T` when
// std::is_trivially_{copy/move}_constructible is instantiated.  This causes
// compilation to fail if we query the trivially copy/move constructible
// property of a class which is not copy/move constructible.
//
// The current implementation of OptionalStorage insists that in order to use
// the trivial specialization, the value_type must be trivially copy
// constructible and trivially copy assignable due to =default implementations
// of the copy/move constructor/assignment.  It does not follow that this is
// necessarily the case std::is_trivially_copyable is true (hence the expanded
// specialization condition).
//
// The move constructible / assignable conditions emulate the remaining behavior
// of std::is_trivially_copyable.
template <typename T,
          bool = (llvm::is_trivially_copy_constructible<T>::value &&
                  std::is_trivially_copy_assignable<T>::value &&
                  (llvm::is_trivially_move_constructible<T>::value ||
                   !std::is_move_constructible<T>::value) &&
                  (std::is_trivially_move_assignable<T>::value ||
                   !std::is_move_assignable<T>::value))>
class OptionalStorage {
  union {
    char empty;
    T val;
  };
  bool hasVal = false;

public:
  ~OptionalStorage() { reset(); }

  constexpr OptionalStorage() noexcept : empty() {}

  constexpr OptionalStorage(OptionalStorage const &other) : OptionalStorage() {
    if (other.has_value()) {
      emplace(other.val);
    }
  }
  constexpr OptionalStorage(OptionalStorage &&other) : OptionalStorage() {
    if (other.has_value()) {
      emplace(std::move(other.val));
    }
  }

  template <class... Args>
  constexpr explicit OptionalStorage(std::in_place_t, Args &&...args)
      : val(std::forward<Args>(args)...), hasVal(true) {}

  void reset() noexcept {
    if (hasVal) {
      val.~T();
      hasVal = false;
    }
  }

  constexpr bool has_value() const noexcept { return hasVal; }

  T &value() &noexcept {
    assert(hasVal);
    return val;
  }
  constexpr T const &value() const &noexcept {
    assert(hasVal);
    return val;
  }
  T &&value() &&noexcept {
    assert(hasVal);
    return std::move(val);
  }

  template <class... Args> void emplace(Args &&...args) {
    reset();
    ::new ((void *)std::addressof(val)) T(std::forward<Args>(args)...);
    hasVal = true;
  }

  OptionalStorage &operator=(T const &y) {
    if (has_value()) {
      val = y;
    } else {
      ::new ((void *)std::addressof(val)) T(y);
      hasVal = true;
    }
    return *this;
  }
  OptionalStorage &operator=(T &&y) {
    if (has_value()) {
      val = std::move(y);
    } else {
      ::new ((void *)std::addressof(val)) T(std::move(y));
      hasVal = true;
    }
    return *this;
  }

  OptionalStorage &operator=(OptionalStorage const &other) {
    if (other.has_value()) {
      if (has_value()) {
        val = other.val;
      } else {
        ::new ((void *)std::addressof(val)) T(other.val);
        hasVal = true;
      }
    } else {
      reset();
    }
    return *this;
  }

  OptionalStorage &operator=(OptionalStorage &&other) {
    if (other.has_value()) {
      if (has_value()) {
        val = std::move(other.val);
      } else {
        ::new ((void *)std::addressof(val)) T(std::move(other.val));
        hasVal = true;
      }
    } else {
      reset();
    }
    return *this;
  }
};

template <typename T> class OptionalStorage<T, true> {
  union {
    char empty;
    T val;
  };
  bool hasVal = false;

public:
  ~OptionalStorage() = default;

  constexpr OptionalStorage() noexcept : empty{} {}

  constexpr OptionalStorage(OptionalStorage const &other) = default;
  constexpr OptionalStorage(OptionalStorage &&other) = default;

  OptionalStorage &operator=(OptionalStorage const &other) = default;
  OptionalStorage &operator=(OptionalStorage &&other) = default;

  template <class... Args>
  constexpr explicit OptionalStorage(std::in_place_t, Args &&...args)
      : val(std::forward<Args>(args)...), hasVal(true) {}

  void reset() noexcept {
    if (hasVal) {
      val.~T();
      hasVal = false;
    }
  }

  constexpr bool has_value() const noexcept { return hasVal; }

  T &value() &noexcept {
    assert(hasVal);
    return val;
  }
  constexpr T const &value() const &noexcept {
    assert(hasVal);
    return val;
  }
  T &&value() &&noexcept {
    assert(hasVal);
    return std::move(val);
  }

  template <class... Args> void emplace(Args &&...args) {
    reset();
    ::new ((void *)std::addressof(val)) T(std::forward<Args>(args)...);
    hasVal = true;
  }

  OptionalStorage &operator=(T const &y) {
    if (has_value()) {
      val = y;
    } else {
      ::new ((void *)std::addressof(val)) T(y);
      hasVal = true;
    }
    return *this;
  }
  OptionalStorage &operator=(T &&y) {
    if (has_value()) {
      val = std::move(y);
    } else {
      ::new ((void *)std::addressof(val)) T(std::move(y));
      hasVal = true;
    }
    return *this;
  }
};

} // namespace optional_detail

template <typename T> class Optional {
  optional_detail::OptionalStorage<T> Storage;

public:
  using value_type = T;

  constexpr Optional() = default;
  constexpr Optional(std::nullopt_t) {}

  constexpr Optional(const T &y) : Storage(std::in_place, y) {}
  constexpr Optional(const Optional &O) = default;

  constexpr Optional(T &&y) : Storage(std::in_place, std::move(y)) {}
  constexpr Optional(Optional &&O) = default;

  template <typename... ArgTypes>
  constexpr Optional(std::in_place_t, ArgTypes &&...Args)
      : Storage(std::in_place, std::forward<ArgTypes>(Args)...) {}

  Optional &operator=(T &&y) {
    Storage = std::move(y);
    return *this;
  }
  Optional &operator=(Optional &&O) = default;

  /// Create a new object by constructing it in place with the given arguments.
  template <typename... ArgTypes> void emplace(ArgTypes &&... Args) {
    Storage.emplace(std::forward<ArgTypes>(Args)...);
  }

  static constexpr Optional create(const T *y) {
    return y ? Optional(*y) : Optional();
  }

  Optional &operator=(const T &y) {
    Storage = y;
    return *this;
  }
  Optional &operator=(const Optional &O) = default;

  void reset() { Storage.reset(); }

  LLVM_DEPRECATED("Use &*X instead.", "&*X")
  constexpr const T *getPointer() const { return &Storage.value(); }
  LLVM_DEPRECATED("Use &*X instead.", "&*X")
  T *getPointer() { return &Storage.value(); }
  constexpr const T &value() const & { return Storage.value(); }
  T &value() & { return Storage.value(); }

  constexpr explicit operator bool() const { return has_value(); }
  constexpr bool has_value() const { return Storage.has_value(); }
  constexpr const T *operator->() const { return &Storage.value(); }
  T *operator->() { return &Storage.value(); }
  constexpr const T &operator*() const & { return value(); }
  T &operator*() & { return value(); }

  template <typename U> constexpr T value_or(U &&alt) const & {
    return has_value() ? value() : std::forward<U>(alt);
  }

  /// Apply a function to the value if present; otherwise return None.
  template <class Function>
  auto transform(const Function &F) const & -> Optional<decltype(F(value()))> {
    if (*this)
      return F(value());
    return None;
  }

  T &&value() && { return std::move(Storage.value()); }
  T &&operator*() && { return std::move(Storage.value()); }

  template <typename U> T value_or(U &&alt) && {
    return has_value() ? std::move(value()) : std::forward<U>(alt);
  }

  /// Apply a function to the value if present; otherwise return None.
  template <class Function>
  auto transform(
      const Function &F) && -> Optional<decltype(F(std::move(*this).value()))> {
    if (*this)
      return F(std::move(*this).value());
    return None;
  }
};

template<typename T>
Optional(const T&) -> Optional<T>;

template <class T> llvm::hash_code hash_value(const Optional<T> &O) {
  return O ? hash_combine(true, *O) : hash_value(false);
}

template <typename T, typename U>
constexpr bool operator==(const Optional<T> &X, const Optional<U> &Y) {
  if (X && Y)
    return *X == *Y;
  return X.has_value() == Y.has_value();
}

template <typename T, typename U>
constexpr bool operator!=(const Optional<T> &X, const Optional<U> &Y) {
  return !(X == Y);
}

template <typename T, typename U>
constexpr bool operator<(const Optional<T> &X, const Optional<U> &Y) {
  if (X && Y)
    return *X < *Y;
  return X.has_value() < Y.has_value();
}

template <typename T, typename U>
constexpr bool operator<=(const Optional<T> &X, const Optional<U> &Y) {
  return !(Y < X);
}

template <typename T, typename U>
constexpr bool operator>(const Optional<T> &X, const Optional<U> &Y) {
  return Y < X;
}

template <typename T, typename U>
constexpr bool operator>=(const Optional<T> &X, const Optional<U> &Y) {
  return !(X < Y);
}

template <typename T>
constexpr bool operator==(const Optional<T> &X, std::nullopt_t) {
  return !X;
}

template <typename T>
constexpr bool operator==(std::nullopt_t, const Optional<T> &X) {
  return X == None;
}

template <typename T>
constexpr bool operator!=(const Optional<T> &X, std::nullopt_t) {
  return !(X == None);
}

template <typename T>
constexpr bool operator!=(std::nullopt_t, const Optional<T> &X) {
  return X != None;
}

template <typename T>
constexpr bool operator<(const Optional<T> &, std::nullopt_t) {
  return false;
}

template <typename T>
constexpr bool operator<(std::nullopt_t, const Optional<T> &X) {
  return X.has_value();
}

template <typename T>
constexpr bool operator<=(const Optional<T> &X, std::nullopt_t) {
  return !(None < X);
}

template <typename T>
constexpr bool operator<=(std::nullopt_t, const Optional<T> &X) {
  return !(X < None);
}

template <typename T>
constexpr bool operator>(const Optional<T> &X, std::nullopt_t) {
  return None < X;
}

template <typename T>
constexpr bool operator>(std::nullopt_t, const Optional<T> &X) {
  return X < None;
}

template <typename T>
constexpr bool operator>=(const Optional<T> &X, std::nullopt_t) {
  return None <= X;
}

template <typename T>
constexpr bool operator>=(std::nullopt_t, const Optional<T> &X) {
  return X <= None;
}

template <typename T>
constexpr bool operator==(const Optional<T> &X, const T &Y) {
  return X && *X == Y;
}

template <typename T>
constexpr bool operator==(const T &X, const Optional<T> &Y) {
  return Y && X == *Y;
}

template <typename T>
constexpr bool operator!=(const Optional<T> &X, const T &Y) {
  return !(X == Y);
}

template <typename T>
constexpr bool operator!=(const T &X, const Optional<T> &Y) {
  return !(X == Y);
}

template <typename T>
constexpr bool operator<(const Optional<T> &X, const T &Y) {
  return !X || *X < Y;
}

template <typename T>
constexpr bool operator<(const T &X, const Optional<T> &Y) {
  return Y && X < *Y;
}

template <typename T>
constexpr bool operator<=(const Optional<T> &X, const T &Y) {
  return !(Y < X);
}

template <typename T>
constexpr bool operator<=(const T &X, const Optional<T> &Y) {
  return !(Y < X);
}

template <typename T>
constexpr bool operator>(const Optional<T> &X, const T &Y) {
  return Y < X;
}

template <typename T>
constexpr bool operator>(const T &X, const Optional<T> &Y) {
  return Y < X;
}

template <typename T>
constexpr bool operator>=(const Optional<T> &X, const T &Y) {
  return !(X < Y);
}

template <typename T>
constexpr bool operator>=(const T &X, const Optional<T> &Y) {
  return !(X < Y);
}

raw_ostream &operator<<(raw_ostream &OS, std::nullopt_t);

template <typename T, typename = decltype(std::declval<raw_ostream &>()
                                          << std::declval<const T &>())>
raw_ostream &operator<<(raw_ostream &OS, const Optional<T> &O) {
  if (O)
    OS << *O;
  else
    OS << None;
  return OS;
}

} // end namespace llvm

#endif // LLVM_ADT_OPTIONAL_H
