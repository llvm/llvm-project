//===- Optional.h - Simple variant for passing optional values --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  This file provides Optional, a template class modeled in the spirit of
//  OCaml's 'opt' variant.  The idea is to strongly type whether or not
//  a value can be optional.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ADT_OPTIONAL_H
#define LLVM_ADT_OPTIONAL_H

#include "llvm/ADT/None.h"
#include "llvm/Support/AlignOf.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/type_traits.h"
#include <algorithm>
#include <cassert>
#include <new>
#include <utility>

namespace llvm {

class raw_ostream;

namespace optional_detail {
template <typename T> struct OptionalStorageBase {
  AlignedCharArrayUnion<T> storage;
  bool hasVal = false;

  OptionalStorageBase() = default;
  OptionalStorageBase(const T &y) : hasVal(true) { new (storage.buffer) T(y); }
  OptionalStorageBase(T &&y) : hasVal(true) {
    new (storage.buffer) T(std::move(y));
  }

  T *getPointer() {
    assert(hasVal);
    return reinterpret_cast<T *>(storage.buffer);
  }
  const T *getPointer() const {
    assert(hasVal);
    return reinterpret_cast<const T *>(storage.buffer);
  }
  OptionalStorageBase &operator=(T &&y) {
    hasVal = true;
    new (this->storage.buffer) T(std::move(y));
    return *this;
  }
  OptionalStorageBase &operator=(const T &y) {
    hasVal = true;
    new (this->storage.buffer) T(y);
    return *this;
  }
  void reset() { this->hasVal = false; }
};

/// Storage for any type.
template <typename T, bool = is_trivially_copyable<T>::value>
struct OptionalStorage : OptionalStorageBase<T> {
  OptionalStorage() = default;

  OptionalStorage(const T &y) : OptionalStorageBase<T>(y) {}
  OptionalStorage(const OptionalStorage &O) : OptionalStorageBase<T>() {
    this->hasVal = O.hasVal;
    if (this->hasVal)
      new (this->storage.buffer) T(*O.getPointer());
  }
  OptionalStorage(T &&y) : OptionalStorageBase<T>(std::move(y)) {}
  OptionalStorage(OptionalStorage &&O) : OptionalStorageBase<T>() {
    this->hasVal = O.hasVal;
    if (O.hasVal) {
      new (this->storage.buffer) T(std::move(*O.getPointer()));
    }
  }

  OptionalStorage &operator=(T &&y) {
    if (this->hasVal)
      *this->getPointer() = std::move(y);
    else {
      OptionalStorageBase<T>::operator=(std::move(y));
    }
    return *this;
  }
  OptionalStorage &operator=(OptionalStorage &&O) {
    if (!O.hasVal)
      reset();
    else {
      *this = std::move(*O.getPointer());
    }
    return *this;
  }

  // FIXME: these assignments (& the equivalent const T&/const Optional& ctors)
  // could be made more efficient by passing by value, possibly unifying them
  // with the rvalue versions above - but this could place a different set of
  // requirements (notably: the existence of a default ctor) when implemented
  // in that way. Careful SFINAE to avoid such pitfalls would be required.
  OptionalStorage &operator=(const T &y) {
    if (this->hasVal)
      *this->getPointer() = y;
    else {
      OptionalStorageBase<T>::operator=(y);
    }
    return *this;
  }
  OptionalStorage &operator=(const OptionalStorage &O) {
    if (!O.hasVal)
      reset();
    else
      *this = *O.getPointer();
    return *this;
  }

  ~OptionalStorage() { reset(); }

  void reset() {
    if (this->hasVal) {
      (*this->getPointer()).~T();
    }
    OptionalStorageBase<T>::reset();
  }
};

template <typename T> struct OptionalStorage<T, true> : OptionalStorageBase<T> {
  OptionalStorage() = default;
  OptionalStorage(const T &y) : OptionalStorageBase<T>(y) {}
  OptionalStorage(const OptionalStorage &O) = default;
  OptionalStorage(T &&y) : OptionalStorageBase<T>(std::move(y)) {}
  OptionalStorage(OptionalStorage &&O) = default;
  OptionalStorage &operator=(T &&y) {
    OptionalStorageBase<T>::operator=(std::move(y));
    return *this;
  }
  OptionalStorage &operator=(OptionalStorage &&O) = default;
  OptionalStorage &operator=(const T &y) {
    OptionalStorageBase<T>::operator=(y);
    return *this;
  }
  OptionalStorage &operator=(const OptionalStorage &O) = default;
  ~OptionalStorage() = default;
};

// Swift-only: The following partial template specialization was removed
// from LLVM because of a bug in GCC, but Swift is relying on it.
// (https://bugs.swift.org/browse/SR-9483)

#if !defined(__GNUC__) || defined(__clang__) // GCC up to GCC7 miscompiles this.
/// Storage for trivially copyable types only.
template <typename T> struct OptionalStorage<T, true> {
  AlignedCharArrayUnion<T> storage;
  bool hasVal = false;

  OptionalStorage() = default;

  OptionalStorage(const T &y) : hasVal(true) { new (storage.buffer) T(y); }
  OptionalStorage &operator=(const T &y) {
    *reinterpret_cast<T *>(storage.buffer) = y;
    hasVal = true;
    return *this;
  }

  void reset() { hasVal = false; }
};
#endif
} // namespace optional_detail

template <typename T> class Optional {
  optional_detail::OptionalStorage<T> Storage;

public:
  using value_type = T;

  constexpr Optional() {}
  constexpr Optional(NoneType) {}

  Optional(const T &y) : Storage(y) {}
  Optional(const Optional &O) = default;

  Optional(T &&y) : Storage(std::forward<T>(y)) {}
  Optional(Optional &&O) = default;

  Optional &operator=(T &&y) {
    Storage = std::move(y);
    return *this;
  }
  Optional &operator=(Optional &&O) = default;

  /// Create a new object by constructing it in place with the given arguments.
  template <typename... ArgTypes> void emplace(ArgTypes &&... Args) {
    reset();
    Storage.hasVal = true;
    new (getPointer()) T(std::forward<ArgTypes>(Args)...);
  }

  static inline Optional create(const T *y) {
    return y ? Optional(*y) : Optional();
  }

  Optional &operator=(const T &y) {
    Storage = y;
    return *this;
  }
  Optional &operator=(const Optional &O) = default;

  void reset() { Storage.reset(); }

  const T *getPointer() const {
    assert(Storage.hasVal);
    return reinterpret_cast<const T *>(Storage.storage.buffer);
  }
  T *getPointer() {
    assert(Storage.hasVal);
    return reinterpret_cast<T *>(Storage.storage.buffer);
  }
  const T &getValue() const LLVM_LVALUE_FUNCTION { return *getPointer(); }
  T &getValue() LLVM_LVALUE_FUNCTION { return *getPointer(); }

  explicit operator bool() const { return Storage.hasVal; }
  bool hasValue() const { return Storage.hasVal; }
  const T *operator->() const { return getPointer(); }
  T *operator->() { return getPointer(); }
  const T &operator*() const LLVM_LVALUE_FUNCTION { return *getPointer(); }
  T &operator*() LLVM_LVALUE_FUNCTION { return *getPointer(); }

  template <typename U>
  constexpr T getValueOr(U &&value) const LLVM_LVALUE_FUNCTION {
    return hasValue() ? getValue() : std::forward<U>(value);
  }

#if LLVM_HAS_RVALUE_REFERENCE_THIS
  T &&getValue() && { return std::move(*getPointer()); }
  T &&operator*() && { return std::move(*getPointer()); }

  template <typename U>
  T getValueOr(U &&value) && {
    return hasValue() ? std::move(getValue()) : std::forward<U>(value);
  }
#endif
};

template <typename T, typename U>
bool operator==(const Optional<T> &X, const Optional<U> &Y) {
  if (X && Y)
    return *X == *Y;
  return X.hasValue() == Y.hasValue();
}

template <typename T, typename U>
bool operator!=(const Optional<T> &X, const Optional<U> &Y) {
  return !(X == Y);
}

template <typename T, typename U>
bool operator<(const Optional<T> &X, const Optional<U> &Y) {
  if (X && Y)
    return *X < *Y;
  return X.hasValue() < Y.hasValue();
}

template <typename T, typename U>
bool operator<=(const Optional<T> &X, const Optional<U> &Y) {
  return !(Y < X);
}

template <typename T, typename U>
bool operator>(const Optional<T> &X, const Optional<U> &Y) {
  return Y < X;
}

template <typename T, typename U>
bool operator>=(const Optional<T> &X, const Optional<U> &Y) {
  return !(X < Y);
}

template<typename T>
bool operator==(const Optional<T> &X, NoneType) {
  return !X;
}

template<typename T>
bool operator==(NoneType, const Optional<T> &X) {
  return X == None;
}

template<typename T>
bool operator!=(const Optional<T> &X, NoneType) {
  return !(X == None);
}

template<typename T>
bool operator!=(NoneType, const Optional<T> &X) {
  return X != None;
}

template <typename T> bool operator<(const Optional<T> &X, NoneType) {
  return false;
}

template <typename T> bool operator<(NoneType, const Optional<T> &X) {
  return X.hasValue();
}

template <typename T> bool operator<=(const Optional<T> &X, NoneType) {
  return !(None < X);
}

template <typename T> bool operator<=(NoneType, const Optional<T> &X) {
  return !(X < None);
}

template <typename T> bool operator>(const Optional<T> &X, NoneType) {
  return None < X;
}

template <typename T> bool operator>(NoneType, const Optional<T> &X) {
  return X < None;
}

template <typename T> bool operator>=(const Optional<T> &X, NoneType) {
  return None <= X;
}

template <typename T> bool operator>=(NoneType, const Optional<T> &X) {
  return X <= None;
}

template <typename T> bool operator==(const Optional<T> &X, const T &Y) {
  return X && *X == Y;
}

template <typename T> bool operator==(const T &X, const Optional<T> &Y) {
  return Y && X == *Y;
}

template <typename T> bool operator!=(const Optional<T> &X, const T &Y) {
  return !(X == Y);
}

template <typename T> bool operator!=(const T &X, const Optional<T> &Y) {
  return !(X == Y);
}

template <typename T> bool operator<(const Optional<T> &X, const T &Y) {
  return !X || *X < Y;
}

template <typename T> bool operator<(const T &X, const Optional<T> &Y) {
  return Y && X < *Y;
}

template <typename T> bool operator<=(const Optional<T> &X, const T &Y) {
  return !(Y < X);
}

template <typename T> bool operator<=(const T &X, const Optional<T> &Y) {
  return !(Y < X);
}

template <typename T> bool operator>(const Optional<T> &X, const T &Y) {
  return Y < X;
}

template <typename T> bool operator>(const T &X, const Optional<T> &Y) {
  return Y < X;
}

template <typename T> bool operator>=(const Optional<T> &X, const T &Y) {
  return !(X < Y);
}

template <typename T> bool operator>=(const T &X, const Optional<T> &Y) {
  return !(X < Y);
}

raw_ostream &operator<<(raw_ostream &OS, NoneType);

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
