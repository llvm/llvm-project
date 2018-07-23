//===- Any.h - Generic type erased holder of any type -----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file provides Any, a non-template class modeled in the spirit of
//  std::any.  The idea is to provide a type-safe replacement for C's void*.
//  It can hold a value of any copy-constructible copy-assignable type
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ADT_ANY_H
#define LLVM_ADT_ANY_H

#include "llvm/ADT/STLExtras.h"

#include <cassert>
#include <memory>
#include <type_traits>

namespace llvm {

class Any {
  template <typename T> struct TypeId { static const char Id; };

  struct StorageBase {
    virtual ~StorageBase() = default;
    virtual std::unique_ptr<StorageBase> clone() const = 0;
    virtual const void *id() const = 0;
  };

  template <typename T> struct StorageImpl : public StorageBase {
    explicit StorageImpl(const T &Value) : Value(Value) {}

    explicit StorageImpl(T &&Value) : Value(std::move(Value)) {}

    std::unique_ptr<StorageBase> clone() const override {
      return llvm::make_unique<StorageImpl<T>>(Value);
    }

    const void *id() const override { return &TypeId<T>::Id; }

    T Value;

  private:
    StorageImpl &operator=(const StorageImpl &Other) = delete;
    StorageImpl(const StorageImpl &Other) = delete;
  };

public:
  Any() = default;

  Any(const Any &Other)
      : Storage(Other.Storage ? Other.Storage->clone() : nullptr) {}

  // When T is Any or T is not copy-constructible we need to explicitly disable
  // the forwarding constructor so that the copy constructor gets selected
  // instead.
  template <
      typename T,
      typename std::enable_if<
          llvm::conjunction<
              llvm::negation<std::is_same<typename std::decay<T>::type, Any>>,
              std::is_copy_constructible<typename std::decay<T>::type>>::value,
          int>::type = 0>
  Any(T &&Value) {
    using U = typename std::decay<T>::type;
    Storage = llvm::make_unique<StorageImpl<U>>(std::forward<T>(Value));
  }

  Any(Any &&Other) : Storage(std::move(Other.Storage)) {}

  Any &swap(Any &Other) {
    std::swap(Storage, Other.Storage);
    return *this;
  }

  Any &operator=(Any Other) {
    Storage = std::move(Other.Storage);
    return *this;
  }

  bool hasValue() const { return !!Storage; }

  void reset() { Storage.reset(); }

private:
  template <class T> friend T any_cast(const Any &Value);
  template <class T> friend T any_cast(Any &Value);
  template <class T> friend T any_cast(Any &&Value);
  template <class T> friend const T *any_cast(const Any *Value);
  template <class T> friend T *any_cast(Any *Value);
  template <typename T> friend bool any_isa(const Any &Value);

  std::unique_ptr<StorageBase> Storage;
};

template <typename T> const char Any::TypeId<T>::Id = 0;


template <typename T> bool any_isa(const Any &Value) {
  if (!Value.Storage)
    return false;
  using U =
      typename std::remove_cv<typename std::remove_reference<T>::type>::type;
  return Value.Storage->id() == &Any::TypeId<U>::Id;
}

template <class T> T any_cast(const Any &Value) {
  using U =
      typename std::remove_cv<typename std::remove_reference<T>::type>::type;
  return static_cast<T>(*any_cast<U>(&Value));
}

template <class T> T any_cast(Any &Value) {
  using U =
      typename std::remove_cv<typename std::remove_reference<T>::type>::type;
  return static_cast<T>(*any_cast<U>(&Value));
}

template <class T> T any_cast(Any &&Value) {
  using U =
      typename std::remove_cv<typename std::remove_reference<T>::type>::type;
  return static_cast<T>(std::move(*any_cast<U>(&Value)));
}

template <class T> const T *any_cast(const Any *Value) {
  using U =
      typename std::remove_cv<typename std::remove_reference<T>::type>::type;
  assert(Value && any_isa<T>(*Value) && "Bad any cast!");
  if (!Value || !any_isa<U>(*Value))
    return nullptr;
  return &static_cast<Any::StorageImpl<U> &>(*Value->Storage).Value;
}

template <class T> T *any_cast(Any *Value) {
  using U = typename std::decay<T>::type;
  assert(Value && any_isa<U>(*Value) && "Bad any cast!");
  if (!Value || !any_isa<U>(*Value))
    return nullptr;
  return &static_cast<Any::StorageImpl<U> &>(*Value->Storage).Value;
}

} // end namespace llvm

#endif // LLVM_ADT_ANY_H
