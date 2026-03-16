//===--------------------- CooperativeFuture.h ------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines a "cooperative" promise/future implementation.
//
// CooperativeFuture::get() runs tasks from a CooperativeFutureWorkQueue until
// the corresponding result is ready. This allows clients to block on a *result*
// without blocking the underlying thread (which will continue to perform work).
//
// Cooperative futures are intended for ORC runtime API clients who want to use
// blocking patterns with a single thread or fixed-sized thread pool.
//
// The CooperativeFuture and CooperativePromise APIs are deliberately similar to
// std::future and std::promise, but there are some differences. In particular:
//   1. Constructing a CooperativePromise requires a
//      CooperativeFutureTaskRunner.
//   2. When building with exceptions turned off, CooperativePromise and
//      CooperativeFuture can only be instantiated with types constructible
//      from Error. This is because CooperativeFutureTaskRunner may return an
//      Error, and CooperativeFuture::get needs a way to return in this case.
//      When exceptions are enabled, CooperativeFuture::get will throw any
//      Error received as an Exception (making its behavior more like
//      std::future).
//
// Blocking patterns should never be used inside the ORC runtime itself.
//
//===----------------------------------------------------------------------===//

#ifndef ORC_RT_COOPERATIVEFUTURE_H
#define ORC_RT_COOPERATIVEFUTURE_H

#include "orc-rt/BitmaskEnum.h"
#include "orc-rt/Error.h"

#include <atomic>
#include <cassert>
#include <utility>

namespace orc_rt {

/// CooperativeFutureTaskRunner provides an interface for CooperativeFuture to
/// run tasks.
class CooperativeFutureTaskRunner {
public:
  CooperativeFutureTaskRunner() = default;
  CooperativeFutureTaskRunner(const CooperativeFutureTaskRunner &) = delete;
  CooperativeFutureTaskRunner(CooperativeFutureTaskRunner &&) = delete;
  CooperativeFutureTaskRunner &
  operator=(const CooperativeFutureTaskRunner &) = delete;
  CooperativeFutureTaskRunner &
  operator=(CooperativeFutureTaskRunner &&) = delete;

  virtual ~CooperativeFutureTaskRunner();

  // Run the next available task. Should return Error::success if the task was
  // run, or an Error if no further tasks can be run (since in this case the
  // future that requested the work will be left without a value, and we want
  // to report why).
  virtual Error runNextTask() = 0;
};

template <typename T> class CooperativeFuture;
template <typename T> class CooperativePromise;

namespace detail {

// Storage for the value being communicated from promise to future.
class CooperativeFutureStorageBase {
  template <typename T> friend class orc_rt::CooperativeFuture;
  template <typename T> friend class orc_rt::CooperativePromise;

public:
  enum State : int {
    HasValue = 1 << 0,
    PromiseAttached = 1 << 1,
    FutureAttached = 1 << 2,
    LargestValue = FutureAttached,
    ORC_RT_MARK_AS_BITMASK_ENUM(LargestValue)
  };

protected:
  CooperativeFutureStorageBase(CooperativeFutureTaskRunner &WQ) : WQ(WQ) {}

  CooperativeFutureStorageBase(const CooperativeFutureStorageBase &) = delete;
  CooperativeFutureStorageBase &
  operator=(const CooperativeFutureStorageBase &) = delete;
  CooperativeFutureStorageBase(CooperativeFutureStorageBase &&) = delete;
  CooperativeFutureStorageBase &
  operator=(CooperativeFutureStorageBase &&) = delete;

  bool hasValue() { return (S & State::HasValue) == State::HasValue; }

  void setHasValue() {
    assert(!hasValue());
    S.fetch_or(State::HasValue);
  }
  void setHasNoValue() { S.fetch_and(~State::HasValue); }
  Error workUntilHasValue() {
    while (!hasValue())
      if (auto Err = WQ.runNextTask())
        return Err;
    return Error::success();
  }

private:
  void attachPromise() {
    assert((S & State::PromiseAttached) != State::PromiseAttached);
    S |= State::PromiseAttached;
  }

  bool detachPromise() {
    return (S.fetch_and(~State::PromiseAttached) & State::FutureAttached) !=
           State::FutureAttached;
  }

  void attachFuture() {
    assert((S & State::FutureAttached) != State::FutureAttached);
    S |= State::FutureAttached;
  }

  bool detachFuture() {
    return (S.fetch_and(~State::FutureAttached) & State::PromiseAttached) !=
           State::PromiseAttached;
  }

  CooperativeFutureTaskRunner &WQ;
  std::atomic<int> S = 0;
};

template <typename T>
class CooperativeFutureStorage : public CooperativeFutureStorageBase {
public:
  CooperativeFutureStorage(CooperativeFutureTaskRunner &WQ)
      : CooperativeFutureStorageBase(WQ) {}
  ~CooperativeFutureStorage() {
    if (hasValue())
      Storage.Value.~T();
  }

  void setValue(T &&Val) {
    new (&Storage.Value) T(std::move(Val));
    setHasValue();
  }

  T getValue() {
    if (auto Err = workUntilHasValue()) {
#if ORC_RT_ENABLE_EXCEPTIONS
      Err.throwOnFailure();
#else
      return Err;
#endif // ORC_RT_ENABLE_EXCEPTIONS
    }
    setHasNoValue();
    return std::move(Storage.Value);
  }

private:
  union ValueStorage {
    ValueStorage() {}
    ~ValueStorage() {}
    T Value;
  } Storage;
};
}; // namespace detail

/// Cooperative future.
///
/// Similar to std::future, but performs work-stealing while waiting for
/// its result. This allows it to work in contexts where there are no
/// other worker threads running.
///
/// Note that if exceptions are disabled (ORC_RT_ENABLE_EXCEPTIONS=Off) then T
/// must be constructible from orc_rt::Error (so must be Error or Expected<U>).
/// This allows errors such as unfulfilled promises to be reported via Error
/// (since exceptions are not available). When exceptions are enabled
/// (ORC_RT_ENABLE_EXCEPTIONS=On) T may be any type.
template <typename T> class CooperativeFuture {
  template <typename U> friend class CooperativePromise;

public:
  CooperativeFuture() = default;
  CooperativeFuture(CooperativeFuture &&Other) : Storage(Other.Storage) {
    Other.Storage = nullptr;
  }
  CooperativeFuture &operator=(CooperativeFuture &&Other) {
    releaseStorage();
    Storage = Other.Storage;
    Other.Storage = nullptr;
    return *this;
  }
  ~CooperativeFuture() { releaseStorage(); }

  T get() {
    assert(Storage);
    return Storage->getValue();
  }

private:
  CooperativeFuture(detail::CooperativeFutureStorage<T> *Storage)
      : Storage(Storage) {
    Storage->attachFuture();
  }

  void releaseStorage() {
    if (Storage && Storage->detachFuture())
      delete Storage;
  }

  detail::CooperativeFutureStorage<T> *Storage = nullptr;
};

/// Cooperative promise.
///
/// Similar to std::promise, but performs work-stealing while waiting for
/// its result. This allows it to work in contexts where there are no
/// other worker threads running.
///
/// Note that if exceptions are disabled (ORC_RT_ENABLE_EXCEPTIONS=Off) then T
/// must be constructible from orc_rt::Error (so must be Error or Expected<U>).
/// This allows errors such as unfulfilled promises to be reported via Error
/// (since exceptions are not available). When exceptions are enabled
/// (ORC_RT_ENABLE_EXCEPTIONS=On) T may be any type.
template <typename T> class CooperativePromise {
public:
  CooperativePromise() = default;
  CooperativePromise(CooperativeFutureTaskRunner &WQ)
      : Storage(new detail::CooperativeFutureStorage<T>(WQ)) {
    Storage->attachPromise();
  }
  CooperativePromise(CooperativePromise &&Other) : Storage(Other.Storage) {
    Other.Storage = nullptr;
  }
  CooperativePromise &operator=(CooperativePromise &&Other) {
    releaseStorage();
    Storage = Other.Storage;
    Other.Storage = nullptr;
    return *this;
  }
  ~CooperativePromise() { releaseStorage(); }

  CooperativeFuture<T> get_future() { return CooperativeFuture<T>(Storage); }

  template <typename U = T>
  std::enable_if_t<!std::is_void_v<U>> set_value(T &&Value) {
    assert(Storage);
    Storage->setValue(std::move(Value));
  }

  template <typename U = T> std::enable_if_t<std::is_void_v<U>> set_value() {
    assert(Storage);
    Storage->setValue();
  }

private:
  void releaseStorage() {
    if (Storage && Storage->detachPromise())
      delete Storage;
  }

  detail::CooperativeFutureStorage<T> *Storage = nullptr;
};

} // namespace orc_rt

#endif // ORC_RT_COOPERATIVEFUTURE_H
