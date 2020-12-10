//===- AsyncRuntime.cpp - Async runtime reference implementation ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements basic Async runtime API for supporting Async dialect
// to LLVM dialect lowering.
//
//===----------------------------------------------------------------------===//

#include "mlir/ExecutionEngine/AsyncRuntime.h"

#ifdef MLIR_ASYNCRUNTIME_DEFINE_FUNCTIONS

#include <atomic>
#include <cassert>
#include <condition_variable>
#include <functional>
#include <iostream>
#include <mutex>
#include <thread>
#include <vector>

//===----------------------------------------------------------------------===//
// Async runtime API.
//===----------------------------------------------------------------------===//

namespace {

// Forward declare class defined below.
class RefCounted;

// -------------------------------------------------------------------------- //
// AsyncRuntime orchestrates all async operations and Async runtime API is built
// on top of the default runtime instance.
// -------------------------------------------------------------------------- //

class AsyncRuntime {
public:
  AsyncRuntime() : numRefCountedObjects(0) {}

  ~AsyncRuntime() {
    assert(getNumRefCountedObjects() == 0 &&
           "all ref counted objects must be destroyed");
  }

  int32_t getNumRefCountedObjects() {
    return numRefCountedObjects.load(std::memory_order_relaxed);
  }

private:
  friend class RefCounted;

  // Count the total number of reference counted objects in this instance
  // of an AsyncRuntime. For debugging purposes only.
  void addNumRefCountedObjects() {
    numRefCountedObjects.fetch_add(1, std::memory_order_relaxed);
  }
  void dropNumRefCountedObjects() {
    numRefCountedObjects.fetch_sub(1, std::memory_order_relaxed);
  }

  std::atomic<int32_t> numRefCountedObjects;
};

// Returns the default per-process instance of an async runtime.
AsyncRuntime *getDefaultAsyncRuntimeInstance() {
  static auto runtime = std::make_unique<AsyncRuntime>();
  return runtime.get();
}

// -------------------------------------------------------------------------- //
// A base class for all reference counted objects created by the async runtime.
// -------------------------------------------------------------------------- //

class RefCounted {
public:
  RefCounted(AsyncRuntime *runtime, int32_t refCount = 1)
      : runtime(runtime), refCount(refCount) {
    runtime->addNumRefCountedObjects();
  }

  virtual ~RefCounted() {
    assert(refCount.load() == 0 && "reference count must be zero");
    runtime->dropNumRefCountedObjects();
  }

  RefCounted(const RefCounted &) = delete;
  RefCounted &operator=(const RefCounted &) = delete;

  void addRef(int32_t count = 1) { refCount.fetch_add(count); }

  void dropRef(int32_t count = 1) {
    int32_t previous = refCount.fetch_sub(count);
    assert(previous >= count && "reference count should not go below zero");
    if (previous == count)
      destroy();
  }

protected:
  virtual void destroy() { delete this; }

private:
  AsyncRuntime *runtime;
  std::atomic<int32_t> refCount;
};

} // namespace

struct AsyncToken : public RefCounted {
  // AsyncToken created with a reference count of 2 because it will be returned
  // to the `async.execute` caller and also will be later on emplaced by the
  // asynchronously executed task. If the caller immediately will drop its
  // reference we must ensure that the token will be alive until the
  // asynchronous operation is completed.
  AsyncToken(AsyncRuntime *runtime) : RefCounted(runtime, /*count=*/2) {}

  // Internal state below guarded by a mutex.
  std::mutex mu;
  std::condition_variable cv;

  bool ready = false;
  std::vector<std::function<void()>> awaiters;
};

struct AsyncGroup : public RefCounted {
  AsyncGroup(AsyncRuntime *runtime)
      : RefCounted(runtime), pendingTokens(0), rank(0) {}

  std::atomic<int> pendingTokens;
  std::atomic<int> rank;

  // Internal state below guarded by a mutex.
  std::mutex mu;
  std::condition_variable cv;

  std::vector<std::function<void()>> awaiters;
};

// Adds references to reference counted runtime object.
extern "C" void mlirAsyncRuntimeAddRef(RefCountedObjPtr ptr, int32_t count) {
  RefCounted *refCounted = static_cast<RefCounted *>(ptr);
  refCounted->addRef(count);
}

// Drops references from reference counted runtime object.
extern "C" void mlirAsyncRuntimeDropRef(RefCountedObjPtr ptr, int32_t count) {
  RefCounted *refCounted = static_cast<RefCounted *>(ptr);
  refCounted->dropRef(count);
}

// Create a new `async.token` in not-ready state.
extern "C" AsyncToken *mlirAsyncRuntimeCreateToken() {
  AsyncToken *token = new AsyncToken(getDefaultAsyncRuntimeInstance());
  return token;
}

// Create a new `async.group` in empty state.
extern "C" AsyncGroup *mlirAsyncRuntimeCreateGroup() {
  AsyncGroup *group = new AsyncGroup(getDefaultAsyncRuntimeInstance());
  return group;
}

extern "C" int64_t mlirAsyncRuntimeAddTokenToGroup(AsyncToken *token,
                                                   AsyncGroup *group) {
  std::unique_lock<std::mutex> lockToken(token->mu);
  std::unique_lock<std::mutex> lockGroup(group->mu);

  // Get the rank of the token inside the group before we drop the reference.
  int rank = group->rank.fetch_add(1);
  group->pendingTokens.fetch_add(1);

  auto onTokenReady = [group]() {
    // Run all group awaiters if it was the last token in the group.
    if (group->pendingTokens.fetch_sub(1) == 1) {
      group->cv.notify_all();
      for (auto &awaiter : group->awaiters)
        awaiter();
    }
  };

  if (token->ready) {
    // Update group pending tokens immediately and maybe run awaiters.
    onTokenReady();

  } else {
    // Update group pending tokens when token will become ready. Because this
    // will happen asynchronously we must ensure that `group` is alive until
    // then, and re-ackquire the lock.
    group->addRef();

    token->awaiters.push_back([group, onTokenReady]() {
      // Make sure that `dropRef` does not destroy the mutex owned by the lock.
      {
        std::unique_lock<std::mutex> lockGroup(group->mu);
        onTokenReady();
      }
      group->dropRef();
    });
  }

  return rank;
}

// Switches `async.token` to ready state and runs all awaiters.
extern "C" void mlirAsyncRuntimeEmplaceToken(AsyncToken *token) {
  // Make sure that `dropRef` does not destroy the mutex owned by the lock.
  {
    std::unique_lock<std::mutex> lock(token->mu);
    token->ready = true;
    token->cv.notify_all();
    for (auto &awaiter : token->awaiters)
      awaiter();
  }

  // Async tokens created with a ref count `2` to keep token alive until the
  // async task completes. Drop this reference explicitly when token emplaced.
  token->dropRef();
}

extern "C" void mlirAsyncRuntimeAwaitToken(AsyncToken *token) {
  std::unique_lock<std::mutex> lock(token->mu);
  if (!token->ready)
    token->cv.wait(lock, [token] { return token->ready; });
}

extern "C" void mlirAsyncRuntimeAwaitAllInGroup(AsyncGroup *group) {
  std::unique_lock<std::mutex> lock(group->mu);
  if (group->pendingTokens != 0)
    group->cv.wait(lock, [group] { return group->pendingTokens == 0; });
}

extern "C" void mlirAsyncRuntimeExecute(CoroHandle handle, CoroResume resume) {
  (*resume)(handle);
}

extern "C" void mlirAsyncRuntimeAwaitTokenAndExecute(AsyncToken *token,
                                                     CoroHandle handle,
                                                     CoroResume resume) {
  std::unique_lock<std::mutex> lock(token->mu);
  auto execute = [handle, resume]() { (*resume)(handle); };
  if (token->ready)
    execute();
  else
    token->awaiters.push_back([execute]() { execute(); });
}

extern "C" void mlirAsyncRuntimeAwaitAllInGroupAndExecute(AsyncGroup *group,
                                                          CoroHandle handle,
                                                          CoroResume resume) {
  std::unique_lock<std::mutex> lock(group->mu);
  auto execute = [handle, resume]() { (*resume)(handle); };
  if (group->pendingTokens == 0)
    execute();
  else
    group->awaiters.push_back([execute]() { execute(); });
}

//===----------------------------------------------------------------------===//
// Small async runtime support library for testing.
//===----------------------------------------------------------------------===//

extern "C" void mlirAsyncRuntimePrintCurrentThreadId() {
  static thread_local std::thread::id thisId = std::this_thread::get_id();
  std::cout << "Current thread id: " << thisId << std::endl;
}

#endif // MLIR_ASYNCRUNTIME_DEFINE_FUNCTIONS
