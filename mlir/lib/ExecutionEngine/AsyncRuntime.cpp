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

#include "llvm/ADT/StringMap.h"
#include "llvm/Support/ThreadPool.h"

using namespace mlir::runtime;

//===----------------------------------------------------------------------===//
// Async runtime API.
//===----------------------------------------------------------------------===//

namespace mlir {
namespace runtime {
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
    threadPool.wait(); // wait for the completion of all async tasks
    assert(getNumRefCountedObjects() == 0 &&
           "all ref counted objects must be destroyed");
  }

  int32_t getNumRefCountedObjects() {
    return numRefCountedObjects.load(std::memory_order_relaxed);
  }

  llvm::ThreadPool &getThreadPool() { return threadPool; }

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
  llvm::ThreadPool threadPool;
};

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

// Returns the default per-process instance of an async runtime.
static std::unique_ptr<AsyncRuntime> &getDefaultAsyncRuntimeInstance() {
  static auto runtime = std::make_unique<AsyncRuntime>();
  return runtime;
}

static void resetDefaultAsyncRuntime() {
  return getDefaultAsyncRuntimeInstance().reset();
}

static AsyncRuntime *getDefaultAsyncRuntime() {
  return getDefaultAsyncRuntimeInstance().get();
}

// Async token provides a mechanism to signal asynchronous operation completion.
struct AsyncToken : public RefCounted {
  // AsyncToken created with a reference count of 2 because it will be returned
  // to the `async.execute` caller and also will be later on emplaced by the
  // asynchronously executed task. If the caller immediately will drop its
  // reference we must ensure that the token will be alive until the
  // asynchronous operation is completed.
  AsyncToken(AsyncRuntime *runtime)
      : RefCounted(runtime, /*count=*/2), ready(false) {}

  std::atomic<bool> ready;

  // Pending awaiters are guarded by a mutex.
  std::mutex mu;
  std::condition_variable cv;
  std::vector<std::function<void()>> awaiters;
};

// Async value provides a mechanism to access the result of asynchronous
// operations. It owns the storage that is used to store/load the value of the
// underlying type, and a flag to signal if the value is ready or not.
struct AsyncValue : public RefCounted {
  // AsyncValue similar to an AsyncToken created with a reference count of 2.
  AsyncValue(AsyncRuntime *runtime, int32_t size)
      : RefCounted(runtime, /*count=*/2), ready(false), storage(size) {}

  std::atomic<bool> ready;

  // Use vector of bytes to store async value payload.
  std::vector<int8_t> storage;

  // Pending awaiters are guarded by a mutex.
  std::mutex mu;
  std::condition_variable cv;
  std::vector<std::function<void()>> awaiters;
};

// Async group provides a mechanism to group together multiple async tokens or
// values to await on all of them together (wait for the completion of all
// tokens or values added to the group).
struct AsyncGroup : public RefCounted {
  AsyncGroup(AsyncRuntime *runtime)
      : RefCounted(runtime), pendingTokens(0), rank(0) {}

  std::atomic<int> pendingTokens;
  std::atomic<int> rank;

  // Pending awaiters are guarded by a mutex.
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

// Creates a new `async.token` in not-ready state.
extern "C" AsyncToken *mlirAsyncRuntimeCreateToken() {
  AsyncToken *token = new AsyncToken(getDefaultAsyncRuntime());
  return token;
}

// Creates a new `async.value` in not-ready state.
extern "C" AsyncValue *mlirAsyncRuntimeCreateValue(int32_t size) {
  AsyncValue *value = new AsyncValue(getDefaultAsyncRuntime(), size);
  return value;
}

// Create a new `async.group` in empty state.
extern "C" AsyncGroup *mlirAsyncRuntimeCreateGroup() {
  AsyncGroup *group = new AsyncGroup(getDefaultAsyncRuntime());
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

// Switches `async.value` to ready state and runs all awaiters.
extern "C" void mlirAsyncRuntimeEmplaceValue(AsyncValue *value) {
  // Make sure that `dropRef` does not destroy the mutex owned by the lock.
  {
    std::unique_lock<std::mutex> lock(value->mu);
    value->ready = true;
    value->cv.notify_all();
    for (auto &awaiter : value->awaiters)
      awaiter();
  }

  // Async values created with a ref count `2` to keep value alive until the
  // async task completes. Drop this reference explicitly when value emplaced.
  value->dropRef();
}

extern "C" void mlirAsyncRuntimeAwaitToken(AsyncToken *token) {
  std::unique_lock<std::mutex> lock(token->mu);
  if (!token->ready)
    token->cv.wait(lock, [token] { return token->ready.load(); });
}

extern "C" void mlirAsyncRuntimeAwaitValue(AsyncValue *value) {
  std::unique_lock<std::mutex> lock(value->mu);
  if (!value->ready)
    value->cv.wait(lock, [value] { return value->ready.load(); });
}

extern "C" void mlirAsyncRuntimeAwaitAllInGroup(AsyncGroup *group) {
  std::unique_lock<std::mutex> lock(group->mu);
  if (group->pendingTokens != 0)
    group->cv.wait(lock, [group] { return group->pendingTokens == 0; });
}

// Returns a pointer to the storage owned by the async value.
extern "C" ValueStorage mlirAsyncRuntimeGetValueStorage(AsyncValue *value) {
  return value->storage.data();
}

extern "C" void mlirAsyncRuntimeExecute(CoroHandle handle, CoroResume resume) {
  auto *runtime = getDefaultAsyncRuntime();
  runtime->getThreadPool().async([handle, resume]() { (*resume)(handle); });
}

extern "C" void mlirAsyncRuntimeAwaitTokenAndExecute(AsyncToken *token,
                                                     CoroHandle handle,
                                                     CoroResume resume) {
  auto execute = [handle, resume]() { (*resume)(handle); };
  std::unique_lock<std::mutex> lock(token->mu);
  if (token->ready) {
    lock.unlock();
    execute();
  } else {
    token->awaiters.push_back([execute]() { execute(); });
  }
}

extern "C" void mlirAsyncRuntimeAwaitValueAndExecute(AsyncValue *value,
                                                     CoroHandle handle,
                                                     CoroResume resume) {
  auto execute = [handle, resume]() { (*resume)(handle); };
  std::unique_lock<std::mutex> lock(value->mu);
  if (value->ready) {
    lock.unlock();
    execute();
  } else {
    value->awaiters.push_back([execute]() { execute(); });
  }
}

extern "C" void mlirAsyncRuntimeAwaitAllInGroupAndExecute(AsyncGroup *group,
                                                          CoroHandle handle,
                                                          CoroResume resume) {
  auto execute = [handle, resume]() { (*resume)(handle); };
  std::unique_lock<std::mutex> lock(group->mu);
  if (group->pendingTokens == 0) {
    lock.unlock();
    execute();
  } else {
    group->awaiters.push_back([execute]() { execute(); });
  }
}

//===----------------------------------------------------------------------===//
// Small async runtime support library for testing.
//===----------------------------------------------------------------------===//

extern "C" void mlirAsyncRuntimePrintCurrentThreadId() {
  static thread_local std::thread::id thisId = std::this_thread::get_id();
  std::cout << "Current thread id: " << thisId << std::endl;
}

//===----------------------------------------------------------------------===//
// MLIR Runner (JitRunner) dynamic library integration.
//===----------------------------------------------------------------------===//

// Export symbols for the MLIR runner integration. All other symbols are hidden.
#ifdef _WIN32
#define API __declspec(dllexport)
#else
#define API __attribute__((visibility("default")))
#endif

// Visual Studio had a bug that fails to compile nested generic lambdas
// inside an `extern "C"` function.
//   https://developercommunity.visualstudio.com/content/problem/475494/clexe-error-with-lambda-inside-function-templates.html
// The bug is fixed in VS2019 16.1. Separating the declaration and definition is
// a work around for older versions of Visual Studio.
extern "C" API void __mlir_runner_init(llvm::StringMap<void *> &exportSymbols);

void __mlir_runner_init(llvm::StringMap<void *> &exportSymbols) {
  auto exportSymbol = [&](llvm::StringRef name, auto ptr) {
    assert(exportSymbols.count(name) == 0 && "symbol already exists");
    exportSymbols[name] = reinterpret_cast<void *>(ptr);
  };

  exportSymbol("mlirAsyncRuntimeAddRef",
               &mlir::runtime::mlirAsyncRuntimeAddRef);
  exportSymbol("mlirAsyncRuntimeDropRef",
               &mlir::runtime::mlirAsyncRuntimeDropRef);
  exportSymbol("mlirAsyncRuntimeExecute",
               &mlir::runtime::mlirAsyncRuntimeExecute);
  exportSymbol("mlirAsyncRuntimeGetValueStorage",
               &mlir::runtime::mlirAsyncRuntimeGetValueStorage);
  exportSymbol("mlirAsyncRuntimeCreateToken",
               &mlir::runtime::mlirAsyncRuntimeCreateToken);
  exportSymbol("mlirAsyncRuntimeCreateValue",
               &mlir::runtime::mlirAsyncRuntimeCreateValue);
  exportSymbol("mlirAsyncRuntimeEmplaceToken",
               &mlir::runtime::mlirAsyncRuntimeEmplaceToken);
  exportSymbol("mlirAsyncRuntimeEmplaceValue",
               &mlir::runtime::mlirAsyncRuntimeEmplaceValue);
  exportSymbol("mlirAsyncRuntimeAwaitToken",
               &mlir::runtime::mlirAsyncRuntimeAwaitToken);
  exportSymbol("mlirAsyncRuntimeAwaitValue",
               &mlir::runtime::mlirAsyncRuntimeAwaitValue);
  exportSymbol("mlirAsyncRuntimeAwaitTokenAndExecute",
               &mlir::runtime::mlirAsyncRuntimeAwaitTokenAndExecute);
  exportSymbol("mlirAsyncRuntimeAwaitValueAndExecute",
               &mlir::runtime::mlirAsyncRuntimeAwaitValueAndExecute);
  exportSymbol("mlirAsyncRuntimeCreateGroup",
               &mlir::runtime::mlirAsyncRuntimeCreateGroup);
  exportSymbol("mlirAsyncRuntimeAddTokenToGroup",
               &mlir::runtime::mlirAsyncRuntimeAddTokenToGroup);
  exportSymbol("mlirAsyncRuntimeAwaitAllInGroup",
               &mlir::runtime::mlirAsyncRuntimeAwaitAllInGroup);
  exportSymbol("mlirAsyncRuntimeAwaitAllInGroupAndExecute",
               &mlir::runtime::mlirAsyncRuntimeAwaitAllInGroupAndExecute);
  exportSymbol("mlirAsyncRuntimePrintCurrentThreadId",
               &mlir::runtime::mlirAsyncRuntimePrintCurrentThreadId);
}

extern "C" API void __mlir_runner_destroy() { resetDefaultAsyncRuntime(); }

} // namespace runtime
} // namespace mlir

#endif // MLIR_ASYNCRUNTIME_DEFINE_FUNCTIONS
