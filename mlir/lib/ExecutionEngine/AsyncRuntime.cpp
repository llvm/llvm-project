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
#include <condition_variable>
#include <functional>
#include <iostream>
#include <mutex>
#include <thread>
#include <vector>

//===----------------------------------------------------------------------===//
// Async runtime API.
//===----------------------------------------------------------------------===//

struct AsyncToken {
  bool ready = false;
  std::mutex mu;
  std::condition_variable cv;
  std::vector<std::function<void()>> awaiters;
};

struct AsyncGroup {
  std::atomic<int> pendingTokens{0};
  std::atomic<int> rank{0};
  std::mutex mu;
  std::condition_variable cv;
  std::vector<std::function<void()>> awaiters;
};

// Create a new `async.token` in not-ready state.
extern "C" AsyncToken *mlirAsyncRuntimeCreateToken() {
  AsyncToken *token = new AsyncToken;
  return token;
}

// Create a new `async.group` in empty state.
extern "C" MLIR_ASYNCRUNTIME_EXPORT AsyncGroup *mlirAsyncRuntimeCreateGroup() {
  AsyncGroup *group = new AsyncGroup;
  return group;
}

extern "C" MLIR_ASYNCRUNTIME_EXPORT int64_t
mlirAsyncRuntimeAddTokenToGroup(AsyncToken *token, AsyncGroup *group) {
  std::unique_lock<std::mutex> lockToken(token->mu);
  std::unique_lock<std::mutex> lockGroup(group->mu);

  group->pendingTokens.fetch_add(1);

  auto onTokenReady = [group]() {
    // Run all group awaiters if it was the last token in the group.
    if (group->pendingTokens.fetch_sub(1) == 1) {
      group->cv.notify_all();
      for (auto &awaiter : group->awaiters)
        awaiter();
    }
  };

  if (token->ready)
    onTokenReady();
  else
    token->awaiters.push_back([onTokenReady]() { onTokenReady(); });

  return group->rank.fetch_add(1);
}

// Switches `async.token` to ready state and runs all awaiters.
extern "C" void mlirAsyncRuntimeEmplaceToken(AsyncToken *token) {
  std::unique_lock<std::mutex> lock(token->mu);
  token->ready = true;
  token->cv.notify_all();
  for (auto &awaiter : token->awaiters)
    awaiter();
}

extern "C" void mlirAsyncRuntimeAwaitToken(AsyncToken *token) {
  std::unique_lock<std::mutex> lock(token->mu);
  if (!token->ready)
    token->cv.wait(lock, [token] { return token->ready; });
}

extern "C" MLIR_ASYNCRUNTIME_EXPORT void
mlirAsyncRuntimeAwaitAllInGroup(AsyncGroup *group) {
  std::unique_lock<std::mutex> lock(group->mu);
  if (group->pendingTokens != 0)
    group->cv.wait(lock, [group] { return group->pendingTokens == 0; });
}

extern "C" void mlirAsyncRuntimeExecute(CoroHandle handle, CoroResume resume) {
#if LLVM_ENABLE_THREADS
  std::thread thread([handle, resume]() { (*resume)(handle); });
  thread.detach();
#else
  (*resume)(handle);
#endif
}

extern "C" void mlirAsyncRuntimeAwaitTokenAndExecute(AsyncToken *token,
                                                     CoroHandle handle,
                                                     CoroResume resume) {
  std::unique_lock<std::mutex> lock(token->mu);

  auto execute = [handle, resume]() {
    mlirAsyncRuntimeExecute(handle, resume);
  };

  if (token->ready)
    execute();
  else
    token->awaiters.push_back([execute]() { execute(); });
}

extern "C" MLIR_ASYNCRUNTIME_EXPORT void
mlirAsyncRuntimeAwaitAllInGroupAndExecute(AsyncGroup *group, CoroHandle handle,
                                          CoroResume resume) {
  std::unique_lock<std::mutex> lock(group->mu);

  auto execute = [handle, resume]() {
    mlirAsyncRuntimeExecute(handle, resume);
  };

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
  std::cout << "Current thread id: " << thisId << "\n";
}

#endif // MLIR_ASYNCRUNTIME_DEFINE_FUNCTIONS
