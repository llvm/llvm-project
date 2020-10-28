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

// Create a new `async.token` in not-ready state.
extern "C" MLIR_ASYNCRUNTIME_EXPORT AsyncToken *mlirAsyncRuntimeCreateToken() {
  AsyncToken *token = new AsyncToken;
  return token;
}

// Switches `async.token` to ready state and runs all awaiters.
extern "C" MLIR_ASYNCRUNTIME_EXPORT void
mlirAsyncRuntimeEmplaceToken(AsyncToken *token) {
  std::unique_lock<std::mutex> lock(token->mu);
  token->ready = true;
  token->cv.notify_all();
  for (auto &awaiter : token->awaiters)
    awaiter();
}

extern "C" MLIR_ASYNCRUNTIME_EXPORT void
mlirAsyncRuntimeAwaitToken(AsyncToken *token) {
  std::unique_lock<std::mutex> lock(token->mu);
  if (!token->ready)
    token->cv.wait(lock, [token] { return token->ready; });
  delete token;
}

extern "C" MLIR_ASYNCRUNTIME_EXPORT void
mlirAsyncRuntimeExecute(CoroHandle handle, CoroResume resume) {
#if LLVM_ENABLE_THREADS
  std::thread thread([handle, resume]() { (*resume)(handle); });
  thread.detach();
#else
  (*resume)(handle);
#endif
}

extern "C" MLIR_ASYNCRUNTIME_EXPORT void
mlirAsyncRuntimeAwaitTokenAndExecute(AsyncToken *token, CoroHandle handle,
                                     CoroResume resume) {
  std::unique_lock<std::mutex> lock(token->mu);

  auto execute = [token, handle, resume]() {
    mlirAsyncRuntimeExecute(handle, resume);
    delete token;
  };

  if (token->ready)
    execute();
  else
    token->awaiters.push_back([execute]() { execute(); });
}

//===----------------------------------------------------------------------===//
// Small async runtime support library for testing.
//===----------------------------------------------------------------------===//

extern "C" MLIR_ASYNCRUNTIME_EXPORT void
mlirAsyncRuntimePrintCurrentThreadId() {
  static thread_local std::thread::id thisId = std::this_thread::get_id();
  std::cout << "Current thread id: " << thisId << "\n";
}

#endif // MLIR_ASYNCRUNTIME_DEFINE_FUNCTIONS
