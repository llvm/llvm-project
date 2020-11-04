//===- AsyncRuntime.h - Async runtime reference implementation ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares basic Async runtime API for supporting Async dialect
// to LLVM dialect lowering.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_EXECUTIONENGINE_ASYNCRUNTIME_H_
#define MLIR_EXECUTIONENGINE_ASYNCRUNTIME_H_

#ifdef _WIN32
#ifndef MLIR_ASYNCRUNTIME_EXPORT
#ifdef mlir_c_runner_utils_EXPORTS
// We are building this library
#define MLIR_ASYNCRUNTIME_EXPORT __declspec(dllexport)
#define MLIR_ASYNCRUNTIME_DEFINE_FUNCTIONS
#else
// We are using this library
#define MLIR_ASYNCRUNTIME_EXPORT __declspec(dllimport)
#endif // mlir_c_runner_utils_EXPORTS
#endif // MLIR_ASYNCRUNTIME_EXPORT
#else
#define MLIR_ASYNCRUNTIME_EXPORT
#define MLIR_ASYNCRUNTIME_DEFINE_FUNCTIONS
#endif // _WIN32

//===----------------------------------------------------------------------===//
// Async runtime API.
//===----------------------------------------------------------------------===//

// Runtime implementation of `async.token` data type.
typedef struct AsyncToken MLIR_AsyncToken;

// Async runtime uses LLVM coroutines to represent asynchronous tasks. Task
// function is a coroutine handle and a resume function that continue coroutine
// execution from a suspension point.
using CoroHandle = void *;           // coroutine handle
using CoroResume = void (*)(void *); // coroutine resume function

// Create a new `async.token` in not-ready state.
extern "C" MLIR_ASYNCRUNTIME_EXPORT AsyncToken *mlirAsyncRuntimeCreateToken();

// Switches `async.token` to ready state and runs all awaiters.
extern "C" MLIR_ASYNCRUNTIME_EXPORT void
mlirAsyncRuntimeEmplaceToken(AsyncToken *);

// Blocks the caller thread until the token becomes ready.
extern "C" MLIR_ASYNCRUNTIME_EXPORT void
mlirAsyncRuntimeAwaitToken(AsyncToken *);

// Executes the task (coro handle + resume function) in one of the threads
// managed by the runtime.
extern "C" MLIR_ASYNCRUNTIME_EXPORT void mlirAsyncRuntimeExecute(CoroHandle,
                                                                 CoroResume);

// Executes the task (coro handle + resume function) in one of the threads
// managed by the runtime after the token becomes ready.
extern "C" MLIR_ASYNCRUNTIME_EXPORT void
mlirAsyncRuntimeAwaitTokenAndExecute(AsyncToken *, CoroHandle, CoroResume);

//===----------------------------------------------------------------------===//
// Small async runtime support library for testing.
//===----------------------------------------------------------------------===//

extern "C" MLIR_ASYNCRUNTIME_EXPORT void mlirAsyncRuntimePrintCurrentThreadId();

#endif // MLIR_EXECUTIONENGINE_ASYNCRUNTIME_H_
