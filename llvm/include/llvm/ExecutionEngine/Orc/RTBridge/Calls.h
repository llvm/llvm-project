//===------- Calls.h - Runtime-agnostic executor call APIs ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Runtime-agnostic interfaces for invoking executor-side operations. These
// abstract over how a call reaches the executor, so clients can be written
// once and used whether the operation is provided by a full ORC runtime or by
// LLVM's own ORC-runtime-lite. Concrete implementations live in subdirectories
// (e.g. RTBridge/SPS).
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_EXECUTIONENGINE_ORC_RTBRIDGE_CALLS_H
#define LLVM_EXECUTIONENGINE_ORC_RTBRIDGE_CALLS_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/FunctionExtras.h"
#include "llvm/ExecutionEngine/Orc/Shared/ExecutorAddress.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/MSVCErrorWorkarounds.h"

#include <cstdint>
#include <future>
#include <string>
#include <type_traits>

namespace llvm::orc::rt {

template <typename FnT> class Caller;

/// Runtime-agnostic interface for invoking an executor-side operation with the
/// signature RetT(ArgTs...).
///
/// The operation is identified by an ExecutorAddr (the address of the
/// executor-side function to invoke) and takes ArgTs... as arguments. Two call
/// operators are provided: an asynchronous form that delivers the result to an
/// OnComplete continuation, and a synchronous form that blocks until the result
/// is available.
///
/// A Caller abstracts over how the operation is dispatched to the executor.
/// Concrete implementations (e.g. rt::sps::Caller) supply the dispatch
/// mechanism.
template <typename RetT, typename... ArgTs> class Caller<RetT(ArgTs...)> {
public:
  using FnType = RetT(ArgTs...);

  /// The result type produced by the executor-side function itself.
  using CalleeRetT = RetT;

  /// The result type delivered to callers: Expected<RetT>, or Error when RetT
  /// is void, so that dispatch failures can be reported alongside the result.
  using ErrorRetT =
      std::conditional_t<std::is_void_v<RetT>, Error, Expected<RetT>>;

  virtual ~Caller() = default;

  /// Asynchronously invoke the executor-side function at FnAddr with the given
  /// Args, delivering its result (or an error) to OnComplete.
  virtual void operator()(unique_function<void(ErrorRetT)> OnComplete,
                          ExecutorAddr FnAddr, ArgTs... Args) = 0;

  /// Invoke the executor-side function at FnAddr with the given Args, blocking
  /// until its result (or an error) is available.
  ErrorRetT operator()(ExecutorAddr FnAddr, ArgTs &&...Args) {
    using PromiseValT = std::conditional_t<std::is_void_v<RetT>, MSVCPError,
                                           MSVCPExpected<RetT>>;
    std::promise<PromiseValT> P;
    auto F = P.get_future();
    this->operator()(
        [P = std::move(P)](ErrorRetT R) mutable { P.set_value(std::move(R)); },
        FnAddr, std::forward<ArgTs>(Args)...);
    return F.get();
  }
};

/// Runtime-agnostic interface for running a main-like function
/// (int(int argc, char *argv[])) in the executor.
///
/// The function to run is given by its ExecutorAddr, its arguments as an
/// argument vector, and its int64_t result is returned.
class MainCaller : public Caller<int64_t(ArrayRef<std::string>)> {};

/// Runtime-agnostic interface for running a void() function in the executor.
///
/// The function to run is given by its ExecutorAddr.
///
/// WARNING: This Caller is experimental and may be removed.
class VoidVoidCaller : public Caller<void()> {};

/// Runtime-agnostic interface for running an int32_t() function in the
/// executor.
///
/// The function to run is given by its ExecutorAddr.
///
/// WARNING: This Caller is experimental and may be removed.
class Int32VoidCaller : public Caller<int32_t()> {};

/// Runtime-agnostic interface for running an int32_t(int32_t) function in the
/// executor.
///
/// The function to run is given by its ExecutorAddr.
///
/// WARNING: This Caller is experimental and may be removed.
class Int32Int32Caller : public Caller<int32_t(int32_t)> {};

} // namespace llvm::orc::rt

#endif // LLVM_EXECUTIONENGINE_ORC_RTBRIDGE_CALLS_H
