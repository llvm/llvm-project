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
#include "llvm/ExecutionEngine/Orc/Core.h"
#include "llvm/ExecutionEngine/Orc/Shared/ExecutorAddress.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/MSVCErrorWorkarounds.h"

#include <cstdint>
#include <future>
#include <string>
#include <type_traits>

namespace llvm::orc::rt {

class CallerBase {
public:
  CallerBase() = default;
  CallerBase(ExecutorAddr CalleeAddr) : CalleeAddr(CalleeAddr) {}

  /// Returns the address of the callee in the executor.
  const ExecutorAddr &calleeAddr() const { return CalleeAddr; }

  /// Evaluates to true if the callee is non-null.
  explicit operator bool() const { return !!CalleeAddr; }

private:
  ExecutorAddr CalleeAddr;
};

template <typename FnT> class Caller;

/// Runtime-agnostic interface for invoking an executor-side operation with the
/// signature RetT(ArgTs...).
///
/// Two call operators are provided: an asynchronous form that delivers the
/// result to an OnComplete continuation, and a synchronous form that blocks
/// until the result is available.
///
/// A Caller abstracts over how the operation is dispatched to the executor.
/// Concrete implementations (e.g. rt::sps::Caller) supply the dispatch
/// mechanism.
template <typename RetT, typename... ArgTs>
class Caller<RetT(ArgTs...)> : public CallerBase {
public:
  using FnType = RetT(ArgTs...);

  /// The result type produced by the executor-side function itself.
  using CalleeRetT = RetT;

  /// The result type delivered to callers: Expected<RetT>, or Error when RetT
  /// is void, so that dispatch failures can be reported alongside the result.
  using ErrorRetT =
      std::conditional_t<std::is_void_v<RetT>, Error, Expected<RetT>>;

  using DispatchFn = void (*)(unique_function<void(ErrorRetT)> OnComplete,
                              ExecutionSession &ES, ExecutorAddr Callee,
                              const ArgTs &...Args);

  Caller() = default;
  Caller(DispatchFn Dispatch, ExecutorAddr CalleeAddr)
      : CallerBase(CalleeAddr), Dispatch(Dispatch) {}

  static Expected<Caller> Create(DispatchFn Dispatch, JITDylib &JD,
                                 StringRef Name, SymbolLookupFlags LF) {
    auto &ES = JD.getExecutionSession();
    if (auto CalleeSyms = ES.lookup(makeJITDylibSearchOrder(&JD),
                                    SymbolLookupSet{ES.intern(Name), LF})) {
      if (!CalleeSyms->empty())
        return Caller(Dispatch, CalleeSyms->begin()->second.getAddress());
      assert(LF == SymbolLookupFlags::WeaklyReferencedSymbol);
      return Caller();
    } else
      return CalleeSyms.takeError();
  }

  static Expected<Caller> Create(DispatchFn Dispatch, ExecutionSession &ES,
                                 StringRef Name, SymbolLookupFlags LF) {
    return Create(Dispatch, ES.getBootstrapJITDylib(), Name, LF);
  }

  /// Asynchronously invoke the operation with the given Args, delivering its
  /// result (or an error) to OnComplete.
  void operator()(unique_function<void(ErrorRetT)> OnComplete,
                  ExecutionSession &ES, const ArgTs &...Args) const {
    assert(Dispatch && "Caller's Dispatch member is not set");
    Dispatch(std::move(OnComplete), ES, calleeAddr(), Args...);
  }

  /// Invoke the operation with the given Args, blocking until its result (or an
  /// error) is available.
  ErrorRetT operator()(ExecutionSession &ES, const ArgTs &...Args) const {
    using PromiseValT = std::conditional_t<std::is_void_v<RetT>, MSVCPError,
                                           MSVCPExpected<RetT>>;
    std::promise<PromiseValT> P;
    auto F = P.get_future();
    this->operator()(
        [P = std::move(P)](ErrorRetT R) mutable { P.set_value(std::move(R)); },
        ES, Args...);
    return F.get();
  }

private:
  DispatchFn Dispatch = nullptr;
};

template <typename FnT> struct CallerInit {
  Caller<FnT> *C = nullptr;
  typename Caller<FnT>::DispatchFn Dispatch = nullptr;
  StringRef Name;
  SymbolLookupFlags LookupFlags = SymbolLookupFlags::RequiredSymbol;
};

template <typename FnT>
CallerInit<FnT>
callerInit(Caller<FnT> *C, typename Caller<FnT>::DispatchFn Dispatch,
           StringRef Name,
           SymbolLookupFlags LookupFlags = SymbolLookupFlags::RequiredSymbol) {
  return {C, Dispatch, Name, LookupFlags};
}

template <typename CallerSpecT, typename FnT>
CallerInit<FnT>
callerInit(Caller<FnT> *C,
           SymbolLookupFlags LookupFlags = SymbolLookupFlags::RequiredSymbol) {
  return {C, CallerSpecT::dispatch, CallerSpecT::Name, LookupFlags};
}

template <typename CallerSpecT, typename FnT>
CallerInit<FnT>
callerInit(Caller<FnT> *C, StringRef Name,
           SymbolLookupFlags LookupFlags = SymbolLookupFlags::RequiredSymbol) {
  return {C, CallerSpecT::dispatch, Name, LookupFlags};
}

/// buildCallers base case.
inline Error buildCallers(JITDylib &JD) { return Error::success(); }

/// buildCallers: Given an ExecutionSession, use BootstrapJITDylib.
template <typename... FnTs>
Error buildCallers(ExecutionSession &ES, CallerInit<FnTs>... CIs) {
  return buildCallers(ES.getBootstrapJITDylib(), CIs...);
}

/// Build a sequence of callers from their respective caller-builders.
template <typename FnT, typename... FnTs>
Error buildCallers(JITDylib &JD, CallerInit<FnT> CI, CallerInit<FnTs>... CIs) {
  if (auto COrErr =
          Caller<FnT>::Create(CI.Dispatch, JD, CI.Name, CI.LookupFlags))
    *CI.C = std::move(*COrErr);
  else
    return COrErr.takeError();
  return buildCallers(JD, CIs...);
}

/// Runtime-agnostic interface for running a main-like function
/// (int(int argc, char *argv[])) in the executor.
///
/// The function to run is given by its ExecutorAddr, its arguments as an
/// argument vector, and its int64_t result is returned.
using MainCaller = Caller<int64_t(ExecutorAddr, ArrayRef<std::string>)>;

/// Runtime-agnostic interface for running a void() function in the executor.
///
/// The function to run is given by its ExecutorAddr.
///
/// WARNING: This Caller is experimental and may be removed.
using VoidVoidCaller = Caller<void(ExecutorAddr)>;

/// Runtime-agnostic interface for running an int32_t() function in the
/// executor.
///
/// The function to run is given by its ExecutorAddr.
///
/// WARNING: This Caller is experimental and may be removed.
using Int32VoidCaller = Caller<int32_t(ExecutorAddr)>;

/// Runtime-agnostic interface for running an int32_t(int32_t) function in the
/// executor.
///
/// The function to run is given by its ExecutorAddr.
///
/// WARNING: This Caller is experimental and may be removed.
using Int32Int32Caller = Caller<int32_t(ExecutorAddr, int32_t)>;

} // namespace llvm::orc::rt

#endif // LLVM_EXECUTIONENGINE_ORC_RTBRIDGE_CALLS_H
