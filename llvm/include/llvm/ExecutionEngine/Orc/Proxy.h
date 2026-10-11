//===------- Proxy.h - Protocol-agnostic executor call APIs -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Protocol-agnostic interfaces for invoking executor-side operations. These
// abstract over how a call reaches the executor, so clients can be written
// once and used whether the operation is provided by a full ORC runtime or by
// LLVM's own ORC-runtime-lite.
//
// This header provides only the core Proxy machinery. A Proxy's dispatch
// function is supplied by a spec for some concrete protocol -- see
// SPSProxySpec.h for the Simple Packed Serialization implementation. Named
// proxies for specific operation families live alongside the utilities that use
// them (e.g. CallProxies.h, EPCGenericMemoryAccess.h).
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_EXECUTIONENGINE_ORC_PROXY_H
#define LLVM_EXECUTIONENGINE_ORC_PROXY_H

#include "llvm/ADT/FunctionExtras.h"
#include "llvm/ExecutionEngine/Orc/Core.h"
#include "llvm/ExecutionEngine/Orc/Shared/ExecutorAddress.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/MSVCErrorWorkarounds.h"

#include <future>
#include <type_traits>

namespace llvm::orc {

class ProxyBase {
public:
  ProxyBase() = default;
  ProxyBase(ExecutorAddr CalleeAddr) : CalleeAddr(CalleeAddr) {}

  /// Returns the address of the callee in the executor.
  const ExecutorAddr &calleeAddr() const { return CalleeAddr; }

  /// Evaluates to true if the callee is non-null.
  explicit operator bool() const { return !!CalleeAddr; }

private:
  ExecutorAddr CalleeAddr;
};

template <typename FnT> class Proxy;

namespace detail {

/// Maps a proxy's callee return type to the type delivered to the client, so a
/// dispatch failure can always be reported alongside the result:
///
///          void -> Error
///         Error -> Error
///             T -> Expected<T>
///   Expected<T> -> Expected<T>
template <typename T> struct ProxyErrorRet {
  using type = Expected<T>;
};
template <> struct ProxyErrorRet<void> {
  using type = Error;
};
template <> struct ProxyErrorRet<Error> {
  using type = Error;
};
template <typename T> struct ProxyErrorRet<Expected<T>> {
  using type = Expected<T>;
};

/// Maps a proxy's client-facing return type to the std::promise value type used
/// by the blocking call operator (working around MSVC's std::promise).
template <typename T> struct ProxyRetPromise;
template <> struct ProxyRetPromise<Error> {
  using type = std::promise<MSVCPError>;
};
template <typename T> struct ProxyRetPromise<Expected<T>> {
  using type = std::promise<MSVCPExpected<T>>;
};

} // namespace detail

/// Protocol-agnostic interface for invoking an executor-side operation with the
/// signature RetT(ArgTs...).
///
/// Two call operators are provided: an asynchronous form that delivers the
/// result to an OnComplete continuation, and a synchronous form that blocks
/// until the result is available.
///
/// A Proxy abstracts over how the operation is dispatched to the executor. Its
/// dispatch function is supplied by a spec (e.g. sps::ProxySpec).
template <typename RetT, typename... ArgTs>
class Proxy<RetT(ArgTs...)> : public ProxyBase {
public:
  using FnType = RetT(ArgTs...);

  /// The result type produced by the executor-side function itself.
  using CalleeRetT = RetT;

  /// The result type delivered to the client: Error when the callee returns
  /// void or Error, otherwise Expected<T> (with Expected<T> callees flattened
  /// rather than nested), so that dispatch failures can be reported alongside
  /// the result.
  using ErrorRetT = typename detail::ProxyErrorRet<RetT>::type;

  using DispatchFn = void (*)(unique_function<void(ErrorRetT)> OnComplete,
                              ExecutionSession &ES, ExecutorAddr Callee,
                              const ArgTs &...Args);

  Proxy() = default;
  Proxy(DispatchFn Dispatch, ExecutorAddr CalleeAddr)
      : ProxyBase(CalleeAddr), Dispatch(Dispatch) {}

  /// Asynchronously invoke the operation with the given Args, delivering its
  /// result (or an error) to OnComplete.
  void operator()(unique_function<void(ErrorRetT)> OnComplete,
                  ExecutionSession &ES, const ArgTs &...Args) const {
    assert(Dispatch && "Proxy's Dispatch member is not set");
    Dispatch(std::move(OnComplete), ES, calleeAddr(), Args...);
  }

  /// Invoke the operation with the given Args, blocking until its result (or an
  /// error) is available.
  ErrorRetT operator()(ExecutionSession &ES, const ArgTs &...Args) const {
    typename detail::ProxyRetPromise<ErrorRetT>::type P;
    auto F = P.get_future();
    this->operator()(
        [P = std::move(P)](ErrorRetT R) mutable { P.set_value(std::move(R)); },
        ES, Args...);
    return F.get();
  }

private:
  DispatchFn Dispatch = nullptr;
};

} // namespace llvm::orc

#endif // LLVM_EXECUTIONENGINE_ORC_PROXY_H
