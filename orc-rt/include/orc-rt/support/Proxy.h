//===------ Proxy.h - Protocol-agnostic controller call APIs ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Protocol-agnostic interface for invoking a controller-side operation from the
// executor. A Proxy abstracts over how a call reaches the controller, so
// callers can be written once regardless of the underlying transport or
// serialization.
//
// The callee is named by an opaque tag (typically the address of a
// controller-side global). A Proxy's dispatch function -- supplied by a spec
// for some concrete protocol -- routes the call through the Session (e.g. via
// callController). Named proxies for specific operation families, and their
// specs, live alongside the utilities that use them.
//
//===----------------------------------------------------------------------===//

#ifndef ORC_RT_SUPPORT_PROXY_H
#define ORC_RT_SUPPORT_PROXY_H

#include "orc-rt/support/Error.h"
#include "orc-rt/support/move_only_function.h"

#include <cassert>
#include <utility>

namespace orc_rt {

class Session;

class ProxyBase {
public:
  ProxyBase() = default;
  explicit ProxyBase(const void *CalleeTag) : CalleeTag(CalleeTag) {}

  /// Returns the callee tag.
  const void *calleeTag() const { return CalleeTag; }

  /// Evaluates to true if the callee is non-null.
  explicit operator bool() const { return !!CalleeTag; }

private:
  const void *CalleeTag = nullptr;
};

template <typename FnT> class Proxy;

namespace detail {

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

} // namespace detail

template <typename RetT, typename... ArgTs>
class Proxy<RetT(ArgTs...)> : public ProxyBase {
public:
  using FnType = RetT(ArgTs...);

  using CalleeRetT = RetT;

  using ErrorRetT = typename detail::ProxyErrorRet<RetT>::type;

  using DispatchFn = void (*)(move_only_function<void(ErrorRetT)> OnComplete,
                              Session &S, const void *CalleeTag,
                              const ArgTs &...Args);

  Proxy() = default;
  Proxy(DispatchFn Dispatch, const void *CalleeTag)
      : ProxyBase(CalleeTag), Dispatch(Dispatch) {}

  void operator()(move_only_function<void(ErrorRetT)> OnComplete, Session &S,
                  const ArgTs &...Args) const {
    assert(Dispatch && "Proxy's Dispatch member is not set");
    Dispatch(std::move(OnComplete), S, calleeTag(), Args...);
  }

private:
  DispatchFn Dispatch = nullptr;
};

} // namespace orc_rt

#endif // ORC_RT_SUPPORT_PROXY_H
