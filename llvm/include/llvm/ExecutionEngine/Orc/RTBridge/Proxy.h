//===------- Proxy.h - Runtime-agnostic executor call APIs ------*- C++ -*-===//
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

#ifndef LLVM_EXECUTIONENGINE_ORC_RTBRIDGE_PROXY_H
#define LLVM_EXECUTIONENGINE_ORC_RTBRIDGE_PROXY_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/FunctionExtras.h"
#include "llvm/ExecutionEngine/Orc/Core.h"
#include "llvm/ExecutionEngine/Orc/Shared/ExecutorAddress.h"
#include "llvm/ExecutionEngine/Orc/Shared/TargetProcessControlTypes.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/MSVCErrorWorkarounds.h"

#include <cstdint>
#include <future>
#include <string>
#include <type_traits>
#include <vector>

namespace llvm::orc::rt {

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

/// Runtime-agnostic interface for invoking an executor-side operation with the
/// signature RetT(ArgTs...).
///
/// Two call operators are provided: an asynchronous form that delivers the
/// result to an OnComplete continuation, and a synchronous form that blocks
/// until the result is available.
///
/// A Proxy abstracts over how the operation is dispatched to the executor. Its
/// dispatch function is supplied by a spec (e.g. rt::sps::ProxySpec).
template <typename RetT, typename... ArgTs>
class Proxy<RetT(ArgTs...)> : public ProxyBase {
public:
  using FnType = RetT(ArgTs...);

  /// The result type produced by the executor-side function itself.
  using CalleeRetT = RetT;

  /// The result type delivered to the client: Expected<RetT>, or Error when
  /// RetT is void, so that dispatch failures can be reported alongside the
  /// result.
  using ErrorRetT =
      std::conditional_t<std::is_void_v<RetT>, Error, Expected<RetT>>;

  using DispatchFn = void (*)(unique_function<void(ErrorRetT)> OnComplete,
                              ExecutionSession &ES, ExecutorAddr Callee,
                              const ArgTs &...Args);

  Proxy() = default;
  Proxy(DispatchFn Dispatch, ExecutorAddr CalleeAddr)
      : ProxyBase(CalleeAddr), Dispatch(Dispatch) {}

  static Expected<Proxy> Create(DispatchFn Dispatch, JITDylib &JD,
                                StringRef Name, SymbolLookupFlags LF) {
    auto &ES = JD.getExecutionSession();
    if (auto CalleeSyms = ES.lookup(makeJITDylibSearchOrder(&JD),
                                    SymbolLookupSet{ES.intern(Name), LF})) {
      if (!CalleeSyms->empty())
        return Proxy(Dispatch, CalleeSyms->begin()->second.getAddress());
      assert(LF == SymbolLookupFlags::WeaklyReferencedSymbol);
      return Proxy();
    } else
      return CalleeSyms.takeError();
  }

  static Expected<Proxy> Create(DispatchFn Dispatch, ExecutionSession &ES,
                                StringRef Name, SymbolLookupFlags LF) {
    return Create(Dispatch, ES.getBootstrapJITDylib(), Name, LF);
  }

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

template <typename FnT> struct ProxyInit {
  Proxy<FnT> *P = nullptr;
  typename Proxy<FnT>::DispatchFn Dispatch = nullptr;
  StringRef Name;
  SymbolLookupFlags LookupFlags = SymbolLookupFlags::RequiredSymbol;
};

template <typename FnT>
ProxyInit<FnT>
proxyInit(Proxy<FnT> *P, typename Proxy<FnT>::DispatchFn Dispatch,
          StringRef Name,
          SymbolLookupFlags LookupFlags = SymbolLookupFlags::RequiredSymbol) {
  return {P, Dispatch, Name, LookupFlags};
}

template <typename ProxySpecT, typename FnT>
ProxyInit<FnT>
proxyInit(Proxy<FnT> *P,
          SymbolLookupFlags LookupFlags = SymbolLookupFlags::RequiredSymbol) {
  return {P, ProxySpecT::dispatch, ProxySpecT::Name, LookupFlags};
}

template <typename ProxySpecT, typename FnT>
ProxyInit<FnT>
proxyInit(Proxy<FnT> *P, StringRef Name,
          SymbolLookupFlags LookupFlags = SymbolLookupFlags::RequiredSymbol) {
  return {P, ProxySpecT::dispatch, Name, LookupFlags};
}

/// buildProxies base case.
inline Error buildProxies(JITDylib &JD) { return Error::success(); }

/// buildProxies: Given an ExecutionSession, use BootstrapJITDylib.
template <typename... FnTs>
Error buildProxies(ExecutionSession &ES, ProxyInit<FnTs>... PIs) {
  return buildProxies(ES.getBootstrapJITDylib(), PIs...);
}

/// Build a sequence of proxies from their respective specs.
template <typename FnT, typename... FnTs>
Error buildProxies(JITDylib &JD, ProxyInit<FnT> PI, ProxyInit<FnTs>... PIs) {
  if (auto POrErr =
          Proxy<FnT>::Create(PI.Dispatch, JD, PI.Name, PI.LookupFlags))
    *PI.P = std::move(*POrErr);
  else
    return POrErr.takeError();
  return buildProxies(JD, PIs...);
}

/// Runtime-agnostic interface for running a main-like function
/// (int(int argc, char *argv[])) in the executor.
///
/// The function to run is given by its ExecutorAddr, its arguments as an
/// argument vector, and its int64_t result is returned.
using CallMainProxy = Proxy<int64_t(ExecutorAddr, ArrayRef<std::string>)>;

/// Runtime-agnostic interface for running a void() function in the executor.
///
/// The function to run is given by its ExecutorAddr.
///
/// WARNING: This Proxy is experimental and may be removed.
using CallVoidVoidProxy = Proxy<void(ExecutorAddr)>;

/// Runtime-agnostic interface for running an int32_t() function in the
/// executor.
///
/// The function to run is given by its ExecutorAddr.
///
/// WARNING: This Proxy is experimental and may be removed.
using CallInt32VoidProxy = Proxy<int32_t(ExecutorAddr)>;

/// Runtime-agnostic interface for running an int32_t(int32_t) function in the
/// executor.
///
/// The function to run is given by its ExecutorAddr.
///
/// WARNING: This Proxy is experimental and may be removed.
using CallInt32Int32Proxy = Proxy<int32_t(ExecutorAddr, int32_t)>;

/// Runtime-agnostic interfaces for the memory-access operations. Unlike the
/// Call* proxies above, these target wrappers that perform the operation
/// directly, so they take the operation's data arguments rather than a callee
/// address.
using MemWriteUInt8sProxy = Proxy<void(ArrayRef<tpctypes::UInt8Write>)>;
using MemWriteUInt16sProxy = Proxy<void(ArrayRef<tpctypes::UInt16Write>)>;
using MemWriteUInt32sProxy = Proxy<void(ArrayRef<tpctypes::UInt32Write>)>;
using MemWriteUInt64sProxy = Proxy<void(ArrayRef<tpctypes::UInt64Write>)>;
using MemWritePointersProxy = Proxy<void(ArrayRef<tpctypes::PointerWrite>)>;
using MemWriteBuffersProxy = Proxy<void(ArrayRef<tpctypes::BufferWrite>)>;
using MemReadUInt8sProxy = Proxy<std::vector<uint8_t>(ArrayRef<ExecutorAddr>)>;
using MemReadUInt16sProxy =
    Proxy<std::vector<uint16_t>(ArrayRef<ExecutorAddr>)>;
using MemReadUInt32sProxy =
    Proxy<std::vector<uint32_t>(ArrayRef<ExecutorAddr>)>;
using MemReadUInt64sProxy =
    Proxy<std::vector<uint64_t>(ArrayRef<ExecutorAddr>)>;
using MemReadPointersProxy =
    Proxy<std::vector<ExecutorAddr>(ArrayRef<ExecutorAddr>)>;
using MemReadBuffersProxy =
    Proxy<std::vector<std::vector<uint8_t>>(ArrayRef<ExecutorAddrRange>)>;
using MemReadStringsProxy =
    Proxy<std::vector<std::string>(ArrayRef<ExecutorAddr>)>;

} // namespace llvm::orc::rt

#endif // LLVM_EXECUTIONENGINE_ORC_RTBRIDGE_PROXY_H
