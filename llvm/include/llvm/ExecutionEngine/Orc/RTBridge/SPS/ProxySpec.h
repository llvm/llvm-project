//===------- ProxySpec.h - SPS dispatch for rt::Proxy -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// ProxySpec implements an rt::Proxy's dispatch by invoking an executor-side
// wrapper function in the runtime's controller interface, using Simple Packed
// Serialization to encode arguments and decode results.
//
// Specs for specific operation families live in sibling headers whose specs
// only need Shared/SPS vocabulary types (e.g. CallProxySpecs.h,
// MemoryAccessProxySpecs.h).
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_EXECUTIONENGINE_ORC_RTBRIDGE_SPS_PROXYSPEC_H
#define LLVM_EXECUTIONENGINE_ORC_RTBRIDGE_SPS_PROXYSPEC_H

#include "llvm/ExecutionEngine/Orc/Core.h"
#include "llvm/ExecutionEngine/Orc/RTBridge/Proxy.h"

namespace llvm::orc::rt::sps {

template <typename ProxyT, typename CI,
          typename FnType = typename ProxyT::FnType>
class ProxySpec;

template <typename ProxyT, typename CI, typename RetT, typename... ArgTs>
class ProxySpec<ProxyT, CI, RetT(ArgTs...)> {

  using SPSSigT = typename CI::SPSSig;
  using CalleeRetT = typename ProxyT::CalleeRetT;
  using ErrorRetT = typename ProxyT::ErrorRetT;

  static void consumeResult(Error &Err) { consumeError(std::move(Err)); }

  template <typename T> static void consumeResult(T &V) {}

  template <typename T> static void consumeResult(Expected<T> &E) {
    consumeError(E.takeError());
  }

public:
  static constexpr const char *Name = CI::Name;

  static void dispatch(unique_function<void(ErrorRetT)> OnComplete,
                       ExecutionSession &ES, ExecutorAddr CalleeAddr,
                       const ArgTs &...Args) {
    if constexpr (std::is_void_v<CalleeRetT>) {
      // Void result: the executor-side function produces no value, so the only
      // thing to report is the dispatch error (success if the call ran).
      ES.callSPSWrapperAsync<SPSSigT>(CalleeAddr, std::move(OnComplete),
                                      Args...);
    } else {
      ES.callSPSWrapperAsync<SPSSigT>(
          CalleeAddr,
          [OnComplete = std::move(OnComplete)](Error SerErr,
                                               CalleeRetT Result) mutable {
            if (SerErr) {
              consumeResult(Result);
              return OnComplete(std::move(SerErr));
            }
            // For an Error/Expected callee this forwards the callee's own
            // result; for a plain value it is wrapped into Expected.
            return OnComplete(std::move(Result));
          },
          Args...);
    }
  }
};

} // namespace llvm::orc::rt::sps

#endif // LLVM_EXECUTIONENGINE_ORC_RTBRIDGE_SPS_PROXYSPEC_H
