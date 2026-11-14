//===------- SPSProxySpec.h - SPS dispatch for orc::Proxy -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// ProxySpec implements an orc::Proxy's dispatch by invoking an executor-side
// wrapper function in the runtime's controller interface, using Simple Packed
// Serialization to encode arguments and decode results.
//
// A ProxySpec is built from a controller-interface descriptor from
// Shared/SPSCI, which supplies the wrapper name and wire signature. It can
// dispatch for any Proxy whose argument and result types are serializable
// with that signature. Specs for specific operation families live alongside
// the utilities that use them (e.g. CallProxiesSPS.h,
// EPCGenericMemoryAccessSPS.h).
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_EXECUTIONENGINE_ORC_SPSPROXYSPEC_H
#define LLVM_EXECUTIONENGINE_ORC_SPSPROXYSPEC_H

#include "llvm/ExecutionEngine/Orc/Core.h"
#include "llvm/ExecutionEngine/Orc/Proxy.h"
#include "llvm/ExecutionEngine/Orc/Shared/SymbolNameSpec.h"

#include <type_traits>

namespace llvm::orc::sps {

/// Dispatches proxies for the call described by CI, which must provide:
///
///   static constexpr SymbolNameSpec Name; // The default wrapper name.
///   using SPSSig = ...;                   // The call's SPS signature.
template <typename CI, typename SPSSigT = typename CI::SPSSig> class ProxySpec;

template <typename CI, typename SPSRetTagT, typename... SPSArgTagTs>
class ProxySpec<CI, SPSRetTagT(SPSArgTagTs...)> {

  using SPSSigT = SPSRetTagT(SPSArgTagTs...);

  template <typename X> struct IsSPSExpected : std::false_type {};
  template <typename X>
  struct IsSPSExpected<shared::SPSExpected<X>> : std::true_type {};

  /// Recovers a proxy's callee return type from the result type delivered to
  /// the client (see Proxy::ErrorRetT) and the SPS return tag:
  ///
  ///          Error with void         -> void
  ///          Error with SPSError     -> Error
  ///    Expected<T> with SPSExpected  -> Expected<T>
  ///    Expected<T> with any other    -> T
  ///
  /// The primary template handles Error.
  template <typename ErrorRetT> struct CalleeRet {
    static_assert(std::is_same_v<ErrorRetT, Error>,
                  "ErrorRetT must be Error or Expected<T>");
    using type = std::conditional_t<std::is_void_v<SPSRetTagT>, void, Error>;
  };
  template <typename T> struct CalleeRet<Expected<T>> {
    using type =
        std::conditional_t<IsSPSExpected<SPSRetTagT>::value, Expected<T>, T>;
  };

  static void consumeResult(Error &Err) { consumeError(std::move(Err)); }

  template <typename T> static void consumeResult(T &V) {}

  template <typename T> static void consumeResult(Expected<T> &E) {
    consumeError(E.takeError());
  }

public:
  static constexpr SymbolNameSpec Name = CI::Name;

  /// Dispatch function for any Proxy whose arguments and result are
  /// serializable with CI::SPSSig. Its template arguments are deduced when it
  /// is converted to the Proxy's DispatchFn type.
  template <typename ErrorRetT, typename... ArgTs>
  static void dispatch(unique_function<void(ErrorRetT)> OnComplete,
                       ExecutionSession &ES, ExecutorAddr CalleeAddr,
                       const ArgTs &...Args) {
    using CalleeRetT = typename CalleeRet<ErrorRetT>::type;
    static_assert(
        std::is_same_v<typename orc::detail::ProxyErrorRet<CalleeRetT>::type,
                       ErrorRetT>,
        "CalleeRet must invert ProxyErrorRet");

    auto Caller = [&ES, CalleeAddr](auto &&SendResult, const char *ArgData,
                                    size_t ArgSize) {
      ES.callWrapperAsync(CalleeAddr,
                          std::forward<decltype(SendResult)>(SendResult),
                          ArrayRef<char>(ArgData, ArgSize));
    };

    if constexpr (std::is_void_v<CalleeRetT>) {
      // Void result: the executor-side function produces no value, so the only
      // thing to report is the dispatch error (success if the call ran).
      shared::WrapperFunction<SPSSigT>::callAsync(Caller, std::move(OnComplete),
                                                  Args...);
    } else {
      shared::WrapperFunction<SPSSigT>::callAsync(
          Caller,
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

} // namespace llvm::orc::sps

#endif // LLVM_EXECUTIONENGINE_ORC_SPSPROXYSPEC_H
