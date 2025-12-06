//===------ CallViaEPC.h - Call wrapper functions via EPC -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Call executor functions with common signatures via
// ExecutorProcessControl::callWrapperAsync.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_EXECUTIONENGINE_ORC_CALLVIAEPC_H
#define LLVM_EXECUTIONENGINE_ORC_CALLVIAEPC_H

#include "llvm/ExecutionEngine/Orc/CallableTraitsHelper.h"
#include "llvm/ExecutionEngine/Orc/ExecutorProcessControl.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/MSVCErrorWorkarounds.h"

#include <type_traits>

namespace llvm::orc {

namespace detail {

template <typename HandlerArgT> struct CallViaEPCRetValueTraits;

template <typename RetT> struct CallViaEPCRetValueTraits<Expected<RetT>> {
  using value_type = RetT;
};

template <> struct CallViaEPCRetValueTraits<Error> {
  using value_type = void;
};

template <typename RetT> struct CallViaEPCRetValueTraits<MSVCPExpected<RetT>> {
  using value_type = RetT;
};

template <> struct CallViaEPCRetValueTraits<MSVCPError> {
  using value_type = void;
};

// Helper to extract the argument type from a handler callable.
template <typename HandlerT> struct CallViaEPCHandlerTraits {
  using ArgInfo = CallableArgInfo<HandlerT>;
  using ArgsTuple = typename ArgInfo::ArgsTupleType;
  static_assert(std::tuple_size_v<ArgsTuple> == 1,
                "Handler must take exactly one argument");
  using HandlerArgT = std::tuple_element_t<0, ArgsTuple>;
  using RetT = typename CallViaEPCRetValueTraits<
      std::remove_cv_t<std::remove_reference_t<HandlerArgT>>>::value_type;
};

} // namespace detail

/// Call a wrapper function via EPC asynchronously.
template <typename HandlerFn, typename Serializer, typename... ArgTs>
std::enable_if_t<std::is_invocable_v<HandlerFn, Error>>
callViaEPC(HandlerFn &&H, ExecutorProcessControl &EPC, Serializer S,
           ExecutorSymbolDef Fn, ArgTs &&...Args) {
  using RetT = typename detail::CallViaEPCHandlerTraits<HandlerFn>::RetT;

  if (auto ArgBytes = S.serialize(std::forward<ArgTs>(Args)...))
    EPC.callWrapperAsync(
        Fn.getAddress(),
        [S = std::move(S), H = std::forward<HandlerFn>(H)](
            shared::WrapperFunctionResult R) mutable {
          if (const char *ErrMsg = R.getOutOfBandError())
            H(make_error<StringError>(ErrMsg, inconvertibleErrorCode()));
          else
            H(S.template deserialize<RetT>(std::move(R)));
        },
        {ArgBytes->data(), ArgBytes->size()});
  else
    H(ArgBytes.takeError());
}

/// Call a wrapper function via EPC synchronously using the given promise.
///
/// This performs a blocking call by making an asynchronous call to set the
/// promise and waiting on a future.
///
/// Blocking calls should only be used for convenience by ORC clients, never
/// internally.
template <typename PromiseT, typename Serializer, typename... ArgTs>
std::enable_if_t<!std::is_invocable_v<PromiseT, Error>,
                 decltype(std::declval<PromiseT>().get_future().get())>
callViaEPC(PromiseT &&P, ExecutorProcessControl &EPC, Serializer S,
           ExecutorSymbolDef Fn, ArgTs &&...Args) {
  auto F = P.get_future();
  using RetT = decltype(F.get());
  callViaEPC([P = std::move(P)](RetT R) mutable { P.set_value(std::move(R)); },
             EPC, std::move(S), std::move(Fn), std::forward<ArgTs>(Args)...);
  return F.get();
}

/// Encapsulates calls via EPC to any function that's compatible with the given
/// serialization scheme.
template <typename Serializer> class EPCCaller {
public:
  EPCCaller(ExecutorProcessControl &EPC, Serializer &&S)
      : EPC(EPC), S(std::move(S)) {}

  // TODO: Add an ExecutionSession constructor once ExecutionSession has been
  //       moved to its own header.

  // Make a call to the given function using callViaEPC.
  //
  // The PromiseOrHandlerT value is forwarded. Its type will determine both the
  // return value type and the dispatch method (asynchronous vs synchronous).
  template <typename PromiseOrHandlerT, typename... ArgTs>
  decltype(auto) operator()(PromiseOrHandlerT &&R, ExecutorSymbolDef Fn,
                            ArgTs &&...Args) {
    return callViaEPC(std::forward<PromiseOrHandlerT>(R), EPC, S, Fn,
                      std::forward<ArgTs>(Args)...);
  }

private:
  ExecutorProcessControl &EPC;
  Serializer S;
};

/// Encapsulates calls via EPC to a specific function, using the given
/// serialization scheme.
template <typename Serializer> class EPCCall {
public:
  EPCCall(ExecutorProcessControl &EPC, Serializer &&S, ExecutorSymbolDef Fn)
      : Caller(EPC, std::move(S)), Fn(std::move(Fn)) {}

  // TODO: Add an ExecutionSession constructor once ExecutionSession has been
  //       moved to its own header.

  // Make a call using callViaEPC.
  //
  // The PromiseOrHandlerT value is forwarded. Its type will determine both the
  // return value type and the dispatch method (asynchronous vs synchronous).
  template <typename PromiseOrHandlerT, typename... ArgTs>
  decltype(auto) operator()(PromiseOrHandlerT &&R, ArgTs &&...Args) {
    return Caller(std::forward<PromiseOrHandlerT>(R), Fn,
                  std::forward<ArgTs>(Args)...);
  }

private:
  EPCCaller<Serializer> Caller;
  ExecutorSymbolDef Fn;
};

} // namespace llvm::orc

#endif // LLVM_EXECUTIONENGINE_ORC_CALLVIAEPC_H
