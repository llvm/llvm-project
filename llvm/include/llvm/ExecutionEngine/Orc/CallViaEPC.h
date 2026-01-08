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

namespace llvm::orc {

namespace detail {

// Helper to extract the Expected<T> argument type from a handler callable.
template <typename HandlerT> struct HandlerTraits {
  using ArgInfo = CallableArgInfo<HandlerT>;
  using ArgsTuple = typename ArgInfo::ArgsTupleType;
  static_assert(std::tuple_size_v<ArgsTuple> == 1,
                "Handler must take exactly one argument");
  using ExpectedArgType = std::tuple_element_t<0, ArgsTuple>;
  using RetT = typename std::remove_cv_t<
      std::remove_reference_t<ExpectedArgType>>::value_type;
};

} // namespace detail

/// Call a wrapper function via EPC.
template <typename HandlerT, typename Serializer, typename... ArgTs>
void callViaEPC(HandlerT &&H, ExecutorProcessControl &EPC, Serializer S,
                ExecutorSymbolDef Fn, ArgTs &&...Args) {
  using RetT = typename detail::HandlerTraits<HandlerT>::RetT;

  if (auto ArgBytes = S.serialize(std::forward<ArgTs>(Args)...))
    EPC.callWrapperAsync(
        Fn.getAddress(),
        [S = std::move(S), H = std::forward<HandlerT>(H)](
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

/// Encapsulates calls via EPC to any function that's compatible with the given
/// serialization scheme.
template <typename Serializer> class EPCCaller {
public:
  EPCCaller(ExecutorProcessControl &EPC, Serializer &&S)
      : EPC(EPC), S(std::move(S)) {}

  // TODO: Add an ExecutionSession constructor once ExecutionSession has been
  //       moved to its own header.

  // Async call version.
  template <typename HandlerT, typename... ArgTs>
  void operator()(HandlerT &&H, ExecutorSymbolDef Fn, ArgTs &&...Args) {
    callViaEPC(std::forward<HandlerT>(H), EPC, S, Fn,
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

  template <typename HandlerT, typename... ArgTs>
  void operator()(HandlerT &&H, ArgTs &&...Args) {
    Caller(std::forward<HandlerT>(H), Fn, std::forward<ArgTs>(Args)...);
  }

private:
  EPCCaller<Serializer> Caller;
  ExecutorSymbolDef Fn;
};

} // namespace llvm::orc

#endif // LLVM_EXECUTIONENGINE_ORC_CALLVIAEPC_H
