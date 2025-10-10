//===--- SPSWrapperFunction.h -- SPS-serializing Wrapper utls ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Utilities for calling / handling wrapper functions that use SPS
// serialization.
//
//===----------------------------------------------------------------------===//

#ifndef ORC_RT_SPSWRAPPERFUNCTION_H
#define ORC_RT_SPSWRAPPERFUNCTION_H

#include "orc-rt/Compiler.h"
#include "orc-rt/SimplePackedSerialization.h"
#include "orc-rt/WrapperFunction.h"

#define ORC_RT_SPS_INTERFACE ORC_RT_INTERFACE

namespace orc_rt {
namespace detail {

template <typename... SPSArgTs> struct WFSPSHelper {
private:
  template <typename... SerializableArgTs>
  std::optional<WrapperFunctionBuffer>
  serializeImpl(const SerializableArgTs &...Args) {
    auto R =
        WrapperFunctionBuffer::allocate(SPSArgList<SPSArgTs...>::size(Args...));
    SPSOutputBuffer OB(R.data(), R.size());
    if (!SPSArgList<SPSArgTs...>::serialize(OB, Args...))
      return std::nullopt;
    return std::move(R);
  }

  template <typename T> struct Serializable {
    typedef std::decay_t<T> serializable_type;
    static const T &to(const T &Arg) noexcept { return Arg; }
    static T &&from(T &&Arg) noexcept { return std::forward<T>(Arg); }
  };

  template <typename T> struct Serializable<T *> {
    typedef ExecutorAddr serializable_type;
    static ExecutorAddr to(T *Arg) { return ExecutorAddr::fromPtr(Arg); }
    static T *from(ExecutorAddr A) { return A.toPtr<T *>(); }
  };

  template <> struct Serializable<Error> {
    typedef SPSSerializableError serializable_type;
    static SPSSerializableError to(Error Err) {
      return SPSSerializableError(std::move(Err));
    }
    static Error from(SPSSerializableError Err) { return Err.toError(); }
  };

  template <typename T> struct Serializable<Expected<T>> {
    typedef SPSSerializableExpected<T> serializable_type;
    static SPSSerializableExpected<T> to(Expected<T> Val) {
      return SPSSerializableExpected<T>(std::move(Val));
    }
    static Expected<T> from(SPSSerializableExpected<T> Val) {
      return Val.toExpected();
    }
  };

  template <typename T> struct Serializable<Expected<T *>> {
    typedef SPSSerializableExpected<ExecutorAddr> serializable_type;
    static SPSSerializableExpected<ExecutorAddr> to(Expected<T *> Val) {
      return SPSSerializableExpected<ExecutorAddr>(
          Val ? Expected<ExecutorAddr>(ExecutorAddr::fromPtr(*Val))
              : Expected<ExecutorAddr>(Val.takeError()));
    }
    static Expected<T *> from(SPSSerializableExpected<ExecutorAddr> Val) {
      if (auto Tmp = Val.toExpected())
        return Tmp->toPtr<T *>();
      else
        return Tmp.takeError();
    }
  };

  template <typename... Ts> struct DeserializableTuple;

  template <typename... Ts> struct DeserializableTuple<std::tuple<Ts...>> {
    typedef std::tuple<typename Serializable<Ts>::serializable_type...> type;
  };

  template <typename... Ts>
  using DeserializableTuple_t = typename DeserializableTuple<Ts...>::type;

  template <typename ArgTuple, typename... SerializableArgs, std::size_t... Is>
  std::optional<ArgTuple>
  applySerializationConversions(std::tuple<SerializableArgs...> &Inputs,
                                std::index_sequence<Is...>) {
    static_assert(sizeof...(SerializableArgs) ==
                      std::index_sequence<Is...>::size(),
                  "Tuple sizes don't match");
    return std::optional<ArgTuple>(
        std::in_place, Serializable<std::tuple_element_t<Is, ArgTuple>>::from(
                           std::move(std::get<Is>(Inputs)))...);
  }

public:
  template <typename... ArgTs>
  std::optional<WrapperFunctionBuffer> serialize(ArgTs &&...Args) {
    return serializeImpl(
        Serializable<std::decay_t<ArgTs>>::to(std::forward<ArgTs>(Args))...);
  }

  template <typename ArgTuple>
  std::optional<ArgTuple> deserialize(const WrapperFunctionBuffer &ArgBytes) {
    assert(!ArgBytes.getOutOfBandError() &&
           "Should not attempt to deserialize out-of-band error");
    SPSInputBuffer IB(ArgBytes.data(), ArgBytes.size());
    DeserializableTuple_t<ArgTuple> Args;
    if (!SPSSerializationTraits<SPSTuple<SPSArgTs...>,
                                decltype(Args)>::deserialize(IB, Args))
      return std::nullopt;
    return applySerializationConversions<ArgTuple>(
        Args, std::make_index_sequence<std::tuple_size_v<ArgTuple>>());
  }
};

} // namespace detail

template <typename SPSSig> struct WrapperFunctionSPSSerializer;

template <typename SPSRetT, typename... SPSArgTs>
struct WrapperFunctionSPSSerializer<SPSRetT(SPSArgTs...)> {
  static detail::WFSPSHelper<SPSArgTs...> arguments() noexcept { return {}; }
  static detail::WFSPSHelper<SPSRetT> result() noexcept { return {}; }
};

/// Provides call and handle utilities to simplify writing and invocation of
/// wrapper functions that use SimplePackedSerialization to serialize and
/// deserialize their arguments and return values.
template <typename SPSSig> struct SPSWrapperFunction {
  template <typename Caller, typename ResultHandler, typename... ArgTs>
  static void call(Caller &&C, ResultHandler &&RH, ArgTs &&...Args) {
    WrapperFunction::call(
        std::forward<Caller>(C), WrapperFunctionSPSSerializer<SPSSig>(),
        std::forward<ResultHandler>(RH), std::forward<ArgTs>(Args)...);
  }

  template <typename Handler>
  static void handle(orc_rt_SessionRef Session, void *CallCtx,
                     orc_rt_WrapperFunctionReturn Return,
                     WrapperFunctionBuffer ArgBytes, Handler &&H) {
    WrapperFunction::handle(Session, CallCtx, Return, std::move(ArgBytes),
                            WrapperFunctionSPSSerializer<SPSSig>(),
                            std::forward<Handler>(H));
  }
};

} // namespace orc_rt

#endif // ORC_RT_SPSWRAPPERFUNCTION_H
