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

#include "orc-rt/SimplePackedSerialization.h"
#include "orc-rt/WrapperFunction.h"

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

  template <typename T> static const T &toSerializable(const T &Arg) noexcept {
    return Arg;
  }

  static SPSSerializableError toSerializable(Error Err) noexcept {
    return SPSSerializableError(std::move(Err));
  }

  template <typename T>
  static SPSSerializableExpected<T> toSerializable(Expected<T> Arg) noexcept {
    return SPSSerializableExpected<T>(std::move(Arg));
  }

  template <typename... Ts> struct DeserializableTuple;

  template <typename... Ts> struct DeserializableTuple<std::tuple<Ts...>> {
    typedef std::tuple<
        std::decay_t<decltype(toSerializable(std::declval<Ts>()))>...>
        type;
  };

  template <typename... Ts>
  using DeserializableTuple_t = typename DeserializableTuple<Ts...>::type;

  template <typename T> static T &&fromSerializable(T &&Arg) noexcept {
    return std::forward<T>(Arg);
  }

  static Error fromSerializable(SPSSerializableError Err) noexcept {
    return Err.toError();
  }

  template <typename T>
  static Expected<T> fromSerializable(SPSSerializableExpected<T> Val) noexcept {
    return Val.toExpected();
  }

public:
  template <typename... ArgTs>
  std::optional<WrapperFunctionBuffer> serialize(ArgTs &&...Args) {
    return serializeImpl(toSerializable(std::forward<ArgTs>(Args))...);
  }

  template <typename ArgTuple>
  std::optional<ArgTuple> deserialize(WrapperFunctionBuffer ArgBytes) {
    assert(!ArgBytes.getOutOfBandError() &&
           "Should not attempt to deserialize out-of-band error");
    SPSInputBuffer IB(ArgBytes.data(), ArgBytes.size());
    DeserializableTuple_t<ArgTuple> Args;
    if (!SPSSerializationTraits<SPSTuple<SPSArgTs...>,
                                decltype(Args)>::deserialize(IB, Args))
      return std::nullopt;
    return std::apply(
        [](auto &&...A) {
          return std::optional<ArgTuple>(std::in_place,
                                         std::move(fromSerializable(A))...);
        },
        std::move(Args));
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
