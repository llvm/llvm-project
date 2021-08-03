//===- WrapperFunctionUtils.h - Utilities for wrapper functions -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A buffer for serialized results.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_EXECUTIONENGINE_ORC_SHARED_WRAPPERFUNCTIONUTILS_H
#define LLVM_EXECUTIONENGINE_ORC_SHARED_WRAPPERFUNCTIONUTILS_H

#include "llvm/ExecutionEngine/Orc/Shared/SimplePackedSerialization.h"
#include "llvm/Support/Error.h"

#include <type_traits>

namespace llvm {
namespace orc {
namespace shared {

namespace detail {

// DO NOT USE DIRECTLY.
// Must be kept in-sync with compiler-rt/lib/orc/c-api.h.
union CWrapperFunctionResultDataUnion {
  char *ValuePtr;
  char Value[sizeof(ValuePtr)];
};

// DO NOT USE DIRECTLY.
// Must be kept in-sync with compiler-rt/lib/orc/c-api.h.
typedef struct {
  CWrapperFunctionResultDataUnion Data;
  size_t Size;
} CWrapperFunctionResult;

} // end namespace detail

/// C++ wrapper function result: Same as CWrapperFunctionResult but
/// auto-releases memory.
class WrapperFunctionResult {
public:
  /// Create a default WrapperFunctionResult.
  WrapperFunctionResult() { init(R); }

  /// Create a WrapperFunctionResult by taking ownership of a
  /// detail::CWrapperFunctionResult.
  ///
  /// Warning: This should only be used by clients writing wrapper-function
  /// caller utilities (like TargetProcessControl).
  WrapperFunctionResult(detail::CWrapperFunctionResult R) : R(R) {
    // Reset R.
    init(R);
  }

  WrapperFunctionResult(const WrapperFunctionResult &) = delete;
  WrapperFunctionResult &operator=(const WrapperFunctionResult &) = delete;

  WrapperFunctionResult(WrapperFunctionResult &&Other) {
    init(R);
    std::swap(R, Other.R);
  }

  WrapperFunctionResult &operator=(WrapperFunctionResult &&Other) {
    WrapperFunctionResult Tmp(std::move(Other));
    std::swap(R, Tmp.R);
    return *this;
  }

  ~WrapperFunctionResult() {
    if ((R.Size > sizeof(R.Data.Value)) ||
        (R.Size == 0 && R.Data.ValuePtr != nullptr))
      free(R.Data.ValuePtr);
  }

  /// Release ownership of the contained detail::CWrapperFunctionResult.
  /// Warning: Do not use -- this method will be removed in the future. It only
  /// exists to temporarily support some code that will eventually be moved to
  /// the ORC runtime.
  detail::CWrapperFunctionResult release() {
    detail::CWrapperFunctionResult Tmp;
    init(Tmp);
    std::swap(R, Tmp);
    return Tmp;
  }

  /// Get a pointer to the data contained in this instance.
  const char *data() const {
    assert((R.Size != 0 || R.Data.ValuePtr == nullptr) &&
           "Cannot get data for out-of-band error value");
    return R.Size > sizeof(R.Data.Value) ? R.Data.ValuePtr : R.Data.Value;
  }

  /// Returns the size of the data contained in this instance.
  size_t size() const {
    assert((R.Size != 0 || R.Data.ValuePtr == nullptr) &&
           "Cannot get data for out-of-band error value");
    return R.Size;
  }

  /// Returns true if this value is equivalent to a default-constructed
  /// WrapperFunctionResult.
  bool empty() const { return R.Size == 0 && R.Data.ValuePtr == nullptr; }

  /// Create a WrapperFunctionResult with the given size and return a pointer
  /// to the underlying memory.
  static char *allocate(WrapperFunctionResult &WFR, size_t Size) {
    // Reset.
    WFR = WrapperFunctionResult();
    WFR.R.Size = Size;
    char *DataPtr;
    if (WFR.R.Size > sizeof(WFR.R.Data.Value)) {
      DataPtr = (char *)malloc(WFR.R.Size);
      WFR.R.Data.ValuePtr = DataPtr;
    } else
      DataPtr = WFR.R.Data.Value;
    return DataPtr;
  }

  /// Copy from the given char range.
  static WrapperFunctionResult copyFrom(const char *Source, size_t Size) {
    WrapperFunctionResult WFR;
    char *DataPtr = allocate(WFR, Size);
    memcpy(DataPtr, Source, Size);
    return WFR;
  }

  /// Copy from the given null-terminated string (includes the null-terminator).
  static WrapperFunctionResult copyFrom(const char *Source) {
    return copyFrom(Source, strlen(Source) + 1);
  }

  /// Copy from the given std::string (includes the null terminator).
  static WrapperFunctionResult copyFrom(const std::string &Source) {
    return copyFrom(Source.c_str());
  }

  /// Create an out-of-band error by copying the given string.
  static WrapperFunctionResult createOutOfBandError(const char *Msg) {
    // Reset.
    WrapperFunctionResult WFR;
    char *Tmp = (char *)malloc(strlen(Msg) + 1);
    strcpy(Tmp, Msg);
    WFR.R.Data.ValuePtr = Tmp;
    return WFR;
  }

  /// Create an out-of-band error by copying the given string.
  static WrapperFunctionResult createOutOfBandError(const std::string &Msg) {
    return createOutOfBandError(Msg.c_str());
  }

  /// If this value is an out-of-band error then this returns the error message,
  /// otherwise returns nullptr.
  const char *getOutOfBandError() const {
    return R.Size == 0 ? R.Data.ValuePtr : nullptr;
  }

private:
  static void init(detail::CWrapperFunctionResult &R) {
    R.Data.ValuePtr = nullptr;
    R.Size = 0;
  }

  detail::CWrapperFunctionResult R;
};

namespace detail {

template <typename SPSArgListT, typename... ArgTs>
WrapperFunctionResult
serializeViaSPSToWrapperFunctionResult(const ArgTs &...Args) {
  WrapperFunctionResult Result;
  char *DataPtr =
      WrapperFunctionResult::allocate(Result, SPSArgListT::size(Args...));
  SPSOutputBuffer OB(DataPtr, Result.size());
  if (!SPSArgListT::serialize(OB, Args...))
    return WrapperFunctionResult::createOutOfBandError(
        "Error serializing arguments to blob in call");
  return Result;
}

template <typename RetT> class WrapperFunctionHandlerCaller {
public:
  template <typename HandlerT, typename ArgTupleT, std::size_t... I>
  static decltype(auto) call(HandlerT &&H, ArgTupleT &Args,
                             std::index_sequence<I...>) {
    return std::forward<HandlerT>(H)(std::get<I>(Args)...);
  }
};

template <> class WrapperFunctionHandlerCaller<void> {
public:
  template <typename HandlerT, typename ArgTupleT, std::size_t... I>
  static SPSEmpty call(HandlerT &&H, ArgTupleT &Args,
                       std::index_sequence<I...>) {
    std::forward<HandlerT>(H)(std::get<I>(Args)...);
    return SPSEmpty();
  }
};

template <typename WrapperFunctionImplT,
          template <typename> class ResultSerializer, typename... SPSTagTs>
class WrapperFunctionHandlerHelper
    : public WrapperFunctionHandlerHelper<
          decltype(&std::remove_reference_t<WrapperFunctionImplT>::operator()),
          ResultSerializer, SPSTagTs...> {};

template <typename RetT, typename... ArgTs,
          template <typename> class ResultSerializer, typename... SPSTagTs>
class WrapperFunctionHandlerHelper<RetT(ArgTs...), ResultSerializer,
                                   SPSTagTs...> {
public:
  using ArgTuple = std::tuple<std::decay_t<ArgTs>...>;
  using ArgIndices = std::make_index_sequence<std::tuple_size<ArgTuple>::value>;

  template <typename HandlerT>
  static WrapperFunctionResult apply(HandlerT &&H, const char *ArgData,
                                     size_t ArgSize) {
    ArgTuple Args;
    if (!deserialize(ArgData, ArgSize, Args, ArgIndices{}))
      return WrapperFunctionResult::createOutOfBandError(
          "Could not deserialize arguments for wrapper function call");

    auto HandlerResult = WrapperFunctionHandlerCaller<RetT>::call(
        std::forward<HandlerT>(H), Args, ArgIndices{});

    return ResultSerializer<decltype(HandlerResult)>::serialize(
        std::move(HandlerResult));
  }

private:
  template <std::size_t... I>
  static bool deserialize(const char *ArgData, size_t ArgSize, ArgTuple &Args,
                          std::index_sequence<I...>) {
    SPSInputBuffer IB(ArgData, ArgSize);
    return SPSArgList<SPSTagTs...>::deserialize(IB, std::get<I>(Args)...);
  }
};

// Map function pointers to function types.
template <typename RetT, typename... ArgTs,
          template <typename> class ResultSerializer, typename... SPSTagTs>
class WrapperFunctionHandlerHelper<RetT (*)(ArgTs...), ResultSerializer,
                                   SPSTagTs...>
    : public WrapperFunctionHandlerHelper<RetT(ArgTs...), ResultSerializer,
                                          SPSTagTs...> {};

// Map non-const member function types to function types.
template <typename ClassT, typename RetT, typename... ArgTs,
          template <typename> class ResultSerializer, typename... SPSTagTs>
class WrapperFunctionHandlerHelper<RetT (ClassT::*)(ArgTs...), ResultSerializer,
                                   SPSTagTs...>
    : public WrapperFunctionHandlerHelper<RetT(ArgTs...), ResultSerializer,
                                          SPSTagTs...> {};

// Map const member function types to function types.
template <typename ClassT, typename RetT, typename... ArgTs,
          template <typename> class ResultSerializer, typename... SPSTagTs>
class WrapperFunctionHandlerHelper<RetT (ClassT::*)(ArgTs...) const,
                                   ResultSerializer, SPSTagTs...>
    : public WrapperFunctionHandlerHelper<RetT(ArgTs...), ResultSerializer,
                                          SPSTagTs...> {};

template <typename WrapperFunctionImplT,
          template <typename> class ResultSerializer, typename... SPSTagTs>
class WrapperFunctionAsyncHandlerHelper
    : public WrapperFunctionAsyncHandlerHelper<
          decltype(&std::remove_reference_t<WrapperFunctionImplT>::operator()),
          ResultSerializer, SPSTagTs...> {};

template <typename RetT, typename SendResultT, typename... ArgTs,
          template <typename> class ResultSerializer, typename... SPSTagTs>
class WrapperFunctionAsyncHandlerHelper<RetT(SendResultT, ArgTs...),
                                        ResultSerializer, SPSTagTs...> {
public:
  using ArgTuple = std::tuple<std::decay_t<ArgTs>...>;
  using ArgIndices = std::make_index_sequence<std::tuple_size<ArgTuple>::value>;

  template <typename HandlerT, typename SendWrapperFunctionResultT>
  static void applyAsync(HandlerT &&H,
                         SendWrapperFunctionResultT &&SendWrapperFunctionResult,
                         const char *ArgData, size_t ArgSize) {
    ArgTuple Args;
    if (!deserialize(ArgData, ArgSize, Args, ArgIndices{})) {
      SendWrapperFunctionResult(WrapperFunctionResult::createOutOfBandError(
          "Could not deserialize arguments for wrapper function call"));
      return;
    }

    auto SendResult =
        [SendWFR = std::move(SendWrapperFunctionResult)](auto Result) mutable {
          using ResultT = decltype(Result);
          SendWFR(ResultSerializer<ResultT>::serialize(std::move(Result)));
        };

    callAsync(std::forward<HandlerT>(H), std::move(SendResult), std::move(Args),
              ArgIndices{});
  }

private:
  template <std::size_t... I>
  static bool deserialize(const char *ArgData, size_t ArgSize, ArgTuple &Args,
                          std::index_sequence<I...>) {
    SPSInputBuffer IB(ArgData, ArgSize);
    return SPSArgList<SPSTagTs...>::deserialize(IB, std::get<I>(Args)...);
  }

  template <typename HandlerT, typename SerializeAndSendResultT,
            typename ArgTupleT, std::size_t... I>
  static void callAsync(HandlerT &&H,
                        SerializeAndSendResultT &&SerializeAndSendResult,
                        ArgTupleT Args, std::index_sequence<I...>) {
    return std::forward<HandlerT>(H)(std::move(SerializeAndSendResult),
                                     std::move(std::get<I>(Args))...);
  }
};

// Map function pointers to function types.
template <typename RetT, typename... ArgTs,
          template <typename> class ResultSerializer, typename... SPSTagTs>
class WrapperFunctionAsyncHandlerHelper<RetT (*)(ArgTs...), ResultSerializer,
                                        SPSTagTs...>
    : public WrapperFunctionAsyncHandlerHelper<RetT(ArgTs...), ResultSerializer,
                                               SPSTagTs...> {};

// Map non-const member function types to function types.
template <typename ClassT, typename RetT, typename... ArgTs,
          template <typename> class ResultSerializer, typename... SPSTagTs>
class WrapperFunctionAsyncHandlerHelper<RetT (ClassT::*)(ArgTs...),
                                        ResultSerializer, SPSTagTs...>
    : public WrapperFunctionAsyncHandlerHelper<RetT(ArgTs...), ResultSerializer,
                                               SPSTagTs...> {};

// Map const member function types to function types.
template <typename ClassT, typename RetT, typename... ArgTs,
          template <typename> class ResultSerializer, typename... SPSTagTs>
class WrapperFunctionAsyncHandlerHelper<RetT (ClassT::*)(ArgTs...) const,
                                        ResultSerializer, SPSTagTs...>
    : public WrapperFunctionAsyncHandlerHelper<RetT(ArgTs...), ResultSerializer,
                                               SPSTagTs...> {};

template <typename SPSRetTagT, typename RetT> class ResultSerializer {
public:
  static WrapperFunctionResult serialize(RetT Result) {
    return serializeViaSPSToWrapperFunctionResult<SPSArgList<SPSRetTagT>>(
        Result);
  }
};

template <typename SPSRetTagT> class ResultSerializer<SPSRetTagT, Error> {
public:
  static WrapperFunctionResult serialize(Error Err) {
    return serializeViaSPSToWrapperFunctionResult<SPSArgList<SPSRetTagT>>(
        toSPSSerializable(std::move(Err)));
  }
};

template <typename SPSRetTagT, typename T>
class ResultSerializer<SPSRetTagT, Expected<T>> {
public:
  static WrapperFunctionResult serialize(Expected<T> E) {
    return serializeViaSPSToWrapperFunctionResult<SPSArgList<SPSRetTagT>>(
        toSPSSerializable(std::move(E)));
  }
};

template <typename SPSRetTagT, typename RetT> class ResultDeserializer {
public:
  static RetT makeValue() { return RetT(); }
  static void makeSafe(RetT &Result) {}

  static Error deserialize(RetT &Result, const char *ArgData, size_t ArgSize) {
    SPSInputBuffer IB(ArgData, ArgSize);
    if (!SPSArgList<SPSRetTagT>::deserialize(IB, Result))
      return make_error<StringError>(
          "Error deserializing return value from blob in call",
          inconvertibleErrorCode());
    return Error::success();
  }
};

template <> class ResultDeserializer<SPSError, Error> {
public:
  static Error makeValue() { return Error::success(); }
  static void makeSafe(Error &Err) { cantFail(std::move(Err)); }

  static Error deserialize(Error &Err, const char *ArgData, size_t ArgSize) {
    SPSInputBuffer IB(ArgData, ArgSize);
    SPSSerializableError BSE;
    if (!SPSArgList<SPSError>::deserialize(IB, BSE))
      return make_error<StringError>(
          "Error deserializing return value from blob in call",
          inconvertibleErrorCode());
    Err = fromSPSSerializable(std::move(BSE));
    return Error::success();
  }
};

template <typename SPSTagT, typename T>
class ResultDeserializer<SPSExpected<SPSTagT>, Expected<T>> {
public:
  static Expected<T> makeValue() { return T(); }
  static void makeSafe(Expected<T> &E) { cantFail(E.takeError()); }

  static Error deserialize(Expected<T> &E, const char *ArgData,
                           size_t ArgSize) {
    SPSInputBuffer IB(ArgData, ArgSize);
    SPSSerializableExpected<T> BSE;
    if (!SPSArgList<SPSExpected<SPSTagT>>::deserialize(IB, BSE))
      return make_error<StringError>(
          "Error deserializing return value from blob in call",
          inconvertibleErrorCode());
    E = fromSPSSerializable(std::move(BSE));
    return Error::success();
  }
};

template <typename SPSRetTagT, typename RetT> class AsyncCallResultHelper {
  // Did you forget to use Error / Expected in your handler?
};

} // end namespace detail

template <typename SPSSignature> class WrapperFunction;

template <typename SPSRetTagT, typename... SPSTagTs>
class WrapperFunction<SPSRetTagT(SPSTagTs...)> {
private:
  template <typename RetT>
  using ResultSerializer = detail::ResultSerializer<SPSRetTagT, RetT>;

public:
  /// Call a wrapper function. Caller should be callable as
  /// WrapperFunctionResult Fn(const char *ArgData, size_t ArgSize);
  template <typename CallerFn, typename RetT, typename... ArgTs>
  static Error call(const CallerFn &Caller, RetT &Result,
                    const ArgTs &...Args) {

    // RetT might be an Error or Expected value. Set the checked flag now:
    // we don't want the user to have to check the unused result if this
    // operation fails.
    detail::ResultDeserializer<SPSRetTagT, RetT>::makeSafe(Result);

    auto ArgBuffer =
        detail::serializeViaSPSToWrapperFunctionResult<SPSArgList<SPSTagTs...>>(
            Args...);
    if (const char *ErrMsg = ArgBuffer.getOutOfBandError())
      return make_error<StringError>(ErrMsg, inconvertibleErrorCode());

    WrapperFunctionResult ResultBuffer =
        Caller(ArgBuffer.data(), ArgBuffer.size());
    if (auto ErrMsg = ResultBuffer.getOutOfBandError())
      return make_error<StringError>(ErrMsg, inconvertibleErrorCode());

    return detail::ResultDeserializer<SPSRetTagT, RetT>::deserialize(
        Result, ResultBuffer.data(), ResultBuffer.size());
  }

  /// Call an async wrapper function.
  /// Caller should be callable as
  /// void Fn(unique_function<void(WrapperFunctionResult)> SendResult,
  ///         WrapperFunctionResult ArgBuffer);
  template <typename AsyncCallerFn, typename SendDeserializedResultFn,
            typename... ArgTs>
  static void callAsync(AsyncCallerFn &&Caller,
                        SendDeserializedResultFn &&SendDeserializedResult,
                        const ArgTs &...Args) {
    using RetT = typename std::tuple_element<
        1, typename detail::WrapperFunctionHandlerHelper<
               std::remove_reference_t<SendDeserializedResultFn>,
               ResultSerializer, SPSRetTagT>::ArgTuple>::type;

    auto ArgBuffer =
        detail::serializeViaSPSToWrapperFunctionResult<SPSArgList<SPSTagTs...>>(
            Args...);
    if (auto *ErrMsg = ArgBuffer.getOutOfBandError()) {
      SendDeserializedResult(
          make_error<StringError>(ErrMsg, inconvertibleErrorCode()),
          detail::ResultDeserializer<SPSRetTagT, RetT>::makeValue());
      return;
    }

    auto SendSerializedResult = [SDR = std::move(SendDeserializedResult)](
                                    WrapperFunctionResult R) {
      RetT RetVal = detail::ResultDeserializer<SPSRetTagT, RetT>::makeValue();
      detail::ResultDeserializer<SPSRetTagT, RetT>::makeSafe(RetVal);

      SPSInputBuffer IB(R.data(), R.size());
      if (auto Err = detail::ResultDeserializer<SPSRetTagT, RetT>::deserialize(
              RetVal, R.data(), R.size()))
        SDR(std::move(Err), std::move(RetVal));

      SDR(Error::success(), std::move(RetVal));
    };

    Caller(std::move(SendSerializedResult), ArgBuffer.data(), ArgBuffer.size());
  }

  /// Handle a call to a wrapper function.
  template <typename HandlerT>
  static WrapperFunctionResult handle(const char *ArgData, size_t ArgSize,
                                      HandlerT &&Handler) {
    using WFHH =
        detail::WrapperFunctionHandlerHelper<std::remove_reference_t<HandlerT>,
                                             ResultSerializer, SPSTagTs...>;
    return WFHH::apply(std::forward<HandlerT>(Handler), ArgData, ArgSize);
  }

  /// Handle a call to an async wrapper function.
  template <typename HandlerT, typename SendResultT>
  static void handleAsync(const char *ArgData, size_t ArgSize,
                          HandlerT &&Handler, SendResultT &&SendResult) {
    using WFAHH = detail::WrapperFunctionAsyncHandlerHelper<
        std::remove_reference_t<HandlerT>, ResultSerializer, SPSTagTs...>;
    WFAHH::applyAsync(std::forward<HandlerT>(Handler),
                      std::forward<SendResultT>(SendResult), ArgData, ArgSize);
  }

private:
  template <typename T> static const T &makeSerializable(const T &Value) {
    return Value;
  }

  static detail::SPSSerializableError makeSerializable(Error Err) {
    return detail::toSPSSerializable(std::move(Err));
  }

  template <typename T>
  static detail::SPSSerializableExpected<T> makeSerializable(Expected<T> E) {
    return detail::toSPSSerializable(std::move(E));
  }
};

template <typename... SPSTagTs>
class WrapperFunction<void(SPSTagTs...)>
    : private WrapperFunction<SPSEmpty(SPSTagTs...)> {

public:
  template <typename CallerFn, typename... ArgTs>
  static Error call(const CallerFn &Caller, const ArgTs &...Args) {
    SPSEmpty BE;
    return WrapperFunction<SPSEmpty(SPSTagTs...)>::call(Caller, BE, Args...);
  }

  using WrapperFunction<SPSEmpty(SPSTagTs...)>::handle;
  using WrapperFunction<SPSEmpty(SPSTagTs...)>::handleAsync;
};

} // end namespace shared
} // end namespace orc
} // end namespace llvm

#endif // LLVM_EXECUTIONENGINE_ORC_SHARED_WRAPPERFUNCTIONUTILS_H
