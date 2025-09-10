//===-------- WrapperFunction.h - Wrapper function utils --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines WrapperFunctionBuffer and related APIs.
//
//===----------------------------------------------------------------------===//

#ifndef ORC_RT_WRAPPERFUNCTION_H
#define ORC_RT_WRAPPERFUNCTION_H

#include "orc-rt-c/WrapperFunction.h"
#include "orc-rt/Error.h"
#include "orc-rt/bind.h"

#include <utility>

namespace orc_rt {

/// A C++ convenience wrapper for orc_rt_WrapperFunctionBuffer. Auto-disposes
/// the contained result on destruction.
class WrapperFunctionBuffer {
public:
  /// Create a default WrapperFunctionBuffer.
  WrapperFunctionBuffer() { orc_rt_WrapperFunctionBufferInit(&B); }

  /// Create a WrapperFunctionBuffer from a WrapperFunctionBuffer. This
  /// instance takes ownership of the result object and will automatically
  /// call dispose on the result upon destruction.
  WrapperFunctionBuffer(orc_rt_WrapperFunctionBuffer B) : B(B) {}

  WrapperFunctionBuffer(const WrapperFunctionBuffer &) = delete;
  WrapperFunctionBuffer &operator=(const WrapperFunctionBuffer &) = delete;

  WrapperFunctionBuffer(WrapperFunctionBuffer &&Other) {
    orc_rt_WrapperFunctionBufferInit(&B);
    std::swap(B, Other.B);
  }

  WrapperFunctionBuffer &operator=(WrapperFunctionBuffer &&Other) {
    orc_rt_WrapperFunctionBufferDispose(&B);
    orc_rt_WrapperFunctionBufferInit(&B);
    std::swap(B, Other.B);
    return *this;
  }

  ~WrapperFunctionBuffer() { orc_rt_WrapperFunctionBufferDispose(&B); }

  /// Relinquish ownership of and return the
  /// orc_rt_WrapperFunctionBuffer.
  orc_rt_WrapperFunctionBuffer release() {
    orc_rt_WrapperFunctionBuffer Tmp;
    orc_rt_WrapperFunctionBufferInit(&Tmp);
    std::swap(B, Tmp);
    return Tmp;
  }

  /// Get a pointer to the data contained in this instance.
  char *data() { return orc_rt_WrapperFunctionBufferData(&B); }

  /// Get a pointer to the data contained is this instance.
  const char *data() const { return orc_rt_WrapperFunctionBufferConstData(&B); }

  /// Returns the size of the data contained in this instance.
  size_t size() const { return orc_rt_WrapperFunctionBufferSize(&B); }

  /// Returns true if this value is equivalent to a default-constructed
  /// WrapperFunctionBuffer.
  bool empty() const { return orc_rt_WrapperFunctionBufferEmpty(&B); }

  /// Create a WrapperFunctionBuffer with the given size and return a pointer
  /// to the underlying memory.
  static WrapperFunctionBuffer allocate(size_t Size) {
    return orc_rt_WrapperFunctionBufferAllocate(Size);
  }

  /// Copy from the given char range.
  static WrapperFunctionBuffer copyFrom(const char *Source, size_t Size) {
    return orc_rt_CreateWrapperFunctionBufferFromRange(Source, Size);
  }

  /// Copy from the given null-terminated string (includes the null-terminator).
  static WrapperFunctionBuffer copyFrom(const char *Source) {
    return orc_rt_CreateWrapperFunctionBufferFromString(Source);
  }

  /// Create an out-of-band error by copying the given string.
  static WrapperFunctionBuffer createOutOfBandError(const char *Msg) {
    return orc_rt_CreateWrapperFunctionBufferFromOutOfBandError(Msg);
  }

  /// If this value is an out-of-band error then this returns the error message,
  /// otherwise returns nullptr.
  const char *getOutOfBandError() const {
    return orc_rt_WrapperFunctionBufferGetOutOfBandError(&B);
  }

private:
  orc_rt_WrapperFunctionBuffer B;
};

namespace detail {

template <typename C>
struct WFCallableTraits
    : public WFCallableTraits<
          decltype(&std::remove_cv_t<std::remove_reference_t<C>>::operator())> {
};

template <typename RetT> struct WFCallableTraits<RetT()> {
  typedef void HeadArgType;
};

template <typename RetT, typename ArgT, typename... ArgTs>
struct WFCallableTraits<RetT(ArgT, ArgTs...)> {
  typedef ArgT HeadArgType;
  typedef std::tuple<ArgTs...> TailArgTuple;
};

template <typename ClassT, typename RetT, typename... ArgTs>
struct WFCallableTraits<RetT (ClassT::*)(ArgTs...)>
    : public WFCallableTraits<RetT(ArgTs...)> {};

template <typename ClassT, typename RetT, typename... ArgTs>
struct WFCallableTraits<RetT (ClassT::*)(ArgTs...) const>
    : public WFCallableTraits<RetT(ArgTs...)> {};

template <typename Serializer> class StructuredYieldBase {
public:
  StructuredYieldBase(orc_rt_SessionRef Session, void *CallCtx,
                      orc_rt_WrapperFunctionReturn Return, Serializer &&S)
      : Session(Session), CallCtx(CallCtx), Return(Return),
        S(std::forward<Serializer>(S)) {}

protected:
  orc_rt_SessionRef Session;
  void *CallCtx;
  orc_rt_WrapperFunctionReturn Return;
  std::decay_t<Serializer> S;
};

template <typename RetT, typename Serializer>
class StructuredYield : public StructuredYieldBase<Serializer> {
public:
  using StructuredYieldBase<Serializer>::StructuredYieldBase;
  void operator()(RetT &&R) {
    if (auto ResultBytes = this->S.resultSerializer()(std::forward<RetT>(R)))
      this->Return(this->Session, this->CallCtx, ResultBytes->release());
    else
      this->Return(this->Session, this->CallCtx,
                   WrapperFunctionBuffer::createOutOfBandError(
                       "Could not serialize wrapper function result data")
                       .release());
  }
};

template <typename Serializer>
class StructuredYield<void, Serializer>
    : public StructuredYieldBase<Serializer> {
public:
  using StructuredYieldBase<Serializer>::StructuredYieldBase;
  void operator()() {
    this->Return(this->Session, this->CallCtx,
                 WrapperFunctionBuffer().release());
  }
};

template <typename T, typename Serializer> struct ResultDeserializer;

template <typename T, typename Serializer>
struct ResultDeserializer<Expected<T>, Serializer> {
  static Expected<T> deserialize(WrapperFunctionBuffer ResultBytes,
                                 Serializer &S) {
    T Val;
    if (S.resultDeserializer()(ResultBytes, Val))
      return std::move(Val);
    else
      return make_error<StringError>("Could not deserialize result");
  }
};

template <typename Serializer> struct ResultDeserializer<Error, Serializer> {
  static Error deserialize(WrapperFunctionBuffer ResultBytes, Serializer &S) {
    assert(ResultBytes.empty());
    return Error::success();
  }
};

} // namespace detail

/// Provides call and handle utilities to simplify writing and invocation of
/// wrapper functions in C++.
struct WrapperFunction {

  /// Make a call to a wrapper function.
  ///
  /// This utility serializes and deserializes arguments and return values
  /// (using the given Serializer), and calls the wrapper function via the
  /// given Caller object.
  template <typename Caller, typename Serializer, typename ResultHandler,
            typename... ArgTs>
  static void call(Caller &&C, Serializer &&S, ResultHandler &&RH,
                   ArgTs &&...Args) {
    typedef detail::WFCallableTraits<ResultHandler> ResultHandlerTraits;
    static_assert(
        std::tuple_size_v<typename ResultHandlerTraits::TailArgTuple> == 0,
        "Expected one argument to result-handler");
    typedef typename ResultHandlerTraits::HeadArgType ResultType;

    if (auto ArgBytes = S.argumentSerializer()(std::forward<ArgTs>(Args)...)) {
      C(
          [RH = std::move(RH),
           S = std::move(S)](orc_rt_SessionRef Session,
                             WrapperFunctionBuffer ResultBytes) mutable {
            if (const char *ErrMsg = ResultBytes.getOutOfBandError())
              RH(make_error<StringError>(ErrMsg));
            else
              RH(detail::ResultDeserializer<
                  ResultType, Serializer>::deserialize(std::move(ResultBytes),
                                                       S));
          },
          std::move(*ArgBytes));
    } else
      RH(make_error<StringError>(
          "Could not serialize wrapper function call arguments"));
  }

  /// Simplifies implementation of wrapper functions in C++.
  ///
  /// This utility deserializes and serializes arguments and return values
  /// (using the given Serializer), and calls the given handler.
  template <typename Serializer, typename Handler>
  static void handle(orc_rt_SessionRef Session, void *CallCtx,
                     orc_rt_WrapperFunctionReturn Return,
                     WrapperFunctionBuffer ArgBytes, Serializer &&S,
                     Handler &&H) {
    typedef detail::WFCallableTraits<Handler> HandlerTraits;
    typedef typename HandlerTraits::HeadArgType Yield;
    typedef typename HandlerTraits::TailArgTuple ArgTuple;
    typedef typename detail::WFCallableTraits<Yield>::HeadArgType RetType;

    if (ArgBytes.getOutOfBandError())
      return Return(Session, CallCtx, ArgBytes.release());

    ArgTuple Args;
    if (std::apply(bind_front(S.argumentDeserializer(), std::move(ArgBytes)),
                   Args))
      std::apply(bind_front(std::forward<Handler>(H),
                            detail::StructuredYield<RetType, Serializer>(
                                Session, CallCtx, Return, std::move(S))),
                 std::move(Args));
    else
      Return(Session, CallCtx,
             WrapperFunctionBuffer::createOutOfBandError(
                 "Could not deserialize wrapper function arg data")
                 .release());
  }
};

} // namespace orc_rt

#endif // ORC_RT_WRAPPERFUNCTION_H
