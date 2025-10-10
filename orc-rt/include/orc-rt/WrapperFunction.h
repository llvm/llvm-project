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
#include "orc-rt/CallableTraitsHelper.h"
#include "orc-rt/Error.h"
#include "orc-rt/ExecutorAddress.h"
#include "orc-rt/bind.h"
#include "orc-rt/move_only_function.h"

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

template <typename RetT, typename ReturnT, typename... ArgTs>
struct WFHandlerTraitsImpl {
  static_assert(std::is_void_v<RetT>,
                "Async wrapper function handler must return void");
  typedef ReturnT YieldType;
  typedef std::tuple<std::decay_t<ArgTs>...> ArgTupleType;

  // Forwards arguments based on the parameter types of the handler.
  template <typename FnT> class ForwardArgsAsRequested {
  public:
    ForwardArgsAsRequested(FnT &&Fn) : Fn(std::move(Fn)) {}
    void operator()(ArgTs &...Args) { Fn(std::forward<ArgTs>(Args)...); }

  private:
    FnT Fn;
  };

  template <typename FnT>
  static ForwardArgsAsRequested<std::decay_t<FnT>>
  forwardArgsAsRequested(FnT &&Fn) {
    return ForwardArgsAsRequested<std::decay_t<FnT>>(std::forward<FnT>(Fn));
  }
};

template <typename C>
using WFHandlerTraits = CallableTraitsHelper<WFHandlerTraitsImpl, C>;

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

template <typename RetT, typename Serializer> class StructuredYield;

template <typename RetT, typename Serializer>
class StructuredYield<std::tuple<RetT>, Serializer>
    : public StructuredYieldBase<Serializer> {
public:
  using StructuredYieldBase<Serializer>::StructuredYieldBase;
  void operator()(RetT &&R) {
    if (auto ResultBytes = this->S.result().serialize(std::forward<RetT>(R)))
      this->Return(this->Session, this->CallCtx, ResultBytes->release());
    else
      this->Return(this->Session, this->CallCtx,
                   WrapperFunctionBuffer::createOutOfBandError(
                       "Could not serialize wrapper function result data")
                       .release());
  }
};

template <typename Serializer>
class StructuredYield<std::tuple<>, Serializer>
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
struct ResultDeserializer<std::tuple<Expected<T>>, Serializer> {
  static Expected<T> deserialize(WrapperFunctionBuffer ResultBytes,
                                 Serializer &S) {
    if (auto Val = S.result().template deserialize<std::tuple<T>>(
            std::move(ResultBytes)))
      return Expected<T>(std::move(std::get<0>(*Val)),
                         ForceExpectedSuccessValue());
    else
      return make_error<StringError>("Could not deserialize result");
  }
};

template <typename Serializer>
struct ResultDeserializer<std::tuple<Error>, Serializer> {
  static Error deserialize(WrapperFunctionBuffer ResultBytes, Serializer &S) {
    assert(ResultBytes.empty());
    return Error::success();
  }
};

} // namespace detail

/// Provides call and handle utilities to simplify writing and invocation of
/// wrapper functions in C++.
struct WrapperFunction {

  /// Wraps an asynchronous method (a method returning void, and taking a
  /// return callback as its first argument) for use with
  /// WrapperFunction::handle.
  ///
  /// AsyncMethod's call operator takes an ExecutorAddr as its second argument,
  /// casts it to a ClassT*, and then calls the wrapped method on that pointer,
  /// forwarding the return callback and any subsequent arguments (after the
  /// second argument representing the object address).
  ///
  /// This utility removes some of the boilerplate from writing wrappers for
  /// method calls.
  template <typename ClassT, typename ReturnT, typename... ArgTs>
  struct AsyncMethod {
    AsyncMethod(void (ClassT::*M)(ReturnT, ArgTs...)) : M(M) {}
    void operator()(ReturnT &&Return, ExecutorAddr Obj, ArgTs &&...Args) {
      (Obj.toPtr<ClassT *>()->*M)(std::forward<ReturnT>(Return),
                                  std::forward<ArgTs>(Args)...);
    }

  private:
    void (ClassT::*M)(ReturnT, ArgTs...);
  };

  /// Create an AsyncMethod wrapper for the given method pointer. The given
  /// method should be asynchronous: returning void, and taking a return
  /// callback as its first argument.
  ///
  /// The handWithAsyncMethod function can be used to remove some of the
  /// boilerplate from writing wrappers for method calls:
  ///
  ///   @code{.cpp}
  ///   class MyClass {
  ///   public:
  ///     void myMethod(move_only_function<void(std::string)> Return,
  //                    uint32_t X, bool Y) { ... }
  ///   };
  ///
  ///   // SPS Method signature -- note MyClass object address as first
  ///   // argument.
  ///   using SPSMyMethodWrapperSignature =
  ///     SPSString(SPSExecutorAddr, uint32_t, bool);
  ///
  ///
  ///   static void adder_add_async_sps_wrapper(
  ///       orc_rt_SessionRef Session, void *CallCtx,
  ///       orc_rt_WrapperFunctionReturn Return,
  ///       orc_rt_WrapperFunctionBuffer ArgBytes) {
  ///     using SPSSig = SPSString(SPSExecutorAddr, int32_t, bool);
  ///     SPSWrapperFunction<SPSSig>::handle(
  ///         Session, CallCtx, Return, ArgBytes,
  ///         WrapperFunction::handleWithAsyncMethod(&MyClass::myMethod));
  ///   }
  ///   @endcode
  ///
  template <typename ClassT, typename ReturnT, typename... ArgTs>
  static AsyncMethod<ClassT, ReturnT, ArgTs...>
  handleWithAsyncMethod(void (ClassT::*M)(ReturnT, ArgTs...)) {
    return AsyncMethod<ClassT, ReturnT, ArgTs...>(M);
  }

  /// Wraps a synchronous method (an ordinary method that returns its result,
  /// as opposed to an asynchronous method, see AsyncMethod) for use with
  /// WrapperFunction::handle.
  ///
  /// SyncMethod's call operator takes a return callback as its first argument
  /// and an ExecutorAddr as its second argument. The ExecutorAddr argument is
  /// cast to a ClassT*, and then called passing the subsequent arguments
  /// (after the second argument representing the object address). The Return
  /// callback is then called on the value returned from the method.
  ///
  /// This utility removes some of the boilerplate from writing wrappers for
  /// method calls.
  template <typename ClassT, typename RetT, typename... ArgTs>
  class SyncMethod {
  public:
    SyncMethod(RetT (ClassT::*M)(ArgTs...)) : M(M) {}

    void operator()(move_only_function<void(RetT)> Return, ExecutorAddr Obj,
                    ArgTs &&...Args) {
      Return((Obj.toPtr<ClassT *>()->*M)(std::forward<ArgTs>(Args)...));
    }

  private:
    RetT (ClassT::*M)(ArgTs...);
  };

  /// Create an SyncMethod wrapper for the given method pointer. The given
  /// method should be synchronous, i.e. returning its result (as opposed to
  /// asynchronous, see AsyncMethod).
  ///
  /// The handWithAsyncMethod function can be used to remove some of the
  /// boilerplate from writing wrappers for method calls:
  ///
  ///   @code{.cpp}
  ///   class MyClass {
  ///   public:
  ///     std::string myMethod(uint32_t X, bool Y) { ... }
  ///   };
  ///
  ///   // SPS Method signature -- note MyClass object address as first
  ///   // argument.
  ///   using SPSMyMethodWrapperSignature =
  ///     SPSString(SPSExecutorAddr, uint32_t, bool);
  ///
  ///
  ///   static void adder_add_sync_sps_wrapper(
  ///       orc_rt_SessionRef Session, void *CallCtx,
  ///       orc_rt_WrapperFunctionReturn Return,
  ///       orc_rt_WrapperFunctionBuffer ArgBytes) {
  ///     using SPSSig = SPSString(SPSExecutorAddr, int32_t, bool);
  ///     SPSWrapperFunction<SPSSig>::handle(
  ///         Session, CallCtx, Return, ArgBytes,
  ///         WrapperFunction::handleWithSyncMethod(&Adder::addSync));
  ///   }
  ///   @endcode
  ///
  template <typename ClassT, typename RetT, typename... ArgTs>
  static SyncMethod<ClassT, RetT, ArgTs...>
  handleWithSyncMethod(RetT (ClassT::*M)(ArgTs...)) {
    return SyncMethod<ClassT, RetT, ArgTs...>(M);
  }

  /// Make a call to a wrapper function.
  ///
  /// This utility serializes and deserializes arguments and return values
  /// (using the given Serializer), and calls the wrapper function via the
  /// given Caller object.
  template <typename Caller, typename Serializer, typename ResultHandler,
            typename... ArgTs>
  static void call(Caller &&C, Serializer &&S, ResultHandler &&RH,
                   ArgTs &&...Args) {
    typedef CallableArgInfo<ResultHandler> ResultHandlerTraits;
    static_assert(std::is_void_v<typename ResultHandlerTraits::return_type>,
                  "Result handler should return void");
    static_assert(
        std::tuple_size_v<typename ResultHandlerTraits::args_tuple_type> == 1,
        "Result-handler should have exactly one argument");
    typedef typename ResultHandlerTraits::args_tuple_type ResultTupleType;

    if (auto ArgBytes = S.arguments().serialize(std::forward<ArgTs>(Args)...)) {
      C(
          [RH = std::move(RH),
           S = std::move(S)](orc_rt_SessionRef Session,
                             WrapperFunctionBuffer ResultBytes) mutable {
            if (const char *ErrMsg = ResultBytes.getOutOfBandError())
              RH(make_error<StringError>(ErrMsg));
            else
              RH(detail::ResultDeserializer<ResultTupleType, Serializer>::
                     deserialize(std::move(ResultBytes), S));
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
    typedef detail::WFHandlerTraits<Handler> HandlerTraits;
    typedef typename HandlerTraits::ArgTupleType ArgTuple;
    typedef typename HandlerTraits::YieldType Yield;
    static_assert(std::is_void_v<typename CallableArgInfo<Yield>::return_type>,
                  "Return callback must return void");
    typedef typename CallableArgInfo<Yield>::args_tuple_type RetTupleType;

    if (ArgBytes.getOutOfBandError())
      return Return(Session, CallCtx, ArgBytes.release());

    if (auto Args = S.arguments().template deserialize<ArgTuple>(ArgBytes))
      std::apply(HandlerTraits::forwardArgsAsRequested(bind_front(
                     std::forward<Handler>(H),
                     detail::StructuredYield<RetTupleType, Serializer>(
                         Session, CallCtx, Return, std::move(S)))),
                 *Args);
    else
      Return(Session, CallCtx,
             WrapperFunctionBuffer::createOutOfBandError(
                 "Could not deserialize wrapper function arg data")
                 .release());
  }
};

} // namespace orc_rt

#endif // ORC_RT_WRAPPERFUNCTION_H
