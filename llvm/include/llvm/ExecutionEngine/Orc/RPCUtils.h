//===----- RPCUTils.h - Basic tilities for building RPC APIs ----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Basic utilities for building RPC APIs.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_EXECUTIONENGINE_ORC_RPCUTILS_H
#define LLVM_EXECUTIONENGINE_ORC_RPCUTILS_H

#include <map>
#include <vector>

#include "llvm/ADT/STLExtras.h"
#include "llvm/ExecutionEngine/Orc/OrcError.h"

#ifdef _MSC_VER
// concrt.h depends on eh.h for __uncaught_exception declaration
// even if we disable exceptions.
#include <eh.h>

// Disable warnings from ppltasks.h transitively included by <future>.
#pragma warning(push)
#pragma warning(disable : 4530)
#pragma warning(disable : 4062)
#endif

#include <future>

#ifdef _MSC_VER
#pragma warning(pop)
#endif

namespace llvm {
namespace orc {
namespace remote {

/// Describes reserved RPC Function Ids.
///
/// The default implementation will serve for integer and enum function id
/// types. If you want to use a custom type as your FunctionId you can
/// specialize this class and provide unique values for InvalidId,
/// ResponseId and FirstValidId.

template <typename T> class RPCFunctionIdTraits {
public:
  static const T InvalidId = static_cast<T>(0);
  static const T ResponseId = static_cast<T>(1);
  static const T FirstValidId = static_cast<T>(2);
};

// Base class containing utilities that require partial specialization.
// These cannot be included in RPC, as template class members cannot be
// partially specialized.
class RPCBase {
protected:

  // RPC Function description type.
  //
  // This class provides the information and operations needed to support the
  // RPC primitive operations (call, expect, etc) for a given function. It
  // is specialized for void and non-void functions to deal with the differences
  // betwen the two. Both specializations have the same interface:
  //
  // Id - The function's unique identifier.
  // ErrorReturn - The return type for blocking calls.
  // readResult - Deserialize a result from a channel.
  // abandon - Abandon a promised result.
  // respond - Retun a result on the channel.
  template <typename FunctionIdT, FunctionIdT FuncId, typename FnT>
  class FunctionHelper {};

  // RPC Function description specialization for non-void functions.
  template <typename FunctionIdT, FunctionIdT FuncId, typename RetT,
            typename... ArgTs>
  class FunctionHelper<FunctionIdT, FuncId, RetT(ArgTs...)> {
  public:
    static_assert(FuncId != RPCFunctionIdTraits<FunctionIdT>::InvalidId &&
                      FuncId != RPCFunctionIdTraits<FunctionIdT>::ResponseId,
                  "Cannot define custom function with InvalidId or ResponseId. "
                  "Please use RPCFunctionTraits<FunctionIdT>::FirstValidId.");

    static const FunctionIdT Id = FuncId;

    typedef Expected<RetT> ErrorReturn;

    template <typename ChannelT>
    static Error readResult(ChannelT &C, std::promise<ErrorReturn> &P) {
      RetT Val;
      auto Err = deserialize(C, Val);
      auto Err2 = endReceiveMessage(C);
      Err = joinErrors(std::move(Err), std::move(Err2));
      if (Err)
        return Err;

      P.set_value(std::move(Val));
      return Error::success();
    }

    static void abandon(std::promise<ErrorReturn> &P) {
      P.set_value(
        make_error<StringError>("RPC function call failed to return",
                                inconvertibleErrorCode()));
    }

    static void consumeAbandoned(std::future<ErrorReturn> &P) {
      consumeError(P.get().takeError());
    }

    template <typename ChannelT, typename SequenceNumberT>
    static Error respond(ChannelT &C, SequenceNumberT SeqNo,
                         ErrorReturn &Result) {
      FunctionIdT ResponseId = RPCFunctionIdTraits<FunctionIdT>::ResponseId;

      // If the handler returned an error then bail out with that.
      if (!Result)
        return Result.takeError();

      // Otherwise open a new message on the channel and send the result.
      if (auto Err = startSendMessage(C))
        return Err;
      if (auto Err = serializeSeq(C, ResponseId, SeqNo, *Result))
        return Err;
      return endSendMessage(C);
    }
  };

  // RPC Function description specialization for void functions.
  template <typename FunctionIdT, FunctionIdT FuncId, typename... ArgTs>
  class FunctionHelper<FunctionIdT, FuncId, void(ArgTs...)> {
  public:
    static_assert(FuncId != RPCFunctionIdTraits<FunctionIdT>::InvalidId &&
                      FuncId != RPCFunctionIdTraits<FunctionIdT>::ResponseId,
                  "Cannot define custom function with InvalidId or ResponseId. "
                  "Please use RPCFunctionTraits<FunctionIdT>::FirstValidId.");

    static const FunctionIdT Id = FuncId;

    typedef Error ErrorReturn;

    template <typename ChannelT>
    static Error readResult(ChannelT &C, std::promise<ErrorReturn> &P) {
      // Void functions don't have anything to deserialize, so we're good.
      P.set_value(Error::success());
      return endReceiveMessage(C);
    }

    static void abandon(std::promise<ErrorReturn> &P) {
      P.set_value(
        make_error<StringError>("RPC function call failed to return",
                                inconvertibleErrorCode()));
    }

    static void consumeAbandoned(std::future<ErrorReturn> &P) {
      consumeError(P.get());
    }

    template <typename ChannelT, typename SequenceNumberT>
    static Error respond(ChannelT &C, SequenceNumberT SeqNo,
                         ErrorReturn &Result) {
      const FunctionIdT ResponseId =
          RPCFunctionIdTraits<FunctionIdT>::ResponseId;

      // If the handler returned an error then bail out with that.
      if (Result)
        return std::move(Result);

      // Otherwise open a new message on the channel and send the result.
      if (auto Err = startSendMessage(C))
        return Err;
      if (auto Err = serializeSeq(C, ResponseId, SeqNo))
        return Err;
      return endSendMessage(C);
    }
  };

  // Helper for the call primitive.
  template <typename ChannelT, typename SequenceNumberT, typename Func>
  class CallHelper;

  template <typename ChannelT, typename SequenceNumberT, typename FunctionIdT,
            FunctionIdT FuncId, typename RetT, typename... ArgTs>
  class CallHelper<ChannelT, SequenceNumberT,
                   FunctionHelper<FunctionIdT, FuncId, RetT(ArgTs...)>> {
  public:
    static Error call(ChannelT &C, SequenceNumberT SeqNo,
                      const ArgTs &... Args) {
      if (auto Err = startSendMessage(C))
        return Err;
      if (auto Err = serializeSeq(C, FuncId, SeqNo, Args...))
        return Err;
      return endSendMessage(C);
    }
  };

  // Helper for handle primitive.
  template <typename ChannelT, typename SequenceNumberT, typename Func>
  class HandlerHelper;

  template <typename ChannelT, typename SequenceNumberT, typename FunctionIdT,
            FunctionIdT FuncId, typename RetT, typename... ArgTs>
  class HandlerHelper<ChannelT, SequenceNumberT,
                      FunctionHelper<FunctionIdT, FuncId, RetT(ArgTs...)>> {
  public:
    template <typename HandlerT>
    static Error handle(ChannelT &C, HandlerT Handler) {
      return readAndHandle(C, Handler, llvm::index_sequence_for<ArgTs...>());
    }

  private:
    typedef FunctionHelper<FunctionIdT, FuncId, RetT(ArgTs...)> Func;

    template <typename HandlerT, size_t... Is>
    static Error readAndHandle(ChannelT &C, HandlerT Handler,
                               llvm::index_sequence<Is...> _) {
      std::tuple<ArgTs...> RPCArgs;
      SequenceNumberT SeqNo;
      // GCC 4.7 and 4.8 incorrectly issue a -Wunused-but-set-variable warning
      // for RPCArgs. Void cast RPCArgs to work around this for now.
      // FIXME: Remove this workaround once we can assume a working GCC version.
      (void)RPCArgs;
      if (auto Err = deserializeSeq(C, SeqNo, std::get<Is>(RPCArgs)...))
        return Err;

      // We've deserialized the arguments, so unlock the channel for reading
      // before we call the handler. This allows recursive RPC calls.
      if (auto Err = endReceiveMessage(C))
        return Err;

      // Run the handler and get the result.
      auto Result = Handler(std::get<Is>(RPCArgs)...);

      // Return the result to the client.
      return Func::template respond<ChannelT, SequenceNumberT>(C, SeqNo,
                                                               Result);
    }
  };

  // Helper for wrapping member functions up as functors.
  template <typename ClassT, typename RetT, typename... ArgTs>
  class MemberFnWrapper {
  public:
    typedef RetT (ClassT::*MethodT)(ArgTs...);
    MemberFnWrapper(ClassT &Instance, MethodT Method)
        : Instance(Instance), Method(Method) {}
    RetT operator()(ArgTs &... Args) { return (Instance.*Method)(Args...); }

  private:
    ClassT &Instance;
    MethodT Method;
  };

  // Helper that provides a Functor for deserializing arguments.
  template <typename... ArgTs> class ReadArgs {
  public:
    Error operator()() { return Error::success(); }
  };

  template <typename ArgT, typename... ArgTs>
  class ReadArgs<ArgT, ArgTs...> : public ReadArgs<ArgTs...> {
  public:
    ReadArgs(ArgT &Arg, ArgTs &... Args)
        : ReadArgs<ArgTs...>(Args...), Arg(Arg) {}

    Error operator()(ArgT &ArgVal, ArgTs &... ArgVals) {
      this->Arg = std::move(ArgVal);
      return ReadArgs<ArgTs...>::operator()(ArgVals...);
    }

  private:
    ArgT &Arg;
  };
};

/// Contains primitive utilities for defining, calling and handling calls to
/// remote procedures. ChannelT is a bidirectional stream conforming to the
/// RPCChannel interface (see RPCChannel.h), and FunctionIdT is a procedure
/// identifier type that must be serializable on ChannelT.
///
/// These utilities support the construction of very primitive RPC utilities.
/// Their intent is to ensure correct serialization and deserialization of
/// procedure arguments, and to keep the client and server's view of the API in
/// sync.
///
/// These utilities do not support return values. These can be handled by
/// declaring a corresponding '.*Response' procedure and expecting it after a
/// call). They also do not support versioning: the client and server *must* be
/// compiled with the same procedure definitions.
///
///
///
/// Overview (see comments individual types/methods for details):
///
/// Function<Id, Args...> :
///
///   associates a unique serializable id with an argument list.
///
///
/// call<Func>(Channel, Args...) :
///
///   Calls the remote procedure 'Func' by serializing Func's id followed by its
/// arguments and sending the resulting bytes to 'Channel'.
///
///
/// handle<Func>(Channel, <functor matching Error(Args...)> :
///
///   Handles a call to 'Func' by deserializing its arguments and calling the
/// given functor. This assumes that the id for 'Func' has already been
/// deserialized.
///
/// expect<Func>(Channel, <functor matching Error(Args...)> :
///
///   The same as 'handle', except that the procedure id should not have been
/// read yet. Expect will deserialize the id and assert that it matches Func's
/// id. If it does not, and unexpected RPC call error is returned.
template <typename ChannelT, typename FunctionIdT = uint32_t,
          typename SequenceNumberT = uint16_t>
class RPC : public RPCBase {
public:
  /// RPC default constructor.
  RPC() = default;

  /// RPC instances cannot be copied.
  RPC(const RPC &) = delete;

  /// RPC instances cannot be copied.
  RPC &operator=(const RPC &) = delete;

  /// RPC move constructor.
  // FIXME: Remove once MSVC can synthesize move ops.
  RPC(RPC &&Other)
      : SequenceNumberMgr(std::move(Other.SequenceNumberMgr)),
        OutstandingResults(std::move(Other.OutstandingResults)) {}

  /// RPC move assignment.
  // FIXME: Remove once MSVC can synthesize move ops.
  RPC &operator=(RPC &&Other) {
    SequenceNumberMgr = std::move(Other.SequenceNumberMgr);
    OutstandingResults = std::move(Other.OutstandingResults);
    return *this;
  }

  /// Utility class for defining/referring to RPC procedures.
  ///
  /// Typedefs of this utility are used when calling/handling remote procedures.
  ///
  /// FuncId should be a unique value of FunctionIdT (i.e. not used with any
  /// other Function typedef in the RPC API being defined.
  ///
  /// the template argument Ts... gives the argument list for the remote
  /// procedure.
  ///
  /// E.g.
  ///
  ///   typedef Function<0, bool> Func1;
  ///   typedef Function<1, std::string, std::vector<int>> Func2;
  ///
  ///   if (auto Err = call<Func1>(Channel, true))
  ///     /* handle Err */;
  ///
  ///   if (auto Err = expect<Func2>(Channel,
  ///         [](std::string &S, std::vector<int> &V) {
  ///           // Stuff.
  ///           return Error::success();
  ///         })
  ///     /* handle Err */;
  ///
  template <FunctionIdT FuncId, typename FnT>
  using Function = FunctionHelper<FunctionIdT, FuncId, FnT>;

  /// Return type for non-blocking call primitives.
  template <typename Func>
  using NonBlockingCallResult = std::future<typename Func::ErrorReturn>;

  /// Return type for non-blocking call-with-seq primitives.
  template <typename Func>
  using NonBlockingCallWithSeqResult =
      std::pair<NonBlockingCallResult<Func>, SequenceNumberT>;

  /// Call Func on Channel C. Does not block, does not call send. Returns a pair
  /// of a future result and the sequence number assigned to the result.
  ///
  /// This utility function is primarily used for single-threaded mode support,
  /// where the sequence number can be used to wait for the corresponding
  /// result. In multi-threaded mode the appendCallNB method, which does not
  /// return the sequence numeber, should be preferred.
  template <typename Func, typename... ArgTs>
  Expected<NonBlockingCallWithSeqResult<Func>>
  appendCallNBWithSeq(ChannelT &C, const ArgTs &... Args) {
    auto SeqNo = SequenceNumberMgr.getSequenceNumber();
    std::promise<typename Func::ErrorReturn> Promise;
    auto Result = Promise.get_future();
    OutstandingResults[SeqNo] =
        createOutstandingResult<Func>(std::move(Promise));

    if (auto Err = CallHelper<ChannelT, SequenceNumberT, Func>::call(C, SeqNo,
                                                                     Args...)) {
      abandonOutstandingResults();
      Func::consumeAbandoned(Result);
      return std::move(Err);
    } else
      return NonBlockingCallWithSeqResult<Func>(std::move(Result), SeqNo);
  }

  /// The same as appendCallNBWithSeq, except that it calls C.send() to
  /// flush the channel after serializing the call.
  template <typename Func, typename... ArgTs>
  Expected<NonBlockingCallWithSeqResult<Func>>
  callNBWithSeq(ChannelT &C, const ArgTs &... Args) {
    auto Result = appendCallNBWithSeq<Func>(C, Args...);
    if (!Result)
      return Result;
    if (auto Err = C.send()) {
      abandonOutstandingResults();
      Func::consumeAbandoned(Result->first);
      return std::move(Err);
    }
    return Result;
  }

  /// Serialize Args... to channel C, but do not call send.
  /// Returns an error if serialization fails, otherwise returns a
  /// std::future<Expected<T>> (or a future<Error> for void functions).
  template <typename Func, typename... ArgTs>
  Expected<NonBlockingCallResult<Func>> appendCallNB(ChannelT &C,
                                                     const ArgTs &... Args) {
    auto FutureResAndSeqOrErr = appendCallNBWithSeq<Func>(C, Args...);
    if (FutureResAndSeqOrErr)
      return std::move(FutureResAndSeqOrErr->first);
    return FutureResAndSeqOrErr.getError();
  }

  /// The same as appendCallNB, except that it calls C.send to flush the
  /// channel after serializing the call.
  template <typename Func, typename... ArgTs>
  Expected<NonBlockingCallResult<Func>> callNB(ChannelT &C,
                                               const ArgTs &... Args) {
    auto FutureResAndSeqOrErr = callNBWithSeq<Func>(C, Args...);
    if (FutureResAndSeqOrErr)
      return std::move(FutureResAndSeqOrErr->first);
    return FutureResAndSeqOrErr.getError();
  }

  /// Call Func on Channel C. Blocks waiting for a result. Returns an Error
  /// for void functions or an Expected<T> for functions returning a T.
  ///
  /// This function is for use in threaded code where another thread is
  /// handling responses and incoming calls.
  template <typename Func, typename... ArgTs>
  typename Func::ErrorReturn callB(ChannelT &C, const ArgTs &... Args) {
    if (auto FutureResOrErr = callNBWithSeq(C, Args...)) {
      if (auto Err = C.send()) {
        abandonOutstandingResults();
        Func::consumeAbandoned(*FutureResOrErr);
        return std::move(Err);
      }
      return FutureResOrErr->get();
    } else
      return FutureResOrErr.takeError();
  }

  /// Call Func on Channel C. Block waiting for a result. While blocked, run
  /// HandleOther to handle incoming calls (Response calls will be handled
  /// implicitly before calling HandleOther). Returns an Error for void
  /// functions or an Expected<T> for functions returning a T.
  ///
  /// This function is for use in single threaded mode when the calling thread
  /// must act as both sender and receiver.
  template <typename Func, typename HandleFtor, typename... ArgTs>
  typename Func::ErrorReturn
  callSTHandling(ChannelT &C, HandleFtor &HandleOther, const ArgTs &... Args) {
    if (auto ResultAndSeqNoOrErr = callNBWithSeq<Func>(C, Args...)) {
      auto &ResultAndSeqNo = *ResultAndSeqNoOrErr;
      if (auto Err = waitForResult(C, ResultAndSeqNo.second, HandleOther))
        return std::move(Err);
      return ResultAndSeqNo.first.get();
    } else
      return ResultAndSeqNoOrErr.takeError();
  }

  /// Call Func on Channel C. Block waiting for a result. Returns an Error for
  /// void functions or an Expected<T> for functions returning a T.
  template <typename Func, typename... ArgTs>
  typename Func::ErrorReturn callST(ChannelT &C, const ArgTs &... Args) {
    return callSTHandling<Func>(C, handleNone, Args...);
  }

  /// Start receiving a new function call.
  ///
  /// Calls startReceiveMessage on the channel, then deserializes a FunctionId
  /// into Id.
  Error startReceivingFunction(ChannelT &C, FunctionIdT &Id) {
    if (auto Err = startReceiveMessage(C))
      return Err;

    return deserialize(C, Id);
  }

  /// Deserialize args for Func from C and call Handler. The signature of
  /// handler must conform to 'Error(Args...)' where Args... matches
  /// the arguments used in the Func typedef.
  template <typename Func, typename HandlerT>
  static Error handle(ChannelT &C, HandlerT Handler) {
    return HandlerHelper<ChannelT, SequenceNumberT, Func>::handle(C, Handler);
  }

  /// Helper version of 'handle' for calling member functions.
  template <typename Func, typename ClassT, typename RetT, typename... ArgTs>
  static Error handle(ChannelT &C, ClassT &Instance,
                      RetT (ClassT::*HandlerMethod)(ArgTs...)) {
    return handle<Func>(
        C, MemberFnWrapper<ClassT, RetT, ArgTs...>(Instance, HandlerMethod));
  }

  /// Deserialize a FunctionIdT from C and verify it matches the id for Func.
  /// If the id does match, deserialize the arguments and call the handler
  /// (similarly to handle).
  /// If the id does not match, return an unexpect RPC call error and do not
  /// deserialize any further bytes.
  template <typename Func, typename HandlerT>
  Error expect(ChannelT &C, HandlerT Handler) {
    FunctionIdT FuncId;
    if (auto Err = startReceivingFunction(C, FuncId))
      return std::move(Err);
    if (FuncId != Func::Id)
      return orcError(OrcErrorCode::UnexpectedRPCCall);
    return handle<Func>(C, Handler);
  }

  /// Helper version of expect for calling member functions.
  template <typename Func, typename ClassT, typename... ArgTs>
  static Error expect(ChannelT &C, ClassT &Instance,
                      Error (ClassT::*HandlerMethod)(ArgTs...)) {
    return expect<Func>(
        C, MemberFnWrapper<ClassT, ArgTs...>(Instance, HandlerMethod));
  }

  /// Helper for handling setter procedures - this method returns a functor that
  /// sets the variables referred to by Args... to values deserialized from the
  /// channel.
  /// E.g.
  ///
  ///   typedef Function<0, bool, int> Func1;
  ///
  ///   ...
  ///   bool B;
  ///   int I;
  ///   if (auto Err = expect<Func1>(Channel, readArgs(B, I)))
  ///     /* Handle Args */ ;
  ///
  template <typename... ArgTs>
  static ReadArgs<ArgTs...> readArgs(ArgTs &... Args) {
    return ReadArgs<ArgTs...>(Args...);
  }

  /// Read a response from Channel.
  /// This should be called from the receive loop to retrieve results.
  Error handleResponse(ChannelT &C, SequenceNumberT *SeqNoRet = nullptr) {
    SequenceNumberT SeqNo;
    if (auto Err = deserialize(C, SeqNo)) {
      abandonOutstandingResults();
      return Err;
    }

    if (SeqNoRet)
      *SeqNoRet = SeqNo;

    auto I = OutstandingResults.find(SeqNo);
    if (I == OutstandingResults.end()) {
      abandonOutstandingResults();
      return orcError(OrcErrorCode::UnexpectedRPCResponse);
    }

    if (auto Err = I->second->readResult(C)) {
      abandonOutstandingResults();
      // FIXME: Release sequence numbers?
      return Err;
    }

    OutstandingResults.erase(I);
    SequenceNumberMgr.releaseSequenceNumber(SeqNo);

    return Error::success();
  }

  // Loop waiting for a result with the given sequence number.
  // This can be used as a receive loop if the user doesn't have a default.
  template <typename HandleOtherFtor>
  Error waitForResult(ChannelT &C, SequenceNumberT TgtSeqNo,
                      HandleOtherFtor &HandleOther = handleNone) {
    bool GotTgtResult = false;

    while (!GotTgtResult) {
      FunctionIdT Id = RPCFunctionIdTraits<FunctionIdT>::InvalidId;
      if (auto Err = startReceivingFunction(C, Id))
        return Err;
      if (Id == RPCFunctionIdTraits<FunctionIdT>::ResponseId) {
        SequenceNumberT SeqNo;
        if (auto Err = handleResponse(C, &SeqNo))
          return Err;
        GotTgtResult = (SeqNo == TgtSeqNo);
      } else if (auto Err = HandleOther(C, Id))
        return Err;
    }

    return Error::success();
  }

  // Default handler for 'other' (non-response) functions when waiting for a
  // result from the channel.
  static Error handleNone(ChannelT &, FunctionIdT) {
    return orcError(OrcErrorCode::UnexpectedRPCCall);
  };

private:
  // Manage sequence numbers.
  class SequenceNumberManager {
  public:
    SequenceNumberManager() = default;

    SequenceNumberManager(const SequenceNumberManager &) = delete;
    SequenceNumberManager &operator=(const SequenceNumberManager &) = delete;

    SequenceNumberManager(SequenceNumberManager &&Other)
        : NextSequenceNumber(std::move(Other.NextSequenceNumber)),
          FreeSequenceNumbers(std::move(Other.FreeSequenceNumbers)) {}

    SequenceNumberManager &operator=(SequenceNumberManager &&Other) {
      NextSequenceNumber = std::move(Other.NextSequenceNumber);
      FreeSequenceNumbers = std::move(Other.FreeSequenceNumbers);
    }

    void reset() {
      std::lock_guard<std::mutex> Lock(SeqNoLock);
      NextSequenceNumber = 0;
      FreeSequenceNumbers.clear();
    }

    SequenceNumberT getSequenceNumber() {
      std::lock_guard<std::mutex> Lock(SeqNoLock);
      if (FreeSequenceNumbers.empty())
        return NextSequenceNumber++;
      auto SequenceNumber = FreeSequenceNumbers.back();
      FreeSequenceNumbers.pop_back();
      return SequenceNumber;
    }

    void releaseSequenceNumber(SequenceNumberT SequenceNumber) {
      std::lock_guard<std::mutex> Lock(SeqNoLock);
      FreeSequenceNumbers.push_back(SequenceNumber);
    }

  private:
    std::mutex SeqNoLock;
    SequenceNumberT NextSequenceNumber = 0;
    std::vector<SequenceNumberT> FreeSequenceNumbers;
  };

  // Base class for results that haven't been returned from the other end of the
  // RPC connection yet.
  class OutstandingResult {
  public:
    virtual ~OutstandingResult() {}
    virtual Error readResult(ChannelT &C) = 0;
    virtual void abandon() = 0;
  };

  // Outstanding results for a specific function.
  template <typename Func>
  class OutstandingResultImpl : public OutstandingResult {
  private:
  public:
    OutstandingResultImpl(std::promise<typename Func::ErrorReturn> &&P)
        : P(std::move(P)) {}

    Error readResult(ChannelT &C) override { return Func::readResult(C, P); }

    void abandon() override { Func::abandon(P); }

  private:
    std::promise<typename Func::ErrorReturn> P;
  };

  // Create an outstanding result for the given function.
  template <typename Func>
  std::unique_ptr<OutstandingResult>
  createOutstandingResult(std::promise<typename Func::ErrorReturn> &&P) {
    return llvm::make_unique<OutstandingResultImpl<Func>>(std::move(P));
  }

  // Abandon all outstanding results.
  void abandonOutstandingResults() {
    for (auto &KV : OutstandingResults)
      KV.second->abandon();
    OutstandingResults.clear();
    SequenceNumberMgr.reset();
  }

  SequenceNumberManager SequenceNumberMgr;
  std::map<SequenceNumberT, std::unique_ptr<OutstandingResult>>
      OutstandingResults;
};

} // end namespace remote
} // end namespace orc
} // end namespace llvm

#endif
