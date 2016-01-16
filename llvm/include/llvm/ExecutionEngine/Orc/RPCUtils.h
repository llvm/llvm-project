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

#include "llvm/ADT/STLExtras.h"

namespace llvm {
namespace orc {
namespace remote {

// Base class containing utilities that require partial specialization.
// These cannot be included in RPC, as template class members cannot be
// partially specialized.
class RPCBase {
protected:
  template <typename ProcedureIdT, ProcedureIdT ProcId, typename... Ts>
  class ProcedureHelper {
  public:
    static const ProcedureIdT Id = ProcId;
  };

  template <typename ChannelT, typename Proc> class CallHelper;

  template <typename ChannelT, typename ProcedureIdT, ProcedureIdT ProcId,
            typename... ArgTs>
  class CallHelper<ChannelT, ProcedureHelper<ProcedureIdT, ProcId, ArgTs...>> {
  public:
    static std::error_code call(ChannelT &C, const ArgTs &... Args) {
      if (auto EC = serialize(C, ProcId))
        return EC;
      // If you see a compile-error on this line you're probably calling a
      // function with the wrong signature.
      return serialize_seq(C, Args...);
    }
  };

  template <typename ChannelT, typename Proc> class HandlerHelper;

  template <typename ChannelT, typename ProcedureIdT, ProcedureIdT ProcId,
            typename... ArgTs>
  class HandlerHelper<ChannelT,
                      ProcedureHelper<ProcedureIdT, ProcId, ArgTs...>> {
  public:
    template <typename HandlerT>
    static std::error_code handle(ChannelT &C, HandlerT Handler) {
      return readAndHandle(C, Handler, llvm::index_sequence_for<ArgTs...>());
    }

  private:
    template <typename HandlerT, size_t... Is>
    static std::error_code readAndHandle(ChannelT &C, HandlerT Handler,
                                         llvm::index_sequence<Is...> _) {
      std::tuple<ArgTs...> RPCArgs;
      // GCC 4.7 and 4.8 incorrectly issue a -Wunused-but-set-variable warning
      // for RPCArgs. Void cast RPCArgs to work around this for now.
      // FIXME: Remove this workaround once we can assume a working GCC version.
      (void)RPCArgs;
      if (auto EC = deserialize_seq(C, std::get<Is>(RPCArgs)...))
        return EC;
      return Handler(std::get<Is>(RPCArgs)...);
    }
  };

  template <typename ClassT, typename... ArgTs> class MemberFnWrapper {
  public:
    typedef std::error_code (ClassT::*MethodT)(ArgTs...);
    MemberFnWrapper(ClassT &Instance, MethodT Method)
        : Instance(Instance), Method(Method) {}
    std::error_code operator()(ArgTs &... Args) {
      return (Instance.*Method)(Args...);
    }

  private:
    ClassT &Instance;
    MethodT Method;
  };

  template <typename... ArgTs> class ReadArgs {
  public:
    std::error_code operator()() { return std::error_code(); }
  };

  template <typename ArgT, typename... ArgTs>
  class ReadArgs<ArgT, ArgTs...> : public ReadArgs<ArgTs...> {
  public:
    ReadArgs(ArgT &Arg, ArgTs &... Args)
        : ReadArgs<ArgTs...>(Args...), Arg(Arg) {}

    std::error_code operator()(ArgT &ArgVal, ArgTs &... ArgVals) {
      this->Arg = std::move(ArgVal);
      return ReadArgs<ArgTs...>::operator()(ArgVals...);
    }

  private:
    ArgT &Arg;
  };
};

/// Contains primitive utilities for defining, calling and handling calls to
/// remote procedures. ChannelT is a bidirectional stream conforming to the
/// RPCChannel interface (see RPCChannel.h), and ProcedureIdT is a procedure
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
/// Procedure<Id, Args...> :
///
///   associates a unique serializable id with an argument list.
///
///
/// call<Proc>(Channel, Args...) :
///
///   Calls the remote procedure 'Proc' by serializing Proc's id followed by its
/// arguments and sending the resulting bytes to 'Channel'.
///
///
/// handle<Proc>(Channel, <functor matching std::error_code(Args...)> :
///
///   Handles a call to 'Proc' by deserializing its arguments and calling the
/// given functor. This assumes that the id for 'Proc' has already been
/// deserialized.
///
/// expect<Proc>(Channel, <functor matching std::error_code(Args...)> :
///
///   The same as 'handle', except that the procedure id should not have been
/// read yet. Expect will deserialize the id and assert that it matches Proc's
/// id. If it does not, and unexpected RPC call error is returned.

template <typename ChannelT, typename ProcedureIdT = uint32_t>
class RPC : public RPCBase {
public:
  /// Utility class for defining/referring to RPC procedures.
  ///
  /// Typedefs of this utility are used when calling/handling remote procedures.
  ///
  /// ProcId should be a unique value of ProcedureIdT (i.e. not used with any
  /// other Procedure typedef in the RPC API being defined.
  ///
  /// the template argument Ts... gives the argument list for the remote
  /// procedure.
  ///
  /// E.g.
  ///
  ///   typedef Procedure<0, bool> Proc1;
  ///   typedef Procedure<1, std::string, std::vector<int>> Proc2;
  ///
  ///   if (auto EC = call<Proc1>(Channel, true))
  ///     /* handle EC */;
  ///
  ///   if (auto EC = expect<Proc2>(Channel,
  ///         [](std::string &S, std::vector<int> &V) {
  ///           // Stuff.
  ///           return std::error_code();
  ///         })
  ///     /* handle EC */;
  ///
  template <ProcedureIdT ProcId, typename... Ts>
  using Procedure = ProcedureHelper<ProcedureIdT, ProcId, Ts...>;

  /// Serialize Args... to channel C, but do not call C.send().
  ///
  /// For buffered channels, this can be used to queue up several calls before
  /// flushing the channel.
  template <typename Proc, typename... ArgTs>
  static std::error_code appendCall(ChannelT &C, const ArgTs &... Args) {
    return CallHelper<ChannelT, Proc>::call(C, Args...);
  }

  /// Serialize Args... to channel C and call C.send().
  template <typename Proc, typename... ArgTs>
  static std::error_code call(ChannelT &C, const ArgTs &... Args) {
    if (auto EC = appendCall<Proc>(C, Args...))
      return EC;
    return C.send();
  }

  /// Deserialize and return an enum whose underlying type is ProcedureIdT.
  static std::error_code getNextProcId(ChannelT &C, ProcedureIdT &Id) {
    return deserialize(C, Id);
  }

  /// Deserialize args for Proc from C and call Handler. The signature of
  /// handler must conform to 'std::error_code(Args...)' where Args... matches
  /// the arguments used in the Proc typedef.
  template <typename Proc, typename HandlerT>
  static std::error_code handle(ChannelT &C, HandlerT Handler) {
    return HandlerHelper<ChannelT, Proc>::handle(C, Handler);
  }

  /// Helper version of 'handle' for calling member functions.
  template <typename Proc, typename ClassT, typename... ArgTs>
  static std::error_code
  handle(ChannelT &C, ClassT &Instance,
         std::error_code (ClassT::*HandlerMethod)(ArgTs...)) {
    return handle<Proc>(
        C, MemberFnWrapper<ClassT, ArgTs...>(Instance, HandlerMethod));
  }

  /// Deserialize a ProcedureIdT from C and verify it matches the id for Proc.
  /// If the id does match, deserialize the arguments and call the handler
  /// (similarly to handle).
  /// If the id does not match, return an unexpect RPC call error and do not
  /// deserialize any further bytes.
  template <typename Proc, typename HandlerT>
  static std::error_code expect(ChannelT &C, HandlerT Handler) {
    ProcedureIdT ProcId;
    if (auto EC = getNextProcId(C, ProcId))
      return EC;
    if (ProcId != Proc::Id)
      return orcError(OrcErrorCode::UnexpectedRPCCall);
    return handle<Proc>(C, Handler);
  }

  /// Helper version of expect for calling member functions.
  template <typename Proc, typename ClassT, typename... ArgTs>
  static std::error_code
  expect(ChannelT &C, ClassT &Instance,
         std::error_code (ClassT::*HandlerMethod)(ArgTs...)) {
    return expect<Proc>(
        C, MemberFnWrapper<ClassT, ArgTs...>(Instance, HandlerMethod));
  }

  /// Helper for handling setter procedures - this method returns a functor that
  /// sets the variables referred to by Args... to values deserialized from the
  /// channel.
  /// E.g.
  ///
  ///   typedef Procedure<0, bool, int> Proc1;
  ///
  ///   ...
  ///   bool B;
  ///   int I;
  ///   if (auto EC = expect<Proc1>(Channel, readArgs(B, I)))
  ///     /* Handle Args */ ;
  ///
  template <typename... ArgTs>
  static ReadArgs<ArgTs...> readArgs(ArgTs &... Args) {
    return ReadArgs<ArgTs...>(Args...);
  }
};

} // end namespace remote
} // end namespace orc
} // end namespace llvm

#endif
