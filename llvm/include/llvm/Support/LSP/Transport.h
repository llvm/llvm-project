//===--- Transport.h - Sending and Receiving LSP messages -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The language server protocol is usually implemented by writing messages as
// JSON-RPC over the stdin/stdout of a subprocess. This file contains a JSON
// transport interface that handles this communication.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_LSP_TRANSPORT_H
#define LLVM_SUPPORT_LSP_TRANSPORT_H

#include "llvm/ADT/FunctionExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/FormatAdapters.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/LSP/Logging.h"
#include "llvm/Support/LSP/Protocol.h"
#include "llvm/Support/raw_ostream.h"
#include <memory>

namespace llvm {
// Simple helper function that returns a string as printed from a op.
template <typename T> static std::string debugString(T &&Op) {
  std::string InstrStr;
  llvm::raw_string_ostream Os(InstrStr);
  Os << Op;
  return Os.str();
}
namespace lsp {
class MessageHandler;

//===----------------------------------------------------------------------===//
// JSONTransport
//===----------------------------------------------------------------------===//

/// The encoding style of the JSON-RPC messages (both input and output).
enum JSONStreamStyle {
  /// Encoding per the LSP specification, with mandatory Content-Length header.
  Standard,
  /// Messages are delimited by a '// -----' line. Comment lines start with //.
  Delimited
};

/// An abstract class used by the JSONTransport to read JSON message.
class JSONTransportInput {
public:
  explicit JSONTransportInput(JSONStreamStyle Style = JSONStreamStyle::Standard)
      : Style(Style) {}
  virtual ~JSONTransportInput() = default;

  virtual bool hasError() const = 0;
  virtual bool isEndOfInput() const = 0;

  /// Read in a message from the input stream.
  LogicalResult readMessage(std::string &Json) {
    return Style == JSONStreamStyle::Delimited ? readDelimitedMessage(Json)
                                               : readStandardMessage(Json);
  }
  virtual LogicalResult readDelimitedMessage(std::string &Json) = 0;
  virtual LogicalResult readStandardMessage(std::string &Json) = 0;

private:
  /// The JSON stream style to use.
  JSONStreamStyle Style;
};

/// Concrete implementation of the JSONTransportInput that reads from a file.
class JSONTransportInputOverFile : public JSONTransportInput {
public:
  explicit JSONTransportInputOverFile(
      std::FILE *In, JSONStreamStyle Style = JSONStreamStyle::Standard)
      : JSONTransportInput(Style), In(In) {}

  bool hasError() const final { return ferror(In); }
  bool isEndOfInput() const final { return feof(In); }

  LogicalResult readDelimitedMessage(std::string &Json) final;
  LogicalResult readStandardMessage(std::string &Json) final;

private:
  std::FILE *In;
};

/// A transport class that performs the JSON-RPC communication with the LSP
/// client.
class JSONTransport {
public:
  JSONTransport(std::unique_ptr<JSONTransportInput> In, raw_ostream &Out,
                bool PrettyOutput = false)
      : In(std::move(In)), Out(Out), PrettyOutput(PrettyOutput) {}

  JSONTransport(std::FILE *In, raw_ostream &Out,
                JSONStreamStyle Style = JSONStreamStyle::Standard,
                bool PrettyOutput = false)
      : In(std::make_unique<JSONTransportInputOverFile>(In, Style)), Out(Out),
        PrettyOutput(PrettyOutput) {}

  /// The following methods are used to send a message to the LSP client.
  void notify(StringRef Method, llvm::json::Value Params);
  void call(StringRef Method, llvm::json::Value Params, llvm::json::Value Id);
  void reply(llvm::json::Value Id, llvm::Expected<llvm::json::Value> Result);

  /// Start executing the JSON-RPC transport.
  llvm::Error run(MessageHandler &Handler);

private:
  /// Dispatches the given incoming json message to the message handler.
  bool handleMessage(llvm::json::Value Msg, MessageHandler &Handler);
  /// Writes the given message to the output stream.
  void sendMessage(llvm::json::Value Msg);

private:
  /// The input to read a message from.
  std::unique_ptr<JSONTransportInput> In;
  SmallVector<char, 0> OutputBuffer;
  /// The output file stream.
  raw_ostream &Out;
  /// If the output JSON should be formatted for easier readability.
  bool PrettyOutput;
};

//===----------------------------------------------------------------------===//
// MessageHandler
//===----------------------------------------------------------------------===//

/// A Callback<T> is a void function that accepts Expected<T>. This is
/// accepted by functions that logically return T.
template <typename T>
using Callback = llvm::unique_function<void(llvm::Expected<T>)>;

/// An OutgoingNotification<T> is a function used for outgoing notifications
/// send to the client.
template <typename T>
using OutgoingNotification = llvm::unique_function<void(const T &)>;

/// An OutgoingRequest<T> is a function used for outgoing requests to send to
/// the client.
template <typename T>
using OutgoingRequest =
    llvm::unique_function<void(const T &, llvm::json::Value Id)>;

/// An `OutgoingRequestCallback` is invoked when an outgoing request to the
/// client receives a response in turn. It is passed the original request's ID,
/// as well as the response result.
template <typename T>
using OutgoingRequestCallback =
    std::function<void(llvm::json::Value, llvm::Expected<T>)>;

/// A handler used to process the incoming transport messages.
class MessageHandler {
public:
  MessageHandler(JSONTransport &Transport) : Transport(Transport) {}

  bool onNotify(StringRef Method, llvm::json::Value Value);
  bool onCall(StringRef Method, llvm::json::Value Params, llvm::json::Value Id);
  bool onReply(llvm::json::Value Id, llvm::Expected<llvm::json::Value> Result);

  template <typename T>
  static llvm::Expected<T> parse(const llvm::json::Value &Raw,
                                 StringRef PayloadName, StringRef PayloadKind) {
    T Result;
    llvm::json::Path::Root Root;
    if (fromJSON(Raw, Result, Root))
      return std::move(Result);

    // Dump the relevant parts of the broken message.
    std::string Context;
    llvm::raw_string_ostream Os(Context);
    Root.printErrorContext(Raw, Os);

    // Report the error (e.g. to the client).
    return llvm::make_error<LSPError>(
        llvm::formatv("failed to decode {0} {1}: {2}", PayloadName, PayloadKind,
                      fmt_consume(Root.getError())),
        ErrorCode::InvalidParams);
  }

  template <typename Param, typename Result, typename ThisT>
  void method(llvm::StringLiteral Method, ThisT *ThisPtr,
              void (ThisT::*Handler)(const Param &, Callback<Result>)) {
    MethodHandlers[Method] = [Method, Handler,
                              ThisPtr](llvm::json::Value RawParams,
                                       Callback<llvm::json::Value> Reply) {
      llvm::Expected<Param> Parameter =
          parse<Param>(RawParams, Method, "request");
      if (!Parameter)
        return Reply(Parameter.takeError());
      (ThisPtr->*Handler)(*Parameter, std::move(Reply));
    };
  }

  template <typename Param, typename ThisT>
  void notification(llvm::StringLiteral Method, ThisT *ThisPtr,
                    void (ThisT::*Handler)(const Param &)) {
    NotificationHandlers[Method] = [Method, Handler,
                                    ThisPtr](llvm::json::Value RawParams) {
      llvm::Expected<Param> Parameter =
          parse<Param>(RawParams, Method, "notification");
      if (!Parameter) {
        return llvm::consumeError(llvm::handleErrors(
            Parameter.takeError(), [](const LSPError &LspError) {
              Logger::error("JSON parsing error: {0}",
                            LspError.message.c_str());
            }));
      }
      (ThisPtr->*Handler)(*Parameter);
    };
  }

  /// Create an OutgoingNotification object used for the given method.
  template <typename T>
  OutgoingNotification<T> outgoingNotification(llvm::StringLiteral Method) {
    return [&, Method](const T &Params) {
      std::lock_guard<std::mutex> TransportLock(TransportOutputMutex);
      Logger::info("--> {0}", Method);
      Transport.notify(Method, llvm::json::Value(Params));
    };
  }

  /// Create an OutgoingRequest function that, when called, sends a request with
  /// the given method via the transport. Should the outgoing request be
  /// met with a response, the result JSON is parsed and the response callback
  /// is invoked.
  template <typename Param, typename Result>
  OutgoingRequest<Param>
  outgoingRequest(llvm::StringLiteral Method,
                  OutgoingRequestCallback<Result> Callback) {
    return [&, Method, Callback](const Param &Parameter, llvm::json::Value Id) {
      auto CallbackWrapper = [Method, Callback = std::move(Callback)](
                                 llvm::json::Value Id,
                                 llvm::Expected<llvm::json::Value> Value) {
        if (!Value)
          return Callback(std::move(Id), Value.takeError());

        std::string ResponseName = llvm::formatv("reply:{0}({1})", Method, Id);
        llvm::Expected<Result> ParseResult =
            parse<Result>(*Value, ResponseName, "response");
        if (!ParseResult)
          return Callback(std::move(Id), ParseResult.takeError());

        return Callback(std::move(Id), *ParseResult);
      };

      {
        std::lock_guard<std::mutex> Lock(ResponseHandlersMutex);
        ResponseHandlers.insert(
            {debugString(Id), std::make_pair(Method.str(), CallbackWrapper)});
      }

      std::lock_guard<std::mutex> TransportLock(TransportOutputMutex);
      Logger::info("--> {0}({1})", Method, Id);
      Transport.call(Method, llvm::json::Value(Parameter), Id);
    };
  }

private:
  template <typename HandlerT>
  using HandlerMap = llvm::StringMap<llvm::unique_function<HandlerT>>;

  HandlerMap<void(llvm::json::Value)> NotificationHandlers;
  HandlerMap<void(llvm::json::Value, Callback<llvm::json::Value>)>
      MethodHandlers;

  /// A pair of (1) the original request's method name, and (2) the callback
  /// function to be invoked for responses.
  using ResponseHandlerTy =
      std::pair<std::string, OutgoingRequestCallback<llvm::json::Value>>;
  /// A mapping from request/response ID to response handler.
  llvm::StringMap<ResponseHandlerTy> ResponseHandlers;
  /// Mutex to guard insertion into the response handler map.
  std::mutex ResponseHandlersMutex;

  JSONTransport &Transport;

  /// Mutex to guard sending output messages to the transport.
  std::mutex TransportOutputMutex;
};

} // namespace lsp
} // namespace llvm

#endif
