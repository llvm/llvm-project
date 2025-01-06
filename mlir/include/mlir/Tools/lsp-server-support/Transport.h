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

#ifndef MLIR_TOOLS_LSPSERVERSUPPORT_TRANSPORT_H
#define MLIR_TOOLS_LSPSERVERSUPPORT_TRANSPORT_H

#include "mlir/Support/DebugStringHelper.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Tools/lsp-server-support/Logging.h"
#include "mlir/Tools/lsp-server-support/Protocol.h"
#include "llvm/ADT/FunctionExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/FormatAdapters.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"
#include <atomic>

namespace mlir {
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

/// A transport class that performs the JSON-RPC communication with the LSP
/// client.
class JSONTransport {
public:
  JSONTransport(std::FILE *in, raw_ostream &out,
                JSONStreamStyle style = JSONStreamStyle::Standard,
                bool prettyOutput = false)
      : in(in), out(out), style(style), prettyOutput(prettyOutput) {}

  /// The following methods are used to send a message to the LSP client.
  void notify(StringRef method, llvm::json::Value params);
  void call(StringRef method, llvm::json::Value params, llvm::json::Value id);
  void reply(llvm::json::Value id, llvm::Expected<llvm::json::Value> result);

  /// Start executing the JSON-RPC transport.
  llvm::Error run(MessageHandler &handler);

private:
  /// Dispatches the given incoming json message to the message handler.
  bool handleMessage(llvm::json::Value msg, MessageHandler &handler);
  /// Writes the given message to the output stream.
  void sendMessage(llvm::json::Value msg);

  /// Read in a message from the input stream.
  LogicalResult readMessage(std::string &json) {
    return style == JSONStreamStyle::Delimited ? readDelimitedMessage(json)
                                               : readStandardMessage(json);
  }
  LogicalResult readDelimitedMessage(std::string &json);
  LogicalResult readStandardMessage(std::string &json);

  /// An output buffer used when building output messages.
  SmallVector<char, 0> outputBuffer;
  /// The input file stream.
  std::FILE *in;
  /// The output file stream.
  raw_ostream &out;
  /// The JSON stream style to use.
  JSONStreamStyle style;
  /// If the output JSON should be formatted for easier readability.
  bool prettyOutput;
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
    llvm::unique_function<void(const T &, llvm::json::Value id)>;

/// An `OutgoingRequestCallback` is invoked when an outgoing request to the
/// client receives a response in turn. It is passed the original request's ID,
/// as well as the response result.
template <typename T>
using OutgoingRequestCallback =
    std::function<void(llvm::json::Value, llvm::Expected<T>)>;

/// A handler used to process the incoming transport messages.
class MessageHandler {
public:
  MessageHandler(JSONTransport &transport) : transport(transport) {}

  bool onNotify(StringRef method, llvm::json::Value value);
  bool onCall(StringRef method, llvm::json::Value params, llvm::json::Value id);
  bool onReply(llvm::json::Value id, llvm::Expected<llvm::json::Value> result);

  template <typename T>
  static llvm::Expected<T> parse(const llvm::json::Value &raw,
                                 StringRef payloadName, StringRef payloadKind) {
    T result;
    llvm::json::Path::Root root;
    if (fromJSON(raw, result, root))
      return std::move(result);

    // Dump the relevant parts of the broken message.
    std::string context;
    llvm::raw_string_ostream os(context);
    root.printErrorContext(raw, os);

    // Report the error (e.g. to the client).
    return llvm::make_error<LSPError>(
        llvm::formatv("failed to decode {0} {1}: {2}", payloadName, payloadKind,
                      fmt_consume(root.getError())),
        ErrorCode::InvalidParams);
  }

  template <typename Param, typename Result, typename ThisT>
  void method(llvm::StringLiteral method, ThisT *thisPtr,
              void (ThisT::*handler)(const Param &, Callback<Result>)) {
    methodHandlers[method] = [method, handler,
                              thisPtr](llvm::json::Value rawParams,
                                       Callback<llvm::json::Value> reply) {
      llvm::Expected<Param> param = parse<Param>(rawParams, method, "request");
      if (!param)
        return reply(param.takeError());
      (thisPtr->*handler)(*param, std::move(reply));
    };
  }

  template <typename Param, typename ThisT>
  void notification(llvm::StringLiteral method, ThisT *thisPtr,
                    void (ThisT::*handler)(const Param &)) {
    notificationHandlers[method] = [method, handler,
                                    thisPtr](llvm::json::Value rawParams) {
      llvm::Expected<Param> param =
          parse<Param>(rawParams, method, "notification");
      if (!param) {
        return llvm::consumeError(
            llvm::handleErrors(param.takeError(), [](const LSPError &lspError) {
              Logger::error("JSON parsing error: {0}",
                            lspError.message.c_str());
            }));
      }
      (thisPtr->*handler)(*param);
    };
  }

  /// Create an OutgoingNotification object used for the given method.
  template <typename T>
  OutgoingNotification<T> outgoingNotification(llvm::StringLiteral method) {
    return [&, method](const T &params) {
      std::lock_guard<std::mutex> transportLock(transportOutputMutex);
      Logger::info("--> {0}", method);
      transport.notify(method, llvm::json::Value(params));
    };
  }

  /// Create an OutgoingRequest function that, when called, sends a request with
  /// the given method via the transport. Should the outgoing request be
  /// met with a response, the result JSON is parsed and the response callback
  /// is invoked.
  template <typename Param, typename Result>
  OutgoingRequest<Param>
  outgoingRequest(llvm::StringLiteral method,
                  OutgoingRequestCallback<Result> callback) {
    return [&, method, callback](const Param &param, llvm::json::Value id) {
      auto callbackWrapper = [method, callback = std::move(callback)](
                                 llvm::json::Value id,
                                 llvm::Expected<llvm::json::Value> value) {
        if (!value)
          return callback(std::move(id), value.takeError());

        std::string responseName = llvm::formatv("reply:{0}({1})", method, id);
        llvm::Expected<Result> result =
            parse<Result>(*value, responseName, "response");
        if (!result)
          return callback(std::move(id), result.takeError());

        return callback(std::move(id), *result);
      };

      {
        std::lock_guard<std::mutex> lock(responseHandlersMutex);
        responseHandlers.insert(
            {debugString(id), std::make_pair(method.str(), callbackWrapper)});
      }

      std::lock_guard<std::mutex> transportLock(transportOutputMutex);
      Logger::info("--> {0}({1})", method, id);
      transport.call(method, llvm::json::Value(param), id);
    };
  }

private:
  template <typename HandlerT>
  using HandlerMap = llvm::StringMap<llvm::unique_function<HandlerT>>;

  HandlerMap<void(llvm::json::Value)> notificationHandlers;
  HandlerMap<void(llvm::json::Value, Callback<llvm::json::Value>)>
      methodHandlers;

  /// A pair of (1) the original request's method name, and (2) the callback
  /// function to be invoked for responses.
  using ResponseHandlerTy =
      std::pair<std::string, OutgoingRequestCallback<llvm::json::Value>>;
  /// A mapping from request/response ID to response handler.
  llvm::StringMap<ResponseHandlerTy> responseHandlers;
  /// Mutex to guard insertion into the response handler map.
  std::mutex responseHandlersMutex;

  JSONTransport &transport;

  /// Mutex to guard sending output messages to the transport.
  std::mutex transportOutputMutex;
};

} // namespace lsp
} // namespace mlir

#endif
