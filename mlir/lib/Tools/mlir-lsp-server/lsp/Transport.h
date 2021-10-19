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

#ifndef LIB_MLIR_TOOLS_MLIRLSPSERVER_LSP_TRANSPORT_H_
#define LIB_MLIR_TOOLS_MLIRLSPSERVER_LSP_TRANSPORT_H_

#include "Logging.h"
#include "Protocol.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
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
      llvm::Expected<Param> param = parse<Param>(rawParams, method, "request");
      if (!param)
        return llvm::consumeError(param.takeError());
      (thisPtr->*handler)(*param);
    };
  }

  /// Create an OutgoingNotification object used for the given method.
  template <typename T>
  OutgoingNotification<T> outgoingNotification(llvm::StringLiteral method) {
    return [&, method](const T &params) {
      Logger::info("--> {0}", method);
      transport.notify(method, llvm::json::Value(params));
    };
  }

private:
  template <typename HandlerT>
  using HandlerMap = llvm::StringMap<llvm::unique_function<HandlerT>>;

  HandlerMap<void(llvm::json::Value)> notificationHandlers;
  HandlerMap<void(llvm::json::Value, Callback<llvm::json::Value>)>
      methodHandlers;

  JSONTransport &transport;
};

} // namespace lsp
} // namespace mlir

#endif
