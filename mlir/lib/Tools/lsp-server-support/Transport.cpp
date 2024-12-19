//===--- JSONTransport.cpp - sending and receiving LSP messages over JSON -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Tools/lsp-server-support/Transport.h"
#include "mlir/Support/ToolUtilities.h"
#include "mlir/Tools/lsp-server-support/Logging.h"
#include "mlir/Tools/lsp-server-support/Protocol.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/Errno.h"
#include "llvm/Support/Error.h"
#include <optional>
#include <system_error>
#include <utility>

using namespace mlir;
using namespace mlir::lsp;

//===----------------------------------------------------------------------===//
// Reply
//===----------------------------------------------------------------------===//

namespace {
/// Function object to reply to an LSP call.
/// Each instance must be called exactly once, otherwise:
///  - if there was no reply, an error reply is sent
///  - if there were multiple replies, only the first is sent
class Reply {
public:
  Reply(const llvm::json::Value &id, StringRef method, JSONTransport &transport,
        std::mutex &transportOutputMutex);
  Reply(Reply &&other);
  Reply &operator=(Reply &&) = delete;
  Reply(const Reply &) = delete;
  Reply &operator=(const Reply &) = delete;

  void operator()(llvm::Expected<llvm::json::Value> reply);

private:
  std::string method;
  std::atomic<bool> replied = {false};
  llvm::json::Value id;
  JSONTransport *transport;
  std::mutex &transportOutputMutex;
};
} // namespace

Reply::Reply(const llvm::json::Value &id, llvm::StringRef method,
             JSONTransport &transport, std::mutex &transportOutputMutex)
    : method(method), id(id), transport(&transport),
      transportOutputMutex(transportOutputMutex) {}

Reply::Reply(Reply &&other)
    : method(other.method), replied(other.replied.load()),
      id(std::move(other.id)), transport(other.transport),
      transportOutputMutex(other.transportOutputMutex) {
  other.transport = nullptr;
}

void Reply::operator()(llvm::Expected<llvm::json::Value> reply) {
  if (replied.exchange(true)) {
    Logger::error("Replied twice to message {0}({1})", method, id);
    assert(false && "must reply to each call only once!");
    return;
  }
  assert(transport && "expected valid transport to reply to");

  std::lock_guard<std::mutex> transportLock(transportOutputMutex);
  if (reply) {
    Logger::info("--> reply:{0}({1})", method, id);
    transport->reply(std::move(id), std::move(reply));
  } else {
    llvm::Error error = reply.takeError();
    Logger::info("--> reply:{0}({1}): {2}", method, id, error);
    transport->reply(std::move(id), std::move(error));
  }
}

//===----------------------------------------------------------------------===//
// MessageHandler
//===----------------------------------------------------------------------===//

bool MessageHandler::onNotify(llvm::StringRef method, llvm::json::Value value) {
  Logger::info("--> {0}", method);

  if (method == "exit")
    return false;
  if (method == "$cancel") {
    // TODO: Add support for cancelling requests.
  } else {
    auto it = notificationHandlers.find(method);
    if (it != notificationHandlers.end())
      it->second(std::move(value));
  }
  return true;
}

bool MessageHandler::onCall(llvm::StringRef method, llvm::json::Value params,
                            llvm::json::Value id) {
  Logger::info("--> {0}({1})", method, id);

  Reply reply(id, method, transport, transportOutputMutex);

  auto it = methodHandlers.find(method);
  if (it != methodHandlers.end()) {
    it->second(std::move(params), std::move(reply));
  } else {
    reply(llvm::make_error<LSPError>("method not found: " + method.str(),
                                     ErrorCode::MethodNotFound));
  }
  return true;
}

bool MessageHandler::onReply(llvm::json::Value id,
                             llvm::Expected<llvm::json::Value> result) {
  // Find the response handler in the mapping. If it exists, move it out of the
  // mapping and erase it.
  ResponseHandlerTy responseHandler;
  {
    std::lock_guard<std::mutex> responseHandlersLock(responseHandlersMutex);
    auto it = responseHandlers.find(debugString(id));
    if (it != responseHandlers.end()) {
      responseHandler = std::move(it->second);
      responseHandlers.erase(it);
    }
  }

  // If we found a response handler, invoke it. Otherwise, log an error.
  if (responseHandler.second) {
    Logger::info("--> reply:{0}({1})", responseHandler.first, id);
    responseHandler.second(std::move(id), std::move(result));
  } else {
    Logger::error(
        "received a reply with ID {0}, but there was no such outgoing request",
        id);
    if (!result)
      llvm::consumeError(result.takeError());
  }
  return true;
}

//===----------------------------------------------------------------------===//
// JSONTransport
//===----------------------------------------------------------------------===//

/// Encode the given error as a JSON object.
static llvm::json::Object encodeError(llvm::Error error) {
  std::string message;
  ErrorCode code = ErrorCode::UnknownErrorCode;
  auto handlerFn = [&](const LSPError &lspError) -> llvm::Error {
    message = lspError.message;
    code = lspError.code;
    return llvm::Error::success();
  };
  if (llvm::Error unhandled = llvm::handleErrors(std::move(error), handlerFn))
    message = llvm::toString(std::move(unhandled));

  return llvm::json::Object{
      {"message", std::move(message)},
      {"code", int64_t(code)},
  };
}

/// Decode the given JSON object into an error.
llvm::Error decodeError(const llvm::json::Object &o) {
  StringRef msg = o.getString("message").value_or("Unspecified error");
  if (std::optional<int64_t> code = o.getInteger("code"))
    return llvm::make_error<LSPError>(msg.str(), ErrorCode(*code));
  return llvm::make_error<llvm::StringError>(llvm::inconvertibleErrorCode(),
                                             msg.str());
}

void JSONTransport::notify(StringRef method, llvm::json::Value params) {
  sendMessage(llvm::json::Object{
      {"jsonrpc", "2.0"},
      {"method", method},
      {"params", std::move(params)},
  });
}
void JSONTransport::call(StringRef method, llvm::json::Value params,
                         llvm::json::Value id) {
  sendMessage(llvm::json::Object{
      {"jsonrpc", "2.0"},
      {"id", std::move(id)},
      {"method", method},
      {"params", std::move(params)},
  });
}
void JSONTransport::reply(llvm::json::Value id,
                          llvm::Expected<llvm::json::Value> result) {
  if (result) {
    return sendMessage(llvm::json::Object{
        {"jsonrpc", "2.0"},
        {"id", std::move(id)},
        {"result", std::move(*result)},
    });
  }

  sendMessage(llvm::json::Object{
      {"jsonrpc", "2.0"},
      {"id", std::move(id)},
      {"error", encodeError(result.takeError())},
  });
}

llvm::Error JSONTransport::run(MessageHandler &handler) {
  std::string json;
  while (!feof(in)) {
    if (ferror(in)) {
      return llvm::errorCodeToError(
          std::error_code(errno, std::system_category()));
    }

    if (succeeded(readMessage(json))) {
      if (llvm::Expected<llvm::json::Value> doc = llvm::json::parse(json)) {
        if (!handleMessage(std::move(*doc), handler))
          return llvm::Error::success();
      } else {
        Logger::error("JSON parse error: {0}", llvm::toString(doc.takeError()));
      }
    }
  }
  return llvm::errorCodeToError(std::make_error_code(std::errc::io_error));
}

void JSONTransport::sendMessage(llvm::json::Value msg) {
  outputBuffer.clear();
  llvm::raw_svector_ostream os(outputBuffer);
  os << llvm::formatv(prettyOutput ? "{0:2}\n" : "{0}", msg);
  out << "Content-Length: " << outputBuffer.size() << "\r\n\r\n"
      << outputBuffer;
  out.flush();
  Logger::debug(">>> {0}\n", outputBuffer);
}

bool JSONTransport::handleMessage(llvm::json::Value msg,
                                  MessageHandler &handler) {
  // Message must be an object with "jsonrpc":"2.0".
  llvm::json::Object *object = msg.getAsObject();
  if (!object ||
      object->getString("jsonrpc") != std::optional<StringRef>("2.0"))
    return false;

  // `id` may be any JSON value. If absent, this is a notification.
  std::optional<llvm::json::Value> id;
  if (llvm::json::Value *i = object->get("id"))
    id = std::move(*i);
  std::optional<StringRef> method = object->getString("method");

  // This is a response.
  if (!method) {
    if (!id)
      return false;
    if (auto *err = object->getObject("error"))
      return handler.onReply(std::move(*id), decodeError(*err));
    // result should be given, use null if not.
    llvm::json::Value result = nullptr;
    if (llvm::json::Value *r = object->get("result"))
      result = std::move(*r);
    return handler.onReply(std::move(*id), std::move(result));
  }

  // Params should be given, use null if not.
  llvm::json::Value params = nullptr;
  if (llvm::json::Value *p = object->get("params"))
    params = std::move(*p);

  if (id)
    return handler.onCall(*method, std::move(params), std::move(*id));
  return handler.onNotify(*method, std::move(params));
}

/// Tries to read a line up to and including \n.
/// If failing, feof(), ferror(), or shutdownRequested() will be set.
LogicalResult readLine(std::FILE *in, SmallVectorImpl<char> &out) {
  // Big enough to hold any reasonable header line. May not fit content lines
  // in delimited mode, but performance doesn't matter for that mode.
  static constexpr int bufSize = 128;
  size_t size = 0;
  out.clear();
  for (;;) {
    out.resize_for_overwrite(size + bufSize);
    if (!std::fgets(&out[size], bufSize, in))
      return failure();

    clearerr(in);

    // If the line contained null bytes, anything after it (including \n) will
    // be ignored. Fortunately this is not a legal header or JSON.
    size_t read = std::strlen(&out[size]);
    if (read > 0 && out[size + read - 1] == '\n') {
      out.resize(size + read);
      return success();
    }
    size += read;
  }
}

// Returns std::nullopt when:
//  - ferror(), feof(), or shutdownRequested() are set.
//  - Content-Length is missing or empty (protocol error)
LogicalResult JSONTransport::readStandardMessage(std::string &json) {
  // A Language Server Protocol message starts with a set of HTTP headers,
  // delimited  by \r\n, and terminated by an empty line (\r\n).
  unsigned long long contentLength = 0;
  llvm::SmallString<128> line;
  while (true) {
    if (feof(in) || ferror(in) || failed(readLine(in, line)))
      return failure();

    // Content-Length is a mandatory header, and the only one we handle.
    StringRef lineRef = line;
    if (lineRef.consume_front("Content-Length: ")) {
      llvm::getAsUnsignedInteger(lineRef.trim(), 0, contentLength);
    } else if (!lineRef.trim().empty()) {
      // It's another header, ignore it.
      continue;
    } else {
      // An empty line indicates the end of headers. Go ahead and read the JSON.
      break;
    }
  }

  // The fuzzer likes crashing us by sending "Content-Length: 9999999999999999"
  if (contentLength == 0 || contentLength > 1 << 30)
    return failure();

  json.resize(contentLength);
  for (size_t pos = 0, read; pos < contentLength; pos += read) {
    read = std::fread(&json[pos], 1, contentLength - pos, in);
    if (read == 0)
      return failure();

    // If we're done, the error was transient. If we're not done, either it was
    // transient or we'll see it again on retry.
    clearerr(in);
    pos += read;
  }
  return success();
}

/// For lit tests we support a simplified syntax:
/// - messages are delimited by '// -----' on a line by itself
/// - lines starting with // are ignored.
/// This is a testing path, so favor simplicity over performance here.
/// When returning failure: feof(), ferror(), or shutdownRequested() will be
/// set.
LogicalResult JSONTransport::readDelimitedMessage(std::string &json) {
  json.clear();
  llvm::SmallString<128> line;
  while (succeeded(readLine(in, line))) {
    StringRef lineRef = line.str().trim();
    if (lineRef.starts_with("//")) {
      // Found a delimiter for the message.
      if (lineRef == kDefaultSplitMarker)
        break;
      continue;
    }

    json += line;
  }

  return failure(ferror(in));
}
