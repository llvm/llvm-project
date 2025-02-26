//===-- Protocol.cpp --------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Protocol.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/JSON.h"
#include <optional>
#include <utility>

namespace llvm {
namespace json {
bool fromJSON(const llvm::json::Value &Params, llvm::json::Value &V,
              llvm::json::Path P) {
  V = std::move(Params);
  return true;
}
} // namespace json
} // namespace llvm

namespace lldb_dap {
namespace protocol {

enum class MessageType { request, response, event };

bool fromJSON(const llvm::json::Value &Params, MessageType &M,
              llvm::json::Path P) {
  auto rawType = Params.getAsString();
  if (!rawType) {
    P.report("expected a string");
    return false;
  }
  std::optional<MessageType> type =
      llvm::StringSwitch<std::optional<MessageType>>(*rawType)
          .Case("request", MessageType::request)
          .Case("response", MessageType::response)
          .Case("event", MessageType::event)
          .Default(std::nullopt);
  if (!type) {
    P.report("unexpected value");
    return false;
  }
  M = *type;
  return true;
}

llvm::json::Value toJSON(const Request &R) {
  llvm::json::Object Result{
      {"type", "request"},
      {"seq", R.seq},
      {"command", R.command},
  };
  if (R.rawArguments)
    Result.insert({"arguments", R.rawArguments});
  return std::move(Result);
}

bool fromJSON(llvm::json::Value const &Params, Request &R, llvm::json::Path P) {
  llvm::json::ObjectMapper O(Params, P);
  MessageType type;
  if (!O.map("type", type)) {
    return false;
  }
  if (type != MessageType::request) {
    P.field("type").report("expected to be 'request'");
    return false;
  }

  return O && O.map("command", R.command) && O.map("seq", R.seq) &&
         O.map("arguments", R.rawArguments);
}

llvm::json::Value toJSON(const Response &R) {
  llvm::json::Object Result{{"type", "response"},
                            {"req", 0},
                            {"command", R.command},
                            {"request_seq", R.request_seq},
                            {"success", R.success}};

  if (R.message)
    Result.insert({"message", R.message});
  if (R.rawBody)
    Result.insert({"body", R.rawBody});

  return std::move(Result);
}

bool fromJSON(llvm::json::Value const &Params, Response &R,
              llvm::json::Path P) {
  llvm::json::ObjectMapper O(Params, P);
  MessageType type;
  if (!O.map("type", type)) {
    return false;
  }
  if (type != MessageType::response) {
    P.field("type").report("expected to be 'response'");
    return false;
  }
  return O && O.map("command", R.command) &&
         O.map("request_seq", R.request_seq) && O.map("success", R.success) &&
         O.mapOptional("message", R.message) &&
         O.mapOptional("body", R.rawBody);
}

llvm::json::Value toJSON(const Event &E) {
  llvm::json::Object Result{
      {"type", "event"},
      {"seq", 0},
      {"event", E.event},
  };
  if (E.rawBody)
    Result.insert({"body", E.rawBody});
  if (E.statistics)
    Result.insert({"statistics", E.statistics});
  return std::move(Result);
}

bool fromJSON(llvm::json::Value const &Params, Event &E, llvm::json::Path P) {
  llvm::json::ObjectMapper O(Params, P);
  MessageType type;
  if (!O.map("type", type)) {
    return false;
  }
  if (type != MessageType::event) {
    P.field("type").report("expected to be 'event'");
    return false;
  }

  return O && O.map("event", E.event) && O.mapOptional("body", E.rawBody) &&
         O.mapOptional("statistics", E.statistics);
}

bool fromJSON(const llvm::json::Value &Params, ProtocolMessage &PM,
              llvm::json::Path P) {
  llvm::json::ObjectMapper O(Params, P);
  if (!O)
    return false;

  MessageType type;
  if (!O.map("type", type))
    return false;

  switch (type) {
  case MessageType::request: {
    Request req;
    if (!fromJSON(Params, req, P)) {
      return false;
    }
    PM = std::move(req);
    return true;
  }
  case MessageType::response: {
    Response resp;
    if (!fromJSON(Params, resp, P)) {
      return false;
    }
    PM = std::move(resp);
    return true;
  }
  case MessageType::event:
    Event evt;
    if (!fromJSON(Params, evt, P)) {
      return false;
    }
    PM = std::move(evt);
    return true;
  }
  llvm_unreachable("Unsupported protocol message");
}

llvm::json::Value toJSON(const ProtocolMessage &PM) {
  if (auto const *Req = std::get_if<Request>(&PM)) {
    return toJSON(*Req);
  }
  if (auto const *Resp = std::get_if<Response>(&PM)) {
    return toJSON(*Resp);
  }
  if (auto const *Evt = std::get_if<Event>(&PM)) {
    return toJSON(*Evt);
  }
  llvm_unreachable("Unsupported protocol message");
}

} // namespace protocol
} // namespace lldb_dap
