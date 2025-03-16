//===-- Protocol.cpp ------------------------------------------------------===//
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

using namespace llvm;

static bool mapRaw(const json::Value &Params, StringLiteral Prop,
                   std::optional<json::Value> &V, json::Path P) {
  const auto *O = Params.getAsObject();
  if (!O) {
    P.report("expected object");
    return false;
  }
  const json::Value *E = O->get(Prop);
  if (E)
    V = std::move(*E);
  return true;
}

namespace lldb_dap {
namespace protocol {

enum class MessageType { request, response, event };

bool fromJSON(const json::Value &Params, MessageType &M, json::Path P) {
  auto rawType = Params.getAsString();
  if (!rawType) {
    P.report("expected a string");
    return false;
  }
  std::optional<MessageType> type =
      StringSwitch<std::optional<MessageType>>(*rawType)
          .Case("request", MessageType::request)
          .Case("response", MessageType::response)
          .Case("event", MessageType::event)
          .Default(std::nullopt);
  if (!type) {
    P.report("unexpected value, expected 'request', 'response' or 'event'");
    return false;
  }
  M = *type;
  return true;
}

json::Value toJSON(const Request &R) {
  json::Object Result{
      {"type", "request"},
      {"seq", R.seq},
      {"command", R.command},
  };

  if (R.rawArguments)
    Result.insert({"arguments", R.rawArguments});

  return std::move(Result);
}

bool fromJSON(json::Value const &Params, Request &R, json::Path P) {
  json::ObjectMapper O(Params, P);
  if (!O)
    return false;

  MessageType type;
  if (!O.map("type", type) || !O.map("command", R.command) ||
      !O.map("seq", R.seq))
    return false;

  if (type != MessageType::request) {
    P.field("type").report("expected to be 'request'");
    return false;
  }

  if (R.command.empty()) {
    P.field("command").report("expected to not be ''");
    return false;
  }

  if (!R.seq) {
    P.field("seq").report("expected to not be '0'");
    return false;
  }

  return mapRaw(Params, "arguments", R.rawArguments, P);
}

json::Value toJSON(const Response &R) {
  json::Object Result{{"type", "response"},
                      {"seq", 0},
                      {"command", R.command},
                      {"request_seq", R.request_seq},
                      {"success", R.success}};

  if (R.message) {
    assert(!R.success && "message can only be used if success is false");
    if (const auto *messageEnum = std::get_if<Response::Message>(&*R.message)) {
      switch (*messageEnum) {
      case Response::Message::cancelled:
        Result.insert({"message", "cancelled"});
        break;
      case Response::Message::notStopped:
        Result.insert({"message", "notStopped"});
        break;
      }
    } else if (const auto *messageString =
                   std::get_if<std::string>(&*R.message)) {
      Result.insert({"message", *messageString});
    }
  }

  if (R.rawBody)
    Result.insert({"body", R.rawBody});

  return std::move(Result);
}

bool fromJSON(json::Value const &Params,
              std::variant<Response::Message, std::string> &M, json::Path P) {
  auto rawMessage = Params.getAsString();
  if (!rawMessage) {
    P.report("expected a string");
    return false;
  }
  std::optional<Response::Message> message =
      StringSwitch<std::optional<Response::Message>>(*rawMessage)
          .Case("cancelled", Response::Message::cancelled)
          .Case("notStopped", Response::Message::notStopped)
          .Default(std::nullopt);
  if (message)
    M = *message;
  else if (!rawMessage->empty())
    M = rawMessage->str();
  return true;
}

bool fromJSON(json::Value const &Params, Response &R, json::Path P) {
  json::ObjectMapper O(Params, P);
  if (!O)
    return false;

  MessageType type;
  int64_t seq;
  if (!O.map("type", type) || !O.map("seq", seq) ||
      !O.map("command", R.command) || !O.map("request_seq", R.request_seq))
    return false;

  if (type != MessageType::response) {
    P.field("type").report("expected to be 'response'");
    return false;
  }

  if (seq != 0) {
    P.field("seq").report("expected to be '0'");
    return false;
  }

  if (R.command.empty()) {
    P.field("command").report("expected to not be ''");
    return false;
  }

  if (R.request_seq == 0) {
    P.field("request_seq").report("expected to not be '0'");
    return false;
  }

  return O.map("success", R.success) && O.mapOptional("message", R.message) &&
         mapRaw(Params, "body", R.rawBody, P);
}

json::Value toJSON(const ErrorMessage &EM) {
  json::Object Result{{"id", EM.id}, {"format", EM.format}};

  if (EM.variables) {
    json::Object variables;
    for (auto &var : *EM.variables)
      variables[var.first] = var.second;
    Result.insert({"variables", std::move(variables)});
  }
  if (EM.sendTelemetry)
    Result.insert({"sendTelemetry", EM.sendTelemetry});
  if (EM.showUser)
    Result.insert({"showUser", EM.showUser});
  if (EM.url)
    Result.insert({"url", EM.url});
  if (EM.urlLabel)
    Result.insert({"urlLabel", EM.urlLabel});

  return std::move(Result);
}

bool fromJSON(json::Value const &Params, ErrorMessage &EM, json::Path P) {
  json::ObjectMapper O(Params, P);
  return O && O.map("id", EM.id) && O.map("format", EM.format) &&
         O.map("variables", EM.variables) &&
         O.map("sendTelemetry", EM.sendTelemetry) &&
         O.map("showUser", EM.showUser) && O.map("url", EM.url) &&
         O.map("urlLabel", EM.urlLabel);
}

json::Value toJSON(const Event &E) {
  json::Object Result{
      {"type", "event"},
      {"seq", 0},
      {"event", E.event},
  };

  if (E.rawBody)
    Result.insert({"body", E.rawBody});

  return std::move(Result);
}

bool fromJSON(json::Value const &Params, Event &E, json::Path P) {
  json::ObjectMapper O(Params, P);
  if (!O)
    return false;

  MessageType type;
  int64_t seq;
  if (!O.map("type", type) || !O.map("seq", seq) || !O.map("event", E.event))
    return false;

  if (type != MessageType::event) {
    P.field("type").report("expected to be 'event'");
    return false;
  }

  if (seq != 0) {
    P.field("seq").report("expected to be '0'");
    return false;
  }

  if (E.event.empty()) {
    P.field("event").report("expected to not be ''");
    return false;
  }

  return mapRaw(Params, "body", E.rawBody, P);
}

bool fromJSON(const json::Value &Params, Message &PM, json::Path P) {
  json::ObjectMapper O(Params, P);
  if (!O)
    return false;

  MessageType type;
  if (!O.map("type", type))
    return false;

  switch (type) {
  case MessageType::request: {
    Request req;
    if (!fromJSON(Params, req, P))
      return false;
    PM = std::move(req);
    return true;
  }
  case MessageType::response: {
    Response resp;
    if (!fromJSON(Params, resp, P))
      return false;
    PM = std::move(resp);
    return true;
  }
  case MessageType::event:
    Event evt;
    if (!fromJSON(Params, evt, P))
      return false;
    PM = std::move(evt);
    return true;
  }
}

json::Value toJSON(const Message &M) {
  return std::visit([](auto &M) { return toJSON(M); }, M);
}

} // namespace protocol
} // namespace lldb_dap
