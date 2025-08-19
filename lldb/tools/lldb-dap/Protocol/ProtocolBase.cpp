//===-- ProtocolBase.cpp --------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Protocol/ProtocolBase.h"
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

namespace lldb_dap::protocol {

enum MessageType : unsigned {
  eMessageTypeRequest,
  eMessageTypeResponse,
  eMessageTypeEvent
};

bool fromJSON(const json::Value &Params, MessageType &M, json::Path P) {
  auto rawType = Params.getAsString();
  if (!rawType) {
    P.report("expected a string");
    return false;
  }
  std::optional<MessageType> type =
      StringSwitch<std::optional<MessageType>>(*rawType)
          .Case("request", eMessageTypeRequest)
          .Case("response", eMessageTypeResponse)
          .Case("event", eMessageTypeEvent)
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

  if (R.arguments)
    Result.insert({"arguments", R.arguments});

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

  if (type != eMessageTypeRequest) {
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

  return mapRaw(Params, "arguments", R.arguments, P);
}

bool operator==(const Request &a, const Request &b) {
  return a.seq == b.seq && a.command == b.command && a.arguments == b.arguments;
}

json::Value toJSON(const Response &R) {
  json::Object Result{{"type", "response"},
                      {"seq", 0},
                      {"command", R.command},
                      {"request_seq", R.request_seq},
                      {"success", R.success}};

  if (R.message) {
    assert(!R.success && "message can only be used if success is false");
    if (const auto *messageEnum = std::get_if<ResponseMessage>(&*R.message)) {
      switch (*messageEnum) {
      case eResponseMessageCancelled:
        Result.insert({"message", "cancelled"});
        break;
      case eResponseMessageNotStopped:
        Result.insert({"message", "notStopped"});
        break;
      }
    } else if (const auto *messageString =
                   std::get_if<std::string>(&*R.message)) {
      Result.insert({"message", *messageString});
    }
  }

  if (R.body)
    Result.insert({"body", R.body});

  return std::move(Result);
}

bool fromJSON(json::Value const &Params,
              std::variant<ResponseMessage, std::string> &M, json::Path P) {
  auto rawMessage = Params.getAsString();
  if (!rawMessage) {
    P.report("expected a string");
    return false;
  }
  std::optional<ResponseMessage> message =
      StringSwitch<std::optional<ResponseMessage>>(*rawMessage)
          .Case("cancelled", eResponseMessageCancelled)
          .Case("notStopped", eResponseMessageNotStopped)
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

  if (type != eMessageTypeResponse) {
    P.field("type").report("expected to be 'response'");
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

  return O.map("success", R.success) && O.map("message", R.message) &&
         mapRaw(Params, "body", R.body, P);
}

bool operator==(const Response &a, const Response &b) {
  return a.request_seq == b.request_seq && a.command == b.command &&
         a.success == b.success && a.message == b.message && a.body == b.body;
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

  if (E.body)
    Result.insert({"body", E.body});

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

  if (type != eMessageTypeEvent) {
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

  return mapRaw(Params, "body", E.body, P);
}

bool operator==(const Event &a, const Event &b) {
  return a.event == b.event && a.body == b.body;
}

bool fromJSON(const json::Value &Params, Message &PM, json::Path P) {
  json::ObjectMapper O(Params, P);
  if (!O)
    return false;

  MessageType type;
  if (!O.map("type", type))
    return false;

  switch (type) {
  case eMessageTypeRequest: {
    Request req;
    if (!fromJSON(Params, req, P))
      return false;
    PM = std::move(req);
    return true;
  }
  case eMessageTypeResponse: {
    Response resp;
    if (!fromJSON(Params, resp, P))
      return false;
    PM = std::move(resp);
    return true;
  }
  case eMessageTypeEvent:
    Event evt;
    if (!fromJSON(Params, evt, P))
      return false;
    PM = std::move(evt);
    return true;
  }
  llvm_unreachable("unhandled message type request.");
}

json::Value toJSON(const Message &M) {
  return std::visit([](auto &M) { return toJSON(M); }, M);
}

json::Value toJSON(const ErrorResponseBody &E) {
  json::Object result{};

  if (E.error)
    result.insert({"error", *E.error});

  return result;
}

} // namespace lldb_dap::protocol
