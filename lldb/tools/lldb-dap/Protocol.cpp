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

  return O && O.map("event", E.event) && O.mapOptional("body", E.rawBody);
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

llvm::json::Value toJSON(lldb_dap::protocol::ErrorResponseBody const &ERB) {
  return llvm::json::Object{{"error", ERB.error}};
}

llvm::json::Value toJSON(std::map<std::string, std::string> const &KV) {
  llvm::json::Object Result;
  for (const auto &[K, V] : KV)
    Result.insert({K, V});
  return std::move(Result);
}

llvm::json::Value toJSON(lldb_dap::protocol::Message const &M) {
  return llvm::json::Object{{"id", M.id},
                            {"format", M.format},
                            {"showUser", M.showUser},
                            {"sendTelemetry", M.sendTelemetry},
                            {"variables", toJSON(*M.variables)},
                            {"url", M.url},
                            {"urlLabel", M.urlLabel}};
}

bool fromJSON(const llvm::json::Value &Params, Source::PresentationHint &PH,
              llvm::json::Path P) {
  auto rawHint = Params.getAsString();
  if (!rawHint) {
    P.report("expected a string");
    return false;
  }
  std::optional<Source::PresentationHint> hint =
      llvm::StringSwitch<std::optional<Source::PresentationHint>>(*rawHint)
          .Case("normal", Source::PresentationHint::normal)
          .Case("emphasize", Source::PresentationHint::emphasize)
          .Case("deemphasize", Source::PresentationHint::deemphasize)
          .Default(std::nullopt);
  if (!hint) {
    P.report("unexpected value");
    return false;
  }
  PH = *hint;
  return true;
}

bool fromJSON(const llvm::json::Value &Params, Source &S, llvm::json::Path P) {
  llvm::json::ObjectMapper O(Params, P);
  return O && O.mapOptional("name", S.name) && O.mapOptional("path", S.path) &&
         O.mapOptional("presentationHint", S.presentationHint) &&
         O.mapOptional("sourceReference", S.sourceReference) &&
         O.mapOptional("origin", S.origin);
}

bool fromJSON(const llvm::json::Value &Params, CancelArguments &CA,
              llvm::json::Path P) {
  llvm::json::ObjectMapper O(Params, P);
  return O && O.mapOptional("requestId", CA.requestId) &&
         O.mapOptional("progressId", CA.progressId);
}

llvm::json::Value toJSON(const Source &S) {
  llvm::json::Object Result;

  if (S.name)
    Result.insert({"name", S.name});
  if (S.path)
    Result.insert({"path", S.path});
  if (S.presentationHint)
    switch (*S.presentationHint) {
    case Source::PresentationHint::normal:
      Result.insert({"presentationHint", "normal"});
      break;
    case Source::PresentationHint::emphasize:
      Result.insert({"presentationHint", "emphasize"});
      break;
    case Source::PresentationHint::deemphasize:
      Result.insert({"presentationHint", "deemphasize"});
      break;
    }
  if (S.sourceReference)
    Result.insert({"sourceReference", S.sourceReference});
  if (S.origin)
    Result.insert({"origin", S.origin});

  return std::move(Result);
}

llvm::json::Value toJSON(const StackFrame &SF) {
  llvm::json::Object Result{{"id", SF.id},
                            {"name", SF.name},
                            {"line", SF.line},
                            {"column", SF.column}};

  if (SF.source)
    Result.insert({"source", SF.source});
  if (SF.endLine)
    Result.insert({"endLine", SF.endLine});
  if (SF.endColumn)
    Result.insert({"endColumn", SF.endColumn});
  if (SF.canRestart)
    Result.insert({"canRestart", SF.canRestart});
  if (SF.instructionPointerReference)
    Result.insert(
        {"instructionPointerReference", SF.instructionPointerReference});
  if (SF.presentationHint)
    switch (*SF.presentationHint) {
    case StackFrame::PresentationHint::normal:
      Result.insert({"presentationHint", "normal"});
      break;
    case StackFrame::PresentationHint::label:
      Result.insert({"presentationHint", "label"});
      break;
    case StackFrame::PresentationHint::subtle:
      Result.insert({"presentationHint", "subtle"});
      break;
    }
  return std::move(Result);
}

bool fromJSON(const llvm::json::Value &Params, SourceArguments &SA,
              llvm::json::Path P) {
  llvm::json::ObjectMapper O(Params, P);
  return O && O.mapOptional("source", SA.source) &&
         O.map("sourceReference", SA.sourceReference);
}

llvm::json::Value toJSON(const SourceResponseBody &SA) {
  llvm::json::Object Result{{"content", SA.content}};

  if (SA.mimeType)
    Result.insert({"mimeType", SA.mimeType});

  return std::move(Result);
}

bool fromJSON(const llvm::json::Value &Params, EvaluateArguments::Context &C,
              llvm::json::Path P) {
  auto rawContext = Params.getAsString();
  if (!rawContext) {
    P.report("expected a string");
    return false;
  }
  std::optional<EvaluateArguments::Context> context =
      llvm::StringSwitch<std::optional<EvaluateArguments::Context>>(*rawContext)
          .Case("repl", EvaluateArguments::Context::repl)
          .Case("watch", EvaluateArguments::Context::watch)
          .Case("clipboard", EvaluateArguments::Context::clipboard)
          .Case("hover", EvaluateArguments::Context::hover)
          .Case("variables", EvaluateArguments::Context::variables)
          .Default(std::nullopt);
  if (!context) {
    P.report("unexpected value");
    return false;
  }
  C = *context;
  return true;
}

bool fromJSON(const llvm::json::Value &Params, EvaluateArguments &EA,
              llvm::json::Path P) {
  llvm::json::ObjectMapper O(Params, P);
  return O && O.map("expression", EA.expression) &&
         O.mapOptional("frameId", EA.frameId) &&
         O.mapOptional("line", EA.line) && O.mapOptional("column", EA.column) &&
         O.mapOptional("source", EA.source) &&
         O.mapOptional("context", EA.context);
}

llvm::json::Value toJSON(const EvaluateResponseBody &ERB) {
  llvm::json::Object Result{{"result", ERB.result},
                            {"variablesReference", ERB.variablesReference}};

  if (ERB.type)
    Result.insert({"type", ERB.type});
  if (ERB.presentationHint)
    Result.insert({"presentationHint", ERB.presentationHint});
  if (ERB.namedVariables)
    Result.insert({"namedVariables", ERB.namedVariables});
  if (ERB.indexedVariables)
    Result.insert({"indexedVariables", ERB.indexedVariables});
  if (ERB.memoryReference)
    Result.insert({"memoryReference", ERB.memoryReference});
  if (ERB.valueLocationReference)
    Result.insert({"valueLocationReference", ERB.valueLocationReference});

  return std::move(Result);
}

llvm::json::Value toJSON(const VariablePresentationHint::Kind &K) {
  switch (K) {
  case VariablePresentationHint::Kind::property_:
    return "property";
  case VariablePresentationHint::Kind::method_:
    return "method";
  case VariablePresentationHint::Kind::class_:
    return "class";
  case VariablePresentationHint::Kind::data_:
    return "data";
  case VariablePresentationHint::Kind::event_:
    return "event";
  case VariablePresentationHint::Kind::baseClass_:
    return "baseClass";
  case VariablePresentationHint::Kind::innerClass_:
    return "innerClass";
  case VariablePresentationHint::Kind::interface_:
    return "interface";
  case VariablePresentationHint::Kind::mostDerivedClass_:
    return "mostDerivedClass";
  case VariablePresentationHint::Kind::virtual_:
    return "virtual";
  case VariablePresentationHint::Kind::dataBreakpoint_:
    return "dataBreakpoint";
  }
}

llvm::json::Value toJSON(const VariablePresentationHint::Attributes &A) {
  switch (A) {
  case VariablePresentationHint::Attributes::static_:
    return "static";
  case VariablePresentationHint::Attributes::constant:
    return "constant";
  case VariablePresentationHint::Attributes::readOnly:
    return "readOnly";
  case VariablePresentationHint::Attributes::rawString:
    return "rawString";
  case VariablePresentationHint::Attributes::hasObjectId:
    return "hasObjectId";
  case VariablePresentationHint::Attributes::canHaveObjectId:
    return "canHaveObjectId";
  case VariablePresentationHint::Attributes::hasSideEffects:
    return "hasSideEffects";
  case VariablePresentationHint::Attributes::hasDataBreakpoint:
    return "hasDataBreakpoint";
  }
}

llvm::json::Value toJSON(const VariablePresentationHint::Visibility &V) {
  switch (V) {
  case VariablePresentationHint::Visibility::public_:
    return "public";
  case VariablePresentationHint::Visibility::private_:
    return "private";
  case VariablePresentationHint::Visibility::protected_:
    return "protected";
  case VariablePresentationHint::Visibility::internal_:
    return "internal";
  case VariablePresentationHint::Visibility::final_:
    return "final";
  }
}

llvm::json::Value toJSON(const VariablePresentationHint &VPH) {
  llvm::json::Object Result;
  if (VPH.kind)
    Result.insert({"kind", VPH.kind});
  if (VPH.attributes)
    Result.insert({"attributes", VPH.attributes});
  if (VPH.visibility)
    Result.insert({"visibility", VPH.visibility});
  if (VPH.lazy)
    Result.insert({"lazy", VPH.lazy});
  return std::move(Result);
}

llvm::json::Value toJSON(const Event &E) {
  llvm::json::Object Result{
      {"type", "event"},
      {"seq", 0},
      {"event", E.event},
  };
  if (E.rawBody)
    Result.insert({"body", E.rawBody});
  return std::move(Result);
}

llvm::json::Value toJSON(const ExitedEventBody &EEB) {
  return llvm::json::Object{{"exitCode", EEB.exitCode}};
}

} // namespace protocol
} // namespace lldb_dap
