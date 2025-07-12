//===- Protocol.cpp -------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Protocol.h"
#include "llvm/Support/JSON.h"

using namespace llvm;

namespace lldb_private::mcp::protocol {

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

llvm::json::Value toJSON(const Request &R) {
  json::Object Result{{"jsonrpc", "2.0"}, {"id", R.id}, {"method", R.method}};
  if (R.params)
    Result.insert({"params", R.params});
  return Result;
}

bool fromJSON(const llvm::json::Value &V, Request &R, llvm::json::Path P) {
  llvm::json::ObjectMapper O(V, P);
  if (!O || !O.map("id", R.id) || !O.map("method", R.method))
    return false;
  return mapRaw(V, "params", R.params, P);
}

llvm::json::Value toJSON(const ErrorInfo &EI) {
  llvm::json::Object Result{{"code", EI.code}, {"message", EI.message}};
  if (!EI.data.empty())
    Result.insert({"data", EI.data});
  return Result;
}

bool fromJSON(const llvm::json::Value &V, ErrorInfo &EI, llvm::json::Path P) {
  llvm::json::ObjectMapper O(V, P);
  return O && O.map("code", EI.code) && O.map("message", EI.message) &&
         O.mapOptional("data", EI.data);
}

llvm::json::Value toJSON(const Error &E) {
  return json::Object{{"jsonrpc", "2.0"}, {"id", E.id}, {"error", E.error}};
}

bool fromJSON(const llvm::json::Value &V, Error &E, llvm::json::Path P) {
  llvm::json::ObjectMapper O(V, P);
  return O && O.map("id", E.id) && O.map("error", E.error);
}

llvm::json::Value toJSON(const Response &R) {
  llvm::json::Object Result{{"jsonrpc", "2.0"}, {"id", R.id}};
  if (R.result)
    Result.insert({"result", R.result});
  if (R.error)
    Result.insert({"error", R.error});
  return Result;
}

bool fromJSON(const llvm::json::Value &V, Response &R, llvm::json::Path P) {
  llvm::json::ObjectMapper O(V, P);
  if (!O || !O.map("id", R.id) || !O.map("error", R.error))
    return false;
  return mapRaw(V, "result", R.result, P);
}

llvm::json::Value toJSON(const Notification &N) {
  llvm::json::Object Result{{"jsonrpc", "2.0"}, {"method", N.method}};
  if (N.params)
    Result.insert({"params", N.params});
  return Result;
}

bool fromJSON(const llvm::json::Value &V, Notification &N, llvm::json::Path P) {
  llvm::json::ObjectMapper O(V, P);
  if (!O || !O.map("method", N.method))
    return false;
  auto *Obj = V.getAsObject();
  if (!Obj)
    return false;
  if (auto *Params = Obj->get("params"))
    N.params = *Params;
  return true;
}

llvm::json::Value toJSON(const ToolCapability &TC) {
  return llvm::json::Object{{"listChanged", TC.listChanged}};
}

bool fromJSON(const llvm::json::Value &V, ToolCapability &TC,
              llvm::json::Path P) {
  llvm::json::ObjectMapper O(V, P);
  return O && O.map("listChanged", TC.listChanged);
}

llvm::json::Value toJSON(const ResourceCapability &RC) {
  return llvm::json::Object{{"listChanged", RC.listChanged},
                            {"subscribe", RC.subscribe}};
}

bool fromJSON(const llvm::json::Value &V, ResourceCapability &RC,
              llvm::json::Path P) {
  llvm::json::ObjectMapper O(V, P);
  return O && O.map("listChanged", RC.listChanged) &&
         O.map("subscribe", RC.subscribe);
}

llvm::json::Value toJSON(const Capabilities &C) {
  return llvm::json::Object{{"tools", C.tools}, {"resources", C.resources}};
}

bool fromJSON(const llvm::json::Value &V, Resource &R, llvm::json::Path P) {
  llvm::json::ObjectMapper O(V, P);
  return O && O.map("uri", R.uri) && O.map("name", R.name) &&
         O.mapOptional("description", R.description) &&
         O.mapOptional("mimeType", R.mimeType);
}

llvm::json::Value toJSON(const Resource &R) {
  llvm::json::Object Result{{"uri", R.uri}, {"name", R.name}};
  if (!R.description.empty())
    Result.insert({"description", R.description});
  if (!R.mimeType.empty())
    Result.insert({"mimeType", R.mimeType});
  return Result;
}

bool fromJSON(const llvm::json::Value &V, Capabilities &C, llvm::json::Path P) {
  llvm::json::ObjectMapper O(V, P);
  return O && O.map("tools", C.tools);
}

llvm::json::Value toJSON(const ResourceContents &RC) {
  llvm::json::Object Result{{"uri", RC.uri}, {"text", RC.text}};
  if (!RC.mimeType.empty())
    Result.insert({"mimeType", RC.mimeType});
  return Result;
}

bool fromJSON(const llvm::json::Value &V, ResourceContents &RC,
              llvm::json::Path P) {
  llvm::json::ObjectMapper O(V, P);
  return O && O.map("uri", RC.uri) && O.map("text", RC.text) &&
         O.mapOptional("mimeType", RC.mimeType);
}

llvm::json::Value toJSON(const ResourceResult &RR) {
  return llvm::json::Object{{"contents", RR.contents}};
}

bool fromJSON(const llvm::json::Value &V, ResourceResult &RR,
              llvm::json::Path P) {
  llvm::json::ObjectMapper O(V, P);
  return O && O.map("contents", RR.contents);
}

llvm::json::Value toJSON(const TextContent &TC) {
  return llvm::json::Object{{"type", "text"}, {"text", TC.text}};
}

bool fromJSON(const llvm::json::Value &V, TextContent &TC, llvm::json::Path P) {
  llvm::json::ObjectMapper O(V, P);
  return O && O.map("text", TC.text);
}

llvm::json::Value toJSON(const TextResult &TR) {
  return llvm::json::Object{{"content", TR.content}, {"isError", TR.isError}};
}

bool fromJSON(const llvm::json::Value &V, TextResult &TR, llvm::json::Path P) {
  llvm::json::ObjectMapper O(V, P);
  return O && O.map("content", TR.content) && O.map("isError", TR.isError);
}

llvm::json::Value toJSON(const ToolDefinition &TD) {
  llvm::json::Object Result{{"name", TD.name}};
  if (!TD.description.empty())
    Result.insert({"description", TD.description});
  if (TD.inputSchema)
    Result.insert({"inputSchema", TD.inputSchema});
  return Result;
}

bool fromJSON(const llvm::json::Value &V, ToolDefinition &TD,
              llvm::json::Path P) {

  llvm::json::ObjectMapper O(V, P);
  if (!O || !O.map("name", TD.name) ||
      !O.mapOptional("description", TD.description))
    return false;
  return mapRaw(V, "inputSchema", TD.inputSchema, P);
}

llvm::json::Value toJSON(const Message &M) {
  return std::visit([](auto &M) { return toJSON(M); }, M);
}

bool fromJSON(const llvm::json::Value &V, Message &M, llvm::json::Path P) {
  const auto *O = V.getAsObject();
  if (!O) {
    P.report("expected object");
    return false;
  }

  if (const json::Value *V = O->get("jsonrpc")) {
    if (V->getAsString().value_or("") != "2.0") {
      P.report("unsupported JSON RPC version");
      return false;
    }
  } else {
    P.report("not a valid JSON RPC message");
    return false;
  }

  // A message without an ID is a Notification.
  if (!O->get("id")) {
    protocol::Notification N;
    if (!fromJSON(V, N, P))
      return false;
    M = std::move(N);
    return true;
  }

  if (O->get("error")) {
    protocol::Error E;
    if (!fromJSON(V, E, P))
      return false;
    M = std::move(E);
    return true;
  }

  if (O->get("result")) {
    protocol::Response R;
    if (!fromJSON(V, R, P))
      return false;
    M = std::move(R);
    return true;
  }

  if (O->get("method")) {
    protocol::Request R;
    if (!fromJSON(V, R, P))
      return false;
    M = std::move(R);
    return true;
  }

  P.report("unrecognized message type");
  return false;
}

} // namespace lldb_private::mcp::protocol
