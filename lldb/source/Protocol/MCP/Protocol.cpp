//===- Protocol.cpp -------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Protocol/MCP/Protocol.h"
#include "llvm/Support/JSON.h"

using namespace llvm;

namespace lldb_protocol::mcp {

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

static llvm::json::Value toJSON(const Id &Id) {
  if (const int64_t *I = std::get_if<int64_t>(&Id))
    return json::Value(*I);
  if (const std::string *S = std::get_if<std::string>(&Id))
    return json::Value(*S);
  llvm_unreachable("unexpected type in protocol::Id");
}

static bool mapId(const llvm::json::Value &V, StringLiteral Prop, Id &Id,
                  llvm::json::Path P) {
  const auto *O = V.getAsObject();
  if (!O) {
    P.report("expected object");
    return false;
  }

  const auto *E = O->get(Prop);
  if (!E) {
    P.field(Prop).report("not found");
    return false;
  }

  if (auto S = E->getAsString()) {
    Id = S->str();
    return true;
  }

  if (auto I = E->getAsInteger()) {
    Id = *I;
    return true;
  }

  P.report("expected string or number");
  return false;
}

llvm::json::Value toJSON(const Request &R) {
  json::Object Result{
      {"jsonrpc", "2.0"}, {"id", toJSON(R.id)}, {"method", R.method}};
  if (R.params)
    Result.insert({"params", R.params});
  return Result;
}

bool fromJSON(const llvm::json::Value &V, Request &R, llvm::json::Path P) {
  llvm::json::ObjectMapper O(V, P);
  return O && mapId(V, "id", R.id, P) && O.map("method", R.method) &&
         mapRaw(V, "params", R.params, P);
}

bool operator==(const Request &a, const Request &b) {
  return a.id == b.id && a.method == b.method && a.params == b.params;
}

llvm::json::Value toJSON(const Error &E) {
  llvm::json::Object Result{{"code", E.code}, {"message", E.message}};
  if (E.data)
    Result.insert({"data", *E.data});
  return Result;
}

bool fromJSON(const llvm::json::Value &V, Error &E, llvm::json::Path P) {
  llvm::json::ObjectMapper O(V, P);
  return O && O.map("code", E.code) && O.map("message", E.message) &&
         mapRaw(V, "data", E.data, P);
}

bool operator==(const Error &a, const Error &b) {
  return a.code == b.code && a.message == b.message && a.data == b.data;
}

llvm::json::Value toJSON(const Response &R) {
  llvm::json::Object Result{{"jsonrpc", "2.0"}, {"id", toJSON(R.id)}};

  if (const Error *error = std::get_if<Error>(&R.result))
    Result.insert({"error", *error});
  if (const json::Value *result = std::get_if<json::Value>(&R.result))
    Result.insert({"result", *result});
  return Result;
}

bool fromJSON(const llvm::json::Value &V, Response &R, llvm::json::Path P) {
  const json::Object *E = V.getAsObject();
  if (!E) {
    P.report("expected object");
    return false;
  }

  const json::Value *result = E->get("result");
  const json::Value *raw_error = E->get("error");

  if (result && raw_error) {
    P.report("'result' and 'error' fields are mutually exclusive");
    return false;
  }

  if (!result && !raw_error) {
    P.report("'result' or 'error' fields are required'");
    return false;
  }

  if (result) {
    R.result = std::move(*result);
  } else {
    Error error;
    if (!fromJSON(*raw_error, error, P))
      return false;
    R.result = std::move(error);
  }

  return mapId(V, "id", R.id, P);
}

bool operator==(const Response &a, const Response &b) {
  return a.id == b.id && a.result == b.result;
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

bool operator==(const Notification &a, const Notification &b) {
  return a.method == b.method && a.params == b.params;
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
    Notification N;
    if (!fromJSON(V, N, P))
      return false;
    M = std::move(N);
    return true;
  }

  if (O->get("method")) {
    Request R;
    if (!fromJSON(V, R, P))
      return false;
    M = std::move(R);
    return true;
  }

  if (O->get("result") || O->get("error")) {
    Response R;
    if (!fromJSON(V, R, P))
      return false;
    M = std::move(R);
    return true;
  }

  P.report("unrecognized message type");
  return false;
}

} // namespace lldb_protocol::mcp
