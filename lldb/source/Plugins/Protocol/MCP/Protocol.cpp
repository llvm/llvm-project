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

llvm::json::Value toJSON(const Error &E) {
  return llvm::json::Object{
      {"code", E.code}, {"message", E.message}, {"data", E.data}};
}

bool fromJSON(const llvm::json::Value &V, Error &E, llvm::json::Path P) {
  llvm::json::ObjectMapper O(V, P);
  return O && O.map("code", E.code) && O.map("message", E.message) &&
         O.map("data", E.data);
}

llvm::json::Value toJSON(const ProtocolError &PE) {
  return llvm::json::Object{{"id", PE.id}, {"error", toJSON(PE.error)}};
}

bool fromJSON(const llvm::json::Value &V, ProtocolError &PE,
              llvm::json::Path P) {
  llvm::json::ObjectMapper O(V, P);
  return O && O.map("id", PE.id) && O.map("error", PE.error);
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

llvm::json::Value toJSON(const Capabilities &C) {
  return llvm::json::Object{{"tools", C.tools}};
}

bool fromJSON(const llvm::json::Value &V, Capabilities &C, llvm::json::Path P) {
  llvm::json::ObjectMapper O(V, P);
  return O && O.map("tools", C.tools);
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

llvm::json::Value toJSON(const ToolAnnotations &TA) {
  llvm::json::Object Result;
  if (TA.title)
    Result.insert({"title", TA.title});
  if (TA.readOnlyHint)
    Result.insert({"readOnlyHint", TA.readOnlyHint});
  if (TA.destructiveHint)
    Result.insert({"destructiveHint", TA.destructiveHint});
  if (TA.idempotentHint)
    Result.insert({"idempotentHint", TA.idempotentHint});
  if (TA.openWorldHint)
    Result.insert({"openWorldHint", TA.openWorldHint});
  return Result;
}

bool fromJSON(const llvm::json::Value &V, ToolAnnotations &TA,
              llvm::json::Path P) {
  llvm::json::ObjectMapper O(V, P);
  return O && O.mapOptional("title", TA.title) &&
         O.mapOptional("readOnlyHint", TA.readOnlyHint) &&
         O.mapOptional("destructiveHint", TA.destructiveHint) &&
         O.mapOptional("idempotentHint", TA.idempotentHint) &&
         O.mapOptional("openWorldHint", TA.openWorldHint);
}

llvm::json::Value toJSON(const ToolDefinition &TD) {
  llvm::json::Object Result{{"name", TD.name}};
  if (TD.description)
    Result.insert({"description", TD.description});
  if (TD.inputSchema)
    Result.insert({"inputSchema", TD.inputSchema});
  if (TD.annotations)
    Result.insert({"annotations", TD.annotations});
  return Result;
}

bool fromJSON(const llvm::json::Value &V, ToolDefinition &TD,
              llvm::json::Path P) {

  llvm::json::ObjectMapper O(V, P);
  if (!O || !O.map("name", TD.name) ||
      !O.mapOptional("description", TD.description) ||
      !O.mapOptional("annotations", TD.annotations))
    return false;
  return mapRaw(V, "inputSchema", TD.inputSchema, P);
}

} // namespace lldb_private::mcp::protocol
