//===- Protocol.h ---------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_PLUGINS_PROTOCOL_MCP_PROTOCOL_H
#define LLDB_PLUGINS_PROTOCOL_MCP_PROTOCOL_H

#include "llvm/Support/JSON.h"
#include <optional>
#include <string>

namespace lldb_private::mcp::protocol {

static llvm::StringLiteral kProtocolVersion = "2025-03-26";

struct Request {
  uint64_t id = 0;
  std::string method;
  std::optional<llvm::json::Value> params;
};

llvm::json::Value toJSON(const Request &);
bool fromJSON(const llvm::json::Value &, Request &, llvm::json::Path);

struct Error {
  int64_t code = 0;
  std::string message;
  std::optional<std::string> data;
};

llvm::json::Value toJSON(const Error &);
bool fromJSON(const llvm::json::Value &, Error &, llvm::json::Path);

struct ProtocolError {
  uint64_t id = 0;
  Error error;
};

llvm::json::Value toJSON(const ProtocolError &);
bool fromJSON(const llvm::json::Value &, ProtocolError &, llvm::json::Path);

struct Response {
  uint64_t id = 0;
  std::optional<llvm::json::Value> result;
  std::optional<Error> error;
};

llvm::json::Value toJSON(const Response &);
bool fromJSON(const llvm::json::Value &, Response &, llvm::json::Path);

struct Notification {
  std::string method;
  std::optional<llvm::json::Value> params;
};

llvm::json::Value toJSON(const Notification &);
bool fromJSON(const llvm::json::Value &, Notification &, llvm::json::Path);

struct ToolCapability {
  bool listChanged = false;
};

llvm::json::Value toJSON(const ToolCapability &);
bool fromJSON(const llvm::json::Value &, ToolCapability &, llvm::json::Path);

struct Capabilities {
  ToolCapability tools;
};

llvm::json::Value toJSON(const Capabilities &);
bool fromJSON(const llvm::json::Value &, Capabilities &, llvm::json::Path);

struct TextContent {
  std::string text;
};

llvm::json::Value toJSON(const TextContent &);
bool fromJSON(const llvm::json::Value &, TextContent &, llvm::json::Path);

struct TextResult {
  std::vector<TextContent> content;
  bool isError = false;
};

llvm::json::Value toJSON(const TextResult &);
bool fromJSON(const llvm::json::Value &, TextResult &, llvm::json::Path);

struct ToolAnnotations {
  // Human-readable title for the tool.
  std::optional<std::string> title;

  /// If true, the tool does not modify its environment.
  std::optional<bool> readOnlyHint;

  /// If true, the tool may perform destructive updates.
  std::optional<bool> destructiveHint;

  /// If true, repeated calls with same args have no additional effect.
  std::optional<bool> idempotentHint;

  /// If true, tool interacts with external entities.
  std::optional<bool> openWorldHint;
};

llvm::json::Value toJSON(const ToolAnnotations &);
bool fromJSON(const llvm::json::Value &, ToolAnnotations &, llvm::json::Path);

struct ToolDefinition {
  /// Unique identifier for the tool.
  std::string name;

  /// Human-readable description.
  std::optional<std::string> description;

  // JSON Schema for the tool's parameters.
  std::optional<llvm::json::Value> inputSchema;

  // Optional hints about tool behavior.
  std::optional<ToolAnnotations> annotations;
};

llvm::json::Value toJSON(const ToolDefinition &);
bool fromJSON(const llvm::json::Value &, ToolDefinition &, llvm::json::Path);

} // namespace lldb_private::mcp::protocol

#endif
