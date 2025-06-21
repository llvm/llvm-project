//===- Protocol.h ---------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains POD structs based on the MCP specification at
// https://github.com/modelcontextprotocol/modelcontextprotocol/blob/main/schema/2024-11-05/schema.json
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_PLUGINS_PROTOCOL_MCP_PROTOCOL_H
#define LLDB_PLUGINS_PROTOCOL_MCP_PROTOCOL_H

#include "llvm/Support/JSON.h"
#include <optional>
#include <string>
#include <variant>

namespace lldb_private::mcp::protocol {

static llvm::StringLiteral kVersion = "2024-11-05";

/// A request that expects a response.
struct Request {
  uint64_t id = 0;
  std::string method;
  std::optional<llvm::json::Value> params;
};

llvm::json::Value toJSON(const Request &);
bool fromJSON(const llvm::json::Value &, Request &, llvm::json::Path);

struct ErrorInfo {
  int64_t code = 0;
  std::string message;
  std::optional<std::string> data;
};

llvm::json::Value toJSON(const ErrorInfo &);
bool fromJSON(const llvm::json::Value &, ErrorInfo &, llvm::json::Path);

struct Error {
  uint64_t id = 0;
  ErrorInfo error;
};

llvm::json::Value toJSON(const Error &);
bool fromJSON(const llvm::json::Value &, Error &, llvm::json::Path);

struct Response {
  uint64_t id = 0;
  std::optional<llvm::json::Value> result;
  std::optional<ErrorInfo> error;
};

llvm::json::Value toJSON(const Response &);
bool fromJSON(const llvm::json::Value &, Response &, llvm::json::Path);

/// A notification which does not expect a response.
struct Notification {
  std::string method;
  std::optional<llvm::json::Value> params;
};

llvm::json::Value toJSON(const Notification &);
bool fromJSON(const llvm::json::Value &, Notification &, llvm::json::Path);

struct ToolCapability {
  /// Whether this server supports notifications for changes to the tool list.
  bool listChanged = false;
};

llvm::json::Value toJSON(const ToolCapability &);
bool fromJSON(const llvm::json::Value &, ToolCapability &, llvm::json::Path);

/// Capabilities that a server may support. Known capabilities are defined here,
/// in this schema, but this is not a closed set: any server can define its own,
/// additional capabilities.
struct Capabilities {
  /// Present if the server offers any tools to call.
  ToolCapability tools;
};

llvm::json::Value toJSON(const Capabilities &);
bool fromJSON(const llvm::json::Value &, Capabilities &, llvm::json::Path);

/// Text provided to or from an LLM.
struct TextContent {
  /// The text content of the message.
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

struct ToolDefinition {
  /// Unique identifier for the tool.
  std::string name;

  /// Human-readable description.
  std::optional<std::string> description;

  // JSON Schema for the tool's parameters.
  std::optional<llvm::json::Value> inputSchema;
};

llvm::json::Value toJSON(const ToolDefinition &);
bool fromJSON(const llvm::json::Value &, ToolDefinition &, llvm::json::Path);

using Message = std::variant<Request, Response, Notification, Error>;

bool fromJSON(const llvm::json::Value &, Message &, llvm::json::Path);
llvm::json::Value toJSON(const Message &);

} // namespace lldb_private::mcp::protocol

#endif
