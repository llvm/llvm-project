//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Protocol/MCP/Tool.h"

using namespace lldb_protocol::mcp;

Tool::Tool(std::string name, std::string description)
    : m_name(std::move(name)), m_description(std::move(description)) {}

lldb_protocol::mcp::ToolDefinition Tool::GetDefinition() const {
  lldb_protocol::mcp::ToolDefinition definition;
  definition.name = m_name;
  definition.description = m_description;

  if (std::optional<llvm::json::Value> input_schema = GetSchema())
    definition.inputSchema = *input_schema;

  return definition;
}
