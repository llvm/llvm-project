//===- Tool.h -------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_PLUGINS_PROTOCOL_MCP_TOOL_H
#define LLDB_PLUGINS_PROTOCOL_MCP_TOOL_H

#include "lldb/Protocol/MCP/Protocol.h"
#include "lldb/Protocol/MCP/Tool.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/JSON.h"
#include <optional>

namespace lldb_private::mcp {

class CommandTool : public lldb_protocol::mcp::Tool {
public:
  using lldb_protocol::mcp::Tool::Tool;
  ~CommandTool() = default;

  llvm::Expected<lldb_protocol::mcp::CallToolResult>
  Call(const lldb_protocol::mcp::ToolArguments &args) override;

  std::optional<llvm::json::Value> GetSchema() const override;
};

} // namespace lldb_private::mcp

#endif
