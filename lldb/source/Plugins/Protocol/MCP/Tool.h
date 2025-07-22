//===- Tool.h -------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_PLUGINS_PROTOCOL_MCP_TOOL_H
#define LLDB_PLUGINS_PROTOCOL_MCP_TOOL_H

#include "Protocol.h"
#include "lldb/Core/Debugger.h"
#include "llvm/Support/JSON.h"
#include <string>

namespace lldb_private::mcp {

class Tool {
public:
  Tool(std::string name, std::string description);
  virtual ~Tool() = default;

  virtual llvm::Expected<protocol::TextResult>
  Call(const protocol::ToolArguments &args) = 0;

  virtual std::optional<llvm::json::Value> GetSchema() const {
    return llvm::json::Object{{"type", "object"}};
  }

  protocol::ToolDefinition GetDefinition() const;

  const std::string &GetName() { return m_name; }

private:
  std::string m_name;
  std::string m_description;
};

class CommandTool : public mcp::Tool {
public:
  using mcp::Tool::Tool;
  ~CommandTool() = default;

  virtual llvm::Expected<protocol::TextResult>
  Call(const protocol::ToolArguments &args) override;

  virtual std::optional<llvm::json::Value> GetSchema() const override;
};

} // namespace lldb_private::mcp

#endif
