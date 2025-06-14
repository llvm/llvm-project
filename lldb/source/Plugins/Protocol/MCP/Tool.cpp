//===- Tool.cpp -----------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Tool.h"
#include "lldb/Interpreter/CommandInterpreter.h"
#include "lldb/Interpreter/CommandReturnObject.h"

using namespace lldb_private::mcp;
using namespace llvm;

Tool::Tool(std::string name, std::string description)
    : m_name(std::move(name)), m_description(std::move(description)) {}

protocol::ToolDefinition Tool::GetDefinition() const {
  protocol::ToolDefinition definition;
  definition.name = m_name;
  definition.description.emplace(m_description);

  if (std::optional<llvm::json::Value> input_schema = GetSchema())
    definition.inputSchema = *input_schema;

  if (m_annotations)
    definition.annotations = m_annotations;

  return definition;
}

LLDBCommandTool::LLDBCommandTool(std::string name, std::string description,
                                 Debugger &debugger)
    : Tool(std::move(name), std::move(description)), m_debugger(debugger) {}

protocol::TextResult LLDBCommandTool::Call(const llvm::json::Value &args) {
  std::string arguments;
  if (const json::Object *args_obj = args.getAsObject()) {
    if (const json::Value *s = args_obj->get("arguments")) {
      arguments = s->getAsString().value_or("");
    }
  }

  CommandReturnObject result(/*colors=*/false);
  m_debugger.GetCommandInterpreter().HandleCommand(arguments.c_str(),
                                                   eLazyBoolYes, result);

  std::string output;
  llvm::StringRef output_str = result.GetOutputString();
  if (!output_str.empty())
    output += output_str.str();

  std::string err_str = result.GetErrorString();
  if (!err_str.empty()) {
    if (!output.empty())
      output += '\n';
    output += err_str;
  }

  mcp::protocol::TextResult text_result;
  text_result.content.emplace_back(mcp::protocol::TextContent{{output}});
  return text_result;
}

std::optional<llvm::json::Value> LLDBCommandTool::GetSchema() const {
  llvm::json::Object str_type{{"type", "string"}};
  llvm::json::Object properties{{"arguments", std::move(str_type)}};
  llvm::json::Object schema{{"type", "object"},
                            {"properties", std::move(properties)}};
  return schema;
}
