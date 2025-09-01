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
#include "lldb/Protocol/MCP/Protocol.h"

using namespace lldb_private;
using namespace lldb_protocol;
using namespace lldb_private::mcp;
using namespace llvm;

namespace {
struct CommandToolArguments {
  uint64_t debugger_id;
  std::string arguments;
};

bool fromJSON(const llvm::json::Value &V, CommandToolArguments &A,
              llvm::json::Path P) {
  llvm::json::ObjectMapper O(V, P);
  return O && O.map("debugger_id", A.debugger_id) &&
         O.mapOptional("arguments", A.arguments);
}

/// Helper function to create a CallToolResult from a string output.
static lldb_protocol::mcp::CallToolResult
createTextResult(std::string output, bool is_error = false) {
  lldb_protocol::mcp::CallToolResult text_result;
  text_result.content.emplace_back(
      lldb_protocol::mcp::TextContent{{std::move(output)}});
  text_result.isError = is_error;
  return text_result;
}

} // namespace

llvm::Expected<lldb_protocol::mcp::CallToolResult>
CommandTool::Call(const lldb_protocol::mcp::ToolArguments &args) {
  if (!std::holds_alternative<json::Value>(args))
    return createStringError("CommandTool requires arguments");

  json::Path::Root root;

  CommandToolArguments arguments;
  if (!fromJSON(std::get<json::Value>(args), arguments, root))
    return root.getError();

  lldb::DebuggerSP debugger_sp =
      Debugger::FindDebuggerWithID(arguments.debugger_id);
  if (!debugger_sp)
    return createStringError(
        llvm::formatv("no debugger with id {0}", arguments.debugger_id));

  // FIXME: Disallow certain commands and their aliases.
  CommandReturnObject result(/*colors=*/false);
  debugger_sp->GetCommandInterpreter().HandleCommand(
      arguments.arguments.c_str(), eLazyBoolYes, result);

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

  return createTextResult(output, !result.Succeeded());
}

std::optional<llvm::json::Value> CommandTool::GetSchema() const {
  llvm::json::Object id_type{{"type", "number"}};
  llvm::json::Object str_type{{"type", "string"}};
  llvm::json::Object properties{{"debugger_id", std::move(id_type)},
                                {"arguments", std::move(str_type)}};
  llvm::json::Array required{"debugger_id"};
  llvm::json::Object schema{{"type", "object"},
                            {"properties", std::move(properties)},
                            {"required", std::move(required)}};
  return schema;
}
