//===- Tool.cpp -----------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Tool.h"
#include "lldb/Core/Module.h"
#include "lldb/Interpreter/CommandInterpreter.h"
#include "lldb/Interpreter/CommandReturnObject.h"

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

/// Helper function to create a TextResult from a string output.
static lldb_private::mcp::protocol::TextResult
createTextResult(std::string output, bool is_error = false) {
  lldb_private::mcp::protocol::TextResult text_result;
  text_result.content.emplace_back(
      lldb_private::mcp::protocol::TextContent{{std::move(output)}});
  text_result.isError = is_error;
  return text_result;
}

} // namespace

Tool::Tool(std::string name, std::string description)
    : m_name(std::move(name)), m_description(std::move(description)) {}

protocol::ToolDefinition Tool::GetDefinition() const {
  protocol::ToolDefinition definition;
  definition.name = m_name;
  definition.description.emplace(m_description);

  if (std::optional<llvm::json::Value> input_schema = GetSchema())
    definition.inputSchema = *input_schema;

  return definition;
}

llvm::Expected<protocol::TextResult>
CommandTool::Call(const protocol::ToolArguments &args) {
  if (!std::holds_alternative<json::Value>(args))
    return createStringError("CommandTool requires arguments");

  json::Path::Root root;

  CommandToolArguments arguments;
  if (!fromJSON(std::get<json::Value>(args), arguments, root))
    return root.getError();

  lldb::DebuggerSP debugger_sp =
      Debugger::GetDebuggerAtIndex(arguments.debugger_id);
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

llvm::Expected<protocol::TextResult>
DebuggerListTool::Call(const protocol::ToolArguments &args) {
  if (!std::holds_alternative<std::monostate>(args))
    return createStringError("DebuggerListTool takes no arguments");

  llvm::json::Path::Root root;

  // Return a nested Markdown list with debuggers and target.
  // Example output:
  //
  // - debugger 0
  //     - target 0 /path/to/foo
  //     - target 1
  // - debugger 1
  //     - target 0 /path/to/bar
  //
  // FIXME: Use Structured Content when we adopt protocol version 2025-06-18.
  std::string output;
  llvm::raw_string_ostream os(output);

  const size_t num_debuggers = Debugger::GetNumDebuggers();
  for (size_t i = 0; i < num_debuggers; ++i) {
    lldb::DebuggerSP debugger_sp = Debugger::GetDebuggerAtIndex(i);
    if (!debugger_sp)
      continue;

    os << "- debugger " << i << '\n';

    TargetList &target_list = debugger_sp->GetTargetList();
    const size_t num_targets = target_list.GetNumTargets();
    for (size_t j = 0; j < num_targets; ++j) {
      lldb::TargetSP target_sp = target_list.GetTargetAtIndex(j);
      if (!target_sp)
        continue;
      os << "    - target " << j;
      if (target_sp == target_list.GetSelectedTarget())
        os << " (selected)";
      // Append the module path if we have one.
      if (Module *exe_module = target_sp->GetExecutableModulePointer())
        os << " " << exe_module->GetFileSpec().GetPath();
      os << '\n';
    }
  }

  return createTextResult(output);
}
