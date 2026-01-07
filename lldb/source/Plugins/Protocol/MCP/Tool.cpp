//===- Tool.cpp -----------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Tool.h"
#include "lldb/Core/Debugger.h"
#include "lldb/Interpreter/CommandInterpreter.h"
#include "lldb/Interpreter/CommandReturnObject.h"
#include "lldb/Protocol/MCP/Protocol.h"
#include "lldb/Utility/UriParser.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include <cstdint>
#include <optional>

using namespace lldb_private;
using namespace lldb_protocol;
using namespace lldb_private::mcp;
using namespace lldb;
using namespace llvm;

namespace {

static constexpr StringLiteral kSchemeAndHost = "lldb-mcp://debugger/";

struct CommandToolArguments {
  /// Either an id like '1' or a uri like 'lldb-mcp://debugger/1'.
  std::string debugger;
  std::string command;
};

bool fromJSON(const json::Value &V, CommandToolArguments &A, json::Path P) {
  json::ObjectMapper O(V, P);
  return O && O.mapOptional("debugger", A.debugger) &&
         O.mapOptional("command", A.command);
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

std::string to_uri(DebuggerSP debugger) {
  return (kSchemeAndHost + std::to_string(debugger->GetID())).str();
}

} // namespace

Expected<lldb_protocol::mcp::CallToolResult>
CommandTool::Call(const lldb_protocol::mcp::ToolArguments &args) {
  if (!std::holds_alternative<json::Value>(args))
    return createStringError("CommandTool requires arguments");

  json::Path::Root root;

  CommandToolArguments arguments;
  if (!fromJSON(std::get<json::Value>(args), arguments, root))
    return root.getError();

  lldb::DebuggerSP debugger_sp;

  if (!arguments.debugger.empty()) {
    llvm::StringRef debugger_specifier = arguments.debugger;
    debugger_specifier.consume_front(kSchemeAndHost);
    uint32_t debugger_id = 0;
    if (debugger_specifier.consumeInteger(10, debugger_id))
      return createStringError(
          formatv("malformed debugger specifier {0}", arguments.debugger));

    debugger_sp = Debugger::FindDebuggerWithID(debugger_id);
  } else {
    for (size_t i = 0; i < Debugger::GetNumDebuggers(); i++) {
      debugger_sp = Debugger::GetDebuggerAtIndex(i);
      if (debugger_sp)
        break;
    }
  }

  if (!debugger_sp)
    return createStringError("no debugger found");

  // FIXME: Disallow certain commands and their aliases.
  CommandReturnObject result(/*colors=*/false);
  debugger_sp->GetCommandInterpreter().HandleCommand(arguments.command.c_str(),
                                                     eLazyBoolYes, result);

  std::string output;
  StringRef output_str = result.GetOutputString();
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

std::optional<json::Value> CommandTool::GetSchema() const {
  using namespace llvm::json;
  Object properties{
      {"debugger",
       Object{{"type", "string"},
              {"description",
               "The debugger ID or URI to a specific debug session. If not "
               "specified, the first debugger will be used."}}},
      {"command",
       Object{{"type", "string"}, {"description", "An lldb command to run."}}}};
  Object schema{{"type", "object"}, {"properties", std::move(properties)}};
  return schema;
}

Expected<lldb_protocol::mcp::CallToolResult>
DebuggerListTool::Call(const lldb_protocol::mcp::ToolArguments &args) {
  llvm::json::Path::Root root;

  // Return a nested Markdown list with debuggers and target.
  // Example output:
  //
  // - lldb-mcp://debugger/1
  // - lldb-mcp://debugger/2
  //
  // FIXME: Use Structured Content when we adopt protocol version 2025-06-18.
  std::string output;
  llvm::raw_string_ostream os(output);

  const size_t num_debuggers = Debugger::GetNumDebuggers();
  for (size_t i = 0; i < num_debuggers; ++i) {
    lldb::DebuggerSP debugger_sp = Debugger::GetDebuggerAtIndex(i);
    if (!debugger_sp)
      continue;

    os << "- " << to_uri(debugger_sp) << '\n';
  }

  return createTextResult(output);
}
