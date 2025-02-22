//===-- Handler.cpp -------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "RequestHandler.h"
#include "DAP.h"
#include "JSONUtils.h"
#include "lldb/API/SBFileSpec.h"

namespace lldb_dap {

void RequestHandler::SendProcessEvent(
    RequestHandler::LaunchMethod launch_method) {
  lldb::SBFileSpec exe_fspec = dap.target.GetExecutable();
  char exe_path[PATH_MAX];
  exe_fspec.GetPath(exe_path, sizeof(exe_path));
  llvm::json::Object event(CreateEventObject("process"));
  llvm::json::Object body;
  EmplaceSafeString(body, "name", std::string(exe_path));
  const auto pid = dap.target.GetProcess().GetProcessID();
  body.try_emplace("systemProcessId", (int64_t)pid);
  body.try_emplace("isLocalProcess", true);
  const char *startMethod = nullptr;
  switch (launch_method) {
  case Launch:
    startMethod = "launch";
    break;
  case Attach:
    startMethod = "attach";
    break;
  case AttachForSuspendedLaunch:
    startMethod = "attachForSuspendedLaunch";
    break;
  }
  body.try_emplace("startMethod", startMethod);
  event.try_emplace("body", std::move(body));
  dap.SendJSON(llvm::json::Value(std::move(event)));
}

// Both attach and launch take a either a sourcePath or sourceMap
// argument (or neither), from which we need to set the target.source-map.
void RequestHandler::SetSourceMapFromArguments(
    const llvm::json::Object &arguments) {
  const char *sourceMapHelp =
      "source must be be an array of two-element arrays, "
      "each containing a source and replacement path string.\n";

  std::string sourceMapCommand;
  llvm::raw_string_ostream strm(sourceMapCommand);
  strm << "settings set target.source-map ";
  const auto sourcePath = GetString(arguments, "sourcePath");

  // sourceMap is the new, more general form of sourcePath and overrides it.
  constexpr llvm::StringRef sourceMapKey = "sourceMap";

  if (const auto *sourceMapArray = arguments.getArray(sourceMapKey)) {
    for (const auto &value : *sourceMapArray) {
      const auto *mapping = value.getAsArray();
      if (mapping == nullptr || mapping->size() != 2 ||
          (*mapping)[0].kind() != llvm::json::Value::String ||
          (*mapping)[1].kind() != llvm::json::Value::String) {
        dap.SendOutput(OutputType::Console, llvm::StringRef(sourceMapHelp));
        return;
      }
      const auto mapFrom = GetAsString((*mapping)[0]);
      const auto mapTo = GetAsString((*mapping)[1]);
      strm << "\"" << mapFrom << "\" \"" << mapTo << "\" ";
    }
  } else if (const auto *sourceMapObj = arguments.getObject(sourceMapKey)) {
    for (const auto &[key, value] : *sourceMapObj) {
      if (value.kind() == llvm::json::Value::String) {
        strm << "\"" << key.str() << "\" \"" << GetAsString(value) << "\" ";
      }
    }
  } else {
    if (ObjectContainsKey(arguments, sourceMapKey)) {
      dap.SendOutput(OutputType::Console, llvm::StringRef(sourceMapHelp));
      return;
    }
    if (sourcePath.empty())
      return;
    // Do any source remapping needed before we create our targets
    strm << "\".\" \"" << sourcePath << "\"";
  }
  if (!sourceMapCommand.empty()) {
    dap.RunLLDBCommands("Setting source map:", {sourceMapCommand});
  }
}

} // namespace lldb_dap
