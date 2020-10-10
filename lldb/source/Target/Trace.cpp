//===-- Trace.cpp ---------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Target/Trace.h"

#include <sstream>

#include "llvm/Support/Format.h"

#include "lldb/Core/PluginManager.h"

using namespace lldb;
using namespace lldb_private;
using namespace llvm;

// Helper structs used to extract the type of a trace session json without
// having to parse the entire object.

struct JSONSimplePluginSettings {
  std::string type;
};

struct JSONSimpleTraceSession {
  JSONSimplePluginSettings trace;
};

namespace llvm {
namespace json {

bool fromJSON(const Value &value, JSONSimplePluginSettings &plugin_settings,
              Path path) {
  json::ObjectMapper o(value, path);
  return o && o.map("type", plugin_settings.type);
}

bool fromJSON(const Value &value, JSONSimpleTraceSession &session, Path path) {
  json::ObjectMapper o(value, path);
  return o && o.map("trace", session.trace);
}

} // namespace json
} // namespace llvm

static Error createInvalidPlugInError(StringRef plugin_name) {
  return createStringError(
      std::errc::invalid_argument,
      "no trace plug-in matches the specified type: \"%s\"",
      plugin_name.data());
}

Expected<lldb::TraceSP> Trace::FindPlugin(Debugger &debugger,
                                          const json::Value &trace_session_file,
                                          StringRef session_file_dir) {
  JSONSimpleTraceSession json_session;
  json::Path::Root root("traceSession");
  if (!json::fromJSON(trace_session_file, json_session, root))
    return root.getError();

  ConstString plugin_name(json_session.trace.type);
  if (auto create_callback = PluginManager::GetTraceCreateCallback(plugin_name))
    return create_callback(trace_session_file, session_file_dir, debugger);

  return createInvalidPlugInError(json_session.trace.type);
}

Expected<StringRef> Trace::FindPluginSchema(StringRef name) {
  ConstString plugin_name(name);
  StringRef schema = PluginManager::GetTraceSchema(plugin_name);
  if (!schema.empty())
    return schema;

  return createInvalidPlugInError(name);
}
