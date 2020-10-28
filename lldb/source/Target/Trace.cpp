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

// Helper structs used to extract the type of a trace settings json without
// having to parse the entire object.

struct JSONSimplePluginSettings {
  std::string type;
};

struct JSONSimpleTraceSettings {
  JSONSimplePluginSettings trace;
};

namespace llvm {
namespace json {

bool fromJSON(const json::Value &value,
              JSONSimplePluginSettings &plugin_settings, json::Path path) {
  json::ObjectMapper o(value, path);
  return o && o.map("type", plugin_settings.type);
}

bool fromJSON(const json::Value &value, JSONSimpleTraceSettings &settings,
              json::Path path) {
  json::ObjectMapper o(value, path);
  return o && o.map("trace", settings.trace);
}

} // namespace json
} // namespace llvm

llvm::Expected<lldb::TraceSP> Trace::FindPlugin(Debugger &debugger,
                                                const json::Value &settings,
                                                StringRef info_dir) {
  JSONSimpleTraceSettings json_settings;
  json::Path::Root root("settings");
  if (!json::fromJSON(settings, json_settings, root))
    return root.getError();

  ConstString plugin_name(json_settings.trace.type);
  auto create_callback = PluginManager::GetTraceCreateCallback(plugin_name);
  if (create_callback) {
    TraceSP instance = create_callback();
    if (llvm::Error err = instance->ParseSettings(debugger, settings, info_dir))
      return std::move(err);
    return instance;
  }

  return createStringError(
      std::errc::invalid_argument,
      "no trace plug-in matches the specified type: \"%s\"",
      plugin_name.AsCString());
}

llvm::Expected<lldb::TraceSP> Trace::FindPlugin(StringRef name) {
  ConstString plugin_name(name);
  auto create_callback = PluginManager::GetTraceCreateCallback(plugin_name);
  if (create_callback)
    return create_callback();

  return createStringError(
      std::errc::invalid_argument,
      "no trace plug-in matches the specified type: \"%s\"",
      plugin_name.AsCString());
}

llvm::Error Trace::ParseSettings(Debugger &debugger,
                                 const llvm::json::Value &settings,
                                 llvm::StringRef settings_dir) {
  if (llvm::Error err =
          CreateParser()->ParseSettings(debugger, settings, settings_dir))
    return err;

  return llvm::Error::success();
}

llvm::StringRef Trace::GetSchema() { return CreateParser()->GetSchema(); }
