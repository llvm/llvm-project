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

llvm::Expected<lldb::TraceSP> Trace::FindPlugin(Debugger &debugger,
                                                const json::Value &settings,
                                                StringRef info_dir) {
  llvm::Expected<const json::Object &> settings_obj =
      json_helpers::ToObjectOrError(settings);
  if (!settings_obj)
    return settings_obj.takeError();

  llvm::Expected<const json::Object &> trace =
      json_helpers::GetObjectOrError(*settings_obj, "trace");
  if (!trace)
    return trace.takeError();

  llvm::Expected<StringRef> type =
      json_helpers::GetStringOrError(*trace, "type");
  if (!type)
    return type.takeError();

  ConstString plugin_name(*type);
  auto create_callback = PluginManager::GetTraceCreateCallback(plugin_name);
  if (create_callback) {
    TraceSP instance = create_callback();
    if (llvm::Error err =
            instance->ParseSettings(debugger, *settings_obj, info_dir))
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
                                 const llvm::json::Object &settings,
                                 llvm::StringRef settings_dir) {
  if (llvm::Error err =
          CreateParser()->ParseSettings(debugger, settings, settings_dir))
    return err;

  return llvm::Error::success();
}

llvm::StringRef Trace::GetSchema() { return CreateParser()->GetSchema(); }
