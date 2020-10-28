//===-- TraceSettingParser.cpp --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Target/TraceSettingsParser.h"

#include <sstream>

#include "Plugins/Process/Utility/HistoryThread.h"
#include "lldb/Core/Debugger.h"
#include "lldb/Target/Process.h"

using namespace lldb;
using namespace lldb_private;
using namespace llvm;

StringRef TraceSettingsParser::GetSchema() {
  static std::string schema;
  if (schema.empty()) {
    std::ostringstream schema_builder;
    schema_builder << "{\n \"trace\": ";
    schema_builder << GetPluginSchema().str() << ",\n";
    schema_builder << R"(  "processes": [
    {
      "pid": integer,
      "triple": string, // llvm-triple
      "threads": [
        {
          "tid": integer,
          "traceFile": string
        }
      ],
      "modules": [
        {
          "systemPath": string, // original path of the module at runtime
          "file"?: string, // copy of the file if not available at "systemPath"
          "loadAddress": string, // string address in hex or decimal form
          "uuid"?: string,
        }
      ]
    }
  ]
}
// Notes:
// All paths are either absolute or relative to the settings file.)";
    schema = schema_builder.str();
  }
  return schema;
}

void TraceSettingsParser::NormalizePath(FileSpec &file_spec) {
  if (file_spec.IsRelative())
    file_spec.PrependPathComponent(m_settings_dir);
}

void TraceSettingsParser::ParseThread(ProcessSP &process_sp,
                                      const JSONThread &thread) {
  lldb::tid_t tid = static_cast<lldb::tid_t>(thread.tid);

  FileSpec spec(thread.trace_file);
  NormalizePath(spec);
  m_thread_to_trace_file_map[process_sp->GetID()][tid] = spec;

  ThreadSP thread_sp(new HistoryThread(*process_sp, tid, /*callstack*/ {}));
  process_sp->GetThreadList().AddThread(thread_sp);
}

llvm::Error TraceSettingsParser::ParseModule(TargetSP &target_sp,
                                             const JSONModule &module) {
  FileSpec system_file_spec(module.system_path);
  NormalizePath(system_file_spec);

  FileSpec local_file_spec(module.file.hasValue() ? *module.file
                                                  : module.system_path);
  NormalizePath(local_file_spec);

  ModuleSpec module_spec;
  module_spec.GetFileSpec() = local_file_spec;
  module_spec.GetPlatformFileSpec() = system_file_spec;
  module_spec.SetObjectOffset(module.load_address.value);

  if (module.uuid.hasValue())
    module_spec.GetUUID().SetFromStringRef(*module.uuid);

  Status error;
  ModuleSP module_sp =
      target_sp->GetOrCreateModule(module_spec, /*notify*/ false, &error);
  return error.ToError();
}

llvm::Error TraceSettingsParser::ParseProcess(Debugger &debugger,
                                              const JSONProcess &process) {
  TargetSP target_sp;
  Status error = debugger.GetTargetList().CreateTarget(
      debugger, /*user_exe_path*/ llvm::StringRef(), process.triple,
      eLoadDependentsNo,
      /*platform_options*/ nullptr, target_sp);

  if (!target_sp)
    return error.ToError();

  m_targets.push_back(target_sp);
  debugger.GetTargetList().SetSelectedTarget(target_sp.get());

  ProcessSP process_sp(target_sp->CreateProcess(
      /*listener*/ nullptr, /*plugin_name*/ llvm::StringRef(),
      /*crash_file*/ nullptr));
  process_sp->SetID(static_cast<lldb::pid_t>(process.pid));

  for (const JSONThread &thread : process.threads)
    ParseThread(process_sp, thread);

  for (const JSONModule &module : process.modules) {
    if (llvm::Error err = ParseModule(target_sp, module))
      return err;
  }
  return llvm::Error::success();
}

llvm::Error
TraceSettingsParser::CreateJSONError(json::Path::Root &root,
                                     const llvm::json::Value &value) {
  std::string err;
  raw_string_ostream os(err);
  root.printErrorContext(value, os);
  return createStringError(std::errc::invalid_argument,
                           "%s\n\nContext:\n%s\n\nSchema:\n%s",
                           llvm::toString(root.getError()).c_str(),
                           os.str().c_str(), GetSchema().data());
}

llvm::Error
TraceSettingsParser::ParseSettingsImpl(Debugger &debugger,
                                       const llvm::json::Value &raw_settings) {
  json::Path::Root root("settings");
  JSONTraceSettings settings;
  if (!json::fromJSON(raw_settings, settings, root))
    return CreateJSONError(root, raw_settings);

  for (const JSONProcess &process : settings.processes) {
    if (llvm::Error err = ParseProcess(debugger, process))
      return err;
  }

  json::Object plugin_obj = *raw_settings.getAsObject()->getObject("trace");
  json::Value plugin_settings(std::move(plugin_obj));
  return ParsePluginSettings(plugin_settings);
}

llvm::Error
TraceSettingsParser::ParseSettings(Debugger &debugger,
                                   const llvm::json::Value &raw_settings,
                                   llvm::StringRef settings_dir) {
  m_settings_dir = settings_dir.str();

  if (llvm::Error err = ParseSettingsImpl(debugger, raw_settings)) {
    // We clean all the targets that were created internally, which should leave
    // the debugger unchanged
    for (auto target_sp : m_targets)
      debugger.GetTargetList().DeleteTarget(target_sp);

    return err;
  }

  m_trace.m_settings = *raw_settings.getAsObject();
  m_trace.m_settings_dir = m_settings_dir;
  m_trace.m_thread_to_trace_file_map = m_thread_to_trace_file_map;
  m_trace.m_targets = m_targets;

  return llvm::Error::success();
}
