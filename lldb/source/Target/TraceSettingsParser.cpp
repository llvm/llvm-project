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

namespace json_helpers {

llvm::Error CreateWrongTypeError(const json::Value &value,
                                 llvm::StringRef type) {
  std::string s;
  llvm::raw_string_ostream os(s);
  os << llvm::formatv("JSON value is expected to be \"{0}\".\nValue:\n{1:2}",
                      type, value);
  os.flush();

  return llvm::createStringError(std::errc::invalid_argument, os.str().c_str());
}

llvm::Expected<int64_t> ToIntegerOrError(const json::Value &value) {
  llvm::Optional<int64_t> v = value.getAsInteger();
  if (v.hasValue())
    return *v;
  return CreateWrongTypeError(value, "integer");
}

llvm::Expected<StringRef> ToStringOrError(const json::Value &value) {
  llvm::Optional<StringRef> v = value.getAsString();
  if (v.hasValue())
    return *v;
  return CreateWrongTypeError(value, "string");
}

llvm::Expected<const json::Array &> ToArrayOrError(const json::Value &value) {
  if (const json::Array *v = value.getAsArray())
    return *v;
  return CreateWrongTypeError(value, "array");
}

llvm::Expected<const json::Object &> ToObjectOrError(const json::Value &value) {
  if (const json::Object *v = value.getAsObject())
    return *v;
  return CreateWrongTypeError(value, "object");
}

llvm::Error CreateMissingKeyError(json::Object obj, llvm::StringRef key) {
  std::string str;
  llvm::raw_string_ostream os(str);
  os << llvm::formatv(
      "JSON object is missing the \"{0}\" field.\nValue:\n{1:2}", key,
      json::Value(std::move(obj)));
  os.flush();

  return llvm::createStringError(std::errc::invalid_argument, os.str().c_str());
}

llvm::Expected<const json::Value &> GetValueOrError(const json::Object &obj,
                                                    StringRef key) {
  if (const json::Value *v = obj.get(key))
    return *v;
  else
    return CreateMissingKeyError(obj, key);
}

llvm::Expected<int64_t> GetIntegerOrError(const json::Object &obj,
                                          StringRef key) {
  if (llvm::Expected<const json::Value &> v = GetValueOrError(obj, key))
    return ToIntegerOrError(*v);
  else
    return v.takeError();
}

llvm::Expected<StringRef> GetStringOrError(const json::Object &obj,
                                           StringRef key) {
  if (llvm::Expected<const json::Value &> v = GetValueOrError(obj, key))
    return ToStringOrError(*v);
  else
    return v.takeError();
}

llvm::Expected<const json::Array &> GetArrayOrError(const json::Object &obj,
                                                    StringRef key) {
  if (llvm::Expected<const json::Value &> v = GetValueOrError(obj, key))
    return ToArrayOrError(*v);
  else
    return v.takeError();
}

llvm::Expected<const json::Object &> GetObjectOrError(const json::Object &obj,
                                                      StringRef key) {
  if (llvm::Expected<const json::Value &> v = GetValueOrError(obj, key))
    return ToObjectOrError(*v);
  else
    return v.takeError();
}

llvm::Expected<llvm::Optional<StringRef>>
GetOptionalStringOrError(const json::Object &obj, StringRef key) {
  if (const json::Value *v = obj.get(key))
    return ToStringOrError(*v);
  return llvm::None;
}

} // namespace json_helpers

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

llvm::Error TraceSettingsParser::ParseThread(ProcessSP &process_sp,
                                             const json::Object &thread) {
  llvm::Expected<lldb::tid_t> raw_tid =
      json_helpers::GetIntegerOrError(thread, "tid");
  if (!raw_tid)
    return raw_tid.takeError();
  lldb::tid_t tid = static_cast<lldb::tid_t>(*raw_tid);

  if (llvm::Expected<StringRef> trace_file =
          json_helpers::GetStringOrError(thread, "traceFile")) {
    FileSpec spec(*trace_file);
    NormalizePath(spec);
    m_thread_to_trace_file_map[process_sp->GetID()][tid] = spec;
  } else
    return trace_file.takeError();

  ThreadSP thread_sp(new HistoryThread(*process_sp, tid, /*callstack*/ {}));
  process_sp->GetThreadList().AddThread(thread_sp);
  return llvm::Error::success();
}

llvm::Error TraceSettingsParser::ParseThreads(ProcessSP &process_sp,
                                              const json::Object &process) {
  llvm::Expected<const json::Array &> threads =
      json_helpers::GetArrayOrError(process, "threads");
  if (!threads)
    return threads.takeError();

  for (const json::Value &thread_val : *threads) {
    llvm::Expected<const json::Object &> thread =
        json_helpers::ToObjectOrError(thread_val);
    if (!thread)
      return thread.takeError();
    if (llvm::Error err = ParseThread(process_sp, *thread))
      return err;
  }
  return llvm::Error::success();
}

static llvm::Expected<addr_t> ParseAddress(StringRef address_str) {
  addr_t address;
  if (address_str.getAsInteger(0, address))
    return createStringError(std::errc::invalid_argument,
                             "\"%s\" does not represent an integer",
                             address_str.data());
  return address;
}

llvm::Error TraceSettingsParser::ParseModule(TargetSP &target_sp,
                                             const json::Object &module) {
  llvm::Expected<StringRef> load_address_str =
      json_helpers::GetStringOrError(module, "loadAddress");
  if (!load_address_str)
    return load_address_str.takeError();
  llvm::Expected<addr_t> load_address = ParseAddress(*load_address_str);
  if (!load_address)
    return load_address.takeError();

  llvm::Expected<StringRef> system_path =
      json_helpers::GetStringOrError(module, "systemPath");
  if (!system_path)
    return system_path.takeError();
  FileSpec system_file_spec(*system_path);
  NormalizePath(system_file_spec);

  llvm::Expected<llvm::Optional<StringRef>> file =
      json_helpers::GetOptionalStringOrError(module, "file");
  if (!file)
    return file.takeError();
  FileSpec local_file_spec(file->hasValue() ? **file : *system_path);
  NormalizePath(local_file_spec);

  ModuleSpec module_spec;
  module_spec.GetFileSpec() = local_file_spec;
  module_spec.GetPlatformFileSpec() = system_file_spec;
  module_spec.SetObjectOffset(*load_address);

  llvm::Expected<llvm::Optional<StringRef>> uuid_str =
      json_helpers::GetOptionalStringOrError(module, "uuid");
  if (!uuid_str)
    return uuid_str.takeError();
  if (uuid_str->hasValue())
    module_spec.GetUUID().SetFromStringRef(**uuid_str);

  Status error;
  ModuleSP module_sp =
      target_sp->GetOrCreateModule(module_spec, /*notify*/ false, &error);
  return error.ToError();
}

llvm::Error TraceSettingsParser::ParseModules(TargetSP &target_sp,
                                              const json::Object &process) {
  llvm::Expected<const json::Array &> modules =
      json_helpers::GetArrayOrError(process, "modules");
  if (!modules)
    return modules.takeError();

  for (const json::Value &module_val : *modules) {
    llvm::Expected<const json::Object &> module =
        json_helpers::ToObjectOrError(module_val);
    if (!module)
      return module.takeError();
    if (llvm::Error err = ParseModule(target_sp, *module))
      return err;
  }
  return llvm::Error::success();
}

llvm::Error TraceSettingsParser::ParseProcess(Debugger &debugger,
                                              const json::Object &process) {
  llvm::Expected<int64_t> pid = json_helpers::GetIntegerOrError(process, "pid");
  if (!pid)
    return pid.takeError();

  llvm::Expected<StringRef> triple =
      json_helpers::GetStringOrError(process, "triple");
  if (!triple)
    return triple.takeError();

  TargetSP target_sp;
  Status error = debugger.GetTargetList().CreateTarget(
      debugger, /*user_exe_path*/ llvm::StringRef(), *triple, eLoadDependentsNo,
      /*platform_options*/ nullptr, target_sp);

  if (!target_sp)
    return error.ToError();

  m_targets.push_back(target_sp);
  debugger.GetTargetList().SetSelectedTarget(target_sp.get());

  ProcessSP process_sp(target_sp->CreateProcess(
      /*listener*/ nullptr, /*plugin_name*/ llvm::StringRef(),
      /*crash_file*/ nullptr));
  process_sp->SetID(static_cast<lldb::pid_t>(*pid));

  if (llvm::Error err = ParseThreads(process_sp, process))
    return err;

  return ParseModules(target_sp, process);
}

llvm::Error TraceSettingsParser::ParseProcesses(Debugger &debugger) {
  llvm::Expected<const json::Array &> processes =
      json_helpers::GetArrayOrError(m_settings, "processes");
  if (!processes)
    return processes.takeError();

  for (const json::Value &process_val : *processes) {
    llvm::Expected<const json::Object &> process =
        json_helpers::ToObjectOrError(process_val);
    if (!process)
      return process.takeError();
    if (llvm::Error err = ParseProcess(debugger, *process))
      return err;
  }
  return llvm::Error::success();
}

llvm::Error TraceSettingsParser::ParseSettingsImpl(Debugger &debugger) {
  if (llvm::Error err = ParseProcesses(debugger))
    return err;
  return ParsePluginSettings();
}

llvm::Error
TraceSettingsParser::ParseSettings(Debugger &debugger,
                                   const llvm::json::Object &settings,
                                   llvm::StringRef settings_dir) {
  m_settings = settings;
  m_settings_dir = settings_dir.str();
  if (llvm::Error err = ParseSettingsImpl(debugger)) {
    // We clean all the targets that were created internally, which should leave
    // the debugger unchanged
    for (auto target_sp : m_targets)
      debugger.GetTargetList().DeleteTarget(target_sp);

    return createStringError(std::errc::invalid_argument, "%s\nSchema:\n%s",
                             llvm::toString(std::move(err)).c_str(),
                             GetSchema().data());
  }

  m_trace.m_settings = m_settings;
  m_trace.m_settings_dir = m_settings_dir;
  m_trace.m_thread_to_trace_file_map = m_thread_to_trace_file_map;
  m_trace.m_targets = m_targets;

  return llvm::Error::success();
}
