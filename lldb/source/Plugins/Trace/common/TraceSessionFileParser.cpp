//===-- TraceSessionFileParser.cpp ---------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===/

#include "TraceSessionFileParser.h"
#include "ThreadPostMortemTrace.h"

#include <sstream>

#include "lldb/Core/Debugger.h"
#include "lldb/Core/Module.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/Target.h"

using namespace lldb;
using namespace lldb_private;
using namespace llvm;

void TraceSessionFileParser::NormalizePath(lldb_private::FileSpec &file_spec) {
  if (file_spec.IsRelative())
    file_spec.PrependPathComponent(m_session_file_dir);
}

Error TraceSessionFileParser::ParseModule(lldb::TargetSP &target_sp,
                                          const JSONModule &module) {
  FileSpec system_file_spec(module.system_path);
  NormalizePath(system_file_spec);

  FileSpec local_file_spec(module.file.hasValue() ? *module.file
                                                  : module.system_path);
  NormalizePath(local_file_spec);

  ModuleSpec module_spec;
  module_spec.GetFileSpec() = local_file_spec;
  module_spec.GetPlatformFileSpec() = system_file_spec;

  if (module.uuid.hasValue())
    module_spec.GetUUID().SetFromStringRef(*module.uuid);

  Status error;
  ModuleSP module_sp =
      target_sp->GetOrCreateModule(module_spec, /*notify*/ false, &error);

  if (error.Fail())
    return error.ToError();

  bool load_addr_changed = false;
  module_sp->SetLoadAddress(*target_sp, module.load_address.value, false,
                            load_addr_changed);
  return llvm::Error::success();
}

Error TraceSessionFileParser::CreateJSONError(json::Path::Root &root,
                                              const json::Value &value) {
  std::string err;
  raw_string_ostream os(err);
  root.printErrorContext(value, os);
  return createStringError(
      std::errc::invalid_argument, "%s\n\nContext:\n%s\n\nSchema:\n%s",
      toString(root.getError()).c_str(), os.str().c_str(), m_schema.data());
}

std::string TraceSessionFileParser::BuildSchema(StringRef plugin_schema) {
  std::ostringstream schema_builder;
  schema_builder << "{\n  \"trace\": ";
  schema_builder << plugin_schema.data() << ",";
  schema_builder << R"(
  "processes": [
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
  // Notes:
  // All paths are either absolute or relative to the session file.
}
)";
  return schema_builder.str();
}

ThreadPostMortemTraceSP
TraceSessionFileParser::ParseThread(ProcessSP &process_sp,
                                    const JSONThread &thread) {
  lldb::tid_t tid = static_cast<lldb::tid_t>(thread.tid);

  FileSpec trace_file(thread.trace_file);
  NormalizePath(trace_file);

  ThreadPostMortemTraceSP thread_sp =
      std::make_shared<ThreadPostMortemTrace>(*process_sp, tid, trace_file);
  process_sp->GetThreadList().AddThread(thread_sp);
  return thread_sp;
}

Expected<TraceSessionFileParser::ParsedProcess>
TraceSessionFileParser::ParseProcess(const JSONProcess &process) {
  TargetSP target_sp;
  Status error = m_debugger.GetTargetList().CreateTarget(
      m_debugger, /*user_exe_path*/ StringRef(), process.triple,
      eLoadDependentsNo,
      /*platform_options*/ nullptr, target_sp);

  if (!target_sp)
    return error.ToError();

  ParsedProcess parsed_process;
  parsed_process.target_sp = target_sp;

  ProcessSP process_sp = target_sp->CreateProcess(
      /*listener*/ nullptr, "trace",
      /*crash_file*/ nullptr,
      /*can_connect*/ false);

  process_sp->SetID(static_cast<lldb::pid_t>(process.pid));

  for (const JSONThread &thread : process.threads)
    parsed_process.threads.push_back(ParseThread(process_sp, thread));

  for (const JSONModule &module : process.modules)
    if (Error err = ParseModule(target_sp, module))
      return std::move(err);

  if (!process.threads.empty())
    process_sp->GetThreadList().SetSelectedThreadByIndexID(0);

  // We invoke DidAttach to create a correct stopped state for the process and
  // its threads.
  ArchSpec process_arch;
  process_sp->DidAttach(process_arch);

  return parsed_process;
}

Expected<std::vector<TraceSessionFileParser::ParsedProcess>>
TraceSessionFileParser::ParseCommonSessionFile(
    const JSONTraceSessionBase &session) {
  std::vector<ParsedProcess> parsed_processes;

  auto onError = [&]() {
    // Delete all targets that were created so far in case of failures
    for (ParsedProcess &parsed_process : parsed_processes)
      m_debugger.GetTargetList().DeleteTarget(parsed_process.target_sp);
  };

  for (const JSONProcess &process : session.processes) {
    if (Expected<ParsedProcess> parsed_process = ParseProcess(process))
      parsed_processes.push_back(std::move(*parsed_process));
    else {
      onError();
      return parsed_process.takeError();
    }
  }
  return parsed_processes;
}
