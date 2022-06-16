//===-- TraceIntelPTSessionFileParser.cpp ---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TraceIntelPTSessionFileParser.h"

#include "../common/ThreadPostMortemTrace.h"
#include "TraceIntelPT.h"
#include "TraceIntelPTJSONStructs.h"

#include "lldb/Core/Debugger.h"
#include "lldb/Core/Module.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/Target.h"

using namespace lldb;
using namespace lldb_private;
using namespace lldb_private::trace_intel_pt;
using namespace llvm;

void TraceIntelPTSessionFileParser::NormalizePath(
    lldb_private::FileSpec &file_spec) {
  if (file_spec.IsRelative())
    file_spec.PrependPathComponent(m_session_file_dir);
}

Error TraceIntelPTSessionFileParser::ParseModule(lldb::TargetSP &target_sp,
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
  module_sp->SetLoadAddress(*target_sp, module.load_address, false,
                            load_addr_changed);
  return llvm::Error::success();
}

Error TraceIntelPTSessionFileParser::CreateJSONError(json::Path::Root &root,
                                                     const json::Value &value) {
  std::string err;
  raw_string_ostream os(err);
  root.printErrorContext(value, os);
  return createStringError(
      std::errc::invalid_argument, "%s\n\nContext:\n%s\n\nSchema:\n%s",
      toString(root.getError()).c_str(), os.str().c_str(), GetSchema().data());
}

ThreadPostMortemTraceSP
TraceIntelPTSessionFileParser::ParseThread(ProcessSP &process_sp,
                                           const JSONThread &thread) {
  lldb::tid_t tid = static_cast<lldb::tid_t>(thread.tid);

  Optional<FileSpec> trace_file;
  if (thread.trace_buffer) {
    trace_file.emplace(*thread.trace_buffer);
    NormalizePath(*trace_file);
  }

  ThreadPostMortemTraceSP thread_sp =
      std::make_shared<ThreadPostMortemTrace>(*process_sp, tid, trace_file);
  process_sp->GetThreadList().AddThread(thread_sp);
  return thread_sp;
}

Expected<TraceIntelPTSessionFileParser::ParsedProcess>
TraceIntelPTSessionFileParser::ParseProcess(const JSONProcess &process) {
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

Expected<std::vector<TraceIntelPTSessionFileParser::ParsedProcess>>
TraceIntelPTSessionFileParser::ParseSessionFile(
    const JSONTraceSession &session) {
  std::vector<ParsedProcess> parsed_processes;

  auto HandleError = [&](Error &&err) {
    // Delete all targets that were created so far in case of failures
    for (ParsedProcess &parsed_process : parsed_processes)
      m_debugger.GetTargetList().DeleteTarget(parsed_process.target_sp);
    return std::move(err);
  };

  for (const JSONProcess &process : session.processes) {
    if (Expected<ParsedProcess> parsed_process = ParseProcess(process))
      parsed_processes.push_back(std::move(*parsed_process));
    else
      return HandleError(parsed_process.takeError());
  }
  return parsed_processes;
}

StringRef TraceIntelPTSessionFileParser::GetSchema() {
  static std::string schema;
  if (schema.empty()) {
    schema = R"({
  "type": "intel-pt",
  "cpuInfo": {
    // CPU information gotten from, for example, /proc/cpuinfo.

    "vendor": "GenuineIntel" | "unknown",
    "family": integer,
    "model": integer,
    "stepping": integer
  },
  "processes": [
    {
      "pid": integer,
      "triple": string,
          // clang/llvm target triple.
      "threads": [
        {
          "tid": integer,
          "traceBuffer"?: string
              // Path to the raw Intel PT buffer file for this thread.
        }
      ],
      "modules": [
        {
          "systemPath": string,
              // Original path of the module at runtime.
          "file"?: string,
              // Path to a copy of the file if not available at "systemPath".
          "loadAddress": integer,
              // Lowest address of the sections of the module loaded on memory.
          "uuid"?: string,
              // Build UUID for the file for sanity checks.
        }
      ]
    }
  ],
  "cores"?: [
    {
      "coreId": integer,
          // Id of this CPU core.
      "traceBuffer": string,
          // Path to the raw Intel PT buffer for this core.
      "contextSwitchTrace": string,
          // Path to the raw perf_event_open context switch trace file for this core.
    }
  ],
  "tscPerfZeroConversion"?: {
    // Values used to convert between TSCs and nanoseconds. See the time_zero
    // section in https://man7.org/linux/man-pages/man2/perf_event_open.2.html
    // for for information.

    "timeMult": integer,
    "timeShift": integer,
    "timeZero": integer,
  }
}

Notes:

- All paths are either absolute or relative to folder containing the session file.
- "cores" is provided if and only if processes[].threads[].traceBuffer is not provided.
- "tscPerfZeroConversion" must be provided if "cores" is provided.
 })";
  }
  return schema;
}

TraceSP TraceIntelPTSessionFileParser::CreateTraceIntelPTInstance(
    JSONTraceSession &session, std::vector<ParsedProcess> &parsed_processes) {
  std::vector<ThreadPostMortemTraceSP> threads;
  std::vector<ProcessSP> processes;
  for (const ParsedProcess &parsed_process : parsed_processes) {
    processes.push_back(parsed_process.target_sp->GetProcessSP());
    threads.insert(threads.end(), parsed_process.threads.begin(),
                   parsed_process.threads.end());
  }

  TraceSP trace_instance(new TraceIntelPT(session, processes, threads));
  for (const ParsedProcess &parsed_process : parsed_processes)
    parsed_process.target_sp->SetTrace(trace_instance);

  return trace_instance;
}

Expected<TraceSP> TraceIntelPTSessionFileParser::Parse() {
  json::Path::Root root("traceSession");
  JSONTraceSession session;
  if (!fromJSON(m_trace_session_file, session, root))
    return CreateJSONError(root, m_trace_session_file);

  if (Expected<std::vector<ParsedProcess>> parsed_processes =
          ParseSessionFile(session))
    return CreateTraceIntelPTInstance(session, *parsed_processes);
  else
    return parsed_processes.takeError();
}
