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

FileSpec TraceIntelPTSessionFileParser::NormalizePath(const std::string &path) {
  FileSpec file_spec(path);
  if (file_spec.IsRelative())
    file_spec.PrependPathComponent(m_session_file_dir);
  return file_spec;
}

Error TraceIntelPTSessionFileParser::ParseModule(Target &target,
                                                 const JSONModule &module) {
  auto do_parse = [&]() -> Error {
    FileSpec system_file_spec(module.system_path);

    FileSpec local_file_spec(module.file.hasValue() ? *module.file
                                                    : module.system_path);

    ModuleSpec module_spec;
    module_spec.GetFileSpec() = local_file_spec;
    module_spec.GetPlatformFileSpec() = system_file_spec;

    if (module.uuid.hasValue())
      module_spec.GetUUID().SetFromStringRef(*module.uuid);

    Status error;
    ModuleSP module_sp =
        target.GetOrCreateModule(module_spec, /*notify*/ false, &error);

    if (error.Fail())
      return error.ToError();

    bool load_addr_changed = false;
    module_sp->SetLoadAddress(target, module.load_address.value, false,
                              load_addr_changed);
    return Error::success();
  };
  if (Error err = do_parse())
    return createStringError(
        inconvertibleErrorCode(), "Error when parsing module %s. %s",
        module.system_path.c_str(), toString(std::move(err)).c_str());
  return Error::success();
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
TraceIntelPTSessionFileParser::ParseThread(Process &process,
                                           const JSONThread &thread) {
  lldb::tid_t tid = static_cast<lldb::tid_t>(thread.tid);

  Optional<FileSpec> trace_file;
  if (thread.ipt_trace)
    trace_file = FileSpec(*thread.ipt_trace);

  ThreadPostMortemTraceSP thread_sp =
      std::make_shared<ThreadPostMortemTrace>(process, tid, trace_file);
  process.GetThreadList().AddThread(thread_sp);
  return thread_sp;
}

Expected<TraceIntelPTSessionFileParser::ParsedProcess>
TraceIntelPTSessionFileParser::ParseProcess(const JSONProcess &process) {
  TargetSP target_sp;
  Status error = m_debugger.GetTargetList().CreateTarget(
      m_debugger, /*user_exe_path*/ StringRef(), process.triple.getValueOr(""),
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
    parsed_process.threads.push_back(ParseThread(*process_sp, thread));

  for (const JSONModule &module : process.modules)
    if (Error err = ParseModule(*target_sp, module))
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
      "triple"?: string,
          // Optional clang/llvm target triple.
      "threads": [
          // A list of known threads for the given process. When context switch
          // data is provided, LLDB will automatically create threads for the
          // this process whenever it finds new threads when traversing the
          // context switches, so passing values to this list in this case is
          // optional.
        {
          "tid": integer,
          "iptTrace"?: string
              // Path to the raw Intel PT buffer file for this thread.
        }
      ],
      "modules": [
        {
          "systemPath": string,
              // Original path of the module at runtime.
          "file"?: string,
              // Path to a copy of the file if not available at "systemPath".
          "loadAddress": integer | string decimal | hex string,
              // Lowest address of the sections of the module loaded on memory.
          "uuid"?: string,
              // Build UUID for the file for sanity checks.
        }
      ]
    }
  ],
  "cpus"?: [
    {
      "id": integer,
          // Id of this CPU core.
      "iptTrace": string,
          // Path to the raw Intel PT buffer for this cpu core.
      "contextSwitchTrace": string,
          // Path to the raw perf_event_open context switch trace file for this cpu core.
          // The perf_event must have been configured with PERF_SAMPLE_TID and
          // PERF_SAMPLE_TIME, as well as sample_id_all = 1.
    }
  ],
  "tscPerfZeroConversion"?: {
    // Values used to convert between TSCs and nanoseconds. See the time_zero
    // section in https://man7.org/linux/man-pages/man2/perf_event_open.2.html
    // for for information.

    "timeMult": integer,
    "timeShift": integer,
    "timeZero": integer | string decimal | hex string,
  }
}

Notes:

- All paths are either absolute or relative to folder containing the session file.
- "cpus" is provided if and only if processes[].threads[].iptTrace is not provided.
- "tscPerfZeroConversion" must be provided if "cpus" is provided.
 })";
  }
  return schema;
}

Error TraceIntelPTSessionFileParser::AugmentThreadsFromContextSwitches(
    JSONTraceSession &session) {
  if (!session.cpus)
    return Error::success();

  if (!session.tsc_perf_zero_conversion)
    return createStringError(inconvertibleErrorCode(),
                             "TSC to nanos conversion values are needed when "
                             "context switch information is provided.");

  DenseMap<lldb::pid_t, JSONProcess *> indexed_processes;
  DenseMap<JSONProcess *, DenseSet<tid_t>> indexed_threads;

  for (JSONProcess &process : session.processes) {
    indexed_processes[process.pid] = &process;
    for (JSONThread &thread : process.threads)
      indexed_threads[&process].insert(thread.tid);
  }

  auto on_thread_seen = [&](lldb::pid_t pid, tid_t tid) {
    auto proc = indexed_processes.find(pid);
    if (proc == indexed_processes.end())
      return;
    if (indexed_threads[proc->second].count(tid))
      return;
    indexed_threads[proc->second].insert(tid);
    proc->second->threads.push_back({tid, /*ipt_trace=*/None});
  };

  for (const JSONCpu &cpu : *session.cpus) {
    Error err = Trace::OnDataFileRead(
        FileSpec(cpu.context_switch_trace),
        [&](ArrayRef<uint8_t> data) -> Error {
          Expected<std::vector<ThreadContinuousExecution>> executions =
              DecodePerfContextSwitchTrace(data, cpu.id,
                                           *session.tsc_perf_zero_conversion);
          if (!executions)
            return executions.takeError();
          for (const ThreadContinuousExecution &execution : *executions)
            on_thread_seen(execution.pid, execution.tid);
          return Error::success();
        });
    if (err)
      return err;
  }
  return Error::success();
}

Expected<TraceSP> TraceIntelPTSessionFileParser::CreateTraceIntelPTInstance(
    JSONTraceSession &session, std::vector<ParsedProcess> &parsed_processes) {
  std::vector<ThreadPostMortemTraceSP> threads;
  std::vector<ProcessSP> processes;
  for (const ParsedProcess &parsed_process : parsed_processes) {
    processes.push_back(parsed_process.target_sp->GetProcessSP());
    threads.insert(threads.end(), parsed_process.threads.begin(),
                   parsed_process.threads.end());
  }

  TraceSP trace_instance = TraceIntelPT::CreateInstanceForPostmortemTrace(
      session, processes, threads);
  for (const ParsedProcess &parsed_process : parsed_processes)
    parsed_process.target_sp->SetTrace(trace_instance);

  return trace_instance;
}

void TraceIntelPTSessionFileParser::NormalizeAllPaths(
    JSONTraceSession &session) {
  for (JSONProcess &process : session.processes) {
    for (JSONModule &module : process.modules) {
      module.system_path = NormalizePath(module.system_path).GetPath();
      if (module.file)
        module.file = NormalizePath(*module.file).GetPath();
    }
    for (JSONThread &thread : process.threads) {
      if (thread.ipt_trace)
        thread.ipt_trace = NormalizePath(*thread.ipt_trace).GetPath();
    }
  }
  if (session.cpus) {
    for (JSONCpu &cpu : *session.cpus) {
      cpu.context_switch_trace =
          NormalizePath(cpu.context_switch_trace).GetPath();
      cpu.ipt_trace = NormalizePath(cpu.ipt_trace).GetPath();
    }
  }
}

Expected<TraceSP> TraceIntelPTSessionFileParser::Parse() {
  json::Path::Root root("traceSession");
  JSONTraceSession session;
  if (!fromJSON(m_trace_session_file, session, root))
    return CreateJSONError(root, m_trace_session_file);

  NormalizeAllPaths(session);

  if (Error err = AugmentThreadsFromContextSwitches(session))
    return std::move(err);

  if (Expected<std::vector<ParsedProcess>> parsed_processes =
          ParseSessionFile(session))
    return CreateTraceIntelPTInstance(session, *parsed_processes);
  else
    return parsed_processes.takeError();
}
