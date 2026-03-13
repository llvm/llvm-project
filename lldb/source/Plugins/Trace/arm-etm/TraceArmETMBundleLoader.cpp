//===-- TraceArmETMBundleLoader.cpp ---------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TraceArmETMBundleLoader.h"

#include "TraceArmETM.h"
#include "lldb/Core/Debugger.h"
#include "lldb/Target/ProcessTrace.h"

using namespace lldb;
using namespace lldb_private;
using namespace lldb_private::trace_arm_etm;
using namespace llvm;

Error TraceArmETMBundleLoader::CreateJSONError(json::Path::Root &root,
                                               const json::Value &value) {
  std::string err;
  raw_string_ostream os(err);
  root.printErrorContext(value, os);
  return createStringError(
      std::errc::invalid_argument, "%s\n\nContext:\n%s\n\nSchema:\n%s",
      toString(root.getError()).c_str(), err.c_str(), GetSchema().data());
}

ThreadPostMortemTraceSP
TraceArmETMBundleLoader::ParseThread(Process &process,
                                     const JSONThread &thread) {
  lldb::tid_t tid = static_cast<lldb::tid_t>(thread.tid);

  std::optional<FileSpec> trace_file;
  if (thread.etm_trace)
    trace_file = FileSpec(*thread.etm_trace);

  ThreadPostMortemTraceSP thread_sp =
      std::make_shared<ThreadPostMortemTrace>(process, tid, trace_file);
  process.GetThreadList().AddThread(thread_sp);
  return thread_sp;
}

Expected<TraceArmETMBundleLoader::ParsedProcess>
TraceArmETMBundleLoader::ParseProcess(const JSONProcess &process) {
  Expected<ParsedProcess> parsed_process =
      CreateEmptyProcess(process.pid, process.triple.value_or(""));

  if (!parsed_process)
    return parsed_process.takeError();

  ProcessSP process_sp = parsed_process->target_sp->GetProcessSP();

  for (const JSONThread &thread : process.threads)
    parsed_process->threads.push_back(ParseThread(*process_sp, thread));

  for (const JSONModule &module : process.modules)
    if (Error err = ParseModule(*parsed_process->target_sp, module))
      return std::move(err);

  if (!process.threads.empty())
    process_sp->GetThreadList().SetSelectedThreadByIndexID(0);

  // We invoke DidAttach to create a correct stopped state for the process and
  // its threads.
  ArchSpec process_arch;
  process_sp->DidAttach(process_arch);

  return parsed_process;
}

Expected<std::vector<TraceArmETMBundleLoader::ParsedProcess>>
TraceArmETMBundleLoader::LoadBundle(
    const JSONTraceBundleDescription &bundle_description) {
  std::vector<ParsedProcess> parsed_processes;

  auto HandleError = [&](Error &&err) {
    // Delete all targets that were created so far in case of failures
    for (ParsedProcess &parsed_process : parsed_processes)
      m_debugger.GetTargetList().DeleteTarget(parsed_process.target_sp);
    return std::move(err);
  };

  if (bundle_description.processes) {
    for (const JSONProcess &process : *bundle_description.processes) {
      if (Expected<ParsedProcess> parsed_process = ParseProcess(process))
        parsed_processes.push_back(std::move(*parsed_process));
      else
        return HandleError(parsed_process.takeError());
    }
  }

  return parsed_processes;
}

StringRef TraceArmETMBundleLoader::GetSchema() {
  static std::string schema;
  if (schema.empty()) {
    schema = R"({
  "type": "arm-etm",
  "processes?": [
    {
      "pid": integer,
      "triple"?: string,
          // Optional clang/llvm target triple.
          // This must be provided if the trace will be created not using the
          // CLI or on a machine other than where the target was traced.
      "threads": [
          // A list of known threads for the given process.
        {
          "tid": integer,
          "etmTrace"?: string
              // Path to the raw ARM ETM buffer file for this thread.
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
  ]
}

Notes:

- All paths are either absolute or relative to folder containing the bundle
  description file.})";
  }
  return schema;
}

Expected<TraceSP> TraceArmETMBundleLoader::CreateTraceArmETMInstance(
    JSONTraceBundleDescription &bundle_description,
    std::vector<ParsedProcess> &parsed_processes) {
  std::vector<ThreadPostMortemTraceSP> threads;
  std::vector<ProcessSP> processes;
  for (const ParsedProcess &parsed_process : parsed_processes) {
    processes.push_back(parsed_process.target_sp->GetProcessSP());
    threads.insert(threads.end(), parsed_process.threads.begin(),
                   parsed_process.threads.end());
  }

  TraceSP trace_instance = TraceArmETM::CreateInstanceForPostmortemTrace(
      bundle_description, processes, threads);
  for (const ParsedProcess &parsed_process : parsed_processes)
    parsed_process.target_sp->SetTrace(trace_instance);

  return trace_instance;
}

void TraceArmETMBundleLoader::NormalizeAllPaths(
    JSONTraceBundleDescription &bundle_description) {
  if (bundle_description.processes) {
    for (JSONProcess &process : *bundle_description.processes) {
      for (JSONModule &module : process.modules) {
        module.system_path = NormalizePath(module.system_path).GetPath();
        if (module.file)
          module.file = NormalizePath(*module.file).GetPath();
      }
      for (JSONThread &thread : process.threads) {
        if (thread.etm_trace)
          thread.etm_trace = NormalizePath(*thread.etm_trace).GetPath();
      }
    }
  }
}

Expected<TraceSP> TraceArmETMBundleLoader::Load() {
  json::Path::Root root("traceBundle");
  JSONTraceBundleDescription bundle_description;
  if (!fromJSON(m_bundle_description, bundle_description, root))
    return CreateJSONError(root, m_bundle_description);

  NormalizeAllPaths(bundle_description);

  if (Expected<std::vector<ParsedProcess>> parsed_processes =
          LoadBundle(bundle_description))
    return CreateTraceArmETMInstance(bundle_description, *parsed_processes);
  else
    return parsed_processes.takeError();
}
