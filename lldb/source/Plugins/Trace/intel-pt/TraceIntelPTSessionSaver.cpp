//===-- TraceIntelPTSessionSaver.cpp --------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TraceIntelPTSessionSaver.h"
#include "TraceIntelPT.h"
#include "TraceIntelPTJSONStructs.h"
#include "lldb/Core/Module.h"
#include "lldb/Core/ModuleList.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/SectionLoadList.h"
#include "lldb/Target/Target.h"
#include "lldb/Target/ThreadList.h"
#include "lldb/lldb-types.h"
#include "llvm/ADT/None.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/JSON.h"

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

using namespace lldb;
using namespace lldb_private;
using namespace lldb_private::trace_intel_pt;
using namespace llvm;

/// Write a stream of bytes from \p data to the given output file.
/// It creates or overwrites the output file, but not append.
static llvm::Error WriteBytesToDisk(FileSpec &output_file,
                                    ArrayRef<uint8_t> data) {
  std::basic_fstream<char> out_fs = std::fstream(
      output_file.GetPath().c_str(), std::ios::out | std::ios::binary);
  if (!data.empty())
    out_fs.write(reinterpret_cast<const char *>(&data[0]), data.size());

  out_fs.close();
  if (!out_fs)
    return createStringError(inconvertibleErrorCode(),
                             formatv("couldn't write to the file {0}",
                                     output_file.GetPath().c_str()));
  return Error::success();
}

/// Save the trace session description JSON object inside the given directory
/// as a file named \a trace.json.
///
/// \param[in] trace_session_json
///     The trace's session, as JSON Object.
///
/// \param[in] directory
///     The directory where the JSON file will be saved.
///
/// \return
///     \a llvm::Success if the operation was successful, or an \a llvm::Error
///     otherwise.
static llvm::Error
WriteSessionToFile(const llvm::json::Value &trace_session_json,
                   const FileSpec &directory) {
  FileSpec trace_path = directory;
  trace_path.AppendPathComponent("trace.json");
  std::ofstream os(trace_path.GetPath());
  os << formatv("{0:2}", trace_session_json).str();
  os.close();
  if (!os)
    return createStringError(inconvertibleErrorCode(),
                             formatv("couldn't write to the file {0}",
                                     trace_path.GetPath().c_str()));
  return Error::success();
}

/// Build the threads sub-section of the trace session description file.
/// Any associated binary files are created inside the given directory.
///
/// \param[in] process
///     The process being traced.
///
/// \param[in] directory
///     The directory where files will be saved when building the threads
///     section.
///
/// \return
///     The threads section or \a llvm::Error in case of failures.
static llvm::Expected<std::vector<JSONThread>>
BuildThreadsSection(Process &process, FileSpec directory) {
  std::vector<JSONThread> json_threads;
  TraceSP trace_sp = process.GetTarget().GetTrace();

  FileSpec threads_dir = directory;
  threads_dir.AppendPathComponent("threads");
  sys::fs::create_directories(threads_dir.GetCString());

  for (ThreadSP thread_sp : process.Threads()) {
    lldb::tid_t tid = thread_sp->GetID();
    if (!trace_sp->IsTraced(tid))
      continue;

    JSONThread json_thread;
    json_thread.tid = tid;

    if (trace_sp->GetTracedCores().empty()) {
      FileSpec output_file = threads_dir;
      output_file.AppendPathComponent(std::to_string(tid) + ".intelpt_trace");
      json_thread.trace_buffer = output_file.GetPath();

      llvm::Error err = process.GetTarget().GetTrace()->OnThreadBinaryDataRead(
          tid, IntelPTDataKinds::kTraceBuffer,
          [&](llvm::ArrayRef<uint8_t> data) -> llvm::Error {
            return WriteBytesToDisk(output_file, data);
          });
      if (err)
        return std::move(err);
    }

    json_threads.push_back(std::move(json_thread));
  }
  return json_threads;
}

static llvm::Expected<llvm::Optional<std::vector<JSONCore>>>
BuildCoresSection(TraceIntelPT &trace_ipt, FileSpec directory) {
  if (trace_ipt.GetTracedCores().empty())
    return None;

  std::vector<JSONCore> json_cores;
  FileSpec cores_dir = directory;
  cores_dir.AppendPathComponent("cores");
  sys::fs::create_directories(cores_dir.GetCString());

  for (lldb::core_id_t core_id : trace_ipt.GetTracedCores()) {
    JSONCore json_core;
    json_core.core_id = core_id;

    {
      FileSpec output_trace = cores_dir;
      output_trace.AppendPathComponent(std::to_string(core_id) +
                                       ".intelpt_trace");
      json_core.trace_buffer = output_trace.GetPath();

      llvm::Error err = trace_ipt.OnCoreBinaryDataRead(
          core_id, IntelPTDataKinds::kTraceBuffer,
          [&](llvm::ArrayRef<uint8_t> data) -> llvm::Error {
            return WriteBytesToDisk(output_trace, data);
          });
      if (err)
        return std::move(err);
    }

    {
      FileSpec output_context_switch_trace = cores_dir;
      output_context_switch_trace.AppendPathComponent(
          std::to_string(core_id) + ".perf_context_switch_trace");
      json_core.context_switch_trace = output_context_switch_trace.GetPath();

      llvm::Error err = trace_ipt.OnCoreBinaryDataRead(
          core_id, IntelPTDataKinds::kPerfContextSwitchTrace,
          [&](llvm::ArrayRef<uint8_t> data) -> llvm::Error {
            return WriteBytesToDisk(output_context_switch_trace, data);
          });
      if (err)
        return std::move(err);
    }
    json_cores.push_back(std::move(json_core));
  }
  return json_cores;
}

/// Build modules sub-section of the trace's session. The original modules
/// will be copied over to the \a <directory/modules> folder. Invalid modules
/// are skipped.
/// Copying the modules has the benefit of making these trace session
/// directories self-contained, as the raw traces and modules are part of the
/// output directory and can be sent to another machine, where lldb can load
/// them and replicate exactly the same trace session.
///
/// \param[in] process
///     The process being traced.
///
/// \param[in] directory
///     The directory where the modules files will be saved when building
///     the modules section.
///     Example: If a module \a libbar.so exists in the path
///     \a /usr/lib/foo/libbar.so, then it will be copied to
///     \a <directory>/modules/usr/lib/foo/libbar.so.
///
/// \return
///     The modules section or \a llvm::Error in case of failures.
static llvm::Expected<std::vector<JSONModule>>
BuildModulesSection(Process &process, FileSpec directory) {
  std::vector<JSONModule> json_modules;
  ModuleList module_list = process.GetTarget().GetImages();
  for (size_t i = 0; i < module_list.GetSize(); ++i) {
    ModuleSP module_sp(module_list.GetModuleAtIndex(i));
    if (!module_sp)
      continue;
    std::string system_path = module_sp->GetPlatformFileSpec().GetPath();
    // TODO: support memory-only libraries like [vdso]
    if (!module_sp->GetFileSpec().IsAbsolute())
      continue;

    std::string file = module_sp->GetFileSpec().GetPath();
    ObjectFile *objfile = module_sp->GetObjectFile();
    if (objfile == nullptr)
      continue;

    lldb::addr_t load_addr = LLDB_INVALID_ADDRESS;
    Address base_addr(objfile->GetBaseAddress());
    if (base_addr.IsValid() &&
        !process.GetTarget().GetSectionLoadList().IsEmpty())
      load_addr = base_addr.GetLoadAddress(&process.GetTarget());

    if (load_addr == LLDB_INVALID_ADDRESS)
      continue;

    FileSpec path_to_copy_module = directory;
    path_to_copy_module.AppendPathComponent("modules");
    path_to_copy_module.AppendPathComponent(system_path);
    sys::fs::create_directories(path_to_copy_module.GetDirectory().AsCString());

    if (std::error_code ec = llvm::sys::fs::copy_file(
            system_path, path_to_copy_module.GetPath()))
      return createStringError(
          inconvertibleErrorCode(),
          formatv("couldn't write to the file. {0}", ec.message()));

    json_modules.push_back(JSONModule{system_path,
                                      path_to_copy_module.GetPath(), load_addr,
                                      module_sp->GetUUID().GetAsString()});
  }
  return json_modules;
}

/// Build the processes section of the trace session description file. Besides
/// returning the processes information, this method saves to disk all modules
/// and raw traces corresponding to the traced threads of the given process.
///
/// \param[in] process
///     The process being traced.
///
/// \param[in] directory
///     The directory where files will be saved when building the processes
///     section.
///
/// \return
///     The processes section or \a llvm::Error in case of failures.
static llvm::Expected<JSONProcess>
BuildProcessSection(Process &process, const FileSpec &directory) {
  Expected<std::vector<JSONThread>> json_threads =
      BuildThreadsSection(process, directory);
  if (!json_threads)
    return json_threads.takeError();

  Expected<std::vector<JSONModule>> json_modules =
      BuildModulesSection(process, directory);
  if (!json_modules)
    return json_modules.takeError();

  return JSONProcess{
      process.GetID(),
      process.GetTarget().GetArchitecture().GetTriple().getTriple(),
      json_threads.get(), json_modules.get()};
}

/// See BuildProcessSection()
static llvm::Expected<std::vector<JSONProcess>>
BuildProcessesSection(TraceIntelPT &trace_ipt, const FileSpec &directory) {
  std::vector<JSONProcess> processes;
  for (Process *process : trace_ipt.GetAllProcesses()) {
    if (llvm::Expected<JSONProcess> json_process =
            BuildProcessSection(*process, directory))
      processes.push_back(std::move(*json_process));
    else
      return json_process.takeError();
  }
  return processes;
}

Error TraceIntelPTSessionSaver::SaveToDisk(TraceIntelPT &trace_ipt,
                                           FileSpec directory) {
  if (std::error_code ec =
          sys::fs::create_directories(directory.GetPath().c_str()))
    return llvm::errorCodeToError(ec);

  Expected<pt_cpu> cpu_info = trace_ipt.GetCPUInfo();
  if (!cpu_info)
    return cpu_info.takeError();

  FileSystem::Instance().Resolve(directory);

  Expected<std::vector<JSONProcess>> json_processes =
      BuildProcessesSection(trace_ipt, directory);

  if (!json_processes)
    return json_processes.takeError();

  Expected<Optional<std::vector<JSONCore>>> json_cores =
      BuildCoresSection(trace_ipt, directory);
  if (!json_cores)
    return json_cores.takeError();

  JSONTraceSession json_intel_pt_session{"intel-pt", *cpu_info, *json_processes,
                                         *json_cores,
                                         trace_ipt.GetPerfZeroTscConversion()};

  return WriteSessionToFile(toJSON(json_intel_pt_session), directory);
}
