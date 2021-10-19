//===-- TraceSessionSaver.cpp ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TraceSessionSaver.h"

#include "lldb/Core/Module.h"
#include "lldb/Core/Value.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/SectionLoadList.h"
#include "lldb/Target/Target.h"
#include "lldb/lldb-types.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/JSON.h"

#include <fstream>

using namespace lldb;
using namespace lldb_private;
using namespace llvm;

llvm::Error TraceSessionSaver::WriteSessionToFile(
    const llvm::json::Value &trace_session_description, FileSpec directory) {

  FileSpec trace_path = directory;
  trace_path.AppendPathComponent("trace.json");
  std::ofstream os(trace_path.GetPath());
  os << std::string(formatv("{0:2}", trace_session_description));
  os.close();
  if (!os)
    return createStringError(inconvertibleErrorCode(),
                             formatv("couldn't write to the file {0}",
                                     trace_path.GetPath().c_str()));
  return Error::success();
}

llvm::Expected<JSONTraceSessionBase> TraceSessionSaver::BuildProcessesSection(
    Process &live_process,
    std::function<
        llvm::Expected<llvm::Optional<std::vector<uint8_t>>>(lldb::tid_t tid)>
        raw_trace_fetcher,
    FileSpec directory) {

  JSONTraceSessionBase json_session_description;
  Expected<std::vector<JSONThread>> json_threads =
      BuildThreadsSection(live_process, raw_trace_fetcher, directory);
  if (!json_threads)
    return json_threads.takeError();

  Expected<std::vector<JSONModule>> json_modules =
      BuildModulesSection(live_process, directory);
  if (!json_modules)
    return json_modules.takeError();

  json_session_description.processes.push_back(JSONProcess{
      static_cast<int64_t>(live_process.GetID()),
      live_process.GetTarget().GetArchitecture().GetTriple().getTriple(),
      json_threads.get(), json_modules.get()});
  return json_session_description;
}

llvm::Expected<std::vector<JSONThread>> TraceSessionSaver::BuildThreadsSection(
    Process &live_process,
    std::function<
        llvm::Expected<llvm::Optional<std::vector<uint8_t>>>(lldb::tid_t tid)>
        raw_trace_fetcher,
    FileSpec directory) {
  std::vector<JSONThread> json_threads;
  for (ThreadSP thread_sp : live_process.Threads()) {
    // resolve the directory just in case
    FileSystem::Instance().Resolve(directory);
    FileSpec raw_trace_path = directory;
    raw_trace_path.AppendPathComponent(std::to_string(thread_sp->GetID()) +
                                       ".trace");
    json_threads.push_back(JSONThread{static_cast<int64_t>(thread_sp->GetID()),
                                      raw_trace_path.GetPath().c_str()});

    llvm::Expected<llvm::Optional<std::vector<uint8_t>>> raw_trace =
        raw_trace_fetcher(thread_sp->GetID());

    if (!raw_trace)
      return raw_trace.takeError();
    if (!raw_trace.get())
      continue;

    std::basic_fstream<char> raw_trace_fs = std::fstream(
        raw_trace_path.GetPath().c_str(), std::ios::out | std::ios::binary);
    raw_trace_fs.write(reinterpret_cast<const char *>(&raw_trace.get()->at(0)),
                       raw_trace.get()->size() * sizeof(uint8_t));
    raw_trace_fs.close();
    if (!raw_trace_fs) {
      return createStringError(inconvertibleErrorCode(),
                               formatv("couldn't write to the file {0}",
                                       raw_trace_path.GetPath().c_str()));
    }
  }
  return json_threads;
}

llvm::Expected<std::vector<JSONModule>>
TraceSessionSaver::BuildModulesSection(Process &live_process,
                                       FileSpec directory) {
  std::vector<JSONModule> json_modules;
  ModuleList module_list = live_process.GetTarget().GetImages();
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
        !live_process.GetTarget().GetSectionLoadList().IsEmpty())
      load_addr = base_addr.GetLoadAddress(&live_process.GetTarget());

    if (load_addr == LLDB_INVALID_ADDRESS)
      continue;

    FileSystem::Instance().Resolve(directory);
    FileSpec path_to_copy_module = directory;
    path_to_copy_module.AppendPathComponent("modules");
    path_to_copy_module.AppendPathComponent(system_path);
    sys::fs::create_directories(path_to_copy_module.GetDirectory().AsCString());

    if (std::error_code ec = llvm::sys::fs::copy_file(
            system_path, path_to_copy_module.GetPath()))
      return createStringError(
          inconvertibleErrorCode(),
          formatv("couldn't write to the file. {0}", ec.message()));

    json_modules.push_back(
        JSONModule{system_path, path_to_copy_module.GetPath(),
                   JSONAddress{load_addr}, module_sp->GetUUID().GetAsString()});
  }
  return json_modules;
}
