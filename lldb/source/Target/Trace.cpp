//===-- Trace.cpp ---------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Target/Trace.h"

#include "llvm/Support/Format.h"

#include "lldb/Core/Module.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/Symbol/Function.h"
#include "lldb/Target/ExecutionContext.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/SectionLoadList.h"
#include "lldb/Target/Thread.h"
#include "lldb/Utility/LLDBLog.h"
#include "lldb/Utility/Stream.h"

using namespace lldb;
using namespace lldb_private;
using namespace llvm;

// Helper structs used to extract the type of a trace session json without
// having to parse the entire object.

struct JSONSimpleTraceSession {
  std::string type;
};

namespace llvm {
namespace json {

bool fromJSON(const Value &value, JSONSimpleTraceSession &session, Path path) {
  json::ObjectMapper o(value, path);
  return o && o.map("type", session.type);
}

} // namespace json
} // namespace llvm

static Error createInvalidPlugInError(StringRef plugin_name) {
  return createStringError(
      std::errc::invalid_argument,
      "no trace plug-in matches the specified type: \"%s\"",
      plugin_name.data());
}

Expected<lldb::TraceSP>
Trace::FindPluginForPostMortemProcess(Debugger &debugger,
                                      const json::Value &trace_session_file,
                                      StringRef session_file_dir) {
  JSONSimpleTraceSession json_session;
  json::Path::Root root("traceSession");
  if (!json::fromJSON(trace_session_file, json_session, root))
    return root.getError();

  if (auto create_callback =
          PluginManager::GetTraceCreateCallback(json_session.type))
    return create_callback(trace_session_file, session_file_dir, debugger);

  return createInvalidPlugInError(json_session.type);
}

Expected<lldb::TraceSP> Trace::FindPluginForLiveProcess(llvm::StringRef name,
                                                        Process &process) {
  if (!process.IsLiveDebugSession())
    return createStringError(inconvertibleErrorCode(),
                             "Can't trace non-live processes");

  if (auto create_callback =
          PluginManager::GetTraceCreateCallbackForLiveProcess(name))
    return create_callback(process);

  return createInvalidPlugInError(name);
}

Expected<StringRef> Trace::FindPluginSchema(StringRef name) {
  StringRef schema = PluginManager::GetTraceSchema(name);
  if (!schema.empty())
    return schema;

  return createInvalidPlugInError(name);
}

Error Trace::Start(const llvm::json::Value &request) {
  if (!m_live_process)
    return createStringError(inconvertibleErrorCode(),
                             "Tracing requires a live process.");
  return m_live_process->TraceStart(request);
}

Error Trace::Stop() {
  if (!m_live_process)
    return createStringError(inconvertibleErrorCode(),
                             "Tracing requires a live process.");
  return m_live_process->TraceStop(TraceStopRequest(GetPluginName()));
}

Error Trace::Stop(llvm::ArrayRef<lldb::tid_t> tids) {
  if (!m_live_process)
    return createStringError(inconvertibleErrorCode(),
                             "Tracing requires a live process.");
  return m_live_process->TraceStop(TraceStopRequest(GetPluginName(), tids));
}

Expected<std::string> Trace::GetLiveProcessState() {
  if (!m_live_process)
    return createStringError(inconvertibleErrorCode(),
                             "Tracing requires a live process.");
  return m_live_process->TraceGetState(GetPluginName());
}

Optional<uint64_t> Trace::GetLiveThreadBinaryDataSize(lldb::tid_t tid,
                                                      llvm::StringRef kind) {
  auto it = m_live_thread_data.find(tid);
  if (it == m_live_thread_data.end())
    return None;
  std::unordered_map<std::string, uint64_t> &single_thread_data = it->second;
  auto single_thread_data_it = single_thread_data.find(kind.str());
  if (single_thread_data_it == single_thread_data.end())
    return None;
  return single_thread_data_it->second;
}

Optional<uint64_t> Trace::GetLiveCoreBinaryDataSize(lldb::core_id_t core_id,
                                                    llvm::StringRef kind) {
  auto it = m_live_core_data.find(core_id);
  if (it == m_live_core_data.end())
    return None;
  std::unordered_map<std::string, uint64_t> &single_core_data = it->second;
  auto single_thread_data_it = single_core_data.find(kind.str());
  if (single_thread_data_it == single_core_data.end())
    return None;
  return single_thread_data_it->second;
}

Optional<uint64_t> Trace::GetLiveProcessBinaryDataSize(llvm::StringRef kind) {
  auto data_it = m_live_process_data.find(kind.str());
  if (data_it == m_live_process_data.end())
    return None;
  return data_it->second;
}

Expected<std::vector<uint8_t>>
Trace::GetLiveThreadBinaryData(lldb::tid_t tid, llvm::StringRef kind) {
  if (!m_live_process)
    return createStringError(inconvertibleErrorCode(),
                             "Tracing requires a live process.");
  llvm::Optional<uint64_t> size = GetLiveThreadBinaryDataSize(tid, kind);
  if (!size)
    return createStringError(
        inconvertibleErrorCode(),
        "Tracing data \"%s\" is not available for thread %" PRIu64 ".",
        kind.data(), tid);

  TraceGetBinaryDataRequest request{GetPluginName().str(), kind.str(),   tid,
                                    /*core_id=*/None,      /*offset=*/0, *size};
  return m_live_process->TraceGetBinaryData(request);
}

Expected<std::vector<uint8_t>>
Trace::GetLiveCoreBinaryData(lldb::core_id_t core_id, llvm::StringRef kind) {
  if (!m_live_process)
    return createStringError(inconvertibleErrorCode(),
                             "Tracing requires a live process.");
  llvm::Optional<uint64_t> size = GetLiveCoreBinaryDataSize(core_id, kind);
  if (!size)
    return createStringError(
        inconvertibleErrorCode(),
        "Tracing data \"%s\" is not available for core_id %" PRIu64 ".",
        kind.data(), core_id);

  TraceGetBinaryDataRequest request{GetPluginName().str(), kind.str(),
                                    /*tid=*/None,          core_id,
                                    /*offset=*/0,          *size};
  return m_live_process->TraceGetBinaryData(request);
}

Expected<std::vector<uint8_t>>
Trace::GetLiveProcessBinaryData(llvm::StringRef kind) {
  if (!m_live_process)
    return createStringError(inconvertibleErrorCode(),
                             "Tracing requires a live process.");
  llvm::Optional<uint64_t> size = GetLiveProcessBinaryDataSize(kind);
  if (!size)
    return createStringError(
        inconvertibleErrorCode(),
        "Tracing data \"%s\" is not available for the process.", kind.data());

  TraceGetBinaryDataRequest request{GetPluginName().str(), kind.str(),
                                    /*tid=*/None,          /*core_id*/ None,
                                    /*offset=*/0,          *size};
  return m_live_process->TraceGetBinaryData(request);
}

const char *Trace::RefreshLiveProcessState() {
  if (!m_live_process)
    return nullptr;

  uint32_t new_stop_id = m_live_process->GetStopID();
  if (new_stop_id == m_stop_id)
    return nullptr;

  Log *log = GetLog(LLDBLog::Target);
  LLDB_LOG(log, "Trace::RefreshLiveProcessState invoked");

  m_stop_id = new_stop_id;
  m_live_thread_data.clear();
  m_live_refresh_error.reset();
  m_cores.reset();

  auto HandleError = [&](Error &&err) -> const char * {
    m_live_refresh_error = toString(std::move(err));
    return m_live_refresh_error->c_str();
  };

  Expected<std::string> json_string = GetLiveProcessState();
  if (!json_string)
    return HandleError(json_string.takeError());

  Expected<TraceGetStateResponse> live_process_state =
      json::parse<TraceGetStateResponse>(*json_string, "TraceGetStateResponse");
  if (!live_process_state)
    return HandleError(live_process_state.takeError());

  if (live_process_state->warnings) {
    for (std::string &warning : *live_process_state->warnings)
      LLDB_LOG(log, "== Warning when fetching the trace state: {0}", warning);
  }

  for (const TraceThreadState &thread_state :
       live_process_state->traced_threads) {
    for (const TraceBinaryData &item : thread_state.binary_data)
      m_live_thread_data[thread_state.tid][item.kind] = item.size;
  }

  LLDB_LOG(log, "== Found {0} threads being traced",
           live_process_state->traced_threads.size());

  if (live_process_state->cores) {
    m_cores.emplace();
    for (const TraceCoreState &core_state : *live_process_state->cores) {
      m_cores->push_back(core_state.core_id);
      for (const TraceBinaryData &item : core_state.binary_data)
        m_live_core_data[core_state.core_id][item.kind] = item.size;
    }
    LLDB_LOG(log, "== Found {0} cpu cores being traced",
            live_process_state->cores->size());
  }


  for (const TraceBinaryData &item : live_process_state->process_binary_data)
    m_live_process_data[item.kind] = item.size;

  if (Error err = DoRefreshLiveProcessState(std::move(*live_process_state),
                                            *json_string))
    return HandleError(std::move(err));

  return nullptr;
}

Trace::Trace(ArrayRef<ProcessSP> postmortem_processes,
             Optional<std::vector<lldb::core_id_t>> postmortem_cores) {
  for (ProcessSP process_sp : postmortem_processes)
    m_postmortem_processes.push_back(process_sp.get());
  m_cores = postmortem_cores;
}

Process *Trace::GetLiveProcess() { return m_live_process; }

ArrayRef<Process *> Trace::GetPostMortemProcesses() {
  return m_postmortem_processes;
}

std::vector<Process *> Trace::GetAllProcesses() {
  if (Process *proc = GetLiveProcess())
    return {proc};
  return GetPostMortemProcesses();
}

uint32_t Trace::GetStopID() {
  RefreshLiveProcessState();
  return m_stop_id;
}

llvm::Expected<FileSpec>
Trace::GetPostMortemThreadDataFile(lldb::tid_t tid, llvm::StringRef kind) {
  auto NotFoundError = [&]() {
    return createStringError(
        inconvertibleErrorCode(),
        formatv("The thread with tid={0} doesn't have the tracing data {1}",
                tid, kind));
  };

  auto it = m_postmortem_thread_data.find(tid);
  if (it == m_postmortem_thread_data.end())
    return NotFoundError();

  std::unordered_map<std::string, FileSpec> &data_kind_to_file_spec_map =
      it->second;
  auto it2 = data_kind_to_file_spec_map.find(kind.str());
  if (it2 == data_kind_to_file_spec_map.end())
    return NotFoundError();
  return it2->second;
}

llvm::Expected<FileSpec>
Trace::GetPostMortemCoreDataFile(lldb::core_id_t core_id,
                                 llvm::StringRef kind) {
  auto NotFoundError = [&]() {
    return createStringError(
        inconvertibleErrorCode(),
        formatv("The core with id={0} doesn't have the tracing data {1}",
                core_id, kind));
  };

  auto it = m_postmortem_core_data.find(core_id);
  if (it == m_postmortem_core_data.end())
    return NotFoundError();

  std::unordered_map<std::string, FileSpec> &data_kind_to_file_spec_map =
      it->second;
  auto it2 = data_kind_to_file_spec_map.find(kind.str());
  if (it2 == data_kind_to_file_spec_map.end())
    return NotFoundError();
  return it2->second;
}

void Trace::SetPostMortemThreadDataFile(lldb::tid_t tid, llvm::StringRef kind,
                                        FileSpec file_spec) {
  m_postmortem_thread_data[tid][kind.str()] = file_spec;
}

void Trace::SetPostMortemCoreDataFile(lldb::core_id_t core_id,
                                      llvm::StringRef kind,
                                      FileSpec file_spec) {
  m_postmortem_core_data[core_id][kind.str()] = file_spec;
}

llvm::Error
Trace::OnLiveThreadBinaryDataRead(lldb::tid_t tid, llvm::StringRef kind,
                                  OnBinaryDataReadCallback callback) {
  Expected<std::vector<uint8_t>> data = GetLiveThreadBinaryData(tid, kind);
  if (!data)
    return data.takeError();
  return callback(*data);
}

llvm::Error Trace::OnLiveCoreBinaryDataRead(lldb::core_id_t core_id,
                                            llvm::StringRef kind,
                                            OnBinaryDataReadCallback callback) {
  Expected<std::vector<uint8_t>> data = GetLiveCoreBinaryData(core_id, kind);
  if (!data)
    return data.takeError();
  return callback(*data);
}

llvm::Error Trace::OnDataFileRead(FileSpec file,
                                  OnBinaryDataReadCallback callback) {
  ErrorOr<std::unique_ptr<MemoryBuffer>> trace_or_error =
      MemoryBuffer::getFile(file.GetPath());
  if (std::error_code err = trace_or_error.getError())
    return errorCodeToError(err);

  MemoryBuffer &data = **trace_or_error;
  ArrayRef<uint8_t> array_ref(
      reinterpret_cast<const uint8_t *>(data.getBufferStart()),
      data.getBufferSize());
  return callback(array_ref);
}

llvm::Error
Trace::OnPostMortemThreadBinaryDataRead(lldb::tid_t tid, llvm::StringRef kind,
                                        OnBinaryDataReadCallback callback) {
  Expected<FileSpec> file = GetPostMortemThreadDataFile(tid, kind);
  if (!file)
    return file.takeError();
  return OnDataFileRead(*file, callback);
}

llvm::Error
Trace::OnPostMortemCoreBinaryDataRead(lldb::core_id_t core_id,
                                      llvm::StringRef kind,
                                      OnBinaryDataReadCallback callback) {
  Expected<FileSpec> file = GetPostMortemCoreDataFile(core_id, kind);
  if (!file)
    return file.takeError();
  return OnDataFileRead(*file, callback);
}

llvm::Error Trace::OnThreadBinaryDataRead(lldb::tid_t tid, llvm::StringRef kind,
                                          OnBinaryDataReadCallback callback) {
  RefreshLiveProcessState();
  if (m_live_process)
    return OnLiveThreadBinaryDataRead(tid, kind, callback);
  else
    return OnPostMortemThreadBinaryDataRead(tid, kind, callback);
}

llvm::Error Trace::OnCoreBinaryDataRead(lldb::core_id_t core_id,
                                        llvm::StringRef kind,
                                        OnBinaryDataReadCallback callback) {
  RefreshLiveProcessState();
  if (m_live_process)
    return OnLiveCoreBinaryDataRead(core_id, kind, callback);
  else
    return OnPostMortemCoreBinaryDataRead(core_id, kind, callback);
}

ArrayRef<lldb::core_id_t> Trace::GetTracedCores() {
  RefreshLiveProcessState();
  if (m_cores)
    return *m_cores;
  return {};
}
