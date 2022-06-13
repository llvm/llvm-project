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

/// Helper functions for fetching data in maps and returning Optionals or
/// pointers instead of iterators for simplicity. It's worth mentioning that the
/// Optionals version can't return the inner data by reference because of
/// limitations in move constructors.
/// \{
template <typename K, typename V>
static Optional<V> Lookup(DenseMap<K, V> &map, K k) {
  auto it = map.find(k);
  if (it == map.end())
    return None;
  return it->second;
}

template <typename K, typename V>
static V *LookupAsPtr(DenseMap<K, V> &map, K k) {
  auto it = map.find(k);
  if (it == map.end())
    return nullptr;
  return &it->second;
}

template <typename K1, typename K2, typename V>
static Optional<V> Lookup2(DenseMap<K1, DenseMap<K2, V>> &map, K1 k1, K2 k2) {
  auto it = map.find(k1);
  if (it == map.end())
    return None;
  return Lookup(it->second, k2);
}

template <typename K1, typename K2, typename V>
static V *Lookup2AsPtr(DenseMap<K1, DenseMap<K2, V>> &map, K1 k1, K2 k2) {
  auto it = map.find(k1);
  if (it == map.end())
    return nullptr;
  return LookupAsPtr(it->second, k2);
}
/// \}

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
    return createStringError(
        inconvertibleErrorCode(),
        "Attempted to start tracing without a live process.");
  return m_live_process->TraceStart(request);
}

Error Trace::Stop() {
  if (!m_live_process)
    return createStringError(
        inconvertibleErrorCode(),
        "Attempted to stop tracing without a live process.");
  return m_live_process->TraceStop(TraceStopRequest(GetPluginName()));
}

Error Trace::Stop(llvm::ArrayRef<lldb::tid_t> tids) {
  if (!m_live_process)
    return createStringError(
        inconvertibleErrorCode(),
        "Attempted to stop tracing without a live process.");
  return m_live_process->TraceStop(TraceStopRequest(GetPluginName(), tids));
}

Expected<std::string> Trace::GetLiveProcessState() {
  if (!m_live_process)
    return createStringError(
        inconvertibleErrorCode(),
        "Attempted to fetch live trace information without a live process.");
  return m_live_process->TraceGetState(GetPluginName());
}

Optional<uint64_t> Trace::GetLiveThreadBinaryDataSize(lldb::tid_t tid,
                                                      llvm::StringRef kind) {
  Storage &storage = GetUpdatedStorage();
  return Lookup2(storage.live_thread_data, tid, ConstString(kind));
}

Optional<uint64_t> Trace::GetLiveCoreBinaryDataSize(lldb::core_id_t core_id,
                                                    llvm::StringRef kind) {
  Storage &storage = GetUpdatedStorage();
  return Lookup2(storage.live_core_data_sizes, core_id, ConstString(kind));
}

Optional<uint64_t> Trace::GetLiveProcessBinaryDataSize(llvm::StringRef kind) {
  Storage &storage = GetUpdatedStorage();
  return Lookup(storage.live_process_data, ConstString(kind));
}

Expected<std::vector<uint8_t>>
Trace::GetLiveTraceBinaryData(const TraceGetBinaryDataRequest &request,
                              uint64_t expected_size) {
  if (!m_live_process)
    return createStringError(
        inconvertibleErrorCode(),
        formatv("Attempted to fetch live trace data without a live process. "
                "Data kind = {0}, tid = {1}, core id = {2}.",
                request.kind, request.tid, request.core_id));

  Expected<std::vector<uint8_t>> data =
      m_live_process->TraceGetBinaryData(request);

  if (!data)
    return data.takeError();

  if (data->size() != expected_size)
    return createStringError(
        inconvertibleErrorCode(),
        formatv("Got incomplete live trace data. Data kind = {0}, expected "
                "size = {1}, actual size = {2}, tid = {3}, core id = {4}",
                request.kind, expected_size, data->size(), request.tid,
                request.core_id));

  return data;
}

Expected<std::vector<uint8_t>>
Trace::GetLiveThreadBinaryData(lldb::tid_t tid, llvm::StringRef kind) {
  llvm::Optional<uint64_t> size = GetLiveThreadBinaryDataSize(tid, kind);
  if (!size)
    return createStringError(
        inconvertibleErrorCode(),
        "Tracing data \"%s\" is not available for thread %" PRIu64 ".",
        kind.data(), tid);

  TraceGetBinaryDataRequest request{GetPluginName().str(), kind.str(), tid,
                                    /*core_id=*/None};
  return GetLiveTraceBinaryData(request, *size);
}

Expected<std::vector<uint8_t>>
Trace::GetLiveCoreBinaryData(lldb::core_id_t core_id, llvm::StringRef kind) {
  if (!m_live_process)
    return createStringError(
        inconvertibleErrorCode(),
        "Attempted to fetch live cpu data without a live process.");
  llvm::Optional<uint64_t> size = GetLiveCoreBinaryDataSize(core_id, kind);
  if (!size)
    return createStringError(
        inconvertibleErrorCode(),
        "Tracing data \"%s\" is not available for core_id %" PRIu64 ".",
        kind.data(), core_id);

  TraceGetBinaryDataRequest request{GetPluginName().str(), kind.str(),
                                    /*tid=*/None, core_id};
  return m_live_process->TraceGetBinaryData(request);
}

Expected<std::vector<uint8_t>>
Trace::GetLiveProcessBinaryData(llvm::StringRef kind) {
  llvm::Optional<uint64_t> size = GetLiveProcessBinaryDataSize(kind);
  if (!size)
    return createStringError(
        inconvertibleErrorCode(),
        "Tracing data \"%s\" is not available for the process.", kind.data());

  TraceGetBinaryDataRequest request{GetPluginName().str(), kind.str(),
                                    /*tid=*/None, /*core_id*/ None};
  return GetLiveTraceBinaryData(request, *size);
}

Trace::Storage &Trace::GetUpdatedStorage() {
  RefreshLiveProcessState();
  return m_storage;
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
  m_storage = Trace::Storage();

  auto do_refresh = [&]() -> Error {
    Expected<std::string> json_string = GetLiveProcessState();
    if (!json_string)
      return json_string.takeError();

    Expected<TraceGetStateResponse> live_process_state =
        json::parse<TraceGetStateResponse>(*json_string,
                                           "TraceGetStateResponse");
    if (!live_process_state)
      return live_process_state.takeError();

    if (live_process_state->warnings) {
      for (std::string &warning : *live_process_state->warnings)
        LLDB_LOG(log, "== Warning when fetching the trace state: {0}", warning);
    }

    for (const TraceThreadState &thread_state :
         live_process_state->traced_threads) {
      for (const TraceBinaryData &item : thread_state.binary_data)
        m_storage.live_thread_data[thread_state.tid].insert(
            {ConstString(item.kind), item.size});
    }

    LLDB_LOG(log, "== Found {0} threads being traced",
             live_process_state->traced_threads.size());

    if (live_process_state->cores) {
      m_storage.cores.emplace();
      for (const TraceCoreState &core_state : *live_process_state->cores) {
        m_storage.cores->push_back(core_state.core_id);
        for (const TraceBinaryData &item : core_state.binary_data)
          m_storage.live_core_data_sizes[core_state.core_id].insert(
              {ConstString(item.kind), item.size});
      }
      LLDB_LOG(log, "== Found {0} cpu cores being traced",
               live_process_state->cores->size());
    }

    for (const TraceBinaryData &item : live_process_state->process_binary_data)
      m_storage.live_process_data.insert({ConstString(item.kind), item.size});

    return DoRefreshLiveProcessState(std::move(*live_process_state),
                                     *json_string);
  };

  if (Error err = do_refresh()) {
    m_storage.live_refresh_error = toString(std::move(err));
    return m_storage.live_refresh_error->c_str();
  }

  return nullptr;
}

Trace::Trace(ArrayRef<ProcessSP> postmortem_processes,
             Optional<std::vector<lldb::core_id_t>> postmortem_cores) {
  for (ProcessSP process_sp : postmortem_processes)
    m_storage.postmortem_processes.push_back(process_sp.get());
  m_storage.cores = postmortem_cores;
}

Process *Trace::GetLiveProcess() { return m_live_process; }

ArrayRef<Process *> Trace::GetPostMortemProcesses() {
  return m_storage.postmortem_processes;
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
  Storage &storage = GetUpdatedStorage();
  if (Optional<FileSpec> file =
          Lookup2(storage.postmortem_thread_data, tid, ConstString(kind)))
    return *file;
  else
    return createStringError(
        inconvertibleErrorCode(),
        formatv("The thread with tid={0} doesn't have the tracing data {1}",
                tid, kind));
}

llvm::Expected<FileSpec>
Trace::GetPostMortemCoreDataFile(lldb::core_id_t core_id,
                                 llvm::StringRef kind) {
  Storage &storage = GetUpdatedStorage();
  if (Optional<FileSpec> file =
          Lookup2(storage.postmortem_core_data, core_id, ConstString(kind)))
    return *file;
  else
    return createStringError(
        inconvertibleErrorCode(),
        formatv("The core with id={0} doesn't have the tracing data {1}",
                core_id, kind));
}

void Trace::SetPostMortemThreadDataFile(lldb::tid_t tid, llvm::StringRef kind,
                                        FileSpec file_spec) {
  Storage &storage = GetUpdatedStorage();
  storage.postmortem_thread_data[tid].insert({ConstString(kind), file_spec});
}

void Trace::SetPostMortemCoreDataFile(lldb::core_id_t core_id,
                                      llvm::StringRef kind,
                                      FileSpec file_spec) {
  Storage &storage = GetUpdatedStorage();
  storage.postmortem_core_data[core_id].insert({ConstString(kind), file_spec});
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
  Storage &storage = GetUpdatedStorage();
  if (std::vector<uint8_t> *core_data =
          Lookup2AsPtr(storage.live_core_data, core_id, ConstString(kind)))
    return callback(*core_data);

  Expected<std::vector<uint8_t>> data = GetLiveCoreBinaryData(core_id, kind);
  if (!data)
    return data.takeError();
  auto it = storage.live_core_data[core_id].insert(
      {ConstString(kind), std::move(*data)});
  return callback(it.first->second);
}

llvm::Error Trace::OnDataFileRead(FileSpec file,
                                  OnBinaryDataReadCallback callback) {
  ErrorOr<std::unique_ptr<MemoryBuffer>> trace_or_error =
      MemoryBuffer::getFile(file.GetPath());
  if (std::error_code err = trace_or_error.getError())
    return createStringError(
        inconvertibleErrorCode(), "Failed fetching trace-related file %s. %s",
        file.GetCString(), toString(errorCodeToError(err)).c_str());

  MemoryBuffer &data = **trace_or_error;
  ArrayRef<uint8_t> array_ref(
      reinterpret_cast<const uint8_t *>(data.getBufferStart()),
      data.getBufferSize());
  return callback(array_ref);
}

llvm::Error
Trace::OnPostMortemThreadBinaryDataRead(lldb::tid_t tid, llvm::StringRef kind,
                                        OnBinaryDataReadCallback callback) {
  if (Expected<FileSpec> file = GetPostMortemThreadDataFile(tid, kind))
    return OnDataFileRead(*file, callback);
  else
    return file.takeError();
}

llvm::Error
Trace::OnPostMortemCoreBinaryDataRead(lldb::core_id_t core_id,
                                      llvm::StringRef kind,
                                      OnBinaryDataReadCallback callback) {
  if (Expected<FileSpec> file = GetPostMortemCoreDataFile(core_id, kind))
    return OnDataFileRead(*file, callback);
  else
    return file.takeError();
}

llvm::Error Trace::OnThreadBinaryDataRead(lldb::tid_t tid, llvm::StringRef kind,
                                          OnBinaryDataReadCallback callback) {
  if (m_live_process)
    return OnLiveThreadBinaryDataRead(tid, kind, callback);
  else
    return OnPostMortemThreadBinaryDataRead(tid, kind, callback);
}

llvm::Error
Trace::OnAllCoresBinaryDataRead(llvm::StringRef kind,
                                OnCoresBinaryDataReadCallback callback) {
  DenseMap<core_id_t, ArrayRef<uint8_t>> buffers;
  Storage &storage = GetUpdatedStorage();
  if (!storage.cores)
    return Error::success();

  std::function<Error(std::vector<core_id_t>::iterator)> process_core =
      [&](std::vector<core_id_t>::iterator core_id) -> Error {
    if (core_id == storage.cores->end())
      return callback(buffers);

    return OnCoreBinaryDataRead(*core_id, kind,
                                [&](ArrayRef<uint8_t> data) -> Error {
                                  buffers.try_emplace(*core_id, data);
                                  auto next_id = core_id;
                                  next_id++;
                                  return process_core(next_id);
                                });
  };
  return process_core(storage.cores->begin());
}

llvm::Error Trace::OnCoreBinaryDataRead(lldb::core_id_t core_id,
                                        llvm::StringRef kind,
                                        OnBinaryDataReadCallback callback) {
  if (m_live_process)
    return OnLiveCoreBinaryDataRead(core_id, kind, callback);
  else
    return OnPostMortemCoreBinaryDataRead(core_id, kind, callback);
}

ArrayRef<lldb::core_id_t> Trace::GetTracedCores() {
  Storage &storage = GetUpdatedStorage();
  if (storage.cores)
    return *storage.cores;
  return {};
}

std::vector<Process *> Trace::GetTracedProcesses() {
  std::vector<Process *> processes;
  Storage &storage = GetUpdatedStorage();

  for (Process *proc : storage.postmortem_processes)
    processes.push_back(proc);

  if (m_live_process)
    processes.push_back(m_live_process);
  return processes;
}
