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
#include "lldb/Utility/Stream.h"

using namespace lldb;
using namespace lldb_private;
using namespace llvm;

// Helper structs used to extract the type of a trace session json without
// having to parse the entire object.

struct JSONSimplePluginSettings {
  std::string type;
};

struct JSONSimpleTraceSession {
  JSONSimplePluginSettings trace;
};

namespace llvm {
namespace json {

bool fromJSON(const Value &value, JSONSimplePluginSettings &plugin_settings,
              Path path) {
  json::ObjectMapper o(value, path);
  return o && o.map("type", plugin_settings.type);
}

bool fromJSON(const Value &value, JSONSimpleTraceSession &session, Path path) {
  json::ObjectMapper o(value, path);
  return o && o.map("trace", session.trace);
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
          PluginManager::GetTraceCreateCallback(json_session.trace.type))
    return create_callback(trace_session_file, session_file_dir, debugger);

  return createInvalidPlugInError(json_session.trace.type);
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

Optional<size_t> Trace::GetLiveThreadBinaryDataSize(lldb::tid_t tid,
                                                    llvm::StringRef kind) {
  auto it = m_live_thread_data.find(tid);
  if (it == m_live_thread_data.end())
    return None;
  std::unordered_map<std::string, size_t> &single_thread_data = it->second;
  auto single_thread_data_it = single_thread_data.find(kind.str());
  if (single_thread_data_it == single_thread_data.end())
    return None;
  return single_thread_data_it->second;
}

Optional<size_t> Trace::GetLiveProcessBinaryDataSize(llvm::StringRef kind) {
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
  llvm::Optional<size_t> size = GetLiveThreadBinaryDataSize(tid, kind);
  if (!size)
    return createStringError(
        inconvertibleErrorCode(),
        "Tracing data \"%s\" is not available for thread %" PRIu64 ".",
        kind.data(), tid);

  TraceGetBinaryDataRequest request{GetPluginName().str(), kind.str(), tid, 0,
                                    *size};
  return m_live_process->TraceGetBinaryData(request);
}

Expected<std::vector<uint8_t>>
Trace::GetLiveProcessBinaryData(llvm::StringRef kind) {
  if (!m_live_process)
    return createStringError(inconvertibleErrorCode(),
                             "Tracing requires a live process.");
  llvm::Optional<size_t> size = GetLiveProcessBinaryDataSize(kind);
  if (!size)
    return createStringError(
        inconvertibleErrorCode(),
        "Tracing data \"%s\" is not available for the process.", kind.data());

  TraceGetBinaryDataRequest request{GetPluginName().str(), kind.str(), None, 0,
                                    *size};
  return m_live_process->TraceGetBinaryData(request);
}

void Trace::RefreshLiveProcessState() {
  if (!m_live_process)
    return;

  uint32_t new_stop_id = m_live_process->GetStopID();
  if (new_stop_id == m_stop_id)
    return;

  m_stop_id = new_stop_id;
  m_live_thread_data.clear();

  Expected<std::string> json_string = GetLiveProcessState();
  if (!json_string) {
    DoRefreshLiveProcessState(json_string.takeError());
    return;
  }
  Expected<TraceGetStateResponse> live_process_state =
      json::parse<TraceGetStateResponse>(*json_string, "TraceGetStateResponse");
  if (!live_process_state) {
    DoRefreshLiveProcessState(live_process_state.takeError());
    return;
  }

  for (const TraceThreadState &thread_state :
       live_process_state->tracedThreads) {
    for (const TraceBinaryData &item : thread_state.binaryData)
      m_live_thread_data[thread_state.tid][item.kind] = item.size;
  }

  for (const TraceBinaryData &item : live_process_state->processBinaryData)
    m_live_process_data[item.kind] = item.size;

  DoRefreshLiveProcessState(std::move(live_process_state));
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

void Trace::SetPostMortemThreadDataFile(lldb::tid_t tid, llvm::StringRef kind,
                                        FileSpec file_spec) {
  m_postmortem_thread_data[tid][kind.str()] = file_spec;
}

llvm::Error
Trace::OnLiveThreadBinaryDataRead(lldb::tid_t tid, llvm::StringRef kind,
                                  OnBinaryDataReadCallback callback) {
  Expected<std::vector<uint8_t>> data = GetLiveThreadBinaryData(tid, kind);
  if (!data)
    return data.takeError();
  return callback(*data);
}

llvm::Error
Trace::OnPostMortemThreadBinaryDataRead(lldb::tid_t tid, llvm::StringRef kind,
                                        OnBinaryDataReadCallback callback) {
  Expected<FileSpec> file = GetPostMortemThreadDataFile(tid, kind);
  if (!file)
    return file.takeError();
  ErrorOr<std::unique_ptr<MemoryBuffer>> trace_or_error =
      MemoryBuffer::getFile(file->GetPath());
  if (std::error_code err = trace_or_error.getError())
    return errorCodeToError(err);

  MemoryBuffer &data = **trace_or_error;
  ArrayRef<uint8_t> array_ref(
      reinterpret_cast<const uint8_t *>(data.getBufferStart()),
      data.getBufferSize());
  return callback(array_ref);
}

llvm::Error Trace::OnThreadBinaryDataRead(lldb::tid_t tid, llvm::StringRef kind,
                                          OnBinaryDataReadCallback callback) {
  if (m_live_process)
    return OnLiveThreadBinaryDataRead(tid, kind, callback);
  else
    return OnPostMortemThreadBinaryDataRead(tid, kind, callback);
}
