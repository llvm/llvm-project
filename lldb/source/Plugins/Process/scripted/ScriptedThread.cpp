//===-- ScriptedThread.cpp ------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ScriptedThread.h"

#include "Plugins/Process/Utility/RegisterContextThreadMemory.h"
#include "lldb/Target/OperatingSystem.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/RegisterContext.h"
#include "lldb/Target/StopInfo.h"
#include "lldb/Target/Unwind.h"
#include "lldb/Utility/DataBufferHeap.h"
#include "lldb/Utility/Log.h"
#include "lldb/Utility/Logging.h"

#include <memory>

using namespace lldb;
using namespace lldb_private;

void ScriptedThread::CheckInterpreterAndScriptObject() const {
  lldbassert(m_script_object_sp && "Invalid Script Object.");
  lldbassert(GetInterface() && "Invalid Scripted Thread Interface.");
}

ScriptedThread::ScriptedThread(ScriptedProcess &process, Status &error)
    : Thread(process, LLDB_INVALID_THREAD_ID), m_scripted_process(process) {
  if (!process.IsValid()) {
    error.SetErrorString("Invalid scripted process");
    return;
  }

  process.CheckInterpreterAndScriptObject();

  auto scripted_thread_interface = GetInterface();
  if (!scripted_thread_interface) {
    error.SetErrorString("Failed to get scripted thread interface.");
    return;
  }

  llvm::Optional<std::string> class_name =
      process.GetInterface().GetScriptedThreadPluginName();
  if (!class_name || class_name->empty()) {
    error.SetErrorString("Failed to get scripted thread class name.");
    return;
  }

  ExecutionContext exe_ctx(process);

  StructuredData::GenericSP object_sp =
      scripted_thread_interface->CreatePluginObject(
          class_name->c_str(), exe_ctx,
          process.m_scripted_process_info.GetArgsSP());
  if (!object_sp || !object_sp->IsValid()) {
    error.SetErrorString("Failed to create valid script object");
    return;
  }

  m_script_object_sp = object_sp;

  SetID(scripted_thread_interface->GetThreadID());
}

ScriptedThread::~ScriptedThread() { DestroyThread(); }

const char *ScriptedThread::GetName() {
  CheckInterpreterAndScriptObject();
  llvm::Optional<std::string> thread_name = GetInterface()->GetName();
  if (!thread_name)
    return nullptr;
  return ConstString(thread_name->c_str()).AsCString();
}

const char *ScriptedThread::GetQueueName() {
  CheckInterpreterAndScriptObject();
  llvm::Optional<std::string> queue_name = GetInterface()->GetQueue();
  if (!queue_name)
    return nullptr;
  return ConstString(queue_name->c_str()).AsCString();
}

void ScriptedThread::WillResume(StateType resume_state) {}

void ScriptedThread::ClearStackFrames() { Thread::ClearStackFrames(); }

RegisterContextSP ScriptedThread::GetRegisterContext() {
  if (!m_reg_context_sp)
    m_reg_context_sp = CreateRegisterContextForFrame(nullptr);
  return m_reg_context_sp;
}

RegisterContextSP
ScriptedThread::CreateRegisterContextForFrame(StackFrame *frame) {
  const uint32_t concrete_frame_idx =
      frame ? frame->GetConcreteFrameIndex() : 0;

  if (concrete_frame_idx)
    return GetUnwinder().CreateRegisterContextForFrame(frame);

  lldb::RegisterContextSP reg_ctx_sp;
  Status error;

  llvm::Optional<std::string> reg_data = GetInterface()->GetRegisterContext();
  if (!reg_data)
    return GetInterface()->ErrorWithMessage<lldb::RegisterContextSP>(
        LLVM_PRETTY_FUNCTION, "Failed to get scripted thread registers data.",
        error, LIBLLDB_LOG_THREAD);

  DataBufferSP data_sp(
      std::make_shared<DataBufferHeap>(reg_data->c_str(), reg_data->size()));

  if (!data_sp->GetByteSize())
    return GetInterface()->ErrorWithMessage<lldb::RegisterContextSP>(
        LLVM_PRETTY_FUNCTION, "Failed to copy raw registers data.", error,
        LIBLLDB_LOG_THREAD);

  std::shared_ptr<RegisterContextMemory> reg_ctx_memory =
      std::make_shared<RegisterContextMemory>(
          *this, 0, *GetDynamicRegisterInfo(), LLDB_INVALID_ADDRESS);
  if (!reg_ctx_memory)
    return GetInterface()->ErrorWithMessage<lldb::RegisterContextSP>(
        LLVM_PRETTY_FUNCTION, "Failed to create a register context.", error,
        LIBLLDB_LOG_THREAD);

  reg_ctx_memory->SetAllRegisterData(data_sp);
  m_reg_context_sp = reg_ctx_memory;

  return m_reg_context_sp;
}

bool ScriptedThread::CalculateStopInfo() {
  StructuredData::DictionarySP dict_sp = GetInterface()->GetStopReason();

  Status error;
  lldb::StopInfoSP stop_info_sp;
  lldb::StopReason stop_reason_type;

  if (!dict_sp->GetValueForKeyAsInteger("type", stop_reason_type))
    return GetInterface()->ErrorWithMessage<bool>(
        LLVM_PRETTY_FUNCTION,
        "Couldn't find value for key 'type' in stop reason dictionary.", error,
        LIBLLDB_LOG_THREAD);

  StructuredData::Dictionary *data_dict;
  if (!dict_sp->GetValueForKeyAsDictionary("data", data_dict))
    return GetInterface()->ErrorWithMessage<bool>(
        LLVM_PRETTY_FUNCTION,
        "Couldn't find value for key 'type' in stop reason dictionary.", error,
        LIBLLDB_LOG_THREAD);

  switch (stop_reason_type) {
  case lldb::eStopReasonNone:
    break;
  case lldb::eStopReasonBreakpoint: {
    lldb::break_id_t break_id;
    data_dict->GetValueForKeyAsInteger("break_id", break_id,
                                       LLDB_INVALID_BREAK_ID);
    stop_info_sp =
        StopInfo::CreateStopReasonWithBreakpointSiteID(*this, break_id);
  } break;
  case lldb::eStopReasonSignal: {
    int signal;
    llvm::StringRef description;
    data_dict->GetValueForKeyAsInteger("signal", signal,
                                       LLDB_INVALID_SIGNAL_NUMBER);
    data_dict->GetValueForKeyAsString("desc", description);
    stop_info_sp =
        StopInfo::CreateStopReasonWithSignal(*this, signal, description.data());
  } break;
  default:
    return GetInterface()->ErrorWithMessage<bool>(
        LLVM_PRETTY_FUNCTION,
        llvm::Twine("Unsupported stop reason type (" +
                    llvm::Twine(stop_reason_type) + llvm::Twine(")."))
            .str(),
        error, LIBLLDB_LOG_THREAD);
  }

  SetStopInfo(stop_info_sp);
  return true;
}

void ScriptedThread::RefreshStateAfterStop() {
  GetRegisterContext()->InvalidateIfNeeded(/*force=*/false);
}

lldb::ScriptedThreadInterfaceSP ScriptedThread::GetInterface() const {
  return m_scripted_process.GetInterface().GetScriptedThreadInterface();
}

std::shared_ptr<DynamicRegisterInfo> ScriptedThread::GetDynamicRegisterInfo() {
  CheckInterpreterAndScriptObject();

  if (!m_register_info_sp) {
    StructuredData::DictionarySP reg_info = GetInterface()->GetRegisterInfo();

    Status error;
    if (!reg_info)
      return GetInterface()
          ->ErrorWithMessage<std::shared_ptr<DynamicRegisterInfo>>(
              LLVM_PRETTY_FUNCTION,
              "Failed to get scripted thread registers info.", error,
              LIBLLDB_LOG_THREAD);

    m_register_info_sp = std::make_shared<DynamicRegisterInfo>(
        *reg_info, m_scripted_process.GetTarget().GetArchitecture());
  }

  return m_register_info_sp;
}
