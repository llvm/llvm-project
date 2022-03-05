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
#include "lldb/Utility/LLDBLog.h"
#include <memory>

using namespace lldb;
using namespace lldb_private;

void ScriptedThread::CheckInterpreterAndScriptObject() const {
  lldbassert(m_script_object_sp && "Invalid Script Object.");
  lldbassert(GetInterface() && "Invalid Scripted Thread Interface.");
}

llvm::Expected<std::shared_ptr<ScriptedThread>>
ScriptedThread::Create(ScriptedProcess &process,
                       StructuredData::Generic *script_object) {
  if (!process.IsValid())
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "Invalid scripted process.");

  process.CheckInterpreterAndScriptObject();

  auto scripted_thread_interface =
      process.GetInterface().CreateScriptedThreadInterface();
  if (!scripted_thread_interface)
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "Failed to create scripted thread interface.");

  llvm::StringRef thread_class_name;
  if (!script_object) {
    llvm::Optional<std::string> class_name =
        process.GetInterface().GetScriptedThreadPluginName();
    if (!class_name || class_name->empty())
      return llvm::createStringError(
          llvm::inconvertibleErrorCode(),
          "Failed to get scripted thread class name.");
    thread_class_name = *class_name;
  }

  ExecutionContext exe_ctx(process);
  StructuredData::GenericSP owned_script_object_sp =
      scripted_thread_interface->CreatePluginObject(
          thread_class_name, exe_ctx,
          process.m_scripted_process_info.GetArgsSP(), script_object);

  if (!owned_script_object_sp)
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "Failed to create script object.");
  if (!owned_script_object_sp->IsValid())
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "Created script object is invalid.");

  lldb::tid_t tid = scripted_thread_interface->GetThreadID();

  return std::make_shared<ScriptedThread>(process, scripted_thread_interface,
                                          tid, owned_script_object_sp);
}

ScriptedThread::ScriptedThread(ScriptedProcess &process,
                               ScriptedThreadInterfaceSP interface_sp,
                               lldb::tid_t tid,
                               StructuredData::GenericSP script_object_sp)
    : Thread(process, tid), m_scripted_process(process),
      m_scripted_thread_interface_sp(interface_sp),
      m_script_object_sp(script_object_sp) {}

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
    return ScriptedInterface::ErrorWithMessage<lldb::RegisterContextSP>(
        LLVM_PRETTY_FUNCTION, "Failed to get scripted thread registers data.",
        error, LLDBLog::Thread);

  DataBufferSP data_sp(
      std::make_shared<DataBufferHeap>(reg_data->c_str(), reg_data->size()));

  if (!data_sp->GetByteSize())
    return ScriptedInterface::ErrorWithMessage<lldb::RegisterContextSP>(
        LLVM_PRETTY_FUNCTION, "Failed to copy raw registers data.", error,
        LLDBLog::Thread);

  std::shared_ptr<RegisterContextMemory> reg_ctx_memory =
      std::make_shared<RegisterContextMemory>(
          *this, 0, *GetDynamicRegisterInfo(), LLDB_INVALID_ADDRESS);
  if (!reg_ctx_memory)
    return ScriptedInterface::ErrorWithMessage<lldb::RegisterContextSP>(
        LLVM_PRETTY_FUNCTION, "Failed to create a register context.", error,
        LLDBLog::Thread);

  reg_ctx_memory->SetAllRegisterData(data_sp);
  m_reg_context_sp = reg_ctx_memory;

  return m_reg_context_sp;
}

bool ScriptedThread::LoadArtificialStackFrames() {
  StructuredData::ArraySP arr_sp = GetInterface()->GetStackFrames();

  Status error;
  if (!arr_sp)
    return ScriptedInterface::ErrorWithMessage<bool>(
        LLVM_PRETTY_FUNCTION, "Failed to get scripted thread stackframes.",
        error, LLDBLog::Thread);

  size_t arr_size = arr_sp->GetSize();
  if (arr_size > std::numeric_limits<uint32_t>::max())
    return ScriptedInterface::ErrorWithMessage<bool>(
        LLVM_PRETTY_FUNCTION,
        llvm::Twine(
            "StackFrame array size (" + llvm::Twine(arr_size) +
            llvm::Twine(
                ") is greater than maximum autorized for a StackFrameList."))
            .str(),
        error, LLDBLog::Thread);

  StackFrameListSP frames = GetStackFrameList();

  for (size_t idx = 0; idx < arr_size; idx++) {

    StructuredData::Dictionary *dict;

    if (!arr_sp->GetItemAtIndexAsDictionary(idx, dict) || !dict)
      return ScriptedInterface::ErrorWithMessage<bool>(
          LLVM_PRETTY_FUNCTION,
          llvm::Twine(
              "Couldn't get artificial stackframe dictionary at index (" +
              llvm::Twine(idx) + llvm::Twine(") from stackframe array."))
              .str(),
          error, LLDBLog::Thread);

    lldb::addr_t pc;
    if (!dict->GetValueForKeyAsInteger("pc", pc))
      return ScriptedInterface::ErrorWithMessage<bool>(
          LLVM_PRETTY_FUNCTION,
          "Couldn't find value for key 'pc' in stackframe dictionary.", error,
          LLDBLog::Thread);

    Address symbol_addr;
    symbol_addr.SetLoadAddress(pc, &this->GetProcess()->GetTarget());

    lldb::addr_t cfa = LLDB_INVALID_ADDRESS;
    bool cfa_is_valid = false;
    const bool behaves_like_zeroth_frame = false;
    SymbolContext sc;
    symbol_addr.CalculateSymbolContext(&sc);

    StackFrameSP synth_frame_sp = std::make_shared<StackFrame>(
        this->shared_from_this(), idx, idx, cfa, cfa_is_valid, pc,
        StackFrame::Kind::Artificial, behaves_like_zeroth_frame, &sc);

    if (!frames->SetFrameAtIndex(static_cast<uint32_t>(idx), synth_frame_sp))
      return ScriptedInterface::ErrorWithMessage<bool>(
          LLVM_PRETTY_FUNCTION,
          llvm::Twine("Couldn't add frame (" + llvm::Twine(idx) +
                      llvm::Twine(") to ScriptedThread StackFrameList."))
              .str(),
          error, LLDBLog::Thread);
  }

  return true;
}

bool ScriptedThread::CalculateStopInfo() {
  StructuredData::DictionarySP dict_sp = GetInterface()->GetStopReason();

  Status error;
  if (!dict_sp)
    return ScriptedInterface::ErrorWithMessage<bool>(
        LLVM_PRETTY_FUNCTION, "Failed to get scripted thread stop info.", error,
        LLDBLog::Thread);

  lldb::StopInfoSP stop_info_sp;
  lldb::StopReason stop_reason_type;

  if (!dict_sp->GetValueForKeyAsInteger("type", stop_reason_type))
    return ScriptedInterface::ErrorWithMessage<bool>(
        LLVM_PRETTY_FUNCTION,
        "Couldn't find value for key 'type' in stop reason dictionary.", error,
        LLDBLog::Thread);

  StructuredData::Dictionary *data_dict;
  if (!dict_sp->GetValueForKeyAsDictionary("data", data_dict))
    return ScriptedInterface::ErrorWithMessage<bool>(
        LLVM_PRETTY_FUNCTION,
        "Couldn't find value for key 'data' in stop reason dictionary.", error,
        LLDBLog::Thread);

  switch (stop_reason_type) {
  case lldb::eStopReasonNone:
    return true;
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
  case lldb::eStopReasonException: {
    llvm::StringRef description;
    data_dict->GetValueForKeyAsString("desc", description);

    stop_info_sp =
        StopInfo::CreateStopReasonWithException(*this, description.data());
  } break;
  default:
    return ScriptedInterface::ErrorWithMessage<bool>(
        LLVM_PRETTY_FUNCTION,
        llvm::Twine("Unsupported stop reason type (" +
                    llvm::Twine(stop_reason_type) + llvm::Twine(")."))
            .str(),
        error, LLDBLog::Thread);
  }

  if (!stop_info_sp)
    return false;

  SetStopInfo(stop_info_sp);
  return true;
}

void ScriptedThread::RefreshStateAfterStop() {
  GetRegisterContext()->InvalidateIfNeeded(/*force=*/false);
  LoadArtificialStackFrames();
}

lldb::ScriptedThreadInterfaceSP ScriptedThread::GetInterface() const {
  return m_scripted_thread_interface_sp;
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
              LLDBLog::Thread);

    m_register_info_sp = std::make_shared<DynamicRegisterInfo>(
        *reg_info, m_scripted_process.GetTarget().GetArchitecture());
  }

  return m_register_info_sp;
}
