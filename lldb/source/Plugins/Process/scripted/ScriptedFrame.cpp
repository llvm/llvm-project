//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ScriptedFrame.h"
#include "Plugins/Process/Utility/RegisterContextMemory.h"

#include "lldb/Core/Address.h"
#include "lldb/Core/Debugger.h"
#include "lldb/Core/Module.h"
#include "lldb/Core/ModuleList.h"
#include "lldb/Host/FileSystem.h"
#include "lldb/Interpreter/Interfaces/ScriptedFrameInterface.h"
#include "lldb/Interpreter/Interfaces/ScriptedThreadInterface.h"
#include "lldb/Interpreter/ScriptInterpreter.h"
#include "lldb/Symbol/CompileUnit.h"
#include "lldb/Symbol/SymbolContext.h"
#include "lldb/Symbol/SymbolFile.h"
#include "lldb/Target/ExecutionContext.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/RegisterContext.h"
#include "lldb/Target/Thread.h"
#include "lldb/Utility/DataBufferHeap.h"
#include "lldb/Utility/LLDBLog.h"
#include "lldb/Utility/Log.h"
#include "lldb/Utility/StructuredData.h"

using namespace lldb;
using namespace lldb_private;

char ScriptedFrame::ID;

void ScriptedFrame::CheckInterpreterAndScriptObject() const {
  lldbassert(m_script_object_sp && "Invalid Script Object.");
  lldbassert(GetInterface() && "Invalid Scripted Frame Interface.");
}

llvm::Expected<std::shared_ptr<ScriptedFrame>>
ScriptedFrame::Create(ThreadSP thread_sp,
                      ScriptedThreadInterfaceSP scripted_thread_interface_sp,
                      StructuredData::DictionarySP args_sp,
                      StructuredData::Generic *script_object) {
  if (!thread_sp || !thread_sp->IsValid())
    return llvm::createStringError("invalid thread");

  ProcessSP process_sp = thread_sp->GetProcess();
  if (!process_sp || !process_sp->IsValid())
    return llvm::createStringError("invalid process");

  ScriptInterpreter *script_interp =
      process_sp->GetTarget().GetDebugger().GetScriptInterpreter();
  if (!script_interp)
    return llvm::createStringError("no script interpreter");

  auto scripted_frame_interface = script_interp->CreateScriptedFrameInterface();
  if (!scripted_frame_interface)
    return llvm::createStringError("failed to create scripted frame interface");

  llvm::StringRef frame_class_name;
  if (!script_object) {
    // If no script object is provided and we have a scripted thread interface,
    // try to get the frame class name from it.
    if (scripted_thread_interface_sp) {
      std::optional<std::string> class_name =
          scripted_thread_interface_sp->GetScriptedFramePluginName();
      if (!class_name || class_name->empty())
        return llvm::createStringError(
            "failed to get scripted frame class name");
      frame_class_name = *class_name;
    } else {
      return llvm::createStringError(
          "no script object provided and no scripted thread interface");
    }
  }

  ExecutionContext exe_ctx(thread_sp);
  auto obj_or_err = scripted_frame_interface->CreatePluginObject(
      frame_class_name, exe_ctx, args_sp, script_object);

  if (!obj_or_err)
    return llvm::createStringError(
        "failed to create script object: %s",
        llvm::toString(obj_or_err.takeError()).c_str());

  StructuredData::GenericSP owned_script_object_sp = *obj_or_err;

  if (!owned_script_object_sp->IsValid())
    return llvm::createStringError("created script object is invalid");

  lldb::user_id_t frame_id = scripted_frame_interface->GetID();

  lldb::addr_t pc = scripted_frame_interface->GetPC();
  SymbolContext sc;
  Address symbol_addr;
  if (pc != LLDB_INVALID_ADDRESS) {
    symbol_addr.SetLoadAddress(pc, &process_sp->GetTarget());
    symbol_addr.CalculateSymbolContext(&sc);
  }

  std::optional<SymbolContext> maybe_sym_ctx =
      scripted_frame_interface->GetSymbolContext();
  if (maybe_sym_ctx)
    sc = *maybe_sym_ctx;

  lldb::RegisterContextSP reg_ctx_sp;
  auto regs_or_err =
      CreateRegisterContext(*scripted_frame_interface, *thread_sp, frame_id);
  if (!regs_or_err)
    LLDB_LOG_ERROR(GetLog(LLDBLog::Thread), regs_or_err.takeError(), "{0}");
  else
    reg_ctx_sp = *regs_or_err;

  return std::make_shared<ScriptedFrame>(thread_sp, scripted_frame_interface,
                                         frame_id, pc, sc, reg_ctx_sp,
                                         owned_script_object_sp);
}

ScriptedFrame::ScriptedFrame(ThreadSP thread_sp,
                             ScriptedFrameInterfaceSP interface_sp,
                             lldb::user_id_t id, lldb::addr_t pc,
                             SymbolContext &sym_ctx,
                             lldb::RegisterContextSP reg_ctx_sp,
                             StructuredData::GenericSP script_object_sp)
    : StackFrame(thread_sp, /*frame_idx=*/id,
                 /*concrete_frame_idx=*/id, /*reg_context_sp=*/reg_ctx_sp,
                 /*cfa=*/0, /*pc=*/pc,
                 /*behaves_like_zeroth_frame=*/!id, /*symbol_ctx=*/&sym_ctx),
      m_scripted_frame_interface_sp(interface_sp),
      m_script_object_sp(script_object_sp) {
  // FIXME: This should be part of the base class constructor.
  m_stack_frame_kind = StackFrame::Kind::Synthetic;
}

ScriptedFrame::~ScriptedFrame() {}

const char *ScriptedFrame::GetFunctionName() {
  CheckInterpreterAndScriptObject();
  std::optional<std::string> function_name = GetInterface()->GetFunctionName();
  if (!function_name)
    return StackFrame::GetFunctionName();
  return ConstString(function_name->c_str()).AsCString();
}

const char *ScriptedFrame::GetDisplayFunctionName() {
  CheckInterpreterAndScriptObject();
  std::optional<std::string> function_name =
      GetInterface()->GetDisplayFunctionName();
  if (!function_name)
    return StackFrame::GetDisplayFunctionName();
  return ConstString(function_name->c_str()).AsCString();
}

bool ScriptedFrame::IsInlined() { return GetInterface()->IsInlined(); }

bool ScriptedFrame::IsArtificial() const {
  return GetInterface()->IsArtificial();
}

bool ScriptedFrame::IsHidden() { return GetInterface()->IsHidden(); }

lldb::ScriptedFrameInterfaceSP ScriptedFrame::GetInterface() const {
  return m_scripted_frame_interface_sp;
}

std::shared_ptr<DynamicRegisterInfo> ScriptedFrame::GetDynamicRegisterInfo() {
  CheckInterpreterAndScriptObject();

  StructuredData::DictionarySP reg_info = GetInterface()->GetRegisterInfo();

  Status error;
  if (!reg_info)
    return ScriptedInterface::ErrorWithMessage<
        std::shared_ptr<DynamicRegisterInfo>>(
        LLVM_PRETTY_FUNCTION, "failed to get scripted frame registers info",
        error, LLDBLog::Thread);

  ThreadSP thread_sp = m_thread_wp.lock();
  if (!thread_sp || !thread_sp->IsValid())
    return ScriptedInterface::ErrorWithMessage<
        std::shared_ptr<DynamicRegisterInfo>>(
        LLVM_PRETTY_FUNCTION,
        "failed to get scripted frame registers info: invalid thread", error,
        LLDBLog::Thread);

  ProcessSP process_sp = thread_sp->GetProcess();
  if (!process_sp || !process_sp->IsValid())
    return ScriptedInterface::ErrorWithMessage<
        std::shared_ptr<DynamicRegisterInfo>>(
        LLVM_PRETTY_FUNCTION,
        "failed to get scripted frame registers info: invalid process", error,
        LLDBLog::Thread);

  return DynamicRegisterInfo::Create(*reg_info,
                                     process_sp->GetTarget().GetArchitecture());
}

llvm::Expected<lldb::RegisterContextSP>
ScriptedFrame::CreateRegisterContext(ScriptedFrameInterface &interface,
                                     Thread &thread, lldb::user_id_t frame_id) {
  StructuredData::DictionarySP reg_info = interface.GetRegisterInfo();

  if (!reg_info)
    return llvm::createStringError(
        "failed to get scripted frame registers info");

  std::shared_ptr<DynamicRegisterInfo> register_info_sp =
      DynamicRegisterInfo::Create(
          *reg_info, thread.GetProcess()->GetTarget().GetArchitecture());

  lldb::RegisterContextSP reg_ctx_sp;

  std::optional<std::string> reg_data = interface.GetRegisterContext();
  if (!reg_data)
    return llvm::createStringError(
        "failed to get scripted frame registers data");

  DataBufferSP data_sp(
      std::make_shared<DataBufferHeap>(reg_data->c_str(), reg_data->size()));

  if (!data_sp->GetByteSize())
    return llvm::createStringError("failed to copy raw registers data");

  std::shared_ptr<RegisterContextMemory> reg_ctx_memory =
      std::make_shared<RegisterContextMemory>(
          thread, frame_id, *register_info_sp, LLDB_INVALID_ADDRESS);

  reg_ctx_memory->SetAllRegisterData(data_sp);
  reg_ctx_sp = reg_ctx_memory;

  return reg_ctx_sp;
}

lldb::RegisterContextSP ScriptedFrame::GetRegisterContext() {
  if (!m_reg_context_sp) {
    Status error;
    if (!m_scripted_frame_interface_sp)
      return ScriptedInterface::ErrorWithMessage<RegisterContextSP>(
          LLVM_PRETTY_FUNCTION,
          "failed to get scripted frame registers context: invalid interface",
          error, LLDBLog::Thread);

    ThreadSP thread_sp = GetThread();
    if (!thread_sp)
      return ScriptedInterface::ErrorWithMessage<RegisterContextSP>(
          LLVM_PRETTY_FUNCTION,
          "failed to get scripted frame registers context: invalid thread",
          error, LLDBLog::Thread);

    auto regs_or_err = CreateRegisterContext(*m_scripted_frame_interface_sp,
                                             *thread_sp, GetFrameIndex());
    if (!regs_or_err) {
      error = Status::FromError(regs_or_err.takeError());
      return ScriptedInterface::ErrorWithMessage<RegisterContextSP>(
          LLVM_PRETTY_FUNCTION,
          "failed to get scripted frame registers context", error,
          LLDBLog::Thread);
    }

    m_reg_context_sp = *regs_or_err;
  }

  return m_reg_context_sp;
}
