//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ScriptedFrame.h"

#include "lldb/Utility/DataBufferHeap.h"

using namespace lldb;
using namespace lldb_private;

void ScriptedFrame::CheckInterpreterAndScriptObject() const {
  lldbassert(m_script_object_sp && "Invalid Script Object.");
  lldbassert(GetInterface() && "Invalid Scripted Frame Interface.");
}

llvm::Expected<std::shared_ptr<ScriptedFrame>>
ScriptedFrame::Create(ScriptedThread &thread,
                      StructuredData::DictionarySP args_sp,
                      StructuredData::Generic *script_object) {
  if (!thread.IsValid())
    return llvm::createStringError("Invalid scripted thread.");

  thread.CheckInterpreterAndScriptObject();

  auto scripted_frame_interface =
      thread.GetInterface()->CreateScriptedFrameInterface();
  if (!scripted_frame_interface)
    return llvm::createStringError("failed to create scripted frame interface");

  llvm::StringRef frame_class_name;
  if (!script_object) {
    std::optional<std::string> class_name =
        thread.GetInterface()->GetScriptedFramePluginName();
    if (!class_name || class_name->empty())
      return llvm::createStringError(
          "failed to get scripted thread class name");
    frame_class_name = *class_name;
  }

  ExecutionContext exe_ctx(thread);
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
    symbol_addr.SetLoadAddress(pc, &thread.GetProcess()->GetTarget());
    symbol_addr.CalculateSymbolContext(&sc);
  }

  std::optional<SymbolContext> maybe_sym_ctx =
      scripted_frame_interface->GetSymbolContext();
  if (maybe_sym_ctx) {
    sc = *maybe_sym_ctx;
  }

  StructuredData::DictionarySP reg_info =
      scripted_frame_interface->GetRegisterInfo();

  if (!reg_info)
    return llvm::createStringError(
        "failed to get scripted thread registers info");

  std::shared_ptr<DynamicRegisterInfo> register_info_sp =
      DynamicRegisterInfo::Create(
          *reg_info, thread.GetProcess()->GetTarget().GetArchitecture());

  lldb::RegisterContextSP reg_ctx_sp;

  std::optional<std::string> reg_data =
      scripted_frame_interface->GetRegisterContext();
  if (reg_data) {
    DataBufferSP data_sp(
        std::make_shared<DataBufferHeap>(reg_data->c_str(), reg_data->size()));

    if (!data_sp->GetByteSize())
      return llvm::createStringError("failed to copy raw registers data");

    std::shared_ptr<RegisterContextMemory> reg_ctx_memory =
        std::make_shared<RegisterContextMemory>(
            thread, frame_id, *register_info_sp, LLDB_INVALID_ADDRESS);
    if (!reg_ctx_memory)
      return llvm::createStringError("failed to create a register context.");

    reg_ctx_memory->SetAllRegisterData(data_sp);
    reg_ctx_sp = reg_ctx_memory;
  }

  return std::make_shared<ScriptedFrame>(
      thread, scripted_frame_interface, frame_id, pc, sc, reg_ctx_sp,
      register_info_sp, owned_script_object_sp);
}

ScriptedFrame::ScriptedFrame(ScriptedThread &thread,
                             ScriptedFrameInterfaceSP interface_sp,
                             lldb::user_id_t id, lldb::addr_t pc,
                             SymbolContext &sym_ctx,
                             lldb::RegisterContextSP reg_ctx_sp,
                             std::shared_ptr<DynamicRegisterInfo> reg_info_sp,
                             StructuredData::GenericSP script_object_sp)
    : StackFrame(thread.shared_from_this(), /*frame_idx=*/id,
                 /*concrete_frame_idx=*/id, /*reg_context_sp=*/reg_ctx_sp,
                 /*cfa=*/0, /*pc=*/pc,
                 /*behaves_like_zeroth_frame=*/!id, /*symbol_ctx=*/&sym_ctx),
      m_scripted_frame_interface_sp(interface_sp),
      m_script_object_sp(script_object_sp), m_register_info_sp(reg_info_sp) {}

ScriptedFrame::~ScriptedFrame() {}

const char *ScriptedFrame::GetFunctionName() {
  CheckInterpreterAndScriptObject();
  std::optional<std::string> function_name = GetInterface()->GetFunctionName();
  if (!function_name)
    return nullptr;
  return ConstString(function_name->c_str()).AsCString();
}

const char *ScriptedFrame::GetDisplayFunctionName() {
  CheckInterpreterAndScriptObject();
  std::optional<std::string> function_name =
      GetInterface()->GetDisplayFunctionName();
  if (!function_name)
    return nullptr;
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

  if (!m_register_info_sp) {
    StructuredData::DictionarySP reg_info = GetInterface()->GetRegisterInfo();

    Status error;
    if (!reg_info)
      return ScriptedInterface::ErrorWithMessage<
          std::shared_ptr<DynamicRegisterInfo>>(
          LLVM_PRETTY_FUNCTION, "Failed to get scripted frame registers info.",
          error, LLDBLog::Thread);

    ThreadSP thread_sp = m_thread_wp.lock();
    if (!thread_sp || !thread_sp->IsValid())
      return ScriptedInterface::ErrorWithMessage<
          std::shared_ptr<DynamicRegisterInfo>>(
          LLVM_PRETTY_FUNCTION,
          "Failed to get scripted frame registers info: invalid thread.", error,
          LLDBLog::Thread);

    ProcessSP process_sp = thread_sp->GetProcess();
    if (!process_sp || !process_sp->IsValid())
      return ScriptedInterface::ErrorWithMessage<
          std::shared_ptr<DynamicRegisterInfo>>(
          LLVM_PRETTY_FUNCTION,
          "Failed to get scripted frame registers info: invalid process.",
          error, LLDBLog::Thread);

    m_register_info_sp = DynamicRegisterInfo::Create(
        *reg_info, process_sp->GetTarget().GetArchitecture());
  }

  return m_register_info_sp;
}
