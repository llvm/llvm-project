//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ScriptedFrameProvider.h"
#include "Plugins/Process/scripted/ScriptedFrame.h"
#include "lldb/Core/Debugger.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/Interpreter/Interfaces/ScriptedFrameProviderInterface.h"
#include "lldb/Interpreter/ScriptInterpreter.h"
#include "lldb/Target/BorrowedStackFrame.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/StackFrame.h"
#include "lldb/Target/Thread.h"
#include "lldb/Utility/ScriptedMetadata.h"
#include "lldb/Utility/Status.h"
#include "llvm/Support/Error.h"
#include <cstdint>

using namespace lldb;
using namespace lldb_private;

void ScriptedFrameProvider::Initialize() {
  PluginManager::RegisterPlugin(GetPluginNameStatic(),
                                "Provides synthetic frames via scripting",
                                nullptr, ScriptedFrameProvider::CreateInstance);
}

void ScriptedFrameProvider::Terminate() {
  PluginManager::UnregisterPlugin(ScriptedFrameProvider::CreateInstance);
}

llvm::Expected<lldb::SyntheticFrameProviderSP>
ScriptedFrameProvider::CreateInstance(
    lldb::StackFrameListSP input_frames,
    const ScriptedFrameProviderDescriptor &descriptor) {
  if (!input_frames)
    return llvm::createStringError(
        "failed to create scripted frame provider: invalid input frames");

  Thread &thread = input_frames->GetThread();
  ProcessSP process_sp = thread.GetProcess();
  if (!process_sp)
    return nullptr;

  if (!descriptor.IsValid())
    return llvm::createStringError(
        "failed to create scripted frame provider: invalid scripted metadata");

  if (!descriptor.AppliesToThread(thread))
    return nullptr;

  ScriptInterpreter *script_interp =
      process_sp->GetTarget().GetDebugger().GetScriptInterpreter();
  if (!script_interp)
    return llvm::createStringError("cannot create scripted frame provider: No "
                                   "script interpreter installed");

  ScriptedFrameProviderInterfaceSP interface_sp =
      script_interp->CreateScriptedFrameProviderInterface();
  if (!interface_sp)
    return llvm::createStringError(
        "cannot create scripted frame provider: script interpreter couldn't "
        "create Scripted Frame Provider Interface");

  const ScriptedMetadataSP scripted_metadata = descriptor.scripted_metadata_sp;

  // If we shouldn't attach a frame provider to this thread, just exit early.
  if (!interface_sp->AppliesToThread(scripted_metadata->GetClassName(),
                                     thread.shared_from_this()))
    return nullptr;

  auto obj_or_err = interface_sp->CreatePluginObject(
      scripted_metadata->GetClassName(), input_frames,
      scripted_metadata->GetArgsSP());
  if (!obj_or_err)
    return obj_or_err.takeError();

  StructuredData::ObjectSP object_sp = *obj_or_err;
  if (!object_sp || !object_sp->IsValid())
    return llvm::createStringError(
        "cannot create scripted frame provider: failed to create valid scripted"
        "frame provider object");

  return std::make_shared<ScriptedFrameProvider>(input_frames, interface_sp,
                                                 descriptor);
}

ScriptedFrameProvider::ScriptedFrameProvider(
    StackFrameListSP input_frames,
    lldb::ScriptedFrameProviderInterfaceSP interface_sp,
    const ScriptedFrameProviderDescriptor &descriptor)
    : SyntheticFrameProvider(input_frames), m_interface_sp(interface_sp),
      m_descriptor(descriptor) {}

ScriptedFrameProvider::~ScriptedFrameProvider() = default;

std::string ScriptedFrameProvider::GetDescription() const {
  if (!m_interface_sp)
    return {};

  return m_interface_sp->GetDescription(m_descriptor.GetName());
}

std::optional<uint32_t> ScriptedFrameProvider::GetPriority() const {
  if (!m_interface_sp)
    return std::nullopt;

  return m_interface_sp->GetPriority(m_descriptor.GetName());
}

llvm::Expected<StackFrameSP>
ScriptedFrameProvider::GetFrameAtIndex(uint32_t idx) {
  if (!m_interface_sp)
    return llvm::createStringError(
        "cannot get stack frame: scripted frame provider not initialized");

  auto create_frame_from_dict =
      [this](StructuredData::Dictionary *dict,
             uint32_t index) -> llvm::Expected<StackFrameSP> {
    lldb::addr_t pc;
    if (!dict->GetValueForKeyAsInteger("pc", pc))
      return llvm::createStringError(
          "missing 'pc' key from scripted frame dictionary");

    Address symbol_addr;
    symbol_addr.SetLoadAddress(pc, &GetThread().GetProcess()->GetTarget());

    const lldb::addr_t cfa = LLDB_INVALID_ADDRESS;
    const bool cfa_is_valid = false;
    const bool artificial = false;
    const bool behaves_like_zeroth_frame = false;
    SymbolContext sc;
    symbol_addr.CalculateSymbolContext(&sc);

    ThreadSP thread_sp = GetThread().shared_from_this();
    return std::make_shared<StackFrame>(thread_sp, index, index, cfa,
                                        cfa_is_valid, pc,
                                        StackFrame::Kind::Synthetic, artificial,
                                        behaves_like_zeroth_frame, &sc);
  };

  auto create_frame_from_script_object =
      [this](
          StructuredData::ObjectSP object_sp) -> llvm::Expected<StackFrameSP> {
    Status error;
    if (!object_sp || !object_sp->GetAsGeneric())
      return llvm::createStringError("invalid script object");

    ThreadSP thread_sp = GetThread().shared_from_this();
    auto frame_or_error = ScriptedFrame::Create(thread_sp, nullptr, nullptr,
                                                object_sp->GetAsGeneric());

    if (!frame_or_error) {
      ScriptedInterface::ErrorWithMessage<bool>(
          LLVM_PRETTY_FUNCTION, toString(frame_or_error.takeError()), error);
      return error.ToError();
    }

    return *frame_or_error;
  };

  StructuredData::ObjectSP obj_sp = m_interface_sp->GetFrameAtIndex(idx);

  // None/null means no more frames or error.
  if (!obj_sp || !obj_sp->IsValid())
    return llvm::createStringError("invalid script object returned for frame " +
                                   llvm::Twine(idx));

  StackFrameSP synth_frame_sp = nullptr;
  if (StructuredData::UnsignedInteger *int_obj =
          obj_sp->GetAsUnsignedInteger()) {
    uint32_t real_frame_index = int_obj->GetValue();
    if (real_frame_index < m_input_frames->GetNumFrames()) {
      StackFrameSP real_frame_sp =
          m_input_frames->GetFrameAtIndex(real_frame_index);
      synth_frame_sp =
          (real_frame_index == idx)
              ? real_frame_sp
              : std::make_shared<BorrowedStackFrame>(real_frame_sp, idx);
    }
  } else if (StructuredData::Dictionary *dict = obj_sp->GetAsDictionary()) {
    // Check if it's a dictionary describing a frame.
    auto frame_from_dict_or_err = create_frame_from_dict(dict, idx);
    if (!frame_from_dict_or_err) {
      return llvm::createStringError(llvm::Twine(
          "couldn't create frame from dictionary at index " + llvm::Twine(idx) +
          ": " + toString(frame_from_dict_or_err.takeError())));
    }
    synth_frame_sp = *frame_from_dict_or_err;
  } else if (obj_sp->GetAsGeneric()) {
    // It's a ScriptedFrame object.
    auto frame_from_script_obj_or_err = create_frame_from_script_object(obj_sp);
    if (!frame_from_script_obj_or_err) {
      return llvm::createStringError(
          llvm::Twine("couldn't create frame from script object at index " +
                      llvm::Twine(idx) + ": " +
                      toString(frame_from_script_obj_or_err.takeError())));
    }
    synth_frame_sp = *frame_from_script_obj_or_err;
  } else {
    return llvm::createStringError(
        llvm::Twine("invalid return type from get_frame_at_index at index " +
                    llvm::Twine(idx)));
  }

  if (!synth_frame_sp)
    return llvm::createStringError(
        llvm::Twine("failed to create frame at index " + llvm::Twine(idx)));

  synth_frame_sp->SetFrameIndex(idx);

  return synth_frame_sp;
}

namespace lldb_private {
void lldb_initialize_ScriptedFrameProvider() {
  ScriptedFrameProvider::Initialize();
}

void lldb_terminate_ScriptedFrameProvider() {
  ScriptedFrameProvider::Terminate();
}
} // namespace lldb_private
