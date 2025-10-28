//===-- ScriptedFrameProvider.cpp ----------------------------------------===//
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
                                ScriptedFrameProvider::CreateInstance);
}

void ScriptedFrameProvider::Terminate() {
  PluginManager::UnregisterPlugin(ScriptedFrameProvider::CreateInstance);
}

llvm::Expected<lldb::SyntheticFrameProviderSP>
ScriptedFrameProvider::CreateInstance(lldb::ThreadSP thread_sp) {
  if (!thread_sp)
    return llvm::createStringError(
        "failed to create scripted frame: invalid thread");

  ProcessSP process_sp = thread_sp->GetProcess();
  if (!process_sp)
    return nullptr;

  Target &target = process_sp->GetTarget();

  Status error;
  if (auto descriptor = target.GetScriptedFrameProviderDescriptor()) {
    if (!descriptor->IsValid())
      return llvm::createStringError(
          "failed to create scripted frame: invalid scripted metadata");

    if (!descriptor->AppliesToThread(thread_sp->GetID()))
      return nullptr;

    auto provider_sp = std::make_shared<ScriptedFrameProvider>(
        thread_sp, *descriptor->scripted_metadata_sp, error);
    if (!provider_sp || error.Fail())
      return error.ToError();

    return provider_sp;
  }

  return llvm::createStringError(
      "failed to create scripted frame: invalid scripted metadata");
}

ScriptedFrameProvider::ScriptedFrameProvider(
    ThreadSP thread_sp, const ScriptedMetadata &scripted_metadata,
    Status &error)
    : SyntheticFrameProvider(thread_sp), m_interface_sp(nullptr) {
  if (!m_thread_sp) {
    error = Status::FromErrorString(
        "cannot create scripted frame provider: Invalid thread");
    return;
  }

  ProcessSP process_sp = m_thread_sp->GetProcess();
  if (!process_sp) {
    error = Status::FromErrorString(
        "cannot create scripted frame provider: Invalid process");
    return;
  }

  ScriptInterpreter *script_interp =
      process_sp->GetTarget().GetDebugger().GetScriptInterpreter();
  if (!script_interp) {
    error = Status::FromErrorString("cannot create scripted frame provider: No "
                                    "script interpreter installed");
    return;
  }

  m_interface_sp = script_interp->CreateScriptedFrameProviderInterface();
  if (!m_interface_sp) {
    error = Status::FromErrorString(
        "cannot create scripted frame provider: Script interpreter couldn't "
        "create Scripted Frame Provider Interface");
    return;
  }

  auto obj_or_err = m_interface_sp->CreatePluginObject(
      scripted_metadata.GetClassName(), m_thread_sp,
      scripted_metadata.GetArgsSP());
  if (!obj_or_err) {
    error = Status::FromError(obj_or_err.takeError());
    return;
  }

  StructuredData::ObjectSP object_sp = *obj_or_err;
  if (!object_sp || !object_sp->IsValid()) {
    error = Status::FromErrorString(
        "cannot create scripted frame provider: Failed to create valid script "
        "object");
    return;
  }

  error.Clear();
}

ScriptedFrameProvider::~ScriptedFrameProvider() = default;

llvm::Expected<StackFrameSP>
ScriptedFrameProvider::GetFrameAtIndex(StackFrameListSP real_frames,
                                       uint32_t idx) {
  if (!m_interface_sp)
    return llvm::createStringError(
        "cannot get stack frame: Scripted frame provider not initialized");

  auto create_frame_from_dict =
      [this](StructuredData::Dictionary *dict,
             uint32_t index) -> llvm::Expected<StackFrameSP> {
    lldb::addr_t pc;
    if (!dict->GetValueForKeyAsInteger("pc", pc))
      return llvm::createStringError(
          "missing 'pc' key from scripted frame dictionary.");

    Address symbol_addr;
    symbol_addr.SetLoadAddress(pc, &m_thread_sp->GetProcess()->GetTarget());

    lldb::addr_t cfa = LLDB_INVALID_ADDRESS;
    bool cfa_is_valid = false;
    const bool artificial = false;
    const bool behaves_like_zeroth_frame = false;
    SymbolContext sc;
    symbol_addr.CalculateSymbolContext(&sc);

    return std::make_shared<StackFrame>(m_thread_sp, index, index, cfa,
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

    auto frame_or_error = ScriptedFrame::Create(m_thread_sp, nullptr, nullptr,
                                                object_sp->GetAsGeneric());

    if (!frame_or_error) {
      ScriptedInterface::ErrorWithMessage<bool>(
          LLVM_PRETTY_FUNCTION, toString(frame_or_error.takeError()), error);
      return error.ToError();
    }

    StackFrameSP frame_sp = frame_or_error.get();
    lldbassert(frame_sp && "Couldn't initialize scripted frame.");

    return frame_sp;
  };

  StructuredData::ObjectSP obj_sp =
      m_interface_sp->GetFrameAtIndex(real_frames, idx);

  // None/null means no more frames or error
  if (!obj_sp || !obj_sp->IsValid())
    return llvm::createStringError("invalid script object returned for frame " +
                                   llvm::Twine(idx));

  StackFrameSP synth_frame_sp = nullptr;
  if (auto *int_obj = obj_sp->GetAsUnsignedInteger()) {
    uint32_t real_frame_index = int_obj->GetValue();
    if (real_frame_index < real_frames->GetNumFrames()) {
      synth_frame_sp = real_frames->GetFrameAtIndex(real_frame_index);
    }
  } else if (auto *dict = obj_sp->GetAsDictionary()) {
    // Check if it's a dictionary describing a frame
    auto frame_from_dict_or_err = create_frame_from_dict(dict, idx);
    if (!frame_from_dict_or_err) {
      return llvm::createStringError(llvm::Twine(
          "Couldn't create frame from dictionary at index " + llvm::Twine(idx) +
          ": " + toString(frame_from_dict_or_err.takeError())));
    }
    synth_frame_sp = *frame_from_dict_or_err;
  } else if (obj_sp->GetAsGeneric()) {
    // It's a ScriptedFrame object
    auto frame_from_script_obj_or_err = create_frame_from_script_object(obj_sp);
    if (!frame_from_script_obj_or_err) {
      return llvm::createStringError(
          llvm::Twine("Couldn't create frame from script object at index " +
                      llvm::Twine(idx) + ": " +
                      toString(frame_from_script_obj_or_err.takeError())));
    }
    synth_frame_sp = *frame_from_script_obj_or_err;
  } else {
    return llvm::createStringError(
        llvm::Twine("Invalid return type from get_frame_at_index at index " +
                    llvm::Twine(idx)));
  }

  if (!synth_frame_sp)
    return llvm::createStringError(
        llvm::Twine("Failed to create frame at index " + llvm::Twine(idx)));

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
