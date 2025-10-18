//===-- ScriptedFrameProvider.cpp ----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Interpreter/ScriptedFrameProvider.h"
#include "lldb/Core/Debugger.h"
#include "lldb/Interpreter/Interfaces/ScriptedFrameProviderInterface.h"
#include "lldb/Interpreter/ScriptInterpreter.h"
#include "lldb/Interpreter/ScriptedFrame.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/StackFrame.h"
#include "lldb/Target/Thread.h"
#include "lldb/Utility/ScriptedMetadata.h"
#include "lldb/Utility/Status.h"
#include "llvm/Support/Error.h"
#include <cstdint>

using namespace lldb;
using namespace lldb_private;

ScriptedFrameProvider::ScriptedFrameProvider(
    ThreadSP thread_sp, const ScriptedMetadata &scripted_metadata,
    Status &error)
    : m_thread_sp(thread_sp), m_interface_sp(nullptr) {
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

llvm::Expected<StackFrameListSP>
ScriptedFrameProvider::GetStackFrames(StackFrameListSP real_frames) {
  if (!m_interface_sp)
    return llvm::createStringError(
        "cannot get stack frames: Scripted frame provider not initialized");

  StructuredData::ArraySP arr_sp = m_interface_sp->GetStackFrames(real_frames);

  Status error;
  if (!arr_sp)
    return llvm::createStringError(
        "Failed to get scripted thread stackframes.");

  size_t arr_size = arr_sp->GetSize();
  if (arr_size > std::numeric_limits<uint32_t>::max())
    return llvm::createStringError(llvm::Twine(
        "StackFrame array size (" + llvm::Twine(arr_size) +
        llvm::Twine(
            ") is greater than maximum authorized for a StackFrameList.")));

  auto create_frame_from_dict =
      [this, arr_sp](size_t iteration_idx) -> llvm::Expected<StackFrameSP> {
    std::optional<StructuredData::Dictionary *> maybe_dict =
        arr_sp->GetItemAtIndexAsDictionary(iteration_idx);
    if (!maybe_dict)
      return llvm::createStringError("invalid scripted frame dictionary.");
    StructuredData::Dictionary *dict = *maybe_dict;

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

    return std::make_shared<StackFrame>(m_thread_sp, iteration_idx,
                                        iteration_idx, cfa, cfa_is_valid, pc,
                                        StackFrame::Kind::Synthetic, artificial,
                                        behaves_like_zeroth_frame, &sc);
  };

  auto create_frame_from_script_object =
      [this, arr_sp](size_t iteration_idx) -> llvm::Expected<StackFrameSP> {
    Status error;
    StructuredData::ObjectSP object_sp = arr_sp->GetItemAtIndex(iteration_idx);
    if (!object_sp || !object_sp->GetAsGeneric())
      return llvm::createStringError("invalid script object at index %u",
                                     iteration_idx);

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

  StackFrameListSP scripted_frames =
      std::make_shared<StackFrameList>(*m_thread_sp, StackFrameListSP(), true);

  for (size_t idx = 0; idx < arr_size; idx++) {
    StackFrameSP synth_frame_sp = nullptr;

    std::optional<uint32_t> maybe_real_frame_index =
        arr_sp->GetItemAtIndexAsInteger<uint32_t>(idx);
    if (maybe_real_frame_index &&
        (*maybe_real_frame_index < real_frames->GetNumFrames())) {
      synth_frame_sp = real_frames->GetFrameAtIndex(*maybe_real_frame_index);
    } else {
      auto frame_from_dict_or_err = create_frame_from_dict(idx);
      if (!frame_from_dict_or_err) {
        auto frame_from_script_obj_or_err =
            create_frame_from_script_object(idx);

        if (!frame_from_script_obj_or_err) {
          llvm::consumeError(frame_from_script_obj_or_err.takeError());
          llvm::consumeError(frame_from_dict_or_err.takeError());
          return llvm::createStringError(
              llvm::Twine("Couldn't add artificial frame (" + llvm::Twine(idx) +
                          llvm::Twine(") to ScriptedThread StackFrameList.")));
        } else {
          llvm::consumeError(frame_from_dict_or_err.takeError());
          synth_frame_sp = *frame_from_script_obj_or_err;
        }
      } else {
        synth_frame_sp = *frame_from_dict_or_err;
      }
    }

    if (!scripted_frames->SetFrameAtIndex(static_cast<uint32_t>(idx),
                                          synth_frame_sp))
      return llvm::createStringError(
          llvm::Twine("Couldn't add frame (" + llvm::Twine(idx) +
                      llvm::Twine(") to ScriptedThread StackFrameList.")));

    synth_frame_sp->SetFrameIndex(idx);
  }

  // Mark that all frames have been fetched to prevent automatic unwinding
  scripted_frames->SetAllFramesFetched();

  return scripted_frames;
}
