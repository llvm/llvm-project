//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "UnwindWasm.h"
#include "Plugins/Process/gdb-remote/ThreadGDBRemote.h"
#include "ProcessWasm.h"
#include "RegisterContextWasm.h"
#include "ThreadWasm.h"
#include "lldb/Utility/LLDBLog.h"
#include "lldb/Utility/Log.h"

using namespace lldb;
using namespace lldb_private;
using namespace process_gdb_remote;
using namespace wasm;

/// Engine frames have no address, so their CFAs are in a space without memory.
static constexpr lldb::addr_t kWasmSyntheticCFABase =
    uint64_t(WasmAddressType::Invalid) << kWasmAddressTypeShift;

lldb::RegisterContextSP
UnwindWasm::DoCreateRegisterContextForFrame(lldb_private::StackFrame *frame) {
  const uint32_t concrete_frame_idx = frame->GetConcreteFrameIndex();
  if (m_frames.size() <= concrete_frame_idx)
    return lldb::RegisterContextSP();

  ThreadSP thread = frame->GetThread();
  ThreadGDBRemote *gdb_thread = static_cast<ThreadGDBRemote *>(thread.get());
  ProcessWasm *wasm_process =
      static_cast<ProcessWasm *>(thread->GetProcess().get());

  return std::make_shared<RegisterContextWasm>(*gdb_thread, concrete_frame_idx,
                                               wasm_process->GetRegisterInfo());
}

uint32_t UnwindWasm::DoGetFrameCount() {
  if (m_unwind_complete)
    return GetVisibleFrameCount();

  m_unwind_complete = true;
  m_frames.clear();

  ThreadWasm &wasm_thread = static_cast<ThreadWasm &>(GetThread());
  llvm::Expected<std::vector<lldb::addr_t>> call_stack_pcs =
      wasm_thread.GetWasmCallStack();
  if (!call_stack_pcs) {
    LLDB_LOG_ERROR(GetLog(LLDBLog::Unwind), call_stack_pcs.takeError(),
                   "Failed to get Wasm callstack: {0}");
    m_frames.clear();
    return 0;
  }

  m_frames = *call_stack_pcs;
  return GetVisibleFrameCount();
}

uint32_t UnwindWasm::GetVisibleFrameCount() {
  // A backtrace goes no deeper than the target asks for, which is what bounds
  // the walk of a stack that recurses without end. The depth bounds what a
  // caller is told rather than what is kept, because it can be raised after a
  // stack has been fetched.
  return std::min<uint64_t>(m_frames.size(),
                            GetThread().GetMaxBacktraceDepth());
}

bool UnwindWasm::DoGetFrameInfoAtIndex(uint32_t frame_idx, lldb::addr_t &cfa,
                                       lldb::addr_t &pc,
                                       bool &behaves_like_zeroth_frame) {
  if (m_frames.size() == 0)
    DoGetFrameCount();

  if (frame_idx >= GetVisibleFrameCount())
    return false;

  behaves_like_zeroth_frame = (frame_idx == 0);
  // The distance to the outermost frame does not change as frames are pushed
  // and popped above it.
  cfa = kWasmSyntheticCFABase + m_frames.size() - 1 - frame_idx;
  pc = m_frames[frame_idx];
  return true;
}
