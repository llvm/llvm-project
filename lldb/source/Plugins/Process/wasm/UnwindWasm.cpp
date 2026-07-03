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

lldb::RegisterContextSP
UnwindWasm::DoCreateRegisterContextForFrame(lldb_private::StackFrame *frame) {
  if (m_frames.size() <= frame->GetFrameIndex())
    return lldb::RegisterContextSP();

  ThreadSP thread = frame->GetThread();
  ThreadGDBRemote *gdb_thread = static_cast<ThreadGDBRemote *>(thread.get());
  ProcessWasm *wasm_process =
      static_cast<ProcessWasm *>(thread->GetProcess().get());

  return std::make_shared<RegisterContextWasm>(*gdb_thread,
                                               frame->GetConcreteFrameIndex(),
                                               wasm_process->GetRegisterInfo());
}

uint32_t UnwindWasm::DoGetFrameCount() {
  if (m_unwind_complete)
    return m_frames.size();

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
  return m_frames.size();
}

bool UnwindWasm::DoGetFrameInfoAtIndex(uint32_t frame_idx, lldb::addr_t &cfa,
                                       lldb::addr_t &pc,
                                       bool &behaves_like_zeroth_frame) {
  if (m_frames.size() == 0)
    DoGetFrameCount();

  if (frame_idx >= m_frames.size())
    return false;

  behaves_like_zeroth_frame = (frame_idx == 0);
  cfa = 0;
  pc = m_frames[frame_idx];
  return true;
}
