//===-- UnwindWasm.cpp -------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "UnwindWasm.h"
#include "Plugins/Process/gdb-remote/GDBRemoteCommunicationClient.h"
#include "Plugins/Process/gdb-remote/GDBRemoteRegisterContext.h"
#include "Plugins/Process/gdb-remote/ProcessGDBRemote.h"
#include "Plugins/Process/gdb-remote/ThreadGDBRemote.h"

using namespace lldb;
using namespace lldb_private;
using namespace process_gdb_remote;

lldb::RegisterContextSP
UnwindWasm::DoCreateRegisterContextForFrame(lldb_private::StackFrame *frame) {
  if (m_frames.size() <= frame->GetFrameIndex()) {
    return lldb::RegisterContextSP();
  }

  ProcessGDBRemote *gdb_process =
      static_cast<ProcessGDBRemote *>(frame->CalculateProcess().get());
  ThreadGDBRemote *gdb_thread =
      static_cast<ThreadGDBRemote *>(frame->CalculateThread().get());
  std::shared_ptr<GDBRemoteRegisterContext> reg_ctx_sp =
      std::make_shared<GDBRemoteRegisterContext>(
          *gdb_thread, frame->GetFrameIndex(), gdb_process->m_register_info,
          false, false);
  reg_ctx_sp->PrivateSetRegisterValue(0, m_frames[frame->GetFrameIndex()]);
  return reg_ctx_sp;
}

uint32_t UnwindWasm::DoGetFrameCount() {
  if (!m_unwind_complete) {
    m_unwind_complete = true;
    m_frames.clear();

    process_gdb_remote::ProcessGDBRemote *process =
        (process_gdb_remote::ProcessGDBRemote *)GetThread().GetProcess().get();
    if (process) {
      process_gdb_remote::GDBRemoteCommunicationClient *gdb_comm =
          &process->GetGDBRemote();
      if (gdb_comm) {
        if (!gdb_comm->GetWasmCallStack(m_frames)) {
          m_frames.clear();
        }
      }
    }
  }
  return m_frames.size();
}

bool UnwindWasm::DoGetFrameInfoAtIndex(uint32_t frame_idx, lldb::addr_t &cfa,
                                       lldb::addr_t &pc,
                                       bool &behaves_like_zeroth_frame) {
  cfa = 0;

  if (m_frames.size() == 0) {
    DoGetFrameCount();
  }

  if (frame_idx == 0) {
    lldb::RegisterContextSP reg_ctx_sp = GetThread().GetRegisterContext();
    pc = reg_ctx_sp->GetPC();
    return true;
  } else if (frame_idx < m_frames.size()) {
    pc = m_frames[frame_idx];
    return true;
  } else {
    pc = 0;
    return false;
  }
}