//===-- UnwindDPU.cpp -------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Core/Module.h"
#include "lldb/Core/RegisterValue.h"
#include "lldb/Symbol/FuncUnwinders.h"
#include "lldb/Symbol/Function.h"
#include "lldb/Symbol/UnwindPlan.h"
#include "lldb/Target/ABI.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/RegisterContext.h"
#include "lldb/Target/Target.h"
#include "lldb/Target/Thread.h"
#include "lldb/Utility/Log.h"

#include "UnwindDPU.h"

using namespace lldb;
using namespace lldb_private;

UnwindDPU::UnwindDPU(Thread &thread) : Unwind(thread), m_frames() {}

void UnwindDPU::DoClear() { m_frames.clear(); }

uint32_t UnwindDPU::DoGetFrameCount() {
  if (!m_frames.empty())
    return m_frames.size();

  Status error;
  ProcessSP process_sp(m_thread.GetProcess());
  CursorSP first_frame(new Cursor());
  lldb::RegisterContextSP reg_ctx_sp = m_thread.GetRegisterContext();
  const RegisterInfo *reg_info_r22 = reg_ctx_sp->GetRegisterInfoByName("r22");
  const RegisterInfo *reg_info_pc = reg_ctx_sp->GetRegisterInfoByName("pc");
  RegisterValue reg_r22, reg_pc;

  reg_ctx_sp->ReadRegister(reg_info_r22, reg_r22);
  reg_ctx_sp->ReadRegister(reg_info_pc, reg_pc);

  first_frame->cfa = reg_r22.GetAsUInt32();
  first_frame->start_pc = reg_pc.GetAsUInt32();
  first_frame->reg_ctx_sp = reg_ctx_sp;
  m_frames.push_back(first_frame);

  int r22 = reg_r22.GetAsUInt32();

  while (r22 != 0) {
    CursorSP next_frame(new Cursor());
    process_sp->ReadMemory(r22 - 4, &next_frame->cfa, 4, error);
    process_sp->ReadMemory(r22 - 8, &next_frame->start_pc, 4, error);

    next_frame->reg_ctx_sp = reg_ctx_sp;
    next_frame->start_pc =
        0xffffffff & (0x80000000 | ((next_frame->start_pc - 1) * 8));
    next_frame->cfa = 0xffffffff & next_frame->cfa;

    r22 = next_frame->cfa;
    if (r22 != 0)
      m_frames.push_back(next_frame);
  }

  return m_frames.size();
}

bool UnwindDPU::DoGetFrameInfoAtIndex(uint32_t frame_idx, lldb::addr_t &cfa,
                                      lldb::addr_t &start_pc) {
  if (frame_idx >= DoGetFrameCount())
    return false;

  cfa = m_frames[frame_idx]->cfa;
  start_pc = m_frames[frame_idx]->start_pc;
  return true;
}

lldb::RegisterContextSP
UnwindDPU::DoCreateRegisterContextForFrame(lldb_private::StackFrame *frame) {
  lldb::RegisterContextSP reg_ctx_sp;
  uint32_t frame_idx = frame->GetConcreteFrameIndex();

  if (frame_idx >= DoGetFrameCount())
    return reg_ctx_sp;

  Cursor *frame_cursor = m_frames[frame_idx].get();
  reg_ctx_sp = frame_cursor->reg_ctx_sp;
  return reg_ctx_sp;
}
