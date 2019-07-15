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

void UnwindDPU::SetFrame(CursorSP &frame, lldb::addr_t cfa, lldb::addr_t pc,
                         lldb::RegisterContextSP reg_ctx_sp) {
  frame->cfa = cfa;
  frame->pc = pc;
  frame->reg_ctx_sp = reg_ctx_sp;
  m_frames.push_back(frame);
}

static void CleanLLDBAddr(lldb::addr_t &addr) { addr = addr & 0xffffffff; }

uint32_t UnwindDPU::DoGetFrameCount() {
  if (!m_frames.empty())
    return m_frames.size();

  Status error;
  ProcessSP process_sp(m_thread.GetProcess());
  CursorSP first_frame(new Cursor());
  CursorSP &prev_frame = first_frame;
  lldb::RegisterContextSP reg_ctx_sp = m_thread.GetRegisterContext();
  RegisterValue reg_r22, reg_pc;

  reg_ctx_sp->ReadRegister(reg_ctx_sp->GetRegisterInfoByName("r22"), reg_r22);
  reg_ctx_sp->ReadRegister(reg_ctx_sp->GetRegisterInfoByName("pc"), reg_pc);

  SetFrame(first_frame, reg_r22.GetAsUInt32(), reg_pc.GetAsUInt32(),
           reg_ctx_sp);

  while (true) {
    CursorSP next_frame(new Cursor());
    process_sp->ReadMemory(prev_frame->cfa - 4, &next_frame->cfa, 4, error);
    process_sp->ReadMemory(prev_frame->cfa - 8, &next_frame->pc, 4, error);

    CleanLLDBAddr(next_frame->cfa);
    CleanLLDBAddr(next_frame->pc);

    if (next_frame->cfa == 0xdb9 || next_frame->cfa == 0 ||
        next_frame->cfa > (64 * 1024) || next_frame->pc > (64 * 1024) ||
        m_frames.size() > (8 * 1024))
      break;

    SetFrame(next_frame, next_frame->cfa,
             0x80000000 | ((next_frame->pc - 1) * 8), reg_ctx_sp);

    prev_frame = next_frame;
  }

  return m_frames.size();
}

bool UnwindDPU::DoGetFrameInfoAtIndex(uint32_t frame_idx, lldb::addr_t &cfa,
                                      lldb::addr_t &pc) {
  if (frame_idx >= DoGetFrameCount())
    return false;

  cfa = m_frames[frame_idx]->cfa;
  pc = m_frames[frame_idx]->pc;
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
