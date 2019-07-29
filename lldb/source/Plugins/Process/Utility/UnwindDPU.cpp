//===-- UnwindDPU.cpp -------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Core/Module.h"
#include "lldb/Symbol/FuncUnwinders.h"
#include "lldb/Symbol/Function.h"
#include "lldb/Symbol/UnwindPlan.h"
#include "lldb/Target/ABI.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/RegisterContext.h"
#include "lldb/Target/Target.h"
#include "lldb/Target/Thread.h"
#include "lldb/Utility/Log.h"
#include "lldb/Utility/RegisterValue.h"

#include "UnwindDPU.h"
#include "RegisterContextDPU.h"

using namespace lldb;
using namespace lldb_private;

UnwindDPU::UnwindDPU(Thread &thread) : Unwind(thread), m_frames() {}

void UnwindDPU::DoClear() { m_frames.clear(); }

void UnwindDPU::SetFrame(CursorSP *prev_frame, lldb::addr_t cfa,
                         lldb::addr_t pc) {
  CursorSP new_frame(new Cursor());
  RegisterContextDPUSP prev_reg_ctx_sp =
      *prev_frame != NULL ? (*prev_frame)->reg_ctx_sp : NULL;
  RegisterContextDPUSP new_reg_ctx_sp(new RegisterContextDPU(
      m_thread, prev_reg_ctx_sp, cfa, pc, m_frames.size()));
  new_frame->cfa = cfa;
  new_frame->pc = pc;
  new_frame->reg_ctx_sp = new_reg_ctx_sp;
  m_frames.push_back(new_frame);
  *prev_frame = new_frame;
}

#define NB_FRAME_MAX (8 * 1024)
#define WRAM_SIZE (64 * 1024)
#define NB_INSTRUCTION_MAX (4 * 1024)
#define STACK_BACKTRACE_STOP_VALUE (0xdb9)

#define FORMAT_PC(pc) (0x80000000 | ((pc) * 8))

uint32_t UnwindDPU::DoGetFrameCount() {
  if (!m_frames.empty())
    return m_frames.size();

  lldb::RegisterContextSP reg_ctx_sp = m_thread.GetRegisterContext();

  RegisterValue reg_r22, reg_pc;
  reg_ctx_sp->ReadRegister(reg_ctx_sp->GetRegisterInfoByName("r22"), reg_r22);
  reg_ctx_sp->ReadRegister(reg_ctx_sp->GetRegisterInfoByName("pc"), reg_pc);

  CursorSP prev_frame = NULL;
  lldb::addr_t first_pc_addr = reg_pc.GetAsUInt32();
  lldb::addr_t first_r22_value = reg_r22.GetAsUInt32();
  SetFrame(&prev_frame, first_r22_value, first_pc_addr);

  // If we are in the 2 first instruction of the function, or in the return
  // instruction of the function, the information to get the next frame are not
  // the same as usual. r22 is already the good one (previous frame does not
  // have the good value for r22, but it's normal as its frame is not yet set,
  // let's just add 1 to it so that we are sure that it will be seen as the
  // youngest one when comparing StackID). pc is in r23.
  Function *fct = NULL;
  lldb::addr_t start_addr = 0;
  prev_frame->reg_ctx_sp->GetFunction(&fct, first_pc_addr);
  if (fct != NULL) {
    start_addr = fct->GetAddressRange().GetBaseAddress().GetFileAddress();
  }
  if (((first_pc_addr >= start_addr) && (first_pc_addr < (start_addr + 16))) ||
      prev_frame->reg_ctx_sp->PCIsInstructionReturn(first_pc_addr)) {
    prev_frame->cfa++;
    RegisterValue reg_r23;
    reg_ctx_sp->ReadRegister(reg_ctx_sp->GetRegisterInfoByName("r23"), reg_r23);
    SetFrame(&prev_frame, first_r22_value, FORMAT_PC(reg_r23.GetAsUInt32()));
  }

  while (true) {
    Status error;
    lldb::addr_t cfa_addr = 0;
    lldb::addr_t pc_addr = 0;
    m_thread.GetProcess()->ReadMemory(prev_frame->cfa - 4, &cfa_addr, 4, error);
    m_thread.GetProcess()->ReadMemory(prev_frame->cfa - 8, &pc_addr, 4, error);

    if (cfa_addr == STACK_BACKTRACE_STOP_VALUE || cfa_addr == 0 ||
        cfa_addr > WRAM_SIZE || pc_addr > NB_INSTRUCTION_MAX ||
        m_frames.size() > NB_FRAME_MAX)
      break;

    SetFrame(&prev_frame, cfa_addr, FORMAT_PC(pc_addr));
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
