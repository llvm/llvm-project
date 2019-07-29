//===-- RegisterContextDPU.cpp --------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Core/Address.h"
#include "lldb/Core/AddressRange.h"
#include "lldb/Core/Module.h"
#include "lldb/Core/Value.h"
#include "lldb/Expression/DWARFExpression.h"
#include "lldb/Symbol/ArmUnwindInfo.h"
#include "lldb/Symbol/DWARFCallFrameInfo.h"
#include "lldb/Symbol/FuncUnwinders.h"
#include "lldb/Symbol/Function.h"
#include "lldb/Symbol/ObjectFile.h"
#include "lldb/Symbol/Symbol.h"
#include "lldb/Symbol/SymbolContext.h"
#include "lldb/Target/ABI.h"
#include "lldb/Target/DynamicLoader.h"
#include "lldb/Target/ExecutionContext.h"
#include "lldb/Target/Platform.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/SectionLoadList.h"
#include "lldb/Target/StackFrame.h"
#include "lldb/Target/Target.h"
#include "lldb/Target/Thread.h"
#include "lldb/Utility/DataBufferHeap.h"
#include "lldb/Utility/Log.h"
#include "lldb/Utility/RegisterValue.h"
#include "lldb/lldb-private.h"

#include "RegisterContextDPU.h"
#include "lldb-dpu-register-enums.h"

using namespace lldb;
using namespace lldb_private;

RegisterContextDPU::RegisterContextDPU(
    Thread &thread, RegisterContextDPUSP prev_frame_reg_ctx_sp,
    lldb::addr_t cfa, lldb::addr_t pc, uint32_t frame_number)
    : RegisterContext(thread, frame_number), m_thread(thread),
      reg_ctx_sp(thread.GetRegisterContext()), m_frame_number(frame_number),
      m_cfa(cfa), m_pc(pc), m_prev_frame(prev_frame_reg_ctx_sp) {}

void RegisterContextDPU::InvalidateAllRegisters() {
  reg_ctx_sp->InvalidateAllRegisters();
}

size_t RegisterContextDPU::GetRegisterCount() {
  return reg_ctx_sp->GetRegisterCount();
}

const RegisterInfo *RegisterContextDPU::GetRegisterInfoAtIndex(size_t reg) {
  return reg_ctx_sp->GetRegisterInfoAtIndex(reg);
}

size_t RegisterContextDPU::GetRegisterSetCount() {
  return reg_ctx_sp->GetRegisterSetCount();
}

const RegisterSet *RegisterContextDPU::GetRegisterSet(size_t reg_set) {
  return reg_ctx_sp->GetRegisterSet(reg_set);
}

bool RegisterContextDPU::ReadRegister(const RegisterInfo *reg_info,
                                      RegisterValue &value) {
  if (m_frame_number == 0) {
    return reg_ctx_sp->ReadRegister(reg_info, value);
  }
  // Do not try to get r22 from the saved location, we alreay have the good one
  // here. The one from the saved location can be false depending on our
  // position in the prologue
  if (reg_info == reg_ctx_sp->GetRegisterInfoByName("r22")) {
    value = m_cfa;
    return true;
  } else if (reg_info == reg_ctx_sp->GetRegisterInfoByName("pc")) {
    value = m_pc;
    return true;
  } else if (reg_info == reg_ctx_sp->GetRegisterInfoByName("zf") ||
             reg_info == reg_ctx_sp->GetRegisterInfoByName("cf")) {
    return false;
  }

  return m_prev_frame->ReadRegisterFromSavedLocation(reg_info, value);
}

bool RegisterContextDPU::WriteRegister(const RegisterInfo *reg_info,
                                       const RegisterValue &value) {
  if (m_frame_number == 0) {
    return reg_ctx_sp->WriteRegister(reg_info, value);
  }
  if (reg_info == reg_ctx_sp->GetRegisterInfoByName("r22") ||
      reg_info == reg_ctx_sp->GetRegisterInfoByName("pc") ||
      reg_info == reg_ctx_sp->GetRegisterInfoByName("zf") ||
      reg_info == reg_ctx_sp->GetRegisterInfoByName("cf")) {
    return false;
  }
  return m_prev_frame->WriteRegisterToSavedLocation(reg_info, value);
}

uint32_t
RegisterContextDPU::ConvertRegisterKindToRegisterNumber(lldb::RegisterKind kind,
                                                        uint32_t num) {
  return reg_ctx_sp->ConvertRegisterKindToRegisterNumber(kind, num);
}

bool RegisterContextDPU::PCInPrologue(lldb::addr_t start_addr,
                                      uint32_t nb_callee_saved_regs) {
  // We consider here only a subset of the prologue
  // The subset is the 2 first instruction (stack managment) and all the saving
  // of callee register
  // We store callee register with double store (instruction of 8 bytes for 2
  // callee registers = 4 bytes per callee register)
  // We need to add 8 bytes (1 instruction) for the add of the stack pointer
  lldb::addr_t end_addr = (start_addr + nb_callee_saved_regs * 4 + 8);
  return m_pc >= start_addr && m_pc < end_addr;
}

bool RegisterContextDPU::LookForRegisterLocation(const RegisterInfo *reg_info,
                                                 lldb::addr_t &addr) {
  Function *fct = NULL;
  GetFunction(&fct, m_pc);
  if (fct == NULL) {
    return false;
  }

  Address pc_addr = fct->GetAddressRange().GetBaseAddress();
  ModuleSP module_sp(pc_addr.GetModule());
  DWARFCallFrameInfo *debug_frame =
      module_sp->GetObjectFile()->GetUnwindTable().GetDebugFrameInfo();

  if (debug_frame) {
    uint32_t row_id = 0;
    UnwindPlanSP unwind_plan_sp(new UnwindPlan(lldb::eRegisterKindGeneric));

    addr = LLDB_INVALID_ADDRESS;

    debug_frame->GetUnwindPlan(pc_addr, *unwind_plan_sp);
    while (unwind_plan_sp->IsValidRowIndex(row_id)) {
      uint32_t nb_callee_saved_regs = 0;
      UnwindPlan::RowSP row = unwind_plan_sp->GetRowAtIndex(row_id++);
      for (unsigned int each_reg = 0;
           each_reg < lldb_private::k_num_registers_dpu; each_reg++) {
        UnwindPlan::Row::RegisterLocation reg_loc;
        row->GetRegisterInfo(each_reg, reg_loc);
        if (!reg_loc.IsUndefined() && !reg_loc.IsUnspecified()) {
          nb_callee_saved_regs++;
          if (reg_info->byte_offset / 4 == each_reg) {
            addr = reg_loc.GetOffset() + m_cfa;
          }
        }
      }
      if (addr != LLDB_INVALID_ADDRESS &&
          !PCInPrologue(pc_addr.GetFileAddress(), nb_callee_saved_regs) &&
          !PCIsInstructionReturn(m_pc))
        return true;
    }
  }

  return false;
}

bool RegisterContextDPU::ReadRegisterFromSavedLocation(
    const RegisterInfo *reg_info, RegisterValue &value) {
  lldb::addr_t reg_addr;
  if (LookForRegisterLocation(reg_info, reg_addr)) {
    Status error;
    uint32_t val;
    m_thread.GetProcess()->ReadMemory(reg_addr, &val, 4, error);
    value.SetUInt32(val);
    return true;
  } else if (m_frame_number == 0) {
    return reg_ctx_sp->ReadRegister(reg_info, value);
  }
  return m_prev_frame->ReadRegisterFromSavedLocation(reg_info, value);
}

bool RegisterContextDPU::WriteRegisterToSavedLocation(
    const RegisterInfo *reg_info, const RegisterValue &value) {
  lldb::addr_t reg_addr;
  if (LookForRegisterLocation(reg_info, reg_addr)) {
    Status error;
    uint32_t val = value.GetAsUInt32();
    m_thread.GetProcess()->WriteMemory(reg_addr, &val, 4, error);
    return true;
  } else if (m_frame_number == 0) {
    return reg_ctx_sp->WriteRegister(reg_info, value);
  }
  return m_prev_frame->WriteRegisterToSavedLocation(reg_info, value);
}

void RegisterContextDPU::GetFunction(Function **fct, lldb::addr_t pc) {
  Address resolved_addr;
  m_thread.GetProcess()->GetTarget().ResolveLoadAddress(pc, resolved_addr);

  SymbolContext sc;
  ModuleSP module_sp(resolved_addr.GetModule());
  module_sp->ResolveSymbolContextForAddress(resolved_addr,
                                            eSymbolContextFunction, sc);
  *fct = sc.function;
}

bool RegisterContextDPU::PCIsInstructionReturn(lldb::addr_t pc) {
  Status error;
  uint64_t instruction;
  m_thread.GetProcess()->ReadMemory(pc, &instruction, 8, error);
  return instruction == 0x8c5f00000000ULL; // 0x8c5f00000000 => 'jump r23'
}
