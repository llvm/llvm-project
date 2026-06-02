//===-- RegisterContextEZH.cpp --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "RegisterContextEZH.h"
#include "ProcessEZH.h"
#include "EZHRegisters.h"
#include "lldb/Core/Module.h"
#include "lldb/Symbol/Variable.h"
#include "lldb/Symbol/VariableList.h"
#include "lldb/Target/Target.h"
#include "lldb/Core/Value.h"
#include "lldb/Core/Debugger.h"

#include "lldb/Target/Process.h"
#include "lldb/Target/Thread.h"
#include "lldb/Target/RegisterFlags.h"
#include "lldb/Utility/DataBufferHeap.h"
#include "lldb/Utility/RegisterValue.h"
#include "lldb/Utility/Status.h"

#define DEFINE_REG_NAME(reg_num)      lldb_private::ConstString(#reg_num).GetCString()
#define DEFINE_REG_NAME_STR(reg_name) lldb_private::ConstString(reg_name).GetCString()

#define DEFINE_GENERIC_REGISTER_STUB(dwarf_num, str_name, generic_num)        \
  {                                                                           \
    DEFINE_REG_NAME(dwarf_num), DEFINE_REG_NAME_STR(str_name),                \
    4, dwarf_num * 4, lldb::eEncodingUint, lldb::eFormatHex,                  \
    { dwarf_num, dwarf_num, generic_num, dwarf_num, dwarf_num },              \
    nullptr, nullptr, nullptr,                                                \
  }

#define DEFINE_FLAGS_REGISTER_STUB(dwarf_num, str_name, generic_num) \
  {                                                                           \
    DEFINE_REG_NAME(dwarf_num), DEFINE_REG_NAME_STR(str_name),                \
    64, dwarf_num * 4, lldb::eEncodingVector, lldb::eFormatCString,           \
    { dwarf_num, dwarf_num, generic_num, dwarf_num, dwarf_num },              \
    nullptr, nullptr, nullptr,                                                \
  }

#define DEFINE_REGISTER_STUB(dwarf_num, str_name) \
  DEFINE_GENERIC_REGISTER_STUB(dwarf_num, str_name, LLDB_INVALID_REGNUM)

using namespace lldb;
using namespace lldb_private;

namespace {
namespace dwarf {
enum regnums {
  r0, r1, r2, r3, r4, r5, r6, r7,
  gpo, gpd, cfs, cfm,
  sp, pc, gpi, ra, flags
};

static const lldb_private::RegisterFlags g_ezh_flags_type("flags", 4, {
    {"EX", 15}, {"BS", 14}, {"NEX", 13}, {"NBS", 12},
    {"SNE", 11}, {"SPO", 10}, {"CZ", 9}, {"NC", 8},
    {"CA", 7}, {"ZB", 6}, {"AZ", 5}, {"NE", 4},
    {"PO", 3}, {"NZ", 2}, {"ZE", 1}, {"EU", 0}
});

static const std::array<RegisterInfo, 17> g_register_infos = { {
    DEFINE_GENERIC_REGISTER_STUB(r0, nullptr, LLDB_REGNUM_GENERIC_ARG1),
    DEFINE_GENERIC_REGISTER_STUB(r1, nullptr, LLDB_REGNUM_GENERIC_ARG2),
    DEFINE_GENERIC_REGISTER_STUB(r2, nullptr, LLDB_REGNUM_GENERIC_ARG3),
    DEFINE_GENERIC_REGISTER_STUB(r3, nullptr, LLDB_REGNUM_GENERIC_ARG4),
    DEFINE_REGISTER_STUB(r4, nullptr),
    DEFINE_REGISTER_STUB(r5, nullptr),
    DEFINE_REGISTER_STUB(r6, nullptr),
    DEFINE_REGISTER_STUB(r7, nullptr),
    DEFINE_REGISTER_STUB(gpo, nullptr),
    DEFINE_REGISTER_STUB(gpd, nullptr),
    DEFINE_REGISTER_STUB(cfs, nullptr),
    DEFINE_REGISTER_STUB(cfm, nullptr),
    DEFINE_GENERIC_REGISTER_STUB(sp, nullptr, LLDB_REGNUM_GENERIC_SP),
    DEFINE_GENERIC_REGISTER_STUB(pc, nullptr, LLDB_REGNUM_GENERIC_PC),
    DEFINE_REGISTER_STUB(gpi, nullptr),
    DEFINE_GENERIC_REGISTER_STUB(ra, nullptr, LLDB_REGNUM_GENERIC_RA),
    DEFINE_FLAGS_REGISTER_STUB(flags, nullptr, LLDB_REGNUM_GENERIC_FLAGS)} };

static uint32_t g_reg_nums[] = {
  0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16
};

static const RegisterSet g_reg_set = {
    "General Purpose Registers", "gpr", 17, g_reg_nums
};
} // namespace dwarf
} // namespace

RegisterContextEZH::RegisterContextEZH(Thread &thread,
                                       uint32_t concrete_frame_idx)
    : RegisterContext(thread, concrete_frame_idx) {}

size_t RegisterContextEZH::GetRegisterCount() {
  return dwarf::g_register_infos.size();
}

const RegisterInfo *RegisterContextEZH::GetRegisterInfoAtIndex(size_t reg) {
  if (reg < dwarf::g_register_infos.size())
    return &dwarf::g_register_infos[reg];
  return nullptr;
}

const RegisterSet *RegisterContextEZH::GetRegisterSet(size_t reg_set) {
  if (reg_set == 0)
    return &dwarf::g_reg_set;
  return nullptr;
}

bool RegisterContextEZH::ReadAllRegisterValuesBytes() {
  ProcessSP process_sp = m_thread.GetProcess();
  if (!process_sp)
    return false;

  if (!process_sp->GetTarget().GetExecutableModule())
    return false; // Block stack frame reads if no ELF is loaded!

  Status error;
  ProcessEZH *ezh_process = static_cast<ProcessEZH*>(process_sp.get());
  if (!ezh_process)
    return false;

  lldb::addr_t debug_frame_addr = ezh_process->GetDebugFrameAddr();
  if (debug_frame_addr == LLDB_INVALID_ADDRESS)
    return false;

  uint32_t sp_addr = 0;
  size_t bytes_read = 0;
  if (debug_frame_addr != LLDB_INVALID_ADDRESS) {
    // 1. Read active stack pointer directly from dynamic RAM global variable debug_frame!
    ezh_process->InvalidateMemoryCache();
    bytes_read = ezh_process->DoReadMemoryDirect(debug_frame_addr, &sp_addr, 4, error);
  }
  if (bytes_read != 4 || error.Fail() || sp_addr == 0) {
    if (error.Fail()) {
      char dbg_err[256];
      snprintf(dbg_err, sizeof(dbg_err), "RegisterContextEZH: Failed to read debug_frame from EZH RAM over JTAG: %s", error.AsCString());
      lldb_private::Debugger::ReportWarning(dbg_err, process_sp->GetTarget().GetDebugger().GetID());
    }
    return false;
  }

  // 3. Read 68-byte stack frame from target RAM into local cache (sp_addr - EZH_FRAME_SIZE)
  ezh_process->InvalidateMemoryCache();
  bytes_read = ezh_process->DoReadMemoryDirect(sp_addr - EZH_FRAME_SIZE, m_reg_values, EZH_FRAME_SIZE, error);
  if (bytes_read != EZH_FRAME_SIZE || error.Fail()) {
    if (error.Fail()) {
      char dbg_err[256];
      snprintf(dbg_err, sizeof(dbg_err), "RegisterContextEZH: Failed to read EZH stack frame from RAM over JTAG: %s", error.AsCString());
      lldb_private::Debugger::ReportWarning(dbg_err, process_sp->GetTarget().GetDebugger().GetID());
    }
    return false;
  }

  m_reg_values_valid = true;
  


  return true;
}

bool RegisterContextEZH::ReadRegister(const RegisterInfo *reg_info,
                                     RegisterValue &value) {
  if (!reg_info)
    return false;

  uint32_t reg_idx = reg_info->kinds[eRegisterKindLLDB];
  if (reg_idx >= EZH_NUM_REGS)
    return false;

  ProcessSP process_sp = m_thread.GetProcess();
  if (!process_sp)
    return false;

  Status error;
  ProcessEZH *ezh_process = static_cast<ProcessEZH*>(process_sp.get());
  
  // 1. Check Ignition bit in EZHB_CTRL
  uint32_t ctrl_val = 0;
  error = ezh_process->ReadEZHRegister(EZHB_CTRL_OFFSET, ctrl_val);
  if (error.Fail()) {
    char dbg_err[256];
    snprintf(dbg_err, sizeof(dbg_err), "RegisterContextEZH::ReadRegister: Failed to read EZH CTRL over JTAG: %s", error.AsCString());
    lldb_private::Debugger::ReportWarning(dbg_err, ezh_process->GetTarget().GetDebugger().GetID());
    return false;
  }

  bool ignition_set = ((ctrl_val & 1) != 0);

  if (!ignition_set) {
    // State 1: True Halted (Ignition = 0)
    if (reg_idx == EZH_REG_IDX_PC) {
      uint32_t boot_addr = 0;
      error = ezh_process->ReadEZHRegister(EZHB_BOOT_OFFSET, boot_addr);
      if (error.Fail()) {
        char dbg_err[256];
        snprintf(dbg_err, sizeof(dbg_err), "RegisterContextEZH::ReadRegister: Failed to read EZH BOOTADR over JTAG: %s", error.AsCString());
        lldb_private::Debugger::ReportWarning(dbg_err, ezh_process->GetTarget().GetDebugger().GetID());
        return false;
      }
      value.SetUInt32(boot_addr);
      return true;
    }
    value.SetUInt32(0);
    return true;
  }

  // 2. Check debug_frame to distinguish Executing vs Virtually Halted
  lldb::addr_t debug_frame_addr = ezh_process->GetDebugFrameAddr();
  if (debug_frame_addr == LLDB_INVALID_ADDRESS)
    return false;

  uint32_t sp_val = 0;
  ezh_process->InvalidateMemoryCache();
  ezh_process->DoReadMemoryDirect(debug_frame_addr, &sp_val, 4, error);

  if (sp_val == 0) {
    // State 2: Executing (Running) -> Registers are not accessible!
    value.SetUInt32(0);
    return true;
  }

  // State 3: Virtually Halted (Ignition = 1, debug_sp != 0)
  if (!m_reg_values_valid && !ReadAllRegisterValuesBytes())
    return false;

  uint32_t reg_val = m_reg_values[reg_idx];
  if (reg_idx == EZH_REG_IDX_PC) {
    if (reg_val >= 0xFFFFFFF0 && reg_val <= 0xFFFFFFFF && ezh_process) {
      uint32_t slot = reg_val - 0xFFFFFFF0;
      lldb::addr_t sw_bp_addr = ezh_process->GetActiveSoftwareBreakpointAddr(slot);
      if (sw_bp_addr != LLDB_INVALID_ADDRESS)
        reg_val = static_cast<uint32_t>(sw_bp_addr);
    } else if (reg_val == 0) {
      reg_val = m_reg_values[EZH_REG_IDX_RA];
    }
  }

  if (reg_idx == EZH_REG_IDX_FLAGS) {
    char flags_str[64];
    snprintf(flags_str, sizeof(flags_str), "0x%08x (", reg_val);
    bool first = true;
    for (const auto &field : dwarf::g_ezh_flags_type.GetFields()) {
      if (!field.GetName().empty() && field.GetName() != "EU" && field.GetValue(reg_val) != 0) {
        if (!first) strncat(flags_str, ", ", sizeof(flags_str) - strlen(flags_str) - 1);
        strncat(flags_str, field.GetName().c_str(), sizeof(flags_str) - strlen(flags_str) - 1);
        first = false;
      }
    }
    if (first) strncat(flags_str, "None", sizeof(flags_str) - strlen(flags_str) - 1);
    strncat(flags_str, ")", sizeof(flags_str) - strlen(flags_str) - 1);

    value.SetBytes(flags_str, strlen(flags_str) + 1, endian::InlHostByteOrder());
    return true;
  }

  value.SetUInt32(reg_val);
  return true;
}

bool RegisterContextEZH::WriteRegister(const RegisterInfo *reg_info,
                                      const RegisterValue &value) {
  Status error;

  if (!reg_info)
    return false;

  uint32_t reg_idx = reg_info->kinds[eRegisterKindLLDB];
  if (reg_idx >= EZH_NUM_REGS)
    return false;

  // flags is read-only
  if (reg_idx == EZH_REG_IDX_FLAGS)
    return false;

  ProcessSP process_sp = m_thread.GetProcess();
  if (!process_sp)
    return false;

  ProcessEZH *ezh_process = static_cast<ProcessEZH*>(process_sp.get());
  if (!ezh_process)
    return false;
  
  // 1. Check Ignition bit in EZHB_CTRL
  uint32_t ctrl_val = 0;
  error = ezh_process->ReadEZHRegister(EZHB_CTRL_OFFSET, ctrl_val);
  if (error.Fail()) {
    char dbg_err[256];
    snprintf(dbg_err, sizeof(dbg_err), "RegisterContextEZH::WriteRegister: Failed to read EZH CTRL over JTAG: %s", error.AsCString());
    lldb_private::Debugger::ReportWarning(dbg_err, ezh_process->GetTarget().GetDebugger().GetID());
    return false;
  }
  bool ignition_set = ((ctrl_val & 1) != 0);
  if (!ignition_set) {
    // State 1: True Halted (Ignition = 0)
    if (reg_idx != EZH_REG_IDX_PC)
      return false;

    uint32_t val = value.GetAsUInt32();
    error = ezh_process->WriteEZHRegister(EZHB_BOOT_OFFSET, val);
    if (error.Fail()) {
      char dbg_err[256];
      snprintf(dbg_err, sizeof(dbg_err), "RegisterContextEZH::WriteRegister: Failed to write EZH BOOTADR over JTAG: %s", error.AsCString());
      lldb_private::Debugger::ReportWarning(dbg_err, ezh_process->GetTarget().GetDebugger().GetID());
      return false;
    }
    return true;
  }



  // 2. Check debug_frame
  lldb::addr_t debug_frame_addr = ezh_process->GetDebugFrameAddr();
  if (debug_frame_addr == LLDB_INVALID_ADDRESS)
    return false;

  uint32_t sp_val = 0;
  ezh_process->DoReadMemoryDirect(debug_frame_addr, &sp_val, 4, error);

  if (sp_val == 0)
    // State 2: Executing -> Cannot write registers!
    return false;

  // State 3: Virtually Halted (Ignition = 1, debug_sp != 0)
  uint32_t ezh_sp_val = sp_val;

  uint32_t val = value.GetAsUInt32();
  m_reg_values[reg_idx] = val;

  // Write value back to stack RAM
  size_t bytes_written = process_sp->WriteMemory(ezh_sp_val - EZH_FRAME_SIZE + (reg_idx * 4), &val, 4, error);
  return bytes_written == 4 && error.Success();
}

bool RegisterContextEZH::ReadAllRegisterValues(WritableDataBufferSP &data_sp) {
  if (!m_reg_values_valid && !ReadAllRegisterValuesBytes())
    return false;
  data_sp = std::make_shared<DataBufferHeap>(m_reg_values, 68);
  return true;
}

bool RegisterContextEZH::WriteAllRegisterValues(const DataBufferSP &data_sp) {
  if (!data_sp || data_sp->GetByteSize() != 68)
    return false;

  ProcessSP process_sp = m_thread.GetProcess();
  if (!process_sp)
    return false;

  Status error;
  uint32_t ezh_sp_val = 0;
  ProcessEZH *ezh_process = static_cast<ProcessEZH*>(process_sp.get());

  lldb::addr_t debug_frame_addr = ezh_process->GetDebugFrameAddr();
  if (debug_frame_addr == LLDB_INVALID_ADDRESS)
    return false;
  // Read stack frame base from debug_frame RAM variable instead of broken EZHB_SP hardware register!
  size_t bytes_read = ezh_process->DoReadMemoryDirect(debug_frame_addr, &ezh_sp_val, 4, error);
  if (bytes_read != 4 || error.Fail() || ezh_sp_val == 0)
    return false;

  memcpy(m_reg_values, data_sp->GetBytes(), EZH_FRAME_SIZE);

  // Write entire stack frame block back to RAM (ezh_sp_val - EZH_FRAME_SIZE)
  size_t bytes_written = process_sp->WriteMemory(ezh_sp_val - EZH_FRAME_SIZE, m_reg_values, EZH_FRAME_SIZE, error);
  return bytes_written == EZH_FRAME_SIZE && error.Success();
}

uint32_t RegisterContextEZH::ConvertRegisterKindToRegisterNumber(RegisterKind kind,
                                                                 uint32_t num) {
  if (kind == eRegisterKindGeneric) {
    switch (num) {
    case LLDB_REGNUM_GENERIC_PC:    return EZH_REG_IDX_PC; // pc
    case LLDB_REGNUM_GENERIC_SP:    return EZH_REG_IDX_SP; // sp
    case LLDB_REGNUM_GENERIC_RA:    return EZH_REG_IDX_RA; // ra
    case LLDB_REGNUM_GENERIC_ARG1:  return EZH_REG_IDX_R0;  // r0
    case LLDB_REGNUM_GENERIC_ARG2:  return EZH_REG_IDX_R1;  // r1
    case LLDB_REGNUM_GENERIC_ARG3:  return EZH_REG_IDX_R2;  // r2
    case LLDB_REGNUM_GENERIC_ARG4:  return EZH_REG_IDX_R3;  // r3
    case LLDB_REGNUM_GENERIC_FLAGS: return EZH_REG_IDX_FLAGS; // flags
    default:                        return LLDB_INVALID_REGNUM;
    }
  }

  if (kind == eRegisterKindLLDB && num < EZH_NUM_REGS)
    return num;

  if (kind == eRegisterKindDWARF) {
    if (num < EZH_REG_IDX_SP)
      return num; // r0-r7, gpo, gpd, cfs, cfm
    if (num == EZH_REG_IDX_SP)
      return EZH_REG_IDX_SP;  // sp
    if (num == EZH_REG_IDX_PC)
      return EZH_REG_IDX_PC;  // pc
    if (num == 14)
      return 14;  // gpi
    if (num == 15)
      return EZH_REG_IDX_RA;  // DWARF 15 (ra) -> LLDB 15 (ra)
    if (num == 16)
      return EZH_REG_IDX_FLAGS; // DWARF 16 (flags) -> LLDB 16 (flags)
  }
  return LLDB_INVALID_REGNUM;
}
