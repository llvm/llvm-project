//===-- RegisterContextPOSIX_mips64.cpp -------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include <cstring>
#include <errno.h>
#include <stdint.h>

#include "lldb/Core/DataBufferHeap.h"
#include "lldb/Core/DataExtractor.h"
#include "lldb/Core/RegisterValue.h"
#include "lldb/Core/Scalar.h"
#include "lldb/Host/Endian.h"
#include "lldb/Target/Target.h"
#include "lldb/Target/Thread.h"
#include "llvm/Support/Compiler.h"

#include "Plugins/Process/elf-core/ProcessElfCore.h"
#include "RegisterContextPOSIX_mips64.h"

using namespace lldb_private;
using namespace lldb;

static const uint32_t g_gpr_regnums[] = {
    gpr_zero_mips64,  gpr_r1_mips64,    gpr_r2_mips64,    gpr_r3_mips64,
    gpr_r4_mips64,    gpr_r5_mips64,    gpr_r6_mips64,    gpr_r7_mips64,
    gpr_r8_mips64,    gpr_r9_mips64,    gpr_r10_mips64,   gpr_r11_mips64,
    gpr_r12_mips64,   gpr_r13_mips64,   gpr_r14_mips64,   gpr_r15_mips64,
    gpr_r16_mips64,   gpr_r17_mips64,   gpr_r18_mips64,   gpr_r19_mips64,
    gpr_r20_mips64,   gpr_r21_mips64,   gpr_r22_mips64,   gpr_r23_mips64,
    gpr_r24_mips64,   gpr_r25_mips64,   gpr_r26_mips64,   gpr_r27_mips64,
    gpr_gp_mips64,    gpr_sp_mips64,    gpr_r30_mips64,   gpr_ra_mips64,
    gpr_sr_mips64,    gpr_mullo_mips64, gpr_mulhi_mips64, gpr_badvaddr_mips64,
    gpr_cause_mips64, gpr_pc_mips64,    gpr_ic_mips64,    gpr_dummy_mips64};

// Number of register sets provided by this context.
enum { k_num_register_sets = 1 };

static const RegisterSet g_reg_sets_mips64[k_num_register_sets] = {
    {"General Purpose Registers", "gpr", k_num_gpr_registers_mips64,
     g_gpr_regnums},
};

bool RegisterContextPOSIX_mips64::IsGPR(unsigned reg) {
  return reg <= k_num_gpr_registers_mips64; // GPR's come first.
}

bool RegisterContextPOSIX_mips64::IsFPR(unsigned reg) {
  // XXX
  return false;
}

RegisterContextPOSIX_mips64::RegisterContextPOSIX_mips64(
    Thread &thread, uint32_t concrete_frame_idx,
    RegisterInfoInterface *register_info)
    : RegisterContext(thread, concrete_frame_idx) {
  m_register_info_ap.reset(register_info);

  // elf-core yet to support ReadFPR()
  ProcessSP base = CalculateProcess();
  if (base.get()->GetPluginName() == ProcessElfCore::GetPluginNameStatic())
    return;
}

RegisterContextPOSIX_mips64::~RegisterContextPOSIX_mips64() {}

void RegisterContextPOSIX_mips64::Invalidate() {}

void RegisterContextPOSIX_mips64::InvalidateAllRegisters() {}

unsigned RegisterContextPOSIX_mips64::GetRegisterOffset(unsigned reg) {
  assert(reg < k_num_registers_mips64 && "Invalid register number.");
  return GetRegisterInfo()[reg].byte_offset;
}

unsigned RegisterContextPOSIX_mips64::GetRegisterSize(unsigned reg) {
  assert(reg < k_num_registers_mips64 && "Invalid register number.");
  return GetRegisterInfo()[reg].byte_size;
}

size_t RegisterContextPOSIX_mips64::GetRegisterCount() {
  size_t num_registers = k_num_registers_mips64;
  return num_registers;
}

size_t RegisterContextPOSIX_mips64::GetGPRSize() {
  return m_register_info_ap->GetGPRSize();
}

const RegisterInfo *RegisterContextPOSIX_mips64::GetRegisterInfo() {
  // Commonly, this method is overridden and g_register_infos is copied and
  // specialized.
  // So, use GetRegisterInfo() rather than g_register_infos in this scope.
  return m_register_info_ap->GetRegisterInfo();
}

const RegisterInfo *
RegisterContextPOSIX_mips64::GetRegisterInfoAtIndex(size_t reg) {
  if (reg < k_num_registers_mips64)
    return &GetRegisterInfo()[reg];
  else
    return NULL;
}

size_t RegisterContextPOSIX_mips64::GetRegisterSetCount() {
  size_t sets = 0;
  for (size_t set = 0; set < k_num_register_sets; ++set) {
    if (IsRegisterSetAvailable(set))
      ++sets;
  }

  return sets;
}

const RegisterSet *RegisterContextPOSIX_mips64::GetRegisterSet(size_t set) {
  if (IsRegisterSetAvailable(set))
    return &g_reg_sets_mips64[set];
  else
    return NULL;
}

const char *RegisterContextPOSIX_mips64::GetRegisterName(unsigned reg) {
  assert(reg < k_num_registers_mips64 && "Invalid register offset.");
  return GetRegisterInfo()[reg].name;
}

lldb::ByteOrder RegisterContextPOSIX_mips64::GetByteOrder() {
  // Get the target process whose privileged thread was used for the register
  // read.
  lldb::ByteOrder byte_order = eByteOrderInvalid;
  Process *process = CalculateProcess().get();

  if (process)
    byte_order = process->GetByteOrder();
  return byte_order;
}

bool RegisterContextPOSIX_mips64::IsRegisterSetAvailable(size_t set_index) {
  size_t num_sets = k_num_register_sets;

  return (set_index < num_sets);
}

// Used when parsing DWARF and EH frame information and any other
// object file sections that contain register numbers in them.
uint32_t RegisterContextPOSIX_mips64::ConvertRegisterKindToRegisterNumber(
    lldb::RegisterKind kind, uint32_t num) {
  const uint32_t num_regs = GetRegisterCount();

  assert(kind < kNumRegisterKinds);
  for (uint32_t reg_idx = 0; reg_idx < num_regs; ++reg_idx) {
    const RegisterInfo *reg_info = GetRegisterInfoAtIndex(reg_idx);

    if (reg_info->kinds[kind] == num)
      return reg_idx;
  }

  return LLDB_INVALID_REGNUM;
}
