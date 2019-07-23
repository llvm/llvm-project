//===-- RegisterContextDpu.cpp --------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "RegisterContextDpu.h"

#include "Plugins/Process/Dpu/ProcessDpu.h"
#include "Plugins/Process/Dpu/RegisterInfo_dpu.h"
#include "Plugins/Process/POSIX/ProcessPOSIXLog.h"
#include "lldb/Host/common/NativeProcessProtocol.h"
#include "lldb/Host/common/NativeThreadProtocol.h"
#include "lldb/Utility/DataBufferHeap.h"
#include "lldb/Utility/Log.h"
#include "lldb/Utility/RegisterValue.h"
#include "lldb/Utility/Status.h"

#include <elf.h>

using namespace lldb;
using namespace lldb_private;
using namespace lldb_private::process_dpu;

namespace {

constexpr lldb::addr_t k_dpu_iram_base = 0x80000000;

} // end of anonymous namespace

// -----------------------------------------------------------------------------
// DPU general purpose registers.
static const uint32_t g_regnums_dpu[] = {
    r0_dpu,
    r1_dpu,
    r2_dpu,
    r3_dpu,
    r4_dpu,
    r5_dpu,
    r6_dpu,
    r7_dpu,
    r8_dpu,
    r9_dpu,
    r10_dpu,
    r11_dpu,
    r12_dpu,
    r13_dpu,
    r14_dpu,
    r15_dpu,
    r16_dpu,
    r17_dpu,
    r18_dpu,
    r19_dpu,
    r20_dpu,
    r21_dpu,
    r22_dpu,
    r23_dpu,
    pc_dpu,
    zf_dpu,
    cf_dpu,
    LLDB_INVALID_REGNUM // register sets need to end with this flag
};
static_assert(((sizeof g_regnums_dpu / sizeof g_regnums_dpu[0]) - 1) ==
                  k_num_registers_dpu,
              "g_regnums_dpu has wrong number of register infos");

static const RegisterSet g_reg_sets_dpu = {"General Purpose Registers", "gpr",
                                           k_num_registers_dpu, g_regnums_dpu};

constexpr size_t k_dpu_reg_context_size =
    k_num_registers_dpu * sizeof(uint32_t);

RegisterContextDpu::RegisterContextDpu(ThreadDpu &thread, ProcessDpu &process)
    : NativeRegisterContextRegisterInfo(thread, new RegisterInfo_dpu()) {
  process.GetThreadContext(thread.GetIndex(), m_context_reg, m_context_pc,
                           m_context_zf, m_context_cf);
}

uint32_t RegisterContextDpu::GetRegisterSetCount() const { return 1; }

uint32_t RegisterContextDpu::GetUserRegisterCount() const {
  return k_num_registers_dpu;
}

const RegisterSet *
RegisterContextDpu::GetRegisterSet(uint32_t set_index) const {
  if (set_index) // single register set
    return nullptr;

  return &g_reg_sets_dpu;
}

Status RegisterContextDpu::ReadRegister(const RegisterInfo *info,
                                        RegisterValue &value) {
  if (!info)
    return Status("RegisterInfo is NULL");

  const uint32_t reg = info->kinds[lldb::eRegisterKindLLDB];

  if (reg == pc_dpu)
    value.SetUInt32(*m_context_pc * 8 /*sizeof(iram_instruction_t)*/ +
                    k_dpu_iram_base);
  else if (reg == zf_dpu)
    value.SetUInt32(*m_context_zf ? 1 : 0);
  else if (reg == cf_dpu)
    value.SetUInt32(*m_context_cf ? 1 : 0);
  else
    value.SetUInt32(m_context_reg[reg]);

  return Status();
}

Status RegisterContextDpu::WriteRegister(const RegisterInfo *info,
                                         const RegisterValue &value) {
  if (!info)
    return Status("RegisterInfo is NULL");

  const uint32_t reg = info->kinds[lldb::eRegisterKindLLDB];

  if (reg == pc_dpu)
    *m_context_pc = (value.GetAsUInt32() - k_dpu_iram_base) /
                    8 /*sizeof(iram_instruction_t)*/;
  else if (reg == zf_dpu)
    *m_context_zf = (value.GetAsUInt32() == 1);
  else if (reg == cf_dpu)
    *m_context_cf = (value.GetAsUInt32() == 1);
  else
    m_context_reg[reg] = value.GetAsUInt32();

  return Status();
}

Status RegisterContextDpu::ReadAllRegisterValues(lldb::DataBufferSP &data_sp) {
  Status error;

  data_sp.reset(new DataBufferHeap(k_dpu_reg_context_size, 0));
  if (!data_sp)
    return Status("failed to allocate DataBufferHeap instance");
  uint8_t *dst = data_sp->GetBytes();
  if (dst == nullptr)
    return Status("invalid DataBufferHeap instance");

  uint32_t pc =
      *m_context_pc * 8 /*sizeof(iram_instruction_t)*/ + k_dpu_iram_base;
  uint32_t zf = m_context_zf ? 1 : 0;
  uint32_t cf = m_context_cf ? 1 : 0;

  ::memcpy(dst, m_context_reg, k_dpu_reg_context_size - sizeof(pc) - sizeof(zf) - sizeof(cf));
  ::memcpy(dst + k_dpu_reg_context_size - sizeof(pc) - sizeof(zf) - sizeof(cf), &pc, sizeof(pc));
  ::memcpy(dst + k_dpu_reg_context_size - sizeof(zf) - sizeof(cf), &zf, sizeof(zf));
  ::memcpy(dst + k_dpu_reg_context_size - sizeof(cf), &cf, sizeof(cf));

  return error;
}

Status
RegisterContextDpu::WriteAllRegisterValues(const lldb::DataBufferSP &data_sp) {
  Status error;

  if (!data_sp)
    return Status("invalid data_sp provided");
  uint8_t *src = data_sp->GetBytes();
  if (src == nullptr)
    return Status("invalid data_sp provided");

  if (data_sp->GetByteSize() != k_dpu_reg_context_size) {
    error.SetErrorStringWithFormat(
        "RegisterContextDpu::%s data_sp contained mismatched "
        "data size, expected %" PRIu64 ", actual %" PRIu64,
        __FUNCTION__, (uint64_t)k_dpu_reg_context_size, data_sp->GetByteSize());
    return error;
  }

  uint32_t pc;
  uint32_t zf, cf;

  ::memcpy(m_context_reg, src, k_dpu_reg_context_size - sizeof(pc) - sizeof(zf) - sizeof(cf));
  ::memcpy(&pc, src + k_dpu_reg_context_size - sizeof(pc) - sizeof(zf) - sizeof(cf), sizeof(pc));
  ::memcpy(&zf, src + k_dpu_reg_context_size - sizeof(zf) - sizeof(cf), sizeof(zf));
  ::memcpy(&cf, src + k_dpu_reg_context_size - sizeof(cf), sizeof(cf));

  *m_context_pc = (pc - k_dpu_iram_base) / 8 /*sizeof(iram_instruction_t)*/;
  *m_context_zf = zf == 1;
  *m_context_cf = cf == 1;

  return error;
}
