//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "RegisterContextMockAccelerator.h"

#include "lldb/Utility/DataBufferHeap.h"
#include "lldb/Utility/RegisterValue.h"

using namespace lldb;
using namespace lldb_private;
using namespace lldb_private::lldb_server;

// LLDB register numbers must start at 0 and be contiguous. This minimal set is
// just enough for the mock accelerator process to be debugged over GDB remote.
enum LLDBRegNum : uint32_t {
  LLDB_R0 = 0,
  LLDB_R1,
  LLDB_SP,
  LLDB_FP,
  LLDB_PC,
  LLDB_Flags,
  kNumRegs
};

#define DEFINE_REG(name, idx, generic)                                         \
  {name,                                                                       \
   nullptr,                                                                    \
   sizeof(uint64_t),                                                           \
   idx * sizeof(uint64_t),                                                     \
   eEncodingUint,                                                              \
   eFormatHex,                                                                 \
   {LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, generic, LLDB_INVALID_REGNUM,    \
    idx},                                                                      \
   nullptr,                                                                    \
   nullptr,                                                                    \
   nullptr}

static const RegisterInfo g_register_infos[] = {
    DEFINE_REG("r0", LLDB_R0, LLDB_INVALID_REGNUM),
    DEFINE_REG("r1", LLDB_R1, LLDB_INVALID_REGNUM),
    DEFINE_REG("sp", LLDB_SP, LLDB_REGNUM_GENERIC_SP),
    DEFINE_REG("fp", LLDB_FP, LLDB_REGNUM_GENERIC_FP),
    DEFINE_REG("pc", LLDB_PC, LLDB_REGNUM_GENERIC_PC),
    DEFINE_REG("flags", LLDB_Flags, LLDB_INVALID_REGNUM),
};

// The set's member register numbers. This must be a valid array (not null):
// the stop-reply path reads it to expedite the set's registers.
static const uint32_t g_register_nums[] = {LLDB_R0, LLDB_R1, LLDB_SP,
                                           LLDB_FP, LLDB_PC, LLDB_Flags};

static const RegisterSet g_register_set = {"General Purpose Registers", "gpr",
                                           kNumRegs, g_register_nums};

RegisterContextMockAccelerator::RegisterContextMockAccelerator(
    NativeThreadProtocol &native_thread)
    : NativeRegisterContext(native_thread) {
  // Give each register a distinct, constant value so reads are deterministic.
  for (uint32_t i = 0; i < kNumRegs; ++i)
    m_regs[i] = 0x1000 + i;
}

uint32_t RegisterContextMockAccelerator::GetRegisterCount() const {
  return kNumRegs;
}

uint32_t RegisterContextMockAccelerator::GetUserRegisterCount() const {
  return kNumRegs;
}

const RegisterInfo *
RegisterContextMockAccelerator::GetRegisterInfoAtIndex(uint32_t reg) const {
  if (reg < kNumRegs)
    return &g_register_infos[reg];
  return nullptr;
}

uint32_t RegisterContextMockAccelerator::GetRegisterSetCount() const {
  return 1;
}

const RegisterSet *
RegisterContextMockAccelerator::GetRegisterSet(uint32_t set_index) const {
  if (set_index == 0)
    return &g_register_set;
  return nullptr;
}

Status
RegisterContextMockAccelerator::ReadRegister(const RegisterInfo *reg_info,
                                             RegisterValue &reg_value) {
  if (!reg_info)
    return Status::FromErrorString("invalid register info");
  const uint32_t reg = reg_info->kinds[eRegisterKindLLDB];
  if (reg >= kNumRegs)
    return Status::FromErrorString("invalid register number");
  reg_value.SetUInt64(m_regs[reg]);
  return Status();
}

Status
RegisterContextMockAccelerator::WriteRegister(const RegisterInfo *reg_info,
                                              const RegisterValue &reg_value) {
  if (!reg_info)
    return Status::FromErrorString("invalid register info");
  const uint32_t reg = reg_info->kinds[eRegisterKindLLDB];
  if (reg >= kNumRegs)
    return Status::FromErrorString("invalid register number");
  m_regs[reg] = reg_value.GetAsUInt64();
  return Status();
}

Status RegisterContextMockAccelerator::ReadAllRegisterValues(
    lldb::WritableDataBufferSP &data_sp) {
  data_sp = std::make_shared<DataBufferHeap>(
      reinterpret_cast<const uint8_t *>(m_regs.data()),
      m_regs.size() * sizeof(uint64_t));
  return Status();
}

Status RegisterContextMockAccelerator::WriteAllRegisterValues(
    const lldb::DataBufferSP &data_sp) {
  if (!data_sp || data_sp->GetByteSize() != m_regs.size() * sizeof(uint64_t))
    return Status::FromErrorString("invalid register data");
  ::memcpy(m_regs.data(), data_sp->GetBytes(),
           m_regs.size() * sizeof(uint64_t));
  return Status();
}
