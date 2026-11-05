//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <cerrno>
#include "lldb/Register/Arch/lldb-s390x-register-enums.h"
#include <cstdint>
#include <cstring>

#include "lldb/Target/Process.h"
#include "lldb/Target/Target.h"
#include "lldb/Target/Thread.h"
#include "lldb/Utility/DataBufferHeap.h"
#include "lldb/Utility/DataExtractor.h"
#include "lldb/Utility/Endian.h"
#include "lldb/Utility/RegisterValue.h"
#include "lldb/Utility/Scalar.h"
#include "llvm/Support/Compiler.h"

#include "RegisterContextCore_s390x.h"

namespace reg_ids = lldb_private::registers::s390x::id;
using namespace lldb_private;
using namespace lldb;

// s390x 64-bit general purpose registers.
static const uint32_t g_gpr_regnums_s390x[] = {
    reg_ids::r0,        reg_ids::r1,    reg_ids::r2,    reg_ids::r3,
    reg_ids::r4,        reg_ids::r5,    reg_ids::r6,    reg_ids::r7,
    reg_ids::r8,        reg_ids::r9,    reg_ids::r10,   reg_ids::r11,
    reg_ids::r12,       reg_ids::r13,   reg_ids::r14,   reg_ids::r15,
    reg_ids::acr0,      reg_ids::acr1,  reg_ids::acr2,  reg_ids::acr3,
    reg_ids::acr4,      reg_ids::acr5,  reg_ids::acr6,  reg_ids::acr7,
    reg_ids::acr8,      reg_ids::acr9,  reg_ids::acr10, reg_ids::acr11,
    reg_ids::acr12,     reg_ids::acr13, reg_ids::acr14, reg_ids::acr15,
    reg_ids::pswm,      reg_ids::pswa,
    LLDB_INVALID_REGNUM // register sets need to end with this flag
};
static_assert((sizeof(g_gpr_regnums_s390x) / sizeof(g_gpr_regnums_s390x[0])) -
                      1 ==
                  reg_ids::k_num_gpr_registers,
              "g_gpr_regnums_s390x has wrong number of register infos");

// s390x 64-bit floating point registers.
static const uint32_t g_fpu_regnums_s390x[] = {
    reg_ids::f0,        reg_ids::f1,  reg_ids::f2,  reg_ids::f3,  reg_ids::f4,
    reg_ids::f5,        reg_ids::f6,  reg_ids::f7,  reg_ids::f8,  reg_ids::f9,
    reg_ids::f10,       reg_ids::f11, reg_ids::f12, reg_ids::f13, reg_ids::f14,
    reg_ids::f15,       reg_ids::fpc,
    LLDB_INVALID_REGNUM // register sets need to end with this flag
};
static_assert((sizeof(g_fpu_regnums_s390x) / sizeof(g_fpu_regnums_s390x[0])) -
                      1 ==
                  reg_ids::k_num_fpr_registers,
              "g_fpu_regnums_s390x has wrong number of register infos");

// Number of register sets provided by this context.
enum { k_num_register_sets = 2 };

// Register sets for s390x 64-bit.
static const RegisterSet g_reg_sets_s390x[k_num_register_sets] = {
    {"General Purpose Registers", "gpr", reg_ids::k_num_gpr_registers,
     g_gpr_regnums_s390x},
    {"Floating Point Registers", "fpr", reg_ids::k_num_fpr_registers,
     g_fpu_regnums_s390x},
};

bool RegisterContextCore_s390x::IsGPR(unsigned reg) {
  return reg <= m_reg_info.last_gpr; // GPRs come first.
}

bool RegisterContextCore_s390x::IsFPR(unsigned reg) {
  return (m_reg_info.first_fpr <= reg && reg <= m_reg_info.last_fpr);
}

RegisterContextCore_s390x::RegisterContextCore_s390x(
    Thread &thread, uint32_t concrete_frame_idx,
    RegisterInfoInterface *register_info)
    : RegisterContext(thread, concrete_frame_idx) {
  m_register_info_up.reset(register_info);

  switch (register_info->GetTargetArchitecture().GetMachine()) {
  case llvm::Triple::systemz:
    m_reg_info.num_registers = reg_ids::k_num_registers;
    m_reg_info.num_gpr_registers = reg_ids::k_num_gpr_registers;
    m_reg_info.num_fpr_registers = reg_ids::k_num_fpr_registers;
    m_reg_info.last_gpr = reg_ids::k_last_gpr;
    m_reg_info.first_fpr = reg_ids::k_first_fpr;
    m_reg_info.last_fpr = reg_ids::k_last_fpr;
    break;
  default:
    assert(false && "Unhandled target architecture.");
    break;
  }
}

RegisterContextCore_s390x::~RegisterContextCore_s390x() = default;

void RegisterContextCore_s390x::Invalidate() {}

void RegisterContextCore_s390x::InvalidateAllRegisters() {}

const RegisterInfo *RegisterContextCore_s390x::GetRegisterInfo() {
  return m_register_info_up->GetRegisterInfo();
}

const RegisterInfo *
RegisterContextCore_s390x::GetRegisterInfoAtIndex(size_t reg) {
  if (reg < m_reg_info.num_registers)
    return &GetRegisterInfo()[reg];
  else
    return nullptr;
}

size_t RegisterContextCore_s390x::GetRegisterCount() {
  return m_reg_info.num_registers;
}

unsigned RegisterContextCore_s390x::GetRegisterOffset(unsigned reg) {
  assert(reg < m_reg_info.num_registers && "Invalid register number.");
  return GetRegisterInfo()[reg].byte_offset;
}

unsigned RegisterContextCore_s390x::GetRegisterSize(unsigned reg) {
  assert(reg < m_reg_info.num_registers && "Invalid register number.");
  return GetRegisterInfo()[reg].byte_size;
}

const char *RegisterContextCore_s390x::GetRegisterName(unsigned reg) {
  assert(reg < m_reg_info.num_registers && "Invalid register offset.");
  return GetRegisterInfo()[reg].name;
}

bool RegisterContextCore_s390x::IsRegisterSetAvailable(size_t set_index) {
  return set_index < k_num_register_sets;
}

size_t RegisterContextCore_s390x::GetRegisterSetCount() {
  size_t sets = 0;
  for (size_t set = 0; set < k_num_register_sets; ++set) {
    if (IsRegisterSetAvailable(set))
      ++sets;
  }

  return sets;
}

const RegisterSet *RegisterContextCore_s390x::GetRegisterSet(size_t set) {
  if (IsRegisterSetAvailable(set)) {
    switch (m_register_info_up->GetTargetArchitecture().GetMachine()) {
    case llvm::Triple::systemz:
      return &g_reg_sets_s390x[set];
    default:
      assert(false && "Unhandled target architecture.");
      return nullptr;
    }
  }
  return nullptr;
}
