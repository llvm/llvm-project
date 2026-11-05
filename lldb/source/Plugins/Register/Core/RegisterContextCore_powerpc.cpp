//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <cerrno>
#include "lldb/Register/Arch/lldb-powerpc-register-enums.h"
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

#include "RegisterContextCore_powerpc.h"


namespace reg_ids = lldb_private::registers::powerpc::id;
using namespace lldb_private;
using namespace lldb;

static const uint32_t g_gpr_regnums[] = {
    reg_ids::gpr_r0,  reg_ids::gpr_r1,  reg_ids::gpr_r2,  reg_ids::gpr_r3,
    reg_ids::gpr_r4,  reg_ids::gpr_r5,  reg_ids::gpr_r6,  reg_ids::gpr_r7,
    reg_ids::gpr_r8,  reg_ids::gpr_r9,  reg_ids::gpr_r10, reg_ids::gpr_r11,
    reg_ids::gpr_r12, reg_ids::gpr_r13, reg_ids::gpr_r14, reg_ids::gpr_r15,
    reg_ids::gpr_r16, reg_ids::gpr_r17, reg_ids::gpr_r18, reg_ids::gpr_r19,
    reg_ids::gpr_r20, reg_ids::gpr_r21, reg_ids::gpr_r22, reg_ids::gpr_r23,
    reg_ids::gpr_r24, reg_ids::gpr_r25, reg_ids::gpr_r26, reg_ids::gpr_r27,
    reg_ids::gpr_r28, reg_ids::gpr_r29, reg_ids::gpr_r30, reg_ids::gpr_r31,
    reg_ids::gpr_lr,  reg_ids::gpr_cr,  reg_ids::gpr_xer, reg_ids::gpr_ctr,
    reg_ids::gpr_pc,
};

static const uint32_t g_fpr_regnums[] = {
    reg_ids::fpr_f0,    reg_ids::fpr_f1,  reg_ids::fpr_f2,  reg_ids::fpr_f3,
    reg_ids::fpr_f4,    reg_ids::fpr_f5,  reg_ids::fpr_f6,  reg_ids::fpr_f7,
    reg_ids::fpr_f8,    reg_ids::fpr_f9,  reg_ids::fpr_f10, reg_ids::fpr_f11,
    reg_ids::fpr_f12,   reg_ids::fpr_f13, reg_ids::fpr_f14, reg_ids::fpr_f15,
    reg_ids::fpr_f16,   reg_ids::fpr_f17, reg_ids::fpr_f18, reg_ids::fpr_f19,
    reg_ids::fpr_f20,   reg_ids::fpr_f21, reg_ids::fpr_f22, reg_ids::fpr_f23,
    reg_ids::fpr_f24,   reg_ids::fpr_f25, reg_ids::fpr_f26, reg_ids::fpr_f27,
    reg_ids::fpr_f28,   reg_ids::fpr_f29, reg_ids::fpr_f30, reg_ids::fpr_f31,
    reg_ids::fpr_fpscr,
};

static const uint32_t g_vmx_regnums[] = {
    reg_ids::vmx_v0,     reg_ids::vmx_v1,   reg_ids::vmx_v2,  reg_ids::vmx_v3,
    reg_ids::vmx_v4,     reg_ids::vmx_v5,   reg_ids::vmx_v6,  reg_ids::vmx_v7,
    reg_ids::vmx_v8,     reg_ids::vmx_v9,   reg_ids::vmx_v10, reg_ids::vmx_v11,
    reg_ids::vmx_v12,    reg_ids::vmx_v13,  reg_ids::vmx_v14, reg_ids::vmx_v15,
    reg_ids::vmx_v16,    reg_ids::vmx_v17,  reg_ids::vmx_v18, reg_ids::vmx_v19,
    reg_ids::vmx_v20,    reg_ids::vmx_v21,  reg_ids::vmx_v22, reg_ids::vmx_v23,
    reg_ids::vmx_v24,    reg_ids::vmx_v25,  reg_ids::vmx_v26, reg_ids::vmx_v27,
    reg_ids::vmx_v28,    reg_ids::vmx_v29,  reg_ids::vmx_v30, reg_ids::vmx_v31,
    reg_ids::vmx_vrsave, reg_ids::vmx_vscr,
};

// Number of register sets provided by this context.
enum { k_num_register_sets = 3 };

static const RegisterSet g_reg_sets_powerpc[k_num_register_sets] = {
    {"General Purpose Registers", "gpr", reg_ids::k_num_gpr_registers,
     g_gpr_regnums},
    {"Floating Point Registers", "fpr", reg_ids::k_num_fpr_registers,
     g_fpr_regnums},
    {"Altivec/VMX Registers", "vmx", reg_ids::k_num_vmx_registers,
     g_vmx_regnums},
};

static_assert(reg_ids::k_first_gpr == 0,
              "GPRs must index starting at 0, or fix IsGPR()");
bool RegisterContextCore_powerpc::IsGPR(unsigned reg) {
  return (reg <= reg_ids::k_last_gpr); // GPR's come first.
}

bool RegisterContextCore_powerpc::IsFPR(unsigned reg) {
  return (reg >= reg_ids::k_first_fpr) && (reg <= reg_ids::k_last_fpr);
}

bool RegisterContextCore_powerpc::IsVMX(unsigned reg) {
  return (reg >= reg_ids::k_first_vmx) && (reg <= reg_ids::k_last_vmx);
}

RegisterContextCore_powerpc::RegisterContextCore_powerpc(
    Thread &thread, uint32_t concrete_frame_idx,
    RegisterInfoInterface *register_info)
    : RegisterContext(thread, concrete_frame_idx) {
  m_register_info_up.reset(register_info);
}

RegisterContextCore_powerpc::~RegisterContextCore_powerpc() = default;

void RegisterContextCore_powerpc::Invalidate() {}

void RegisterContextCore_powerpc::InvalidateAllRegisters() {}

unsigned RegisterContextCore_powerpc::GetRegisterOffset(unsigned reg) {
  assert(reg < reg_ids::k_num_registers && "Invalid register number.");
  return GetRegisterInfo()[reg].byte_offset;
}

unsigned RegisterContextCore_powerpc::GetRegisterSize(unsigned reg) {
  assert(reg < reg_ids::k_num_registers && "Invalid register number.");
  return GetRegisterInfo()[reg].byte_size;
}

size_t RegisterContextCore_powerpc::GetRegisterCount() {
  size_t num_registers = reg_ids::k_num_registers;
  return num_registers;
}

size_t RegisterContextCore_powerpc::GetGPRSize() {
  return m_register_info_up->GetGPRSize();
}

const RegisterInfo *RegisterContextCore_powerpc::GetRegisterInfo() {
  // Commonly, this method is overridden and g_register_infos is copied and
  // specialized. So, use GetRegisterInfo() rather than g_register_infos in
  // this scope.
  return m_register_info_up->GetRegisterInfo();
}

const RegisterInfo *
RegisterContextCore_powerpc::GetRegisterInfoAtIndex(size_t reg) {
  if (reg < reg_ids::k_num_registers)
    return &GetRegisterInfo()[reg];
  else
    return nullptr;
}

size_t RegisterContextCore_powerpc::GetRegisterSetCount() {
  size_t sets = 0;
  for (size_t set = 0; set < k_num_register_sets; ++set) {
    if (IsRegisterSetAvailable(set))
      ++sets;
  }

  return sets;
}

const RegisterSet *RegisterContextCore_powerpc::GetRegisterSet(size_t set) {
  if (IsRegisterSetAvailable(set))
    return &g_reg_sets_powerpc[set];
  else
    return nullptr;
}

const char *RegisterContextCore_powerpc::GetRegisterName(unsigned reg) {
  assert(reg < reg_ids::k_num_registers && "Invalid register offset.");
  return GetRegisterInfo()[reg].name;
}

bool RegisterContextCore_powerpc::IsRegisterSetAvailable(size_t set_index) {
  size_t num_sets = k_num_register_sets;

  return (set_index < num_sets);
}
