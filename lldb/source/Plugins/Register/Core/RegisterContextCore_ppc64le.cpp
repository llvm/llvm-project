//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <cerrno>
#include "lldb/Register/Arch/lldb-ppc64le-register-enums.h"
#include <cstdint>
#include <cstring>

#include "lldb/Target/Process.h"
#include "lldb/Target/Target.h"
#include "lldb/Target/Thread.h"
#include "lldb/Utility/DataBufferHeap.h"
#include "lldb/Utility/DataExtractor.h"
#include "lldb/Utility/Endian.h"
#include "lldb/Register/RegisterValue.h"
#include "lldb/Utility/Scalar.h"
#include "llvm/Support/Compiler.h"

#include "RegisterContextCore_ppc64le.h"

namespace reg_ids = lldb_private::registers::ppc64le::id;
using namespace lldb_private;
using namespace lldb;

static const uint32_t g_gpr_regnums[] = {
    reg_ids::gpr_r0,    reg_ids::gpr_r1,     reg_ids::gpr_r2,
    reg_ids::gpr_r3,    reg_ids::gpr_r4,     reg_ids::gpr_r5,
    reg_ids::gpr_r6,    reg_ids::gpr_r7,     reg_ids::gpr_r8,
    reg_ids::gpr_r9,    reg_ids::gpr_r10,    reg_ids::gpr_r11,
    reg_ids::gpr_r12,   reg_ids::gpr_r13,    reg_ids::gpr_r14,
    reg_ids::gpr_r15,   reg_ids::gpr_r16,    reg_ids::gpr_r17,
    reg_ids::gpr_r18,   reg_ids::gpr_r19,    reg_ids::gpr_r20,
    reg_ids::gpr_r21,   reg_ids::gpr_r22,    reg_ids::gpr_r23,
    reg_ids::gpr_r24,   reg_ids::gpr_r25,    reg_ids::gpr_r26,
    reg_ids::gpr_r27,   reg_ids::gpr_r28,    reg_ids::gpr_r29,
    reg_ids::gpr_r30,   reg_ids::gpr_r31,    reg_ids::gpr_pc,
    reg_ids::gpr_msr,   reg_ids::gpr_origr3, reg_ids::gpr_ctr,
    reg_ids::gpr_lr,    reg_ids::gpr_xer,    reg_ids::gpr_cr,
    reg_ids::gpr_softe, reg_ids::gpr_trap,
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
    reg_ids::vmx_vr0,    reg_ids::vmx_vr1,  reg_ids::vmx_vr2,
    reg_ids::vmx_vr3,    reg_ids::vmx_vr4,  reg_ids::vmx_vr5,
    reg_ids::vmx_vr6,    reg_ids::vmx_vr7,  reg_ids::vmx_vr8,
    reg_ids::vmx_vr9,    reg_ids::vmx_vr10, reg_ids::vmx_vr11,
    reg_ids::vmx_vr12,   reg_ids::vmx_vr13, reg_ids::vmx_vr14,
    reg_ids::vmx_vr15,   reg_ids::vmx_vr16, reg_ids::vmx_vr17,
    reg_ids::vmx_vr18,   reg_ids::vmx_vr19, reg_ids::vmx_vr20,
    reg_ids::vmx_vr21,   reg_ids::vmx_vr22, reg_ids::vmx_vr23,
    reg_ids::vmx_vr24,   reg_ids::vmx_vr25, reg_ids::vmx_vr26,
    reg_ids::vmx_vr27,   reg_ids::vmx_vr28, reg_ids::vmx_vr29,
    reg_ids::vmx_vr30,   reg_ids::vmx_vr31, reg_ids::vmx_vscr,
    reg_ids::vmx_vrsave,
};

static const uint32_t g_vsx_regnums[] = {
    reg_ids::vsx_vs0,  reg_ids::vsx_vs1,  reg_ids::vsx_vs2,  reg_ids::vsx_vs3,
    reg_ids::vsx_vs4,  reg_ids::vsx_vs5,  reg_ids::vsx_vs6,  reg_ids::vsx_vs7,
    reg_ids::vsx_vs8,  reg_ids::vsx_vs9,  reg_ids::vsx_vs10, reg_ids::vsx_vs11,
    reg_ids::vsx_vs12, reg_ids::vsx_vs13, reg_ids::vsx_vs14, reg_ids::vsx_vs15,
    reg_ids::vsx_vs16, reg_ids::vsx_vs17, reg_ids::vsx_vs18, reg_ids::vsx_vs19,
    reg_ids::vsx_vs20, reg_ids::vsx_vs21, reg_ids::vsx_vs22, reg_ids::vsx_vs23,
    reg_ids::vsx_vs24, reg_ids::vsx_vs25, reg_ids::vsx_vs26, reg_ids::vsx_vs27,
    reg_ids::vsx_vs28, reg_ids::vsx_vs29, reg_ids::vsx_vs30, reg_ids::vsx_vs31,
    reg_ids::vsx_vs32, reg_ids::vsx_vs33, reg_ids::vsx_vs34, reg_ids::vsx_vs35,
    reg_ids::vsx_vs36, reg_ids::vsx_vs37, reg_ids::vsx_vs38, reg_ids::vsx_vs39,
    reg_ids::vsx_vs40, reg_ids::vsx_vs41, reg_ids::vsx_vs42, reg_ids::vsx_vs43,
    reg_ids::vsx_vs44, reg_ids::vsx_vs45, reg_ids::vsx_vs46, reg_ids::vsx_vs47,
    reg_ids::vsx_vs48, reg_ids::vsx_vs49, reg_ids::vsx_vs50, reg_ids::vsx_vs51,
    reg_ids::vsx_vs52, reg_ids::vsx_vs53, reg_ids::vsx_vs54, reg_ids::vsx_vs55,
    reg_ids::vsx_vs56, reg_ids::vsx_vs57, reg_ids::vsx_vs58, reg_ids::vsx_vs59,
    reg_ids::vsx_vs60, reg_ids::vsx_vs61, reg_ids::vsx_vs62, reg_ids::vsx_vs63,
};

// Number of register sets provided by this context.
enum { k_num_register_sets = 4 };

static const RegisterSet g_reg_sets_ppc64le[k_num_register_sets] = {
    {"General Purpose Registers", "gpr", reg_ids::k_num_gpr_registers,
     g_gpr_regnums},
    {"Floating Point Registers", "fpr", reg_ids::k_num_fpr_registers,
     g_fpr_regnums},
    {"Altivec/VMX Registers", "vmx", reg_ids::k_num_vmx_registers,
     g_vmx_regnums},
    {"VSX Registers", "vsx", reg_ids::k_num_vsx_registers, g_vsx_regnums},
};

bool RegisterContextCore_ppc64le::IsGPR(unsigned reg) {
  return (reg <= reg_ids::k_last_gpr); // GPR's come first.
}

bool RegisterContextCore_ppc64le::IsFPR(unsigned reg) {
  return (reg >= reg_ids::k_first_fpr) && (reg <= reg_ids::k_last_fpr);
}

bool RegisterContextCore_ppc64le::IsVMX(unsigned reg) {
  return (reg >= reg_ids::k_first_vmx) && (reg <= reg_ids::k_last_vmx);
}

bool RegisterContextCore_ppc64le::IsVSX(unsigned reg) {
  return (reg >= reg_ids::k_first_vsx) && (reg <= reg_ids::k_last_vsx);
}

RegisterContextCore_ppc64le::RegisterContextCore_ppc64le(
    Thread &thread, uint32_t concrete_frame_idx,
    RegisterInfoInterface *register_info)
    : ThreadRegisterContext(thread, concrete_frame_idx) {
  m_register_info_up.reset(register_info);
}

void RegisterContextCore_ppc64le::InvalidateAllRegisters() {}

unsigned RegisterContextCore_ppc64le::GetRegisterOffset(unsigned reg) {
  assert(reg < reg_ids::k_num_registers && "Invalid register number.");
  return GetRegisterInfo()[reg].byte_offset;
}

unsigned RegisterContextCore_ppc64le::GetRegisterSize(unsigned reg) {
  assert(reg < reg_ids::k_num_registers && "Invalid register number.");
  return GetRegisterInfo()[reg].byte_size;
}

size_t RegisterContextCore_ppc64le::GetRegisterCount() {
  size_t num_registers = reg_ids::k_num_registers;
  return num_registers;
}

size_t RegisterContextCore_ppc64le::GetGPRSize() {
  return m_register_info_up->GetGPRSize();
}

const RegisterInfo *RegisterContextCore_ppc64le::GetRegisterInfo() {
  // Commonly, this method is overridden and g_register_infos is copied and
  // specialized. So, use GetRegisterInfo() rather than g_register_infos in
  // this scope.
  return m_register_info_up->GetRegisterInfo();
}

const RegisterInfo *
RegisterContextCore_ppc64le::GetRegisterInfoAtIndex(size_t reg) {
  if (reg < reg_ids::k_num_registers)
    return &GetRegisterInfo()[reg];
  else
    return nullptr;
}

size_t RegisterContextCore_ppc64le::GetRegisterSetCount() {
  size_t sets = 0;
  for (size_t set = 0; set < k_num_register_sets; ++set) {
    if (IsRegisterSetAvailable(set))
      ++sets;
  }

  return sets;
}

const RegisterSet *RegisterContextCore_ppc64le::GetRegisterSet(size_t set) {
  if (IsRegisterSetAvailable(set))
    return &g_reg_sets_ppc64le[set];
  else
    return nullptr;
}

const char *RegisterContextCore_ppc64le::GetRegisterName(unsigned reg) {
  assert(reg < reg_ids::k_num_registers && "Invalid register offset.");
  return GetRegisterInfo()[reg].name;
}

bool RegisterContextCore_ppc64le::IsRegisterSetAvailable(size_t set_index) {
  size_t num_sets = k_num_register_sets;

  return (set_index < num_sets);
}
