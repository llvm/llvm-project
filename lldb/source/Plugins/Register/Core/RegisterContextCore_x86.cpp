//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <cerrno>
#include "lldb/Register/Arch/lldb-x86_64-register-enums.h"
#include "lldb/Register/Arch/lldb-x86-register-enums.h"
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

#include "RegisterContextCore_x86.h"

using namespace lldb_private;
using namespace lldb;

const uint32_t g_gpr_regnums_i386[] = {
    registers::x86::id::eax, registers::x86::id::ebx,
    registers::x86::id::ecx, registers::x86::id::edx,
    registers::x86::id::edi, registers::x86::id::esi,
    registers::x86::id::ebp, registers::x86::id::esp,
    registers::x86::id::eip, registers::x86::id::eflags,
    registers::x86::id::cs,  registers::x86::id::fs,
    registers::x86::id::gs,  registers::x86::id::ss,
    registers::x86::id::ds,  registers::x86::id::es,
    registers::x86::id::ax,  registers::x86::id::bx,
    registers::x86::id::cx,  registers::x86::id::dx,
    registers::x86::id::di,  registers::x86::id::si,
    registers::x86::id::bp,  registers::x86::id::sp,
    registers::x86::id::ah,  registers::x86::id::bh,
    registers::x86::id::ch,  registers::x86::id::dh,
    registers::x86::id::al,  registers::x86::id::bl,
    registers::x86::id::cl,  registers::x86::id::dl,
    LLDB_INVALID_REGNUM, // Register sets must be terminated with
                         // LLDB_INVALID_REGNUM.
};
static_assert((sizeof(g_gpr_regnums_i386) / sizeof(g_gpr_regnums_i386[0])) -
                      1 ==
                  registers::x86::id::k_num_gpr_registers,
              "g_gpr_regnums_i386 has wrong number of register infos");

const uint32_t g_lldb_regnums_i386[] = {
    registers::x86::id::fctrl, registers::x86::id::fstat,
    registers::x86::id::ftag,  registers::x86::id::fop,
    registers::x86::id::fiseg, registers::x86::id::fioff,
    registers::x86::id::foseg, registers::x86::id::fooff,
    registers::x86::id::mxcsr, registers::x86::id::mxcsrmask,
    registers::x86::id::st0,   registers::x86::id::st1,
    registers::x86::id::st2,   registers::x86::id::st3,
    registers::x86::id::st4,   registers::x86::id::st5,
    registers::x86::id::st6,   registers::x86::id::st7,
    registers::x86::id::mm0,   registers::x86::id::mm1,
    registers::x86::id::mm2,   registers::x86::id::mm3,
    registers::x86::id::mm4,   registers::x86::id::mm5,
    registers::x86::id::mm6,   registers::x86::id::mm7,
    registers::x86::id::xmm0,  registers::x86::id::xmm1,
    registers::x86::id::xmm2,  registers::x86::id::xmm3,
    registers::x86::id::xmm4,  registers::x86::id::xmm5,
    registers::x86::id::xmm6,  registers::x86::id::xmm7,
    LLDB_INVALID_REGNUM // Register sets must be terminated with
                        // LLDB_INVALID_REGNUM.
};
static_assert((sizeof(g_lldb_regnums_i386) / sizeof(g_lldb_regnums_i386[0])) -
                      1 ==
                  registers::x86::id::k_num_fpr_registers,
              "g_lldb_regnums_i386 has wrong number of register infos");

const uint32_t g_avx_regnums_i386[] = {
    registers::x86::id::ymm0, registers::x86::id::ymm1,
    registers::x86::id::ymm2, registers::x86::id::ymm3,
    registers::x86::id::ymm4, registers::x86::id::ymm5,
    registers::x86::id::ymm6, registers::x86::id::ymm7,
    LLDB_INVALID_REGNUM // Register sets must be terminated with
                        // LLDB_INVALID_REGNUM.
};
static_assert((sizeof(g_avx_regnums_i386) / sizeof(g_avx_regnums_i386[0])) -
                      1 ==
                  registers::x86::id::k_num_avx_registers,
              " g_avx_regnums_i386 has wrong number of register infos");

static const uint32_t g_gpr_regnums_x86_64[] = {
    registers::x86_64::id::rax,  registers::x86_64::id::rbx,
    registers::x86_64::id::rcx,  registers::x86_64::id::rdx,
    registers::x86_64::id::rdi,  registers::x86_64::id::rsi,
    registers::x86_64::id::rbp,  registers::x86_64::id::rsp,
    registers::x86_64::id::r8,   registers::x86_64::id::r9,
    registers::x86_64::id::r10,  registers::x86_64::id::r11,
    registers::x86_64::id::r12,  registers::x86_64::id::r13,
    registers::x86_64::id::r14,  registers::x86_64::id::r15,
    registers::x86_64::id::rip,  registers::x86_64::id::rflags,
    registers::x86_64::id::cs,   registers::x86_64::id::fs,
    registers::x86_64::id::gs,   registers::x86_64::id::ss,
    registers::x86_64::id::ds,   registers::x86_64::id::es,
    registers::x86_64::id::eax,  registers::x86_64::id::ebx,
    registers::x86_64::id::ecx,  registers::x86_64::id::edx,
    registers::x86_64::id::edi,  registers::x86_64::id::esi,
    registers::x86_64::id::ebp,  registers::x86_64::id::esp,
    registers::x86_64::id::r8d,  // Low 32 bits or r8
    registers::x86_64::id::r9d,  // Low 32 bits or r9
    registers::x86_64::id::r10d, // Low 32 bits or r10
    registers::x86_64::id::r11d, // Low 32 bits or r11
    registers::x86_64::id::r12d, // Low 32 bits or r12
    registers::x86_64::id::r13d, // Low 32 bits or r13
    registers::x86_64::id::r14d, // Low 32 bits or r14
    registers::x86_64::id::r15d, // Low 32 bits or r15
    registers::x86_64::id::ax,   registers::x86_64::id::bx,
    registers::x86_64::id::cx,   registers::x86_64::id::dx,
    registers::x86_64::id::di,   registers::x86_64::id::si,
    registers::x86_64::id::bp,   registers::x86_64::id::sp,
    registers::x86_64::id::r8w,  // Low 16 bits or r8
    registers::x86_64::id::r9w,  // Low 16 bits or r9
    registers::x86_64::id::r10w, // Low 16 bits or r10
    registers::x86_64::id::r11w, // Low 16 bits or r11
    registers::x86_64::id::r12w, // Low 16 bits or r12
    registers::x86_64::id::r13w, // Low 16 bits or r13
    registers::x86_64::id::r14w, // Low 16 bits or r14
    registers::x86_64::id::r15w, // Low 16 bits or r15
    registers::x86_64::id::ah,     registers::x86_64::id::bh,     registers::x86_64::id::ch,  registers::x86_64::id::dh,
    registers::x86_64::id::al,     registers::x86_64::id::bl,     registers::x86_64::id::cl,  registers::x86_64::id::dl,
    registers::x86_64::id::dil,    registers::x86_64::id::sil,    registers::x86_64::id::bpl, registers::x86_64::id::spl,
    registers::x86_64::id::r8l,    // Low 8 bits or r8
    registers::x86_64::id::r9l,    // Low 8 bits or r9
    registers::x86_64::id::r10l,   // Low 8 bits or r10
    registers::x86_64::id::r11l,   // Low 8 bits or r11
    registers::x86_64::id::r12l,   // Low 8 bits or r12
    registers::x86_64::id::r13l,   // Low 8 bits or r13
    registers::x86_64::id::r14l,   // Low 8 bits or r14
    registers::x86_64::id::r15l,   // Low 8 bits or r15
    LLDB_INVALID_REGNUM // Register sets must be terminated with
                        // LLDB_INVALID_REGNUM.
};
static_assert((sizeof(g_gpr_regnums_x86_64) / sizeof(g_gpr_regnums_x86_64[0])) -
                      1 ==
                  registers::x86_64::id::k_num_gpr_registers,
              "g_gpr_regnums_x86_64 has wrong number of register infos");

static const uint32_t g_lldb_regnums_x86_64[] = {
    registers::x86_64::id::fctrl, registers::x86_64::id::fstat,
    registers::x86_64::id::ftag,  registers::x86_64::id::fop,
    registers::x86_64::id::fiseg, registers::x86_64::id::fioff,
    registers::x86_64::id::fip,   registers::x86_64::id::foseg,
    registers::x86_64::id::fooff, registers::x86_64::id::fdp,
    registers::x86_64::id::mxcsr, registers::x86_64::id::mxcsrmask,
    registers::x86_64::id::st0,   registers::x86_64::id::st1,
    registers::x86_64::id::st2,   registers::x86_64::id::st3,
    registers::x86_64::id::st4,   registers::x86_64::id::st5,
    registers::x86_64::id::st6,   registers::x86_64::id::st7,
    registers::x86_64::id::mm0,   registers::x86_64::id::mm1,
    registers::x86_64::id::mm2,   registers::x86_64::id::mm3,
    registers::x86_64::id::mm4,   registers::x86_64::id::mm5,
    registers::x86_64::id::mm6,   registers::x86_64::id::mm7,
    registers::x86_64::id::xmm0,  registers::x86_64::id::xmm1,
    registers::x86_64::id::xmm2,  registers::x86_64::id::xmm3,
    registers::x86_64::id::xmm4,  registers::x86_64::id::xmm5,
    registers::x86_64::id::xmm6,  registers::x86_64::id::xmm7,
    registers::x86_64::id::xmm8,  registers::x86_64::id::xmm9,
    registers::x86_64::id::xmm10, registers::x86_64::id::xmm11,
    registers::x86_64::id::xmm12, registers::x86_64::id::xmm13,
    registers::x86_64::id::xmm14, registers::x86_64::id::xmm15,
    LLDB_INVALID_REGNUM // Register sets must be terminated with
                        // LLDB_INVALID_REGNUM.
};
static_assert((sizeof(g_lldb_regnums_x86_64) /
               sizeof(g_lldb_regnums_x86_64[0])) -
                      1 ==
                  registers::x86_64::id::k_num_fpr_registers,
              "g_lldb_regnums_x86_64 has wrong number of register infos");

static const uint32_t g_avx_regnums_x86_64[] = {
    registers::x86_64::id::ymm0,  registers::x86_64::id::ymm1,
    registers::x86_64::id::ymm2,  registers::x86_64::id::ymm3,
    registers::x86_64::id::ymm4,  registers::x86_64::id::ymm5,
    registers::x86_64::id::ymm6,  registers::x86_64::id::ymm7,
    registers::x86_64::id::ymm8,  registers::x86_64::id::ymm9,
    registers::x86_64::id::ymm10, registers::x86_64::id::ymm11,
    registers::x86_64::id::ymm12, registers::x86_64::id::ymm13,
    registers::x86_64::id::ymm14, registers::x86_64::id::ymm15,
    LLDB_INVALID_REGNUM // Register sets must be terminated with
                        // LLDB_INVALID_REGNUM.
};
static_assert((sizeof(g_avx_regnums_x86_64) / sizeof(g_avx_regnums_x86_64[0])) -
                      1 ==
                  registers::x86_64::id::k_num_avx_registers,
              "g_avx_regnums_x86_64 has wrong number of register infos");

// Number of register sets provided by this context.
enum { k_num_extended_register_sets = 1, k_num_register_sets = 3 };

static const RegisterSet g_reg_sets_i386[k_num_register_sets] = {
    {"General Purpose Registers", "gpr", registers::x86::id::k_num_gpr_registers,
     g_gpr_regnums_i386},
    {"Floating Point Registers", "fpu", registers::x86::id::k_num_fpr_registers,
     g_lldb_regnums_i386},
    {"Advanced Vector Extensions", "avx", registers::x86::id::k_num_avx_registers,
     g_avx_regnums_i386}};

static const RegisterSet g_reg_sets_x86_64[k_num_register_sets] = {
    {"General Purpose Registers", "gpr", registers::x86_64::id::k_num_gpr_registers,
     g_gpr_regnums_x86_64},
    {"Floating Point Registers", "fpu", registers::x86_64::id::k_num_fpr_registers,
     g_lldb_regnums_x86_64},
    {"Advanced Vector Extensions", "avx", registers::x86_64::id::k_num_avx_registers,
     g_avx_regnums_x86_64}};

bool RegisterContextCore_x86::IsGPR(unsigned reg) {
  return reg <= GetRegInfo().last_gpr; // GPR's come first.
}

bool RegisterContextCore_x86::IsFPR(unsigned reg) {
  return (GetRegInfo().first_fpr <= reg && reg <= GetRegInfo().last_fpr);
}

bool RegisterContextCore_x86::IsAVX(unsigned reg) {
  return (GetRegInfo().first_ymm <= reg && reg <= GetRegInfo().last_ymm);
}

bool RegisterContextCore_x86::IsFPR(unsigned reg, FPRType fpr_type) {
  bool generic_fpr = IsFPR(reg);

  if (fpr_type == eXSAVE)
    return generic_fpr || IsAVX(reg);
  return generic_fpr;
}

RegisterContextCore_x86::RegisterContextCore_x86(
    Thread &thread, uint32_t concrete_frame_idx,
    RegisterInfoInterface *register_info)
    : RegisterContext(thread, concrete_frame_idx) {
  m_register_info_up.reset(register_info);

  ::memset(&m_fpr, 0, sizeof(FPR));
  ::memset(&m_ymm_set, 0, sizeof(YMM));

  m_fpr_type = eNotValid;
}

RegisterContextCore_x86::~RegisterContextCore_x86() = default;

RegisterContextCore_x86::FPRType RegisterContextCore_x86::GetFPRType() {
  if (m_fpr_type == eNotValid) {
    // TODO: Use assembly to call cpuid on the inferior and query ebx or ecx
    m_fpr_type = eXSAVE; // extended floating-point registers, if available
    if (!ReadFPR())
      m_fpr_type = eFXSAVE; // assume generic floating-point registers
  }
  return m_fpr_type;
}

void RegisterContextCore_x86::Invalidate() {}

void RegisterContextCore_x86::InvalidateAllRegisters() {}

unsigned RegisterContextCore_x86::GetRegisterOffset(unsigned reg) {
  assert(reg < GetRegInfo().num_registers && "Invalid register number.");
  return GetRegisterInfo()[reg].byte_offset;
}

RegInfo &RegisterContextCore_x86::GetRegInfo() {
  return GetRegInfoShared(
      m_register_info_up->GetTargetArchitecture().GetMachine(),
      /*with_base=*/false);
}

unsigned RegisterContextCore_x86::GetRegisterSize(unsigned reg) {
  assert(reg < GetRegInfo().num_registers && "Invalid register number.");
  return GetRegisterInfo()[reg].byte_size;
}

size_t RegisterContextCore_x86::GetRegisterCount() {
  size_t num_registers =
      GetRegInfo().num_gpr_registers + GetRegInfo().num_fpr_registers;
  if (GetFPRType() == eXSAVE)
    return num_registers + GetRegInfo().num_avx_registers;
  return num_registers;
}

size_t RegisterContextCore_x86::GetGPRSize() {
  return m_register_info_up->GetGPRSize();
}

size_t RegisterContextCore_x86::GetFXSAVEOffset() {
  return GetRegisterInfo()[GetRegInfo().first_fpr].byte_offset;
}

const RegisterInfo *RegisterContextCore_x86::GetRegisterInfo() {
  // Commonly, this method is overridden and g_register_infos is copied and
  // specialized. So, use GetRegisterInfo() rather than g_register_infos in
  // this scope.
  return m_register_info_up->GetRegisterInfo();
}

const RegisterInfo *
RegisterContextCore_x86::GetRegisterInfoAtIndex(size_t reg) {
  if (reg < GetRegInfo().num_registers)
    return &GetRegisterInfo()[reg];
  else
    return nullptr;
}

size_t RegisterContextCore_x86::GetRegisterSetCount() {
  size_t sets = 0;
  for (size_t set = 0; set < k_num_register_sets; ++set) {
    if (IsRegisterSetAvailable(set))
      ++sets;
  }

  return sets;
}

const RegisterSet *RegisterContextCore_x86::GetRegisterSet(size_t set) {
  if (IsRegisterSetAvailable(set)) {
    switch (m_register_info_up->GetTargetArchitecture().GetMachine()) {
    case llvm::Triple::x86:
      return &g_reg_sets_i386[set];
    case llvm::Triple::x86_64:
      return &g_reg_sets_x86_64[set];
    default:
      assert(false && "Unhandled target architecture.");
      return nullptr;
    }
  }
  return nullptr;
}

const char *RegisterContextCore_x86::GetRegisterName(unsigned reg) {
  assert(reg < GetRegInfo().num_registers && "Invalid register offset.");
  return GetRegisterInfo()[reg].name;
}

// Parse ymm registers and into xmm.bytes and ymmh.bytes.
bool RegisterContextCore_x86::CopyYMMtoXSTATE(uint32_t reg,
                                               lldb::ByteOrder byte_order) {
  if (!IsAVX(reg))
    return false;

  if (byte_order == eByteOrderLittle) {
    uint32_t reg_no = reg - GetRegInfo().first_ymm;
    YMMToXState(m_ymm_set.ymm[reg_no], m_fpr.fxsave.xmm[reg_no].bytes,
                m_fpr.xsave.ymmh[reg_no].bytes);
    return true;
  }

  return false; // unsupported or invalid byte order
}

// Concatenate xmm.bytes with ymmh.bytes
bool RegisterContextCore_x86::CopyXSTATEtoYMM(uint32_t reg,
                                               lldb::ByteOrder byte_order) {
  if (!IsAVX(reg))
    return false;

  if (byte_order == eByteOrderLittle) {
    uint32_t reg_no = reg - GetRegInfo().first_ymm;
    m_ymm_set.ymm[reg_no] = XStateToYMM(m_fpr.fxsave.xmm[reg_no].bytes,
                                        m_fpr.xsave.ymmh[reg_no].bytes);
    return true;
  }

  return false; // unsupported or invalid byte order
}

bool RegisterContextCore_x86::IsRegisterSetAvailable(size_t set_index) {
  // Note: Extended register sets are assumed to be at the end of g_reg_sets...
  size_t num_sets = k_num_register_sets - k_num_extended_register_sets;

  if (GetFPRType() == eXSAVE) // ...and to start with AVX registers.
    ++num_sets;
  return (set_index < num_sets);
}
