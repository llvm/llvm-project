//===-- NativeRegisterContextNetBSD_x86.cpp -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#if defined(__i386__) || defined(__x86_64__)

#include "NativeRegisterContextNetBSD_x86.h"
#include "lldb/Register/Arch/lldb-x86_64-register-enums.h"
#include "lldb/Register/Arch/lldb-x86-register-enums.h"

#include "lldb/Host/HostInfo.h"
#include "lldb/Utility/DataBufferHeap.h"
#include "lldb/Utility/Log.h"
#include "lldb/Register/RegisterValue.h"
#include "lldb/Utility/Status.h"

#include "Register/NetBSD/RegisterInfoNetBSD_i386.h"
#include "Register/NetBSD/RegisterInfoNetBSD_x86_64.h"

// clang-format off
#include <sys/types.h>
#include <sys/ptrace.h>
#include <sys/sysctl.h>
#include <sys/uio.h>
#include <x86/cpu.h>
#include <x86/cpu_extended_state.h>
#include <x86/specialreg.h>
#include <elf.h>
#include <err.h>
#include <cstdint>
#include <cstdlib>
#include <optional>
// clang-format on

using namespace lldb_private;
using namespace lldb_private::process_netbsd;

// x86 64-bit general purpose registers.
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
    LLDB_INVALID_REGNUM // register sets need to end with this flag
};
static_assert((sizeof(g_gpr_regnums_x86_64) / sizeof(g_gpr_regnums_x86_64[0])) -
                      1 ==
                  registers::x86_64::id::k_num_gpr_registers,
              "g_gpr_regnums_x86_64 has wrong number of register infos");

// x86 64-bit floating point registers.
static const uint32_t g_fpu_regnums_x86_64[] = {
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
    LLDB_INVALID_REGNUM // register sets need to end with this flag
};
static_assert((sizeof(g_fpu_regnums_x86_64) / sizeof(g_fpu_regnums_x86_64[0])) -
                      1 ==
                  registers::x86_64::id::k_num_fpr_registers,
              "g_fpu_regnums_x86_64 has wrong number of register infos");

static const uint32_t g_avx_regnums_x86_64[] = {
    registers::x86_64::id::ymm0,  registers::x86_64::id::ymm1,
    registers::x86_64::id::ymm2,  registers::x86_64::id::ymm3,
    registers::x86_64::id::ymm4,  registers::x86_64::id::ymm5,
    registers::x86_64::id::ymm6,  registers::x86_64::id::ymm7,
    registers::x86_64::id::ymm8,  registers::x86_64::id::ymm9,
    registers::x86_64::id::ymm10, registers::x86_64::id::ymm11,
    registers::x86_64::id::ymm12, registers::x86_64::id::ymm13,
    registers::x86_64::id::ymm14, registers::x86_64::id::ymm15,
    LLDB_INVALID_REGNUM // register sets need to end with this flag
};
static_assert((sizeof(g_avx_regnums_x86_64) / sizeof(g_avx_regnums_x86_64[0])) -
                      1 ==
                  registers::x86_64::id::k_num_avx_registers,
              "g_avx_regnums_x86_64 has wrong number of register infos");

static const uint32_t g_mpx_regnums_x86_64[] = {
    // Note: we currently do not provide them but this is needed to avoid
    // unnamed groups in SBFrame::GetRegisterContext().
    registers::x86_64::id::bnd0,
    registers::x86_64::id::bnd1,
    registers::x86_64::id::bnd2,
    registers::x86_64::id::bnd3,
    registers::x86_64::id::bndcfgu,
    registers::x86_64::id::bndstatus,
    LLDB_INVALID_REGNUM // register sets need to end with this flag
};
static_assert((sizeof(g_mpx_regnums_x86_64) / sizeof(g_mpx_regnums_x86_64[0])) -
                      1 ==
                  registers::x86_64::id::k_num_mpx_registers,
              "g_mpx_regnums_x86_64 has wrong number of register infos");

// x86 debug registers.
static const uint32_t g_dbr_regnums_x86_64[] = {
    registers::x86_64::id::dr0, registers::x86_64::id::dr1,
    registers::x86_64::id::dr2, registers::x86_64::id::dr3,
    registers::x86_64::id::dr4, registers::x86_64::id::dr5,
    registers::x86_64::id::dr6, registers::x86_64::id::dr7,
    LLDB_INVALID_REGNUM // register sets need to end with this flag
};
static_assert((sizeof(g_dbr_regnums_x86_64) / sizeof(g_dbr_regnums_x86_64[0])) -
                      1 ==
                  registers::x86_64::id::k_num_dbr_registers,
              "g_dbr_regnums_x86_64 has wrong number of register infos");

// x86 32-bit general purpose registers.
static const uint32_t g_gpr_regnums_i386[] = {
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
    LLDB_INVALID_REGNUM // register sets need to end with this flag
};
static_assert((sizeof(g_gpr_regnums_i386) / sizeof(g_gpr_regnums_i386[0])) -
                      1 ==
                  registers::x86::id::k_num_gpr_registers,
              "g_gpr_regnums_i386 has wrong number of register infos");

// x86 32-bit floating point registers.
static const uint32_t g_fpu_regnums_i386[] = {
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
    LLDB_INVALID_REGNUM // register sets need to end with this flag
};
static_assert((sizeof(g_fpu_regnums_i386) / sizeof(g_fpu_regnums_i386[0])) -
                      1 ==
                  registers::x86::id::k_num_fpr_registers,
              "g_fpu_regnums_i386 has wrong number of register infos");

static const uint32_t g_avx_regnums_i386[] = {
    registers::x86::id::ymm0, registers::x86::id::ymm1,
    registers::x86::id::ymm2, registers::x86::id::ymm3,
    registers::x86::id::ymm4, registers::x86::id::ymm5,
    registers::x86::id::ymm6, registers::x86::id::ymm7,
    LLDB_INVALID_REGNUM // register sets need to end with this flag
};
static_assert((sizeof(g_avx_regnums_i386) / sizeof(g_avx_regnums_i386[0])) -
                      1 ==
                  registers::x86::id::k_num_avx_registers,
              "g_avx_regnums_i386 has wrong number of register infos");

static const uint32_t g_mpx_regnums_i386[] = {
    // Note: we currently do not provide them but this is needed to avoid
    // unnamed groups in SBFrame::GetRegisterContext().
    registers::x86::id::bnd0,    registers::x86::id::bnd1,
    registers::x86::id::bnd2,    registers::x86::id::bnd3,
    registers::x86::id::bndcfgu, registers::x86::id::bndstatus,
    LLDB_INVALID_REGNUM // register sets need to end with this flag
};
static_assert((sizeof(g_mpx_regnums_i386) / sizeof(g_mpx_regnums_i386[0])) -
                      1 ==
                  registers::x86::id::k_num_mpx_registers,
              "g_mpx_regnums_i386 has wrong number of register infos");

// x86 debug registers.
static const uint32_t g_dbr_regnums_i386[] = {
    registers::x86::id::dr0, registers::x86::id::dr1, registers::x86::id::dr2,
    registers::x86::id::dr3, registers::x86::id::dr4, registers::x86::id::dr5,
    registers::x86::id::dr6, registers::x86::id::dr7,
    LLDB_INVALID_REGNUM // register sets need to end with this flag
};
static_assert((sizeof(g_dbr_regnums_i386) / sizeof(g_dbr_regnums_i386[0])) -
                      1 ==
                  registers::x86::id::k_num_dbr_registers,
              "g_dbr_regnums_i386 has wrong number of register infos");

// Number of register sets provided by this context.
enum { k_num_register_sets = 5 };

// Register sets for x86 32-bit.
static const RegisterSet g_reg_sets_i386[k_num_register_sets] = {
    {"General Purpose Registers", "gpr", registers::x86::id::k_num_gpr_registers,
     g_gpr_regnums_i386},
    {"Floating Point Registers", "fpu", registers::x86::id::k_num_fpr_registers,
     g_fpu_regnums_i386},
    {"Debug Registers", "dbr", registers::x86::id::k_num_dbr_registers, g_dbr_regnums_i386},
    {"Advanced Vector Extensions", "avx", registers::x86::id::k_num_avx_registers,
     g_avx_regnums_i386},
    {"Memory Protection Extensions", "mpx", registers::x86::id::k_num_mpx_registers,
     g_mpx_regnums_i386},
};

// Register sets for x86 64-bit.
static const RegisterSet g_reg_sets_x86_64[k_num_register_sets] = {
    {"General Purpose Registers", "gpr", registers::x86_64::id::k_num_gpr_registers,
     g_gpr_regnums_x86_64},
    {"Floating Point Registers", "fpu", registers::x86_64::id::k_num_fpr_registers,
     g_fpu_regnums_x86_64},
    {"Debug Registers", "dbr", registers::x86_64::id::k_num_dbr_registers,
     g_dbr_regnums_x86_64},
    {"Advanced Vector Extensions", "avx", registers::x86_64::id::k_num_avx_registers,
     g_avx_regnums_x86_64},
    {"Memory Protection Extensions", "mpx", registers::x86_64::id::k_num_mpx_registers,
     g_mpx_regnums_x86_64},
};

#define REG_CONTEXT_SIZE (GetRegisterInfoInterface().GetGPRSize())

NativeRegisterContextNetBSD *
NativeRegisterContextNetBSD::CreateHostNativeRegisterContextNetBSD(
    const ArchSpec &target_arch, NativeThreadProtocol &native_thread) {
  return new NativeRegisterContextNetBSD_x86(target_arch, native_thread);
}

// NativeRegisterContextNetBSD_x86 members.

static RegisterInfoInterface *
CreateRegisterInfoInterface(const ArchSpec &target_arch) {
  if (HostInfo::GetArchitecture().GetAddressByteSize() == 4) {
    // 32-bit hosts run with a RegisterInfoNetBSD_i386 context.
    return new RegisterInfoNetBSD_i386(target_arch);
  } else {
    assert((HostInfo::GetArchitecture().GetAddressByteSize() == 8) &&
           "Register setting path assumes this is a 64-bit host");
    // X86_64 hosts know how to work with 64-bit and 32-bit EXEs using the
    // x86_64 register context.
    return new RegisterInfoNetBSD_x86_64(target_arch);
  }
}

NativeRegisterContextNetBSD_x86::NativeRegisterContextNetBSD_x86(
    const ArchSpec &target_arch, NativeThreadProtocol &native_thread)
    : NativeRegisterContextRegisterInfo(
          native_thread, CreateRegisterInfoInterface(target_arch)),
      m_regset_offsets({0}) {
  assert(m_gpr.size() == GetRegisterInfoInterface().GetGPRSize());
  std::array<uint32_t, MaxRegularRegSet + 1> first_regnos;

  switch (GetRegisterInfoInterface().GetTargetArchitecture().GetMachine()) {
  case llvm::Triple::x86:
    first_regnos[FPRegSet] = registers::x86::id::fctrl;
    first_regnos[DBRegSet] = registers::x86::id::dr0;
    break;
  case llvm::Triple::x86_64:
    first_regnos[FPRegSet] = registers::x86_64::id::fctrl;
    first_regnos[DBRegSet] = registers::x86_64::id::dr0;
    break;
  default:
    llvm_unreachable("Unhandled target architecture.");
  }

  for (int i : {FPRegSet, DBRegSet})
    m_regset_offsets[i] = GetRegisterInfoInterface()
                              .GetRegisterInfo()[first_regnos[i]]
                              .byte_offset;
}

uint32_t NativeRegisterContextNetBSD_x86::GetRegisterSetCount() const {
  return k_num_register_sets;
}

const RegisterSet *
NativeRegisterContextNetBSD_x86::GetRegisterSet(uint32_t set_index) const {
  switch (GetRegisterInfoInterface().GetTargetArchitecture().GetMachine()) {
  case llvm::Triple::x86:
    return &g_reg_sets_i386[set_index];
  case llvm::Triple::x86_64:
    return &g_reg_sets_x86_64[set_index];
  default:
    llvm_unreachable("Unhandled target architecture.");
  }
}

std::optional<NativeRegisterContextNetBSD_x86::RegSetKind>
NativeRegisterContextNetBSD_x86::GetSetForNativeRegNum(
    uint32_t reg_num) const {
  switch (GetRegisterInfoInterface().GetTargetArchitecture().GetMachine()) {
  case llvm::Triple::x86:
    if (reg_num >= registers::x86::id::k_first_gpr &&
        reg_num <= registers::x86::id::k_last_gpr)
      return GPRegSet;
    if (reg_num >= registers::x86::id::k_first_fpr &&
        reg_num <= registers::x86::id::k_last_fpr)
      return FPRegSet;
    if (reg_num >= registers::x86::id::k_first_avx &&
        reg_num <= registers::x86::id::k_last_avx)
      return YMMRegSet;
    if (reg_num >= registers::x86::id::k_first_mpxr &&
        reg_num <= registers::x86::id::k_last_mpxr)
      return std::nullopt; // MPXR
    if (reg_num >= registers::x86::id::k_first_mpxc &&
        reg_num <= registers::x86::id::k_last_mpxc)
      return std::nullopt; // MPXC
    if (reg_num >= registers::x86::id::k_first_dbr &&
        reg_num <= registers::x86::id::k_last_dbr)
      return DBRegSet; // DBR
    break;
  case llvm::Triple::x86_64:
    if (reg_num >= registers::x86_64::id::k_first_gpr &&
        reg_num <= registers::x86_64::id::k_last_gpr)
      return GPRegSet;
    if (reg_num >= registers::x86_64::id::k_first_fpr &&
        reg_num <= registers::x86_64::id::k_last_fpr)
      return FPRegSet;
    if (reg_num >= registers::x86_64::id::k_first_avx &&
        reg_num <= registers::x86_64::id::k_last_avx)
      return YMMRegSet;
    if (reg_num >= registers::x86_64::id::k_first_mpxr &&
        reg_num <= registers::x86_64::id::k_last_mpxr)
      return std::nullopt; // MPXR
    if (reg_num >= registers::x86_64::id::k_first_mpxc &&
        reg_num <= registers::x86_64::id::k_last_mpxc)
      return std::nullopt; // MPXC
    if (reg_num >= registers::x86_64::id::k_first_dbr &&
        reg_num <= registers::x86_64::id::k_last_dbr)
      return DBRegSet; // DBR
    break;
  default:
    llvm_unreachable("Unhandled target architecture.");
  }

  llvm_unreachable("Register does not belong to any register set");
}

Status NativeRegisterContextNetBSD_x86::ReadRegisterSet(RegSetKind set) {
  switch (set) {
  case GPRegSet:
    return DoRegisterSet(PT_GETREGS, m_gpr.data());
  case DBRegSet:
    return DoRegisterSet(PT_GETDBREGS, m_dbr.data());
  case FPRegSet:
  case YMMRegSet:
  case MPXRegSet: {
    struct iovec iov = {m_xstate.data(), m_xstate.size()};
    Status ret = DoRegisterSet(PT_GETXSTATE, &iov);
    assert(reinterpret_cast<xstate *>(m_xstate.data())->xs_rfbm & XCR0_X87);
    return ret;
  }
  }
  llvm_unreachable("NativeRegisterContextNetBSD_x86::ReadRegisterSet");
}

Status NativeRegisterContextNetBSD_x86::WriteRegisterSet(RegSetKind set) {
  switch (set) {
  case GPRegSet:
    return DoRegisterSet(PT_SETREGS, m_gpr.data());
  case DBRegSet:
    return DoRegisterSet(PT_SETDBREGS, m_dbr.data());
  case FPRegSet:
  case YMMRegSet:
  case MPXRegSet: {
    struct iovec iov = {&m_xstate, sizeof(m_xstate)};
    return DoRegisterSet(PT_SETXSTATE, &iov);
  }
  }
  llvm_unreachable("NativeRegisterContextNetBSD_x86::WriteRegisterSet");
}

Status
NativeRegisterContextNetBSD_x86::ReadRegister(const RegisterInfo *reg_info,
                                                 RegisterValue &reg_value) {
  Status error;

  if (!reg_info) {
    error = Status::FromErrorString("reg_info NULL");
    return error;
  }

  uint32_t reg = reg_info->kinds[lldb::eRegisterKindLLDB];
  if (reg == LLDB_INVALID_REGNUM) {
    // This is likely an internal register for lldb use only and should not be
    // directly queried.
    error = Status::FromErrorStringWithFormat(
        "register \"%s\" is an internal-only lldb "
        "register, cannot read directly",
        reg_info->name);
    return error;
  }

  std::optional<RegSetKind> opt_set = GetSetForNativeRegNum(reg);
  if (!opt_set) {
    // This is likely an internal register for lldb use only and should not be
    // directly queried.
    error = Status::FromErrorStringWithFormat(
        "register \"%s\" is in unrecognized set", reg_info->name);
    return error;
  }

  RegSetKind set = *opt_set;
  error = ReadRegisterSet(set);
  if (error.Fail())
    return error;

  switch (set) {
  case GPRegSet:
  case FPRegSet:
  case DBRegSet: {
    void *data = GetOffsetRegSetData(set, reg_info->byte_offset);
    FXSAVE *fpr = reinterpret_cast<FXSAVE *>(m_xstate.data() +
                                             offsetof(xstate, xs_fxsave));
    if (data == &fpr->ftag) // ftag
      reg_value.SetUInt16(
          AbridgedToFullTagWord(fpr->ftag, fpr->fstat, fpr->stmm));
    else
      reg_value.SetBytes(data, reg_info->byte_size, endian::InlHostByteOrder());
    break;
  }
  case YMMRegSet: {
    std::optional<YMMSplitPtr> ymm_reg = GetYMMSplitReg(reg);
    if (!ymm_reg) {
      error = Status::FromErrorStringWithFormat(
          "register \"%s\" not supported by CPU/kernel", reg_info->name);
    } else {
      YMMReg ymm = XStateToYMM(ymm_reg->xmm, ymm_reg->ymm_hi);
      reg_value.SetBytes(ymm.bytes, reg_info->byte_size,
                         endian::InlHostByteOrder());
    }
    break;
  }
  case MPXRegSet:
    llvm_unreachable("MPX regset should have returned error");
  }

  return error;
}

Status NativeRegisterContextNetBSD_x86::WriteRegister(
    const RegisterInfo *reg_info, const RegisterValue &reg_value) {

  Status error;

  if (!reg_info) {
    error = Status::FromErrorString("reg_info NULL");
    return error;
  }

  uint32_t reg = reg_info->kinds[lldb::eRegisterKindLLDB];
  if (reg == LLDB_INVALID_REGNUM) {
    // This is likely an internal register for lldb use only and should not be
    // directly queried.
    error = Status::FromErrorStringWithFormat(
        "register \"%s\" is an internal-only lldb "
        "register, cannot read directly",
        reg_info->name);
    return error;
  }

  std::optional<RegSetKind> opt_set = GetSetForNativeRegNum(reg);
  if (!opt_set) {
    // This is likely an internal register for lldb use only and should not be
    // directly queried.
    error = Status::FromErrorStringWithFormat(
        "register \"%s\" is in unrecognized set", reg_info->name);
    return error;
  }

  RegSetKind set = *opt_set;
  uint64_t new_xstate_bv = 0;

  error = ReadRegisterSet(set);
  if (error.Fail())
    return error;

  switch (set) {
  case GPRegSet:
  case DBRegSet:
    ::memcpy(GetOffsetRegSetData(set, reg_info->byte_offset),
             reg_value.GetBytes(), reg_value.GetByteSize());
    break;
  case FPRegSet: {
    void *data = GetOffsetRegSetData(set, reg_info->byte_offset);
    FXSAVE *fpr = reinterpret_cast<FXSAVE *>(m_xstate.data() +
                                             offsetof(xstate, xs_fxsave));
    if (data == &fpr->ftag) // ftag
      fpr->ftag = FullToAbridgedTagWord(reg_value.GetAsUInt16());
    else
      ::memcpy(data, reg_value.GetBytes(), reg_value.GetByteSize());
    if (data >= &fpr->xmm)
      new_xstate_bv |= XCR0_SSE;
    else if (data >= &fpr->mxcsr && data < &fpr->stmm)
      new_xstate_bv |= XCR0_SSE;
    else
      new_xstate_bv |= XCR0_X87;
    break;
  }
  case YMMRegSet: {
    std::optional<YMMSplitPtr> ymm_reg = GetYMMSplitReg(reg);
    if (!ymm_reg) {
      error = Status::FromErrorStringWithFormat(
          "register \"%s\" not supported by CPU/kernel", reg_info->name);
    } else {
      YMMReg ymm;
      ::memcpy(ymm.bytes, reg_value.GetBytes(), reg_value.GetByteSize());
      YMMToXState(ymm, ymm_reg->xmm, ymm_reg->ymm_hi);
      new_xstate_bv |= XCR0_SSE | XCR0_YMM_Hi128;
    }
    break;
  }
  case MPXRegSet:
    llvm_unreachable("MPX regset should have returned error");
  }

  if (new_xstate_bv != 0)
    reinterpret_cast<xstate *>(m_xstate.data())->xs_xstate_bv |= new_xstate_bv;
  return WriteRegisterSet(set);
}

Status NativeRegisterContextNetBSD_x86::ReadAllRegisterValues(
    lldb::WritableDataBufferSP &data_sp) {
  Status error;

  data_sp.reset(new DataBufferHeap(REG_CONTEXT_SIZE, 0));
  error = ReadRegisterSet(GPRegSet);
  if (error.Fail())
    return error;

  uint8_t *dst = data_sp->GetBytes();
  ::memcpy(dst, m_gpr.data(), GetRegisterInfoInterface().GetGPRSize());
  dst += GetRegisterInfoInterface().GetGPRSize();

  return error;
}

Status NativeRegisterContextNetBSD_x86::WriteAllRegisterValues(
    const lldb::DataBufferSP &data_sp) {
  Status error;

  if (!data_sp) {
    error = Status::FromErrorStringWithFormat(
        "NativeRegisterContextNetBSD_x86::%s invalid data_sp provided",
        __FUNCTION__);
    return error;
  }

  if (data_sp->GetByteSize() != REG_CONTEXT_SIZE) {
    error = Status::FromErrorStringWithFormat(
        "NativeRegisterContextNetBSD_x86::%s data_sp contained mismatched "
        "data size, expected %zu, actual %" PRIu64,
        __FUNCTION__, REG_CONTEXT_SIZE, data_sp->GetByteSize());
    return error;
  }

  const uint8_t *src = data_sp->GetBytes();
  if (src == nullptr) {
    error = Status::FromErrorStringWithFormat(
        "NativeRegisterContextNetBSD_x86::%s "
        "DataBuffer::GetBytes() returned a null "
        "pointer",
        __FUNCTION__);
    return error;
  }
  ::memcpy(m_gpr.data(), src, GetRegisterInfoInterface().GetGPRSize());

  error = WriteRegisterSet(GPRegSet);
  if (error.Fail())
    return error;
  src += GetRegisterInfoInterface().GetGPRSize();

  return error;
}

llvm::Error NativeRegisterContextNetBSD_x86::CopyHardwareWatchpointsFrom(
    NativeRegisterContextNetBSD &source) {
  auto &r_source = static_cast<NativeRegisterContextNetBSD_x86 &>(source);
  // NB: This implicitly reads the whole dbreg set.
  RegisterValue dr7;
  Status res = r_source.ReadRegister(GetDR(7), dr7);
  if (!res.Fail()) {
    // copy dbregs only if any watchpoints were set
    if ((dr7.GetAsUInt64() & 0xFF) == 0)
      return llvm::Error::success();

    m_dbr = r_source.m_dbr;
    res = WriteRegisterSet(DBRegSet);
  }
  return res.ToError();
}

uint8_t *
NativeRegisterContextNetBSD_x86::GetOffsetRegSetData(RegSetKind set,
                                                        size_t reg_offset) {
  uint8_t *base;
  switch (set) {
  case GPRegSet:
    base = m_gpr.data();
    break;
  case FPRegSet:
    base = m_xstate.data() + offsetof(xstate, xs_fxsave);
    break;
  case DBRegSet:
    base = m_dbr.data();
    break;
  case YMMRegSet:
    llvm_unreachable("GetRegSetData() is unsuitable for this regset.");
  case MPXRegSet:
    llvm_unreachable("MPX regset should have returned error");
  }
  assert(reg_offset >= m_regset_offsets[set]);
  return base + (reg_offset - m_regset_offsets[set]);
}

std::optional<NativeRegisterContextNetBSD_x86::YMMSplitPtr>
NativeRegisterContextNetBSD_x86::GetYMMSplitReg(uint32_t reg) {
  auto xst = reinterpret_cast<xstate *>(m_xstate.data());
  if (!(xst->xs_rfbm & XCR0_SSE) || !(xst->xs_rfbm & XCR0_YMM_Hi128))
    return std::nullopt;

  uint32_t reg_index;
  switch (GetRegisterInfoInterface().GetTargetArchitecture().GetMachine()) {
  case llvm::Triple::x86:
    reg_index = reg - registers::x86::id::ymm0;
    break;
  case llvm::Triple::x86_64:
    reg_index = reg - registers::x86_64::id::ymm0;
    break;
  default:
    llvm_unreachable("Unhandled target architecture.");
  }

  return YMMSplitPtr{&xst->xs_fxsave.fx_xmm[reg_index],
                     &xst->xs_ymm_hi128.xs_ymm[reg_index]};
}

#endif // defined(__i386__) || defined(__x86_64__)
