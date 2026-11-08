//===-- NativeRegisterContextFreeBSD_x86.cpp ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#if defined(__i386__) || defined(__x86_64__)

#include <set>
#include "lldb/Register/Arch/lldb-x86_64-register-enums.h"
#include "lldb/Register/Arch/lldb-x86-register-enums.h"

#include "NativeRegisterContextFreeBSD_x86.h"

// clang-format off
#include <x86/fpu.h>
#include <x86/specialreg.h>
#include <cpuid.h>
// clang-format on

#include "lldb/Host/HostInfo.h"
#include "lldb/Utility/DataBufferHeap.h"
#include "lldb/Utility/Log.h"
#include "lldb/Register/RegisterValue.h"
#include "lldb/Utility/Status.h"

#include "NativeProcessFreeBSD.h"
#include "lldb/Register/Layouts/FreeBSD/RegisterLayoutFreeBSD_i386.h"
#include "lldb/Register/Layouts/FreeBSD/RegisterLayoutFreeBSD_x86_64.h"
#include <optional>

using namespace lldb_private;
using namespace lldb_private::process_freebsd;

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
struct RegisterGroupNumbers {
  const char *name;
  const char *short_name;
  std::vector<uint32_t> registers;
};

static const RegisterGroupNumbers g_reg_sets_i386[k_num_register_sets] = {
    {"General Purpose Registers", "gpr",
     std::vector<RegisterID>(g_gpr_regnums_i386,
                             g_gpr_regnums_i386 +
                                 registers::x86::id::k_num_gpr_registers)},
    {"Floating Point Registers", "fpu",
     std::vector<RegisterID>(g_fpu_regnums_i386,
                             g_fpu_regnums_i386 +
                                 registers::x86::id::k_num_fpr_registers)},
    {"Debug Registers", "dbr",
     std::vector<RegisterID>(g_dbr_regnums_i386,
                             g_dbr_regnums_i386 +
                                 registers::x86::id::k_num_dbr_registers)},
    {"Advanced Vector Extensions", "avx",
     std::vector<RegisterID>(g_avx_regnums_i386,
                             g_avx_regnums_i386 +
                                 registers::x86::id::k_num_avx_registers)},
    {"Memory Protection Extensions", "mpx",
     std::vector<RegisterID>(g_mpx_regnums_i386,
                             g_mpx_regnums_i386 +
                                 registers::x86::id::k_num_mpx_registers)}};

// Register sets for x86 64-bit.
static const RegisterGroupNumbers g_reg_sets_x86_64[k_num_register_sets] = {
    {"General Purpose Registers", "gpr",
     std::vector<RegisterID>(g_gpr_regnums_x86_64,
                             g_gpr_regnums_x86_64 +
                                 registers::x86_64::id::k_num_gpr_registers)},
    {"Floating Point Registers", "fpu",
     std::vector<RegisterID>(g_fpu_regnums_x86_64,
                             g_fpu_regnums_x86_64 +
                                 registers::x86_64::id::k_num_fpr_registers)},
    {"Debug Registers", "dbr",
     std::vector<RegisterID>(g_dbr_regnums_x86_64,
                             g_dbr_regnums_x86_64 +
                                 registers::x86_64::id::k_num_dbr_registers)},
    {"Advanced Vector Extensions", "avx",
     std::vector<RegisterID>(g_avx_regnums_x86_64,
                             g_avx_regnums_x86_64 +
                                 registers::x86_64::id::k_num_avx_registers)},
    {"Memory Protection Extensions", "mpx",
     std::vector<RegisterID>(g_mpx_regnums_x86_64,
                             g_mpx_regnums_x86_64 +
                                 registers::x86_64::id::k_num_mpx_registers)}};

#define REG_CONTEXT_SIZE (GetBankSize(0))

NativeRegisterContextFreeBSD *
NativeRegisterContextFreeBSD::CreateHostNativeRegisterContextFreeBSD(
    const ArchSpec &target_arch, NativeThreadFreeBSD &native_thread) {
  return new NativeRegisterContextFreeBSD_x86(target_arch, native_thread);
}

// NativeRegisterContextFreeBSD_x86 members.

static RegisterLayout CreateRegisterLayout(const ArchSpec &target_arch) {
  if (HostInfo::GetArchitecture().GetAddressByteSize() == 4) {
    // 32-bit hosts use the i386 saved-state layout.
    return CreateFreeBSDI386RegisterLayout(target_arch);
  } else {
    assert((HostInfo::GetArchitecture().GetAddressByteSize() == 8) &&
           "Register setting path assumes this is a 64-bit host");
    // X86_64 hosts know how to work with 64-bit and 32-bit EXEs using the
    // x86_64 register context.
    return CreateFreeBSDX86_64RegisterLayout(target_arch);
  }
}

NativeRegisterContextFreeBSD_x86::NativeRegisterContextFreeBSD_x86(
    const ArchSpec &target_arch, NativeThreadFreeBSD &native_thread)
    : NativeRegisterContext(native_thread, target_arch,
                            CreateRegisterLayout(target_arch)) {
  assert(m_gpr.size() == GetBankSize(0));

  // Bind native presentation policy to this schema generation once. The
  // private definitions above use the backend's external numbering.
  auto select_group = [&](uint32_t set_index) -> const RegisterGroupNumbers * {
    switch (GetTargetArchitecture().GetMachine()) {
    case llvm::Triple::x86:
      return &g_reg_sets_i386[set_index];
    case llvm::Triple::x86_64:
      return &g_reg_sets_x86_64[set_index];
    default:
      llvm_unreachable("Unhandled target architecture.");
    }
  };
  auto previous = GetDescription();
  std::vector<RegisterGroup> groups;
  std::set<RegisterID> visible;
  for (uint32_t index = 0; index < k_num_register_sets; ++index) {
    const auto *input = select_group(index);
    if (!input)
      continue;
    RegisterGroup group{input->name, input->short_name, {}};
    for (uint32_t number : input->registers) {
      const auto *reg = previous->FindByNumber(lldb::eRegisterKindLLDB, number);
      assert(reg && "native presentation names an unknown register");
      if (reg) {
        group.registers.push_back(reg->id);
        visible.insert(reg->id);
      }
    }
    groups.push_back(std::move(group));
  }
  auto description = llvm::cantFail(RegisterDescription::Create(
      {previous->GetRegisters().begin(), previous->GetRegisters().end()},
      std::move(groups),
      {previous->GetOwnedTypes().begin(), previous->GetOwnedTypes().end()}));
  std::vector<RegisterID> order;
  for (RegisterID id : previous->GetUserRegisters())
    if (visible.count(id))
      order.push_back(id);
  auto presentation = llvm::cantFail(description.WithUserRegisters(order));
  SetRegisterLayout(llvm::cantFail(RegisterLayout::Create(
      std::make_shared<RegisterDescription>(std::move(presentation)),
      GetLayout())));
}

std::optional<NativeRegisterContextFreeBSD_x86::RegSetKind>
NativeRegisterContextFreeBSD_x86::GetSetForNativeRegNum(
    uint32_t reg_num) const {
  switch (GetTargetArchitecture().GetMachine()) {
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

Status NativeRegisterContextFreeBSD_x86::ReadRegisterSet(RegSetKind set) {
  switch (set) {
  case GPRegSet:
    return NativeProcessFreeBSD::PtraceWrapper(PT_GETREGS, m_thread.GetID(),
                                               m_gpr.data());
  case FPRegSet:
#if defined(__x86_64__)
    return NativeProcessFreeBSD::PtraceWrapper(PT_GETFPREGS, m_thread.GetID(),
                                               m_fpr.data());
#else
    return NativeProcessFreeBSD::PtraceWrapper(PT_GETXMMREGS, m_thread.GetID(),
                                               m_fpr.data());
#endif
  case DBRegSet:
    return NativeProcessFreeBSD::PtraceWrapper(PT_GETDBREGS, m_thread.GetID(),
                                               m_dbr.data());
  case YMMRegSet:
  case MPXRegSet: {
    struct ptrace_xstate_info info;
    Status ret = NativeProcessFreeBSD::PtraceWrapper(
        PT_GETXSTATE_INFO, GetProcessPid(), &info, sizeof(info));
    if (!ret.Success())
      return ret;

    assert(info.xsave_mask & XFEATURE_ENABLED_X87);
    assert(info.xsave_mask & XFEATURE_ENABLED_SSE);

    m_xsave_offsets[YMMRegSet] = LLDB_INVALID_XSAVE_OFFSET;
    if (info.xsave_mask & XFEATURE_ENABLED_YMM_HI128) {
      uint32_t eax, ecx, edx;
      __get_cpuid_count(0x0D, 2, &eax, &m_xsave_offsets[YMMRegSet], &ecx, &edx);
    }

    m_xsave.resize(info.xsave_len);
    return NativeProcessFreeBSD::PtraceWrapper(PT_GETXSTATE, GetProcessPid(),
                                               m_xsave.data(), m_xsave.size());
  }
  }
  llvm_unreachable("NativeRegisterContextFreeBSD_x86::ReadRegisterSet");
}

Status NativeRegisterContextFreeBSD_x86::WriteRegisterSet(RegSetKind set) {
  switch (set) {
  case GPRegSet:
    return NativeProcessFreeBSD::PtraceWrapper(PT_SETREGS, m_thread.GetID(),
                                               m_gpr.data());
  case FPRegSet:
#if defined(__x86_64__)
    return NativeProcessFreeBSD::PtraceWrapper(PT_SETFPREGS, m_thread.GetID(),
                                               m_fpr.data());
#else
    return NativeProcessFreeBSD::PtraceWrapper(PT_SETXMMREGS, m_thread.GetID(),
                                               m_fpr.data());
#endif
  case DBRegSet:
    return NativeProcessFreeBSD::PtraceWrapper(PT_SETDBREGS, m_thread.GetID(),
                                               m_dbr.data());
  case YMMRegSet:
  case MPXRegSet:
    // ReadRegisterSet() must always be called before WriteRegisterSet().
    assert(m_xsave.size() > 0);
    return NativeProcessFreeBSD::PtraceWrapper(PT_SETXSTATE, GetProcessPid(),
                                               m_xsave.data(), m_xsave.size());
  }
  llvm_unreachable("NativeRegisterContextFreeBSD_x86::WriteRegisterSet");
}

llvm::Expected<RegisterValue>
NativeRegisterContextFreeBSD_x86::ReadRegister(RegisterID id) {
  RegisterValue reg_value;
  auto description = GetDescription();
  const auto *descriptor = description ? description->Find(id) : nullptr;
  if (!descriptor)
    return Status::FromErrorString("unknown register").ToError();
  const uint32_t reg = descriptor->numbers[lldb::eRegisterKindLLDB];
  auto set = GetSetForNativeRegNum(reg);
  if (!set)
    return Status::FromErrorString("unsupported register set").ToError();
  Status status = ReadRegisterSet(*set);
  if (status.Fail())
    return status.ToError();
  const bool ymm = *set == YMMRegSet;
  const uint32_t first_ymm =
      GetTargetArchitecture().GetMachine() == llvm::Triple::x86
          ? registers::x86::id::ymm0
          : registers::x86_64::id::ymm0;
  auto split = ymm ? GetYMMSplitReg(first_ymm) : std::nullopt;
  if (ymm && !split)
    return Status::FromErrorString("AVX state is unavailable").ToError();
  uint8_t *fp = ymm ? m_xsave.data() : m_fpr.data();
  auto *fxsave = reinterpret_cast<FXSAVE *>(fp);
  if (llvm::StringRef(descriptor->name) == "ftag") {
    reg_value.SetUInt16(
        AbridgedToFullTagWord(fxsave->ftag, fxsave->fstat, fxsave->stmm));
    return reg_value;
  }
  auto value = ReadBankRegister(
      descriptor->id,
      {{x86_layout::GP, m_gpr},
       {x86_layout::FP, {fp, sizeof(FXSAVE)}},
       {x86_layout::Debug, m_dbr},
       {x86_layout::YMMHigh,
        {split ? static_cast<const uint8_t *>(split->ymm_hi) : nullptr,
         split ? 256u : 0u}}});
  return value;
}

llvm::Error NativeRegisterContextFreeBSD_x86::WriteRegister(
    RegisterID id, const RegisterValue &reg_value) {
  auto description = GetDescription();
  const auto *descriptor = description ? description->Find(id) : nullptr;
  if (!descriptor)
    return Status::FromErrorString("unknown register").ToError();
  const uint32_t reg = descriptor->numbers[lldb::eRegisterKindLLDB];
  auto set = GetSetForNativeRegNum(reg);
  if (!set)
    return Status::FromErrorString("unsupported register set").ToError();
  Status status = ReadRegisterSet(*set);
  if (status.Fail())
    return status.ToError();
  const bool ymm = *set == YMMRegSet;
  const uint32_t first_ymm =
      GetTargetArchitecture().GetMachine() == llvm::Triple::x86
          ? registers::x86::id::ymm0
          : registers::x86_64::id::ymm0;
  auto split = ymm ? GetYMMSplitReg(first_ymm) : std::nullopt;
  if (ymm && !split)
    return Status::FromErrorString("AVX state is unavailable").ToError();
  uint8_t *fp = ymm ? m_xsave.data() : m_fpr.data();
  auto *fxsave = reinterpret_cast<FXSAVE *>(fp);
  if (llvm::StringRef(descriptor->name) == "ftag")
    fxsave->ftag = FullToAbridgedTagWord(reg_value.GetAsUInt16());
  else if (auto error = WriteBankRegister(
               descriptor->id, reg_value,
               {{x86_layout::GP, m_gpr},
                {x86_layout::FP, {fp, sizeof(FXSAVE)}},
                {x86_layout::Debug, m_dbr},
                {x86_layout::YMMHigh,
                 {split ? static_cast<uint8_t *>(split->ymm_hi) : nullptr,
                  split ? 256u : 0u}}}))
    return error;
  return WriteRegisterSet(*set).ToError();
}

Status NativeRegisterContextFreeBSD_x86::ReadAllRegisterValues(
    lldb::WritableDataBufferSP &data_sp) {
  Status error;

  data_sp.reset(new DataBufferHeap(REG_CONTEXT_SIZE, 0));
  error = ReadRegisterSet(GPRegSet);
  if (error.Fail())
    return error;

  uint8_t *dst = data_sp->GetBytes();
  ::memcpy(dst, m_gpr.data(), GetBankSize(0));
  dst += GetBankSize(0);

  return error;
}

Status NativeRegisterContextFreeBSD_x86::WriteAllRegisterValues(
    const lldb::DataBufferSP &data_sp) {
  Status error;

  if (!data_sp) {
    error = Status::FromErrorStringWithFormat(
        "NativeRegisterContextFreeBSD_x86::%s invalid data_sp provided",
        __FUNCTION__);
    return error;
  }

  if (data_sp->GetByteSize() != REG_CONTEXT_SIZE) {
    error = Status::FromErrorStringWithFormat(
        "NativeRegisterContextFreeBSD_x86::%s data_sp contained mismatched "
        "data size, expected %zu, actual %" PRIu64,
        __FUNCTION__, REG_CONTEXT_SIZE, data_sp->GetByteSize());
    return error;
  }

  const uint8_t *src = data_sp->GetBytes();
  if (src == nullptr) {
    error = Status::FromErrorStringWithFormat(
        "NativeRegisterContextFreeBSD_x86::%s "
        "DataBuffer::GetBytes() returned a null "
        "pointer",
        __FUNCTION__);
    return error;
  }
  ::memcpy(m_gpr.data(), src, GetBankSize(0));

  error = WriteRegisterSet(GPRegSet);
  if (error.Fail())
    return error;
  src += GetBankSize(0);

  return error;
}

llvm::Error NativeRegisterContextFreeBSD_x86::CopyHardwareWatchpointsFrom(
    NativeRegisterContextFreeBSD &source) {
  auto &r_source = static_cast<NativeRegisterContextFreeBSD_x86 &>(source);
  // NB: This implicitly reads the whole dbreg set.
  auto dr7 = r_source.ReadRegister(r_source.GetDR(7));
  if (!dr7)
    return dr7.takeError();
  // Copy dbregs only if any watchpoints were set.
  if ((dr7->GetAsUInt64() & 0xFF) == 0)
    return llvm::Error::success();
  m_dbr = r_source.m_dbr;
  return WriteRegisterSet(DBRegSet).ToError();
}

std::optional<NativeRegisterContextFreeBSD_x86::YMMSplitPtr>
NativeRegisterContextFreeBSD_x86::GetYMMSplitReg(uint32_t reg) {
  uint32_t offset = m_xsave_offsets[YMMRegSet];
  if (offset == LLDB_INVALID_XSAVE_OFFSET)
    return std::nullopt;

  uint32_t reg_index;
  switch (GetTargetArchitecture().GetMachine()) {
  case llvm::Triple::x86:
    reg_index = reg - registers::x86::id::ymm0;
    break;
  case llvm::Triple::x86_64:
    reg_index = reg - registers::x86_64::id::ymm0;
    break;
  default:
    llvm_unreachable("Unhandled target architecture.");
  }

  auto *fpreg = reinterpret_cast<struct savexmm_ymm *>(m_xsave.data());
  auto *ymmreg = reinterpret_cast<struct ymmacc *>(m_xsave.data() + offset);

  return YMMSplitPtr{&fpreg->sv_xmm[reg_index], &ymmreg[reg_index]};
}

#endif // defined(__i386__) || defined(__x86_64__)
