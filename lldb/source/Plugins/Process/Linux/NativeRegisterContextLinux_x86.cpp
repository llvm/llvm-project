//===-- NativeRegisterContextLinux_x86.cpp --------------------------------===//
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

#include "NativeRegisterContextLinux_x86.h"

#include "Plugins/Process/Linux/NativeProcessLinux.h"
#include "Plugins/Process/Linux/NativeThreadLinux.h"
#include "lldb/Host/HostInfo.h"
#include "lldb/Register/Layouts/Linux/RegisterLayoutLinux_i386.h"
#include "lldb/Register/Layouts/Linux/RegisterLayoutLinux_x86_64.h"
#include "lldb/Utility/DataBufferHeap.h"
#include "lldb/Utility/Log.h"
#include "lldb/Register/RegisterValue.h"
#include "lldb/Utility/Status.h"

#include <algorithm>
#include <cstddef>
#include <optional>

// System includes - They have to be included after framework includes because
// they define some macros which collide with variable names in other modules.
#include <asm/ldt.h>
#include <cpuid.h>
#include <linux/elf.h>
#include <sys/ptrace.h>
#include <sys/user.h>

#ifndef PTRACE_GET_THREAD_AREA
#define PTRACE_GET_THREAD_AREA 25
#endif

// Newer toolchains define __get_cpuid_count in cpuid.h, but some
// older-but-still-supported ones (e.g. gcc 5.4.0) don't, so we
// define it locally here, following the definition in clang/lib/Headers.
static inline int get_cpuid_count(unsigned int __leaf,
                                  unsigned int __subleaf,
                                  unsigned int *__eax, unsigned int *__ebx,
                                  unsigned int *__ecx, unsigned int *__edx)
{
  unsigned int __max_leaf = __get_cpuid_max(__leaf & 0x80000000, nullptr);

  if (__max_leaf == 0 || __max_leaf < __leaf)
    return 0;

  __cpuid_count(__leaf, __subleaf, *__eax, *__ebx, *__ecx, *__edx);
  return 1;
}

using namespace lldb_private;
using namespace lldb_private::process_linux;

// Linux exposes the i386 thread pointer as a synthetic register between the
// always-available FPU registers and the optional extended register sets.
constexpr uint32_t k_i386_thread_pointer_index =
    registers::x86::id::k_first_avx;

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

// x86 32-bit AVX registers.
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
              " g_avx_regnums_i386 has wrong number of register infos");

// x64 32-bit MPX registers.
static const uint32_t g_mpx_regnums_i386[] = {
    registers::x86::id::bnd0,    registers::x86::id::bnd1,
    registers::x86::id::bnd2,    registers::x86::id::bnd3,
    registers::x86::id::bndcfgu, registers::x86::id::bndstatus,
    LLDB_INVALID_REGNUM // register sets need to end with this flag
};
static_assert((sizeof(g_mpx_regnums_i386) / sizeof(g_mpx_regnums_i386[0])) -
                      1 ==
                  registers::x86::id::k_num_mpx_registers,
              "g_mpx_regnums_x86_64 has wrong number of register infos");

// x86 64-bit general purpose registers.
static const uint32_t g_gpr_regnums_x86_64[] = {
    registers::x86_64::id::with_base::rax,
    registers::x86_64::id::with_base::rbx,
    registers::x86_64::id::with_base::rcx,
    registers::x86_64::id::with_base::rdx,
    registers::x86_64::id::with_base::rdi,
    registers::x86_64::id::with_base::rsi,
    registers::x86_64::id::with_base::rbp,
    registers::x86_64::id::with_base::rsp,
    registers::x86_64::id::with_base::r8,
    registers::x86_64::id::with_base::r9,
    registers::x86_64::id::with_base::r10,
    registers::x86_64::id::with_base::r11,
    registers::x86_64::id::with_base::r12,
    registers::x86_64::id::with_base::r13,
    registers::x86_64::id::with_base::r14,
    registers::x86_64::id::with_base::r15,
    registers::x86_64::id::with_base::rip,
    registers::x86_64::id::with_base::rflags,
    registers::x86_64::id::with_base::cs,
    registers::x86_64::id::with_base::fs,
    registers::x86_64::id::with_base::gs,
    registers::x86_64::id::with_base::ss,
    registers::x86_64::id::with_base::fs_base,
    registers::x86_64::id::with_base::gs_base,
    registers::x86_64::id::with_base::ds,
    registers::x86_64::id::with_base::es,
    registers::x86_64::id::with_base::eax,
    registers::x86_64::id::with_base::ebx,
    registers::x86_64::id::with_base::ecx,
    registers::x86_64::id::with_base::edx,
    registers::x86_64::id::with_base::edi,
    registers::x86_64::id::with_base::esi,
    registers::x86_64::id::with_base::ebp,
    registers::x86_64::id::with_base::esp,
    registers::x86_64::id::with_base::r8d,  // Low 32 bits or r8
    registers::x86_64::id::with_base::r9d,  // Low 32 bits or r9
    registers::x86_64::id::with_base::r10d, // Low 32 bits or r10
    registers::x86_64::id::with_base::r11d, // Low 32 bits or r11
    registers::x86_64::id::with_base::r12d, // Low 32 bits or r12
    registers::x86_64::id::with_base::r13d, // Low 32 bits or r13
    registers::x86_64::id::with_base::r14d, // Low 32 bits or r14
    registers::x86_64::id::with_base::r15d, // Low 32 bits or r15
    registers::x86_64::id::with_base::ax,
    registers::x86_64::id::with_base::bx,
    registers::x86_64::id::with_base::cx,
    registers::x86_64::id::with_base::dx,
    registers::x86_64::id::with_base::di,
    registers::x86_64::id::with_base::si,
    registers::x86_64::id::with_base::bp,
    registers::x86_64::id::with_base::sp,
    registers::x86_64::id::with_base::r8w,  // Low 16 bits or r8
    registers::x86_64::id::with_base::r9w,  // Low 16 bits or r9
    registers::x86_64::id::with_base::r10w, // Low 16 bits or r10
    registers::x86_64::id::with_base::r11w, // Low 16 bits or r11
    registers::x86_64::id::with_base::r12w, // Low 16 bits or r12
    registers::x86_64::id::with_base::r13w, // Low 16 bits or r13
    registers::x86_64::id::with_base::r14w, // Low 16 bits or r14
    registers::x86_64::id::with_base::r15w, // Low 16 bits or r15
    registers::x86_64::id::with_base::ah,
    registers::x86_64::id::with_base::bh,
    registers::x86_64::id::with_base::ch,
    registers::x86_64::id::with_base::dh,
    registers::x86_64::id::with_base::al,
    registers::x86_64::id::with_base::bl,
    registers::x86_64::id::with_base::cl,
    registers::x86_64::id::with_base::dl,
    registers::x86_64::id::with_base::dil,
    registers::x86_64::id::with_base::sil,
    registers::x86_64::id::with_base::bpl,
    registers::x86_64::id::with_base::spl,
    registers::x86_64::id::with_base::r8l,  // Low 8 bits or r8
    registers::x86_64::id::with_base::r9l,  // Low 8 bits or r9
    registers::x86_64::id::with_base::r10l, // Low 8 bits or r10
    registers::x86_64::id::with_base::r11l, // Low 8 bits or r11
    registers::x86_64::id::with_base::r12l, // Low 8 bits or r12
    registers::x86_64::id::with_base::r13l, // Low 8 bits or r13
    registers::x86_64::id::with_base::r14l, // Low 8 bits or r14
    registers::x86_64::id::with_base::r15l, // Low 8 bits or r15
    LLDB_INVALID_REGNUM // register sets need to end with this flag
};
static_assert((sizeof(g_gpr_regnums_x86_64) / sizeof(g_gpr_regnums_x86_64[0])) -
                      1 ==
                  registers::x86_64::id::with_base::k_num_gpr_registers,
              "g_gpr_regnums_x86_64 has wrong number of register infos");

// x86 64-bit floating point registers.
static const uint32_t g_fpu_regnums_x86_64[] = {
    registers::x86_64::id::with_base::fctrl,
    registers::x86_64::id::with_base::fstat,
    registers::x86_64::id::with_base::ftag,
    registers::x86_64::id::with_base::fop,
    registers::x86_64::id::with_base::fiseg,
    registers::x86_64::id::with_base::fioff,
    registers::x86_64::id::with_base::fip,
    registers::x86_64::id::with_base::foseg,
    registers::x86_64::id::with_base::fooff,
    registers::x86_64::id::with_base::fdp,
    registers::x86_64::id::with_base::mxcsr,
    registers::x86_64::id::with_base::mxcsrmask,
    registers::x86_64::id::with_base::st0,
    registers::x86_64::id::with_base::st1,
    registers::x86_64::id::with_base::st2,
    registers::x86_64::id::with_base::st3,
    registers::x86_64::id::with_base::st4,
    registers::x86_64::id::with_base::st5,
    registers::x86_64::id::with_base::st6,
    registers::x86_64::id::with_base::st7,
    registers::x86_64::id::with_base::mm0,
    registers::x86_64::id::with_base::mm1,
    registers::x86_64::id::with_base::mm2,
    registers::x86_64::id::with_base::mm3,
    registers::x86_64::id::with_base::mm4,
    registers::x86_64::id::with_base::mm5,
    registers::x86_64::id::with_base::mm6,
    registers::x86_64::id::with_base::mm7,
    registers::x86_64::id::with_base::xmm0,
    registers::x86_64::id::with_base::xmm1,
    registers::x86_64::id::with_base::xmm2,
    registers::x86_64::id::with_base::xmm3,
    registers::x86_64::id::with_base::xmm4,
    registers::x86_64::id::with_base::xmm5,
    registers::x86_64::id::with_base::xmm6,
    registers::x86_64::id::with_base::xmm7,
    registers::x86_64::id::with_base::xmm8,
    registers::x86_64::id::with_base::xmm9,
    registers::x86_64::id::with_base::xmm10,
    registers::x86_64::id::with_base::xmm11,
    registers::x86_64::id::with_base::xmm12,
    registers::x86_64::id::with_base::xmm13,
    registers::x86_64::id::with_base::xmm14,
    registers::x86_64::id::with_base::xmm15,
    LLDB_INVALID_REGNUM // register sets need to end with this flag
};
static_assert((sizeof(g_fpu_regnums_x86_64) / sizeof(g_fpu_regnums_x86_64[0])) -
                      1 ==
                  registers::x86_64::id::with_base::k_num_fpr_registers,
              "g_fpu_regnums_x86_64 has wrong number of register infos");

// x86 64-bit AVX registers.
static const uint32_t g_avx_regnums_x86_64[] = {
    registers::x86_64::id::with_base::ymm0,
    registers::x86_64::id::with_base::ymm1,
    registers::x86_64::id::with_base::ymm2,
    registers::x86_64::id::with_base::ymm3,
    registers::x86_64::id::with_base::ymm4,
    registers::x86_64::id::with_base::ymm5,
    registers::x86_64::id::with_base::ymm6,
    registers::x86_64::id::with_base::ymm7,
    registers::x86_64::id::with_base::ymm8,
    registers::x86_64::id::with_base::ymm9,
    registers::x86_64::id::with_base::ymm10,
    registers::x86_64::id::with_base::ymm11,
    registers::x86_64::id::with_base::ymm12,
    registers::x86_64::id::with_base::ymm13,
    registers::x86_64::id::with_base::ymm14,
    registers::x86_64::id::with_base::ymm15,
    LLDB_INVALID_REGNUM // register sets need to end with this flag
};
static_assert((sizeof(g_avx_regnums_x86_64) / sizeof(g_avx_regnums_x86_64[0])) -
                      1 ==
                  registers::x86_64::id::with_base::k_num_avx_registers,
              "g_avx_regnums_x86_64 has wrong number of register infos");

// x86 64-bit MPX registers.
static const uint32_t g_mpx_regnums_x86_64[] = {
    registers::x86_64::id::with_base::bnd0,
    registers::x86_64::id::with_base::bnd1,
    registers::x86_64::id::with_base::bnd2,
    registers::x86_64::id::with_base::bnd3,
    registers::x86_64::id::with_base::bndcfgu,
    registers::x86_64::id::with_base::bndstatus,
    LLDB_INVALID_REGNUM // register sets need to end with this flag
};
static_assert((sizeof(g_mpx_regnums_x86_64) / sizeof(g_mpx_regnums_x86_64[0])) -
                      1 ==
                  registers::x86_64::id::with_base::k_num_mpx_registers,
              "g_mpx_regnums_x86_64 has wrong number of register infos");

// Number of register sets provided by this context.
constexpr unsigned k_num_extended_register_sets = 2, k_num_register_sets = 4;

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
                                 registers::x86_64::id::with_base::k_num_gpr_registers)},
    {"Floating Point Registers", "fpu",
     std::vector<RegisterID>(g_fpu_regnums_x86_64,
                             g_fpu_regnums_x86_64 +
                                 registers::x86_64::id::with_base::k_num_fpr_registers)},
    {"Advanced Vector Extensions", "avx",
     std::vector<RegisterID>(g_avx_regnums_x86_64,
                             g_avx_regnums_x86_64 +
                                 registers::x86_64::id::with_base::k_num_avx_registers)},
    {"Memory Protection Extensions", "mpx",
     std::vector<RegisterID>(g_mpx_regnums_x86_64,
                             g_mpx_regnums_x86_64 +
                                 registers::x86_64::id::with_base::k_num_mpx_registers)}};

#define REG_CONTEXT_SIZE (GetBankSize(0) + sizeof(FPR))

// Required ptrace defines.

// Support ptrace extensions even when compiled without required kernel support
#ifndef NT_X86_XSTATE
#define NT_X86_XSTATE 0x202
#endif
#ifndef NT_PRXFPREG
#define NT_PRXFPREG 0x46e62b7f
#endif

// On x86_64 NT_PRFPREG is used to access the FXSAVE area. On i386, we need to
// use NT_PRXFPREG.
static inline unsigned int fxsr_regset(const ArchSpec &arch) {
  return arch.GetAddressByteSize() == 8 ? NT_PRFPREG : NT_PRXFPREG;
}

// Required MPX define.

// Support MPX extensions also if compiled with compiler without MPX support.
#ifndef bit_MPX
#define bit_MPX 0x4000
#endif

// XCR0 extended register sets masks.
#define mask_XSTATE_AVX (1ULL << 2)
#define mask_XSTATE_BNDREGS (1ULL << 3)
#define mask_XSTATE_BNDCFG (1ULL << 4)
#define mask_XSTATE_MPX (mask_XSTATE_BNDREGS | mask_XSTATE_BNDCFG)

std::unique_ptr<NativeRegisterContextLinux>
NativeRegisterContextLinux::CreateHostNativeRegisterContextLinux(
    const ArchSpec &target_arch, NativeThreadLinux &native_thread) {
  return std::unique_ptr<NativeRegisterContextLinux>(
      new NativeRegisterContextLinux_x86(target_arch, native_thread));
}

llvm::Expected<ArchSpec>
NativeRegisterContextLinux::DetermineArchitecture(lldb::tid_t tid) {
  return DetermineArchitectureViaGPR(tid, sizeof(linux_x86_64::GPR));
}

// NativeRegisterContextLinux_x86 members.

static RegisterLayout CreateRegisterLayout(const ArchSpec &target_arch) {
  const bool host32 = HostInfo::GetArchitecture().GetAddressByteSize() == 4;
  assert((host32 || HostInfo::GetArchitecture().GetAddressByteSize() == 8) &&
         "unsupported native x86 saved-state layout");
  RegisterLayout layout = host32 ? CreateLinuxI386RegisterLayout(target_arch)
                                 : CreateLinuxX86_64RegisterLayout(target_arch);
  if (target_arch.GetMachine() != llvm::Triple::x86)
    return layout;

  // The thread-area API supplies this logical value; it has no slot in the
  // GPR bank. Preserve its public position without shifting CPU identities.
  auto old = layout.GetDescription();
  std::vector<RegisterDescriptor> registers(old->GetRegisters().begin(),
                                            old->GetRegisters().end());
  RegisterDescriptor thread_pointer;
  thread_pointer.id = 0x10010;
  thread_pointer.name = "gs_base";
  thread_pointer.byte_size = sizeof(uint32_t);
  thread_pointer.numbers[lldb::eRegisterKindGeneric] = LLDB_REGNUM_GENERIC_TP;
  thread_pointer.numbers[lldb::eRegisterKindLLDB] = 0x10010;
  registers.insert(registers.begin() + k_i386_thread_pointer_index,
                   std::move(thread_pointer));
  auto description = std::make_shared<RegisterDescription>(
      llvm::cantFail(RegisterDescription::Create(
          std::move(registers),
          std::vector<RegisterGroup>(old->GetGroups().begin(),
                                     old->GetGroups().end()),
          RegisterDescription::OwnedTypes(old->GetOwnedTypes().begin(),
                                          old->GetOwnedTypes().end()))));
  std::vector<RegisterID> visible(old->GetUserRegisters().begin(),
                                  old->GetUserRegisters().end());
  visible.insert(visible.begin() + k_i386_thread_pointer_index, 0x10010);
  auto presented = std::make_shared<RegisterDescription>(
      llvm::cantFail(description->WithUserRegisters(visible)));
  return llvm::cantFail(
      RegisterLayout::Create(std::move(presented), std::move(layout)));
}

// Return the size of the XSTATE area supported on this cpu. It is necessary to
// allocate the full size of the area even if we do not use/recognise all of it
// because ptrace(PTRACE_SETREGSET, NT_X86_XSTATE) will refuse to write to it if
// we do not pass it a buffer of sufficient size. The size is always at least
// sizeof(FPR) so that the allocated buffer can be safely cast to FPR*.
static std::size_t GetXSTATESize() {
  unsigned int eax, ebx, ecx, edx;
  // First check whether the XSTATE are is supported at all.
  if (!__get_cpuid(1, &eax, &ebx, &ecx, &edx) || !(ecx & bit_XSAVE))
    return sizeof(FPR);

  // Then fetch the maximum size of the area.
  if (!get_cpuid_count(0x0d, 0, &eax, &ebx, &ecx, &edx))
    return sizeof(FPR);
  return std::max<std::size_t>(ecx, sizeof(FPR));
}

NativeRegisterContextLinux_x86::NativeRegisterContextLinux_x86(
    const ArchSpec &target_arch, NativeThreadProtocol &native_thread)
    : NativeRegisterContext(native_thread, target_arch,
                            CreateRegisterLayout(target_arch)),
      NativeRegisterContextLinux(native_thread),
      m_xstate_type(XStateType::Invalid), m_reg_info(), m_gpr_x86_64() {
  const bool host64 = HostInfo::GetArchitecture().GetAddressByteSize() == 8;
  m_orig_ax = {
      host64 ? "orig_rax" : "orig_eax", host64 ? 8u : 4u,
      static_cast<uint32_t>(host64 ? offsetof(linux_x86_64::GPR, orig_rax)
                                   : offsetof(linux_i386::GPR, orig_eax))};
  // Set up data about ranges of valid registers.
  switch (target_arch.GetMachine()) {
  case llvm::Triple::x86: {
    m_reg_info.num_registers = registers::x86::id::k_num_registers + 1;
    m_reg_info.num_gpr_registers = registers::x86::id::k_num_gpr_registers;
    m_reg_info.num_fpr_registers = registers::x86::id::k_num_fpr_registers;
    m_reg_info.num_avx_registers = registers::x86::id::k_num_avx_registers;
    m_reg_info.num_mpx_registers = registers::x86::id::k_num_mpx_registers;
    m_reg_info.last_gpr = registers::x86::id::k_last_gpr;
    m_reg_info.first_fpr = registers::x86::id::k_first_fpr;
    m_reg_info.last_fpr = registers::x86::id::k_last_fpr;
    m_reg_info.first_st = registers::x86::id::st0;
    m_reg_info.last_st = registers::x86::id::st7;
    m_reg_info.first_mm = registers::x86::id::mm0;
    m_reg_info.last_mm = registers::x86::id::mm7;
    m_reg_info.first_xmm = registers::x86::id::xmm0;
    m_reg_info.last_xmm = registers::x86::id::xmm7;
    m_reg_info.first_ymm = registers::x86::id::ymm0;
    m_reg_info.last_ymm = registers::x86::id::ymm7;
    m_reg_info.first_mpxr = registers::x86::id::bnd0;
    m_reg_info.last_mpxr = registers::x86::id::bnd3;
    m_reg_info.first_mpxc = registers::x86::id::bndcfgu;
    m_reg_info.last_mpxc = registers::x86::id::bndstatus;
    m_reg_info.first_dr = registers::x86::id::dr0;
    m_reg_info.last_dr = registers::x86::id::dr7;
    m_reg_info.gpr_flags = registers::x86::id::eflags;
    break;
  }
  case llvm::Triple::x86_64:
    m_reg_info.num_registers =
        registers::x86_64::id::with_base::k_num_registers;
    m_reg_info.num_gpr_registers =
        registers::x86_64::id::with_base::k_num_gpr_registers;
    m_reg_info.num_fpr_registers =
        registers::x86_64::id::with_base::k_num_fpr_registers;
    m_reg_info.num_avx_registers =
        registers::x86_64::id::with_base::k_num_avx_registers;
    m_reg_info.num_mpx_registers =
        registers::x86_64::id::with_base::k_num_mpx_registers;
    m_reg_info.last_gpr = registers::x86_64::id::with_base::k_last_gpr;
    m_reg_info.first_fpr = registers::x86_64::id::with_base::k_first_fpr;
    m_reg_info.last_fpr = registers::x86_64::id::with_base::k_last_fpr;
    m_reg_info.first_st = registers::x86_64::id::with_base::st0;
    m_reg_info.last_st = registers::x86_64::id::with_base::st7;
    m_reg_info.first_mm = registers::x86_64::id::with_base::mm0;
    m_reg_info.last_mm = registers::x86_64::id::with_base::mm7;
    m_reg_info.first_xmm = registers::x86_64::id::with_base::xmm0;
    m_reg_info.last_xmm = registers::x86_64::id::with_base::xmm15;
    m_reg_info.first_ymm = registers::x86_64::id::with_base::ymm0;
    m_reg_info.last_ymm = registers::x86_64::id::with_base::ymm15;
    m_reg_info.first_mpxr = registers::x86_64::id::with_base::bnd0;
    m_reg_info.last_mpxr = registers::x86_64::id::with_base::bnd3;
    m_reg_info.first_mpxc = registers::x86_64::id::with_base::bndcfgu;
    m_reg_info.last_mpxc = registers::x86_64::id::with_base::bndstatus;
    m_reg_info.first_dr = registers::x86_64::id::with_base::dr0;
    m_reg_info.last_dr = registers::x86_64::id::with_base::dr7;
    m_reg_info.gpr_flags = registers::x86_64::id::with_base::rflags;
    break;
  default:
    assert(false && "Unhandled target architecture.");
    break;
  }

  std::size_t xstate_size = GetXSTATESize();
  m_xstate.reset(static_cast<FPR *>(std::malloc(xstate_size)));
  m_iovec.iov_base = m_xstate.get();
  m_iovec.iov_len = xstate_size;

  // Clear out the FPR state.
  ::memset(m_xstate.get(), 0, xstate_size);

  // Bind native presentation policy to this schema generation once. The
  // private definitions above use the backend's external numbering.
  auto select_group = [&](uint32_t set_index) -> const RegisterGroupNumbers * {
    if (!IsRegisterSetAvailable(set_index))
      return nullptr;

    switch (GetTargetArchitecture().GetMachine()) {
    case llvm::Triple::x86:
      return &g_reg_sets_i386[set_index];
    case llvm::Triple::x86_64:
      return &g_reg_sets_x86_64[set_index];
    default:
      assert(false && "Unhandled target architecture.");
      return nullptr;
    }

    return nullptr;
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
  if (GetTargetArchitecture().GetMachine() == llvm::Triple::x86)
    visible.insert(0x10010); // gs_base is an explicitly exposed Linux value.
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

// CONSIDER after local and llgs debugging are merged, register set support can
// be moved into a base x86-64 class with IsRegisterSetAvailable made virtual.

llvm::Expected<RegisterValue>
NativeRegisterContextLinux_x86::ReadRegister(RegisterID id) {
  RegisterValue reg_value;
  auto description = GetDescription();
  const auto *descriptor = description ? description->Find(id) : nullptr;
  if (!descriptor)
    return Status::FromErrorString("unknown register").ToError();
  if (IsThreadPointer(*descriptor)) {
    if (Status status = ReadThreadPointer(reg_value); status.Fail())
      return status.ToError();
    return reg_value;
  }
  const uint32_t reg = descriptor->numbers[lldb::eRegisterKindLLDB];
  if (IsDR(reg)) {
    if (Status status = ReadRegisterRaw(reg, reg_value); status.Fail())
      return status.ToError();
    return reg_value;
  }
  auto required = GetLayout().GetRequiredBanks(descriptor->id, false);
  if (!required)
    return required.takeError();
  for (RegisterBankID bank : *required) {
    Status status;
    switch (bank) {
    case x86_layout::GP:
      status = ReadGPR();
      break;
    case x86_layout::FP:
      status = ReadFPR();
      break;
    case x86_layout::YMMHigh:
      status = ReadFPR();
      if (status.Success() && !IsCPUFeatureAvailable(RegSet::avx))
        return Status::FromErrorString("AVX state is unavailable").ToError();
      break;
    case x86_layout::Bounds:
    case x86_layout::BoundsControl:
      status = ReadFPR();
      if (status.Success() && !IsCPUFeatureAvailable(RegSet::mpx))
        return Status::FromErrorString("MPX state is unavailable").ToError();
      break;
    default:
      return Status::FromErrorString("unsupported register bank").ToError();
    }
    if (status.Fail())
      return status.ToError();
  }
  if (llvm::StringRef(descriptor->name) == "ftag") {
    reg_value.SetUInt16(AbridgedToFullTagWord(
        m_xstate->fxsave.ftag, m_xstate->fxsave.fstat, m_xstate->fxsave.stmm));
    return reg_value;
  }
  auto value = ReadBankRegister(
      descriptor->id,
      {{x86_layout::GP,
        {reinterpret_cast<const uint8_t *>(&m_gpr_x86_64), GetGPRSize()}},
       {x86_layout::FP,
        {reinterpret_cast<const uint8_t *>(&m_xstate->fxsave), sizeof(FXSAVE)}},
       {x86_layout::YMMHigh,
        {reinterpret_cast<const uint8_t *>(&m_xstate->xsave.ymmh),
         sizeof(m_xstate->xsave.ymmh)}},
       {x86_layout::Bounds,
        {reinterpret_cast<const uint8_t *>(&m_xstate->xsave.mpxr),
         sizeof(m_xstate->xsave.mpxr)}},
       {x86_layout::BoundsControl,
        {reinterpret_cast<const uint8_t *>(&m_xstate->xsave.mpxc),
         sizeof(m_xstate->xsave.mpxc)}}});
  return value;
}

void NativeRegisterContextLinux_x86::UpdateXSTATEforWrite(
    uint32_t reg_index) {
  XSAVE_HDR::XFeature &xstate_bv = m_xstate->xsave.header.xstate_bv;
  if (IsFPR(reg_index)) {
    // IsFPR considers both %st and %xmm registers as floating point, but these
    // map to two features. Set both flags, just in case.
    xstate_bv |= XSAVE_HDR::XFeature::FP | XSAVE_HDR::XFeature::SSE;
  } else if (IsAVX(reg_index)) {
    // Lower bytes of some %ymm registers are shared with %xmm registers.
    xstate_bv |= XSAVE_HDR::XFeature::YMM | XSAVE_HDR::XFeature::SSE;
  } else if (IsMPX(reg_index)) {
    // MPX registers map to two XSAVE features.
    xstate_bv |= XSAVE_HDR::XFeature::BNDREGS | XSAVE_HDR::XFeature::BNDCSR;
  }
}

llvm::Error
NativeRegisterContextLinux_x86::WriteRegister(RegisterID id,
                                              const RegisterValue &reg_value) {
  auto description = GetDescription();
  const auto *descriptor = description ? description->Find(id) : nullptr;
  if (!descriptor)
    return Status::FromErrorString("unknown register").ToError();
  if (IsThreadPointer(*descriptor))
    return Status::FromErrorString("the i386 thread pointer is read-only")
        .ToError();
  const uint32_t reg = descriptor->numbers[lldb::eRegisterKindLLDB];
  if (IsDR(reg))
    return WriteRegisterRaw(reg, reg_value).ToError();
  auto required = GetLayout().GetRequiredBanks(descriptor->id, true);
  if (!required)
    return required.takeError();
  bool write_gp = false, write_fp = false;
  for (RegisterBankID bank : *required) {
    Status status;
    switch (bank) {
    case x86_layout::GP:
      status = ReadGPR();
      write_gp = true;
      break;
    case x86_layout::FP:
      status = ReadFPR();
      write_fp = true;
      break;
    case x86_layout::YMMHigh:
      status = ReadFPR();
      write_fp = true;
      if (status.Success() && !IsCPUFeatureAvailable(RegSet::avx))
        return Status::FromErrorString("AVX state is unavailable").ToError();
      break;
    case x86_layout::Bounds:
    case x86_layout::BoundsControl:
      status = ReadFPR();
      write_fp = true;
      if (status.Success() && !IsCPUFeatureAvailable(RegSet::mpx))
        return Status::FromErrorString("MPX state is unavailable").ToError();
      break;
    default:
      return Status::FromErrorString("unsupported register bank").ToError();
    }
    if (status.Fail())
      return status.ToError();
  }
  if (llvm::StringRef(descriptor->name) == "ftag")
    m_xstate->fxsave.ftag = FullToAbridgedTagWord(reg_value.GetAsUInt16());
  else if (auto error = WriteBankRegister(
               descriptor->id, reg_value,
               {{x86_layout::GP,
                 {reinterpret_cast<uint8_t *>(&m_gpr_x86_64), GetGPRSize()}},
                {x86_layout::FP,
                 {reinterpret_cast<uint8_t *>(&m_xstate->fxsave),
                  sizeof(FXSAVE)}},
                {x86_layout::YMMHigh,
                 {reinterpret_cast<uint8_t *>(&m_xstate->xsave.ymmh),
                  sizeof(m_xstate->xsave.ymmh)}},
                {x86_layout::Bounds,
                 {reinterpret_cast<uint8_t *>(&m_xstate->xsave.mpxr),
                  sizeof(m_xstate->xsave.mpxr)}},
                {x86_layout::BoundsControl,
                 {reinterpret_cast<uint8_t *>(&m_xstate->xsave.mpxc),
                  sizeof(m_xstate->xsave.mpxc)}}}))
    return error;
  if (write_gp) {
    Status status = WriteGPR();
    if (status.Fail())
      return status.ToError();
  }
  if (write_fp) {
    UpdateXSTATEforWrite(reg);
    return WriteFPR().ToError();
  }
  return llvm::Error::success();
}

Status NativeRegisterContextLinux_x86::ReadAllRegisterValues(
    lldb::WritableDataBufferSP &data_sp) {
  Status error;

  data_sp.reset(new DataBufferHeap(REG_CONTEXT_SIZE, 0));
  error = ReadGPR();
  if (error.Fail())
    return error;

  error = ReadFPR();
  if (error.Fail())
    return error;

  uint8_t *dst = data_sp->GetBytes();
  ::memcpy(dst, &m_gpr_x86_64, GetBankSize(0));
  dst += GetBankSize(0);
  if (m_xstate_type == XStateType::FXSAVE)
    ::memcpy(dst, &m_xstate->fxsave, sizeof(m_xstate->fxsave));
  else if (m_xstate_type == XStateType::XSAVE) {
    ::memcpy(dst, m_xstate.get(), sizeof(FPR));
  } else {
    assert(false && "how do we save the floating point registers?");
    error = Status::FromErrorString(
        "unsure how to save the floating point registers");
  }
  /** The following code is specific to Linux x86 based architectures,
   *  where the register orig_eax (32 bit)/orig_rax (64 bit) is set to
   *  -1 to solve the bug 23659, such a setting prevents the automatic
   *  decrement of the instruction pointer which was causing the SIGILL
   *  exception.
   * **/

  RegisterValue value((uint64_t)-1);
  const RegisterStorageField &info = m_orig_ax;
  return DoWriteRegisterValue(info.byte_offset, info.name, value);

  return error;
}

Status NativeRegisterContextLinux_x86::WriteAllRegisterValues(
    const lldb::DataBufferSP &data_sp) {
  Status error;

  if (!data_sp) {
    error = Status::FromErrorStringWithFormat(
        "NativeRegisterContextLinux_x86::%s invalid data_sp provided",
        __FUNCTION__);
    return error;
  }

  if (data_sp->GetByteSize() != REG_CONTEXT_SIZE) {
    error = Status::FromErrorStringWithFormatv(
        "data_sp contained mismatched data size, expected {0}, actual {1}",
        REG_CONTEXT_SIZE, data_sp->GetByteSize());
    return error;
  }

  const uint8_t *src = data_sp->GetBytes();
  if (src == nullptr) {
    error = Status::FromErrorStringWithFormat(
        "NativeRegisterContextLinux_x86::%s "
        "DataBuffer::GetBytes() returned a null "
        "pointer",
        __FUNCTION__);
    return error;
  }
  ::memcpy(&m_gpr_x86_64, src, GetBankSize(0));

  error = WriteGPR();
  if (error.Fail())
    return error;

  src += GetBankSize(0);
  if (m_xstate_type == XStateType::FXSAVE)
    ::memcpy(&m_xstate->fxsave, src, sizeof(m_xstate->fxsave));
  else if (m_xstate_type == XStateType::XSAVE)
    ::memcpy(&m_xstate->xsave, src, sizeof(m_xstate->xsave));

  error = WriteFPR();
  if (error.Fail())
    return error;

  return error;
}

bool NativeRegisterContextLinux_x86::IsCPUFeatureAvailable(
    RegSet feature_code) const {
  if (m_xstate_type == XStateType::Invalid) {
    if (const_cast<NativeRegisterContextLinux_x86 *>(this)->ReadFPR().Fail())
      return false;
  }
  switch (feature_code) {
  case RegSet::gpr:
  case RegSet::fpu:
    return true;
  case RegSet::avx: // Check if CPU has AVX and if there is kernel support, by
                    // reading in the XCR0 area of XSAVE.
    if ((m_xstate->xsave.i387.xcr0 & mask_XSTATE_AVX) == mask_XSTATE_AVX)
      return true;
     break;
  case RegSet::mpx: // Check if CPU has MPX and if there is kernel support, by
                    // reading in the XCR0 area of XSAVE.
    if ((m_xstate->xsave.i387.xcr0 & mask_XSTATE_MPX) == mask_XSTATE_MPX)
      return true;
    break;
  }
  return false;
}

bool NativeRegisterContextLinux_x86::IsRegisterSetAvailable(
    uint32_t set_index) const {
  uint32_t num_sets = k_num_register_sets - k_num_extended_register_sets;

  switch (static_cast<RegSet>(set_index)) {
  case RegSet::gpr:
  case RegSet::fpu:
    return (set_index < num_sets);
  case RegSet::avx:
    return IsCPUFeatureAvailable(RegSet::avx);
  case RegSet::mpx:
    return IsCPUFeatureAvailable(RegSet::mpx);
  }
  return false;
}

bool NativeRegisterContextLinux_x86::IsGPR(uint32_t reg_index) const {
  // GPRs come first.
  return reg_index <= m_reg_info.last_gpr;
}

bool NativeRegisterContextLinux_x86::IsFPR(uint32_t reg_index) const {
  return (m_reg_info.first_fpr <= reg_index &&
          reg_index <= m_reg_info.last_fpr);
}

bool NativeRegisterContextLinux_x86::IsDR(uint32_t reg_index) const {
  return (m_reg_info.first_dr <= reg_index &&
          reg_index <= m_reg_info.last_dr);
}

Status NativeRegisterContextLinux_x86::WriteFPR() {
  switch (m_xstate_type) {
  case XStateType::FXSAVE:
    return WriteRegisterSet(&m_iovec, sizeof(m_xstate->fxsave),
                            fxsr_regset(GetTargetArchitecture()));
  case XStateType::XSAVE:
    return WriteRegisterSet(&m_iovec, sizeof(m_xstate->xsave), NT_X86_XSTATE);
  default:
    return Status::FromErrorString("Unrecognized FPR type.");
  }
}

bool NativeRegisterContextLinux_x86::IsAVX(uint32_t reg_index) const {
  if (!IsCPUFeatureAvailable(RegSet::avx))
    return false;
  return (m_reg_info.first_ymm <= reg_index &&
          reg_index <= m_reg_info.last_ymm);
}

void *NativeRegisterContextLinux_x86::GetFPRBuffer() {
  switch (m_xstate_type) {
  case XStateType::FXSAVE:
    return &m_xstate->fxsave;
  case XStateType::XSAVE:
    return &m_iovec;
  default:
    return nullptr;
  }
}

size_t NativeRegisterContextLinux_x86::GetFPRSize() {
  switch (m_xstate_type) {
  case XStateType::FXSAVE:
    return sizeof(m_xstate->fxsave);
  case XStateType::XSAVE:
    return sizeof(m_iovec);
  default:
    return 0;
  }
}

Status NativeRegisterContextLinux_x86::ReadFPR() {
  Status error;

  // Probe XSAVE and if it is not supported fall back to FXSAVE.
  if (m_xstate_type != XStateType::FXSAVE) {
    error = ReadRegisterSet(&m_iovec, sizeof(m_xstate->xsave), NT_X86_XSTATE);
    if (!error.Fail()) {
      m_xstate_type = XStateType::XSAVE;
      return error;
    }
  }
  error = ReadRegisterSet(&m_iovec, sizeof(m_xstate->xsave),
                          fxsr_regset(GetTargetArchitecture()));
  if (!error.Fail()) {
    m_xstate_type = XStateType::FXSAVE;
    return error;
  }
  return Status::FromErrorString("Unrecognized FPR type.");
}

bool NativeRegisterContextLinux_x86::IsMPX(uint32_t reg_index) const {
  if (!IsCPUFeatureAvailable(RegSet::mpx))
    return false;
  return (m_reg_info.first_mpxr <= reg_index &&
          reg_index <= m_reg_info.last_mpxc);
}

bool NativeRegisterContextLinux_x86::IsThreadPointer(
    const RegisterDescriptor &reg_info) const {
  return GetTargetArchitecture().GetMachine() == llvm::Triple::x86 &&
         reg_info.numbers[lldb::eRegisterKindGeneric] == LLDB_REGNUM_GENERIC_TP;
}

Status
NativeRegisterContextLinux_x86::ReadThreadPointer(RegisterValue &reg_value) {
  RegisterValue selector;
  Status error = ReadRegisterRaw(registers::x86::id::gs, selector);
  if (error.Fail())
    return error;

  const uint32_t gs = selector.GetAsUInt32();
  // Selectors 0 through 3 refer to the null GDT descriptor with different
  // requested privilege levels.
  if (gs < 4) {
    reg_value.SetUInt32(0);
    return Status();
  }
  if (gs & 4)
    return Status::FromErrorString(
        "cannot read the i386 thread pointer from an LDT selector");

  const uintptr_t entry_number = gs >> 3;
  user_desc descriptor = {};
  error = NativeProcessLinux::PtraceWrapper(
      PTRACE_GET_THREAD_AREA, m_thread.GetID(),
      reinterpret_cast<void *>(entry_number), &descriptor, sizeof(descriptor));
  if (error.Fail())
    return error;

  reg_value.SetUInt32(descriptor.base_addr);
  return Status();
}

uint32_t NativeRegisterContextLinux_x86::GetPtraceOffset(uint32_t reg_index) {
  const auto *info = GetRegisterByNumber(lldb::eRegisterKindLLDB, reg_index);
  const auto *reg = GetDescription()->Find(info->name);
  const auto *location = GetLayout().Find(reg->id);
  assert(location && location->pieces.size() == 1);
  const auto &piece = location->pieces.front();
  if (piece.bank == x86_layout::Debug) {
    // PEEKUSER/POKEUSER offsets are native ABI details, not packet offsets.
#if defined(__x86_64__)
    return offsetof(struct user, u_debugreg) + piece.byte_offset;
#else
    return 0xfc + piece.byte_offset;
#endif
  }
  assert(piece.bank == x86_layout::GP);
  return piece.byte_offset;
}

std::optional<NativeRegisterContextLinux::SyscallData>
NativeRegisterContextLinux_x86::GetSyscallData() {
  switch (GetTargetArchitecture().GetMachine()) {
  case llvm::Triple::x86: {
    static const uint8_t Int80[] = {0xcd, 0x80};
    static const uint32_t Args[] = {
        registers::x86::id::eax, registers::x86::id::ebx,
        registers::x86::id::ecx, registers::x86::id::edx,
        registers::x86::id::esi, registers::x86::id::edi,
        registers::x86::id::ebp};
    return SyscallData{Int80, Args, registers::x86::id::eax};
  }
  case llvm::Triple::x86_64: {
    static const uint8_t Syscall[] = {0x0f, 0x05};
    static const uint32_t Args[] = {
        registers::x86_64::id::with_base::rax, registers::x86_64::id::with_base::rdi, registers::x86_64::id::with_base::rsi, registers::x86_64::id::with_base::rdx,
        registers::x86_64::id::with_base::r10, registers::x86_64::id::with_base::r8,  registers::x86_64::id::with_base::r9};
    return SyscallData{Syscall, Args, registers::x86_64::id::with_base::rax};
  }
  default:
    llvm_unreachable("Unhandled architecture!");
  }
}

std::optional<NativeRegisterContextLinux::MmapData>
NativeRegisterContextLinux_x86::GetMmapData() {
  switch (GetTargetArchitecture().GetMachine()) {
  case llvm::Triple::x86:
    return MmapData{192, 91};
  case llvm::Triple::x86_64:
    return MmapData{9, 11};
  default:
    llvm_unreachable("Unhandled architecture!");
  }
}

#endif // defined(__i386__) || defined(__x86_64__)
