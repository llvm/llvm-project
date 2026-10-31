//===-- NativeRegisterContextOpenBSD_x86.cpp ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#if defined(__i386__) || defined(__x86_64__)

#include "NativeRegisterContextOpenBSD_x86.h"

#include <cinttypes>
#include <cpuid.h>
#include <cstring>
#include <iterator>

#include "lldb/Utility/DataBufferHeap.h"
#include "lldb/Utility/RegisterValue.h"
#include "lldb/Utility/Status.h"

#include "Register/OpenBSD/RegisterInfoOpenBSD_i386.h"
#include "Register/OpenBSD/RegisterInfoOpenBSD_x86_64.h"

// clang-format off
#include <sys/types.h>
#include <sys/ptrace.h>
#if defined(__x86_64__)
#include <machine/specialreg.h>
#endif
// clang-format on

using namespace lldb_private;
using namespace lldb_private::process_openbsd;

namespace {

enum {
  k_xsave_offset_legacy_region = 160,
  k_xsave_offset_invalid = UINT32_MAX,
};

#if defined(__x86_64__)

// x86 64-bit general purpose registers.
const uint32_t g_gpr_regnums_x86_64[] = {
    lldb_rax_x86_64,    lldb_rbx_x86_64,    lldb_rcx_x86_64,  lldb_rdx_x86_64,
    lldb_rdi_x86_64,    lldb_rsi_x86_64,    lldb_rbp_x86_64,  lldb_rsp_x86_64,
    lldb_r8_x86_64,     lldb_r9_x86_64,     lldb_r10_x86_64,  lldb_r11_x86_64,
    lldb_r12_x86_64,    lldb_r13_x86_64,    lldb_r14_x86_64,  lldb_r15_x86_64,
    lldb_rip_x86_64,    lldb_rflags_x86_64, lldb_cs_x86_64,   lldb_fs_x86_64,
    lldb_gs_x86_64,     lldb_ss_x86_64,     lldb_ds_x86_64,   lldb_es_x86_64,
    lldb_eax_x86_64,    lldb_ebx_x86_64,    lldb_ecx_x86_64,  lldb_edx_x86_64,
    lldb_edi_x86_64,    lldb_esi_x86_64,    lldb_ebp_x86_64,  lldb_esp_x86_64,
    lldb_r8d_x86_64,    lldb_r9d_x86_64,    lldb_r10d_x86_64, lldb_r11d_x86_64,
    lldb_r12d_x86_64,   lldb_r13d_x86_64,   lldb_r14d_x86_64, lldb_r15d_x86_64,
    lldb_ax_x86_64,     lldb_bx_x86_64,     lldb_cx_x86_64,   lldb_dx_x86_64,
    lldb_di_x86_64,     lldb_si_x86_64,     lldb_bp_x86_64,   lldb_sp_x86_64,
    lldb_r8w_x86_64,    lldb_r9w_x86_64,    lldb_r10w_x86_64, lldb_r11w_x86_64,
    lldb_r12w_x86_64,   lldb_r13w_x86_64,   lldb_r14w_x86_64, lldb_r15w_x86_64,
    lldb_ah_x86_64,     lldb_bh_x86_64,     lldb_ch_x86_64,   lldb_dh_x86_64,
    lldb_al_x86_64,     lldb_bl_x86_64,     lldb_cl_x86_64,   lldb_dl_x86_64,
    lldb_dil_x86_64,    lldb_sil_x86_64,    lldb_bpl_x86_64,  lldb_spl_x86_64,
    lldb_r8l_x86_64,    lldb_r9l_x86_64,    lldb_r10l_x86_64, lldb_r11l_x86_64,
    lldb_r12l_x86_64,   lldb_r13l_x86_64,   lldb_r14l_x86_64, lldb_r15l_x86_64,
    LLDB_INVALID_REGNUM // register sets need to end with this flag
};
static_assert((sizeof(g_gpr_regnums_x86_64) / sizeof(g_gpr_regnums_x86_64[0])) -
                      1 ==
                  k_num_gpr_registers_x86_64,
              "g_gpr_regnums_x86_64 has wrong number of register infos");

// x86 64-bit floating point registers.
const uint32_t g_fpu_regnums_x86_64[] = {
    lldb_fctrl_x86_64,  lldb_fstat_x86_64, lldb_ftag_x86_64,
    lldb_fop_x86_64,    lldb_fiseg_x86_64, lldb_fioff_x86_64,
    lldb_fip_x86_64,    lldb_foseg_x86_64, lldb_fooff_x86_64,
    lldb_fdp_x86_64,    lldb_mxcsr_x86_64, lldb_mxcsrmask_x86_64,
    lldb_st0_x86_64,    lldb_st1_x86_64,   lldb_st2_x86_64,
    lldb_st3_x86_64,    lldb_st4_x86_64,   lldb_st5_x86_64,
    lldb_st6_x86_64,    lldb_st7_x86_64,   lldb_mm0_x86_64,
    lldb_mm1_x86_64,    lldb_mm2_x86_64,   lldb_mm3_x86_64,
    lldb_mm4_x86_64,    lldb_mm5_x86_64,   lldb_mm6_x86_64,
    lldb_mm7_x86_64,    lldb_xmm0_x86_64,  lldb_xmm1_x86_64,
    lldb_xmm2_x86_64,   lldb_xmm3_x86_64,  lldb_xmm4_x86_64,
    lldb_xmm5_x86_64,   lldb_xmm6_x86_64,  lldb_xmm7_x86_64,
    lldb_xmm8_x86_64,   lldb_xmm9_x86_64,  lldb_xmm10_x86_64,
    lldb_xmm11_x86_64,  lldb_xmm12_x86_64, lldb_xmm13_x86_64,
    lldb_xmm14_x86_64,  lldb_xmm15_x86_64,
    LLDB_INVALID_REGNUM // register sets need to end with this flag
};
static_assert((sizeof(g_fpu_regnums_x86_64) / sizeof(g_fpu_regnums_x86_64[0])) -
                      1 ==
                  k_num_fpr_registers_x86_64,
              "g_fpu_regnums_x86_64 has wrong number of register infos");

// x86 64-bit AVX registers.
const uint32_t g_avx_regnums_x86_64[] = {
    lldb_ymm0_x86_64,   lldb_ymm1_x86_64,  lldb_ymm2_x86_64,  lldb_ymm3_x86_64,
    lldb_ymm4_x86_64,   lldb_ymm5_x86_64,  lldb_ymm6_x86_64,  lldb_ymm7_x86_64,
    lldb_ymm8_x86_64,   lldb_ymm9_x86_64,  lldb_ymm10_x86_64, lldb_ymm11_x86_64,
    lldb_ymm12_x86_64,  lldb_ymm13_x86_64, lldb_ymm14_x86_64, lldb_ymm15_x86_64,
    LLDB_INVALID_REGNUM // register sets need to end with this flag
};
static_assert((sizeof(g_avx_regnums_x86_64) / sizeof(g_avx_regnums_x86_64[0])) -
                      1 ==
                  k_num_avx_registers_x86_64,
              "g_avx_regnums_x86_64 has wrong number of register infos");

// Register sets for x86 64-bit.
const RegisterSet g_reg_sets[] = {
    {"General Purpose Registers", "gpr", k_num_gpr_registers_x86_64,
     g_gpr_regnums_x86_64},
    {"Floating Point Registers", "fpu", k_num_fpr_registers_x86_64,
     g_fpu_regnums_x86_64},
    {"Advanced Vector Extensions", "avx", k_num_avx_registers_x86_64,
     g_avx_regnums_x86_64},
};

#else // defined(__i386__)

// x86 32-bit general purpose registers.
const uint32_t g_gpr_regnums_i386[] = {
    lldb_eax_i386,      lldb_ebx_i386,    lldb_ecx_i386, lldb_edx_i386,
    lldb_edi_i386,      lldb_esi_i386,    lldb_ebp_i386, lldb_esp_i386,
    lldb_eip_i386,      lldb_eflags_i386, lldb_cs_i386,  lldb_fs_i386,
    lldb_gs_i386,       lldb_ss_i386,     lldb_ds_i386,  lldb_es_i386,
    lldb_ax_i386,       lldb_bx_i386,     lldb_cx_i386,  lldb_dx_i386,
    lldb_di_i386,       lldb_si_i386,     lldb_bp_i386,  lldb_sp_i386,
    lldb_ah_i386,       lldb_bh_i386,     lldb_ch_i386,  lldb_dh_i386,
    lldb_al_i386,       lldb_bl_i386,     lldb_cl_i386,  lldb_dl_i386,
    LLDB_INVALID_REGNUM // register sets need to end with this flag
};
static_assert((sizeof(g_gpr_regnums_i386) / sizeof(g_gpr_regnums_i386[0])) -
                      1 ==
                  k_num_gpr_registers_i386,
              "g_gpr_regnums_i386 has wrong number of register infos");

// x86 32-bit floating point registers.
const uint32_t g_fpu_regnums_i386[] = {
    lldb_fctrl_i386,    lldb_fstat_i386,     lldb_ftag_i386,  lldb_fop_i386,
    lldb_fiseg_i386,    lldb_fioff_i386,     lldb_foseg_i386, lldb_fooff_i386,
    lldb_mxcsr_i386,    lldb_mxcsrmask_i386, lldb_st0_i386,   lldb_st1_i386,
    lldb_st2_i386,      lldb_st3_i386,       lldb_st4_i386,   lldb_st5_i386,
    lldb_st6_i386,      lldb_st7_i386,       lldb_mm0_i386,   lldb_mm1_i386,
    lldb_mm2_i386,      lldb_mm3_i386,       lldb_mm4_i386,   lldb_mm5_i386,
    lldb_mm6_i386,      lldb_mm7_i386,       lldb_xmm0_i386,  lldb_xmm1_i386,
    lldb_xmm2_i386,     lldb_xmm3_i386,      lldb_xmm4_i386,  lldb_xmm5_i386,
    lldb_xmm6_i386,     lldb_xmm7_i386,
    LLDB_INVALID_REGNUM // register sets need to end with this flag
};
static_assert((sizeof(g_fpu_regnums_i386) / sizeof(g_fpu_regnums_i386[0])) -
                      1 ==
                  k_num_fpr_registers_i386,
              "g_fpu_regnums_i386 has wrong number of register infos");

// Register sets for x86 32-bit.  OpenBSD/i386 has no PT_GETXSTATE, so the AVX
// registers are not available.
const RegisterSet g_reg_sets[] = {
    {"General Purpose Registers", "gpr", k_num_gpr_registers_i386,
     g_gpr_regnums_i386},
    {"Floating Point Registers", "fpu", k_num_fpr_registers_i386,
     g_fpu_regnums_i386},
};

#endif

constexpr uint32_t k_num_register_sets = std::size(g_reg_sets);

} // namespace

std::unique_ptr<NativeRegisterContextOpenBSD>
NativeRegisterContextOpenBSD::CreateHostNativeRegisterContextOpenBSD(
    const ArchSpec &target_arch, NativeThreadProtocol &native_thread) {
  return std::make_unique<NativeRegisterContextOpenBSD_x86>(target_arch,
                                                            native_thread);
}

// NativeRegisterContextOpenBSD_x86 members.

static RegisterInfoInterface *
CreateRegisterInfoInterface(const ArchSpec &target_arch) {
  // OpenBSD/amd64 cannot run i386 executables, so the register layout always
  // matches the host.
#if defined(__x86_64__)
  return new RegisterInfoOpenBSD_x86_64(target_arch);
#else
  return new RegisterInfoOpenBSD_i386(target_arch);
#endif
}

NativeRegisterContextOpenBSD_x86::NativeRegisterContextOpenBSD_x86(
    const ArchSpec &target_arch, NativeThreadProtocol &native_thread)
    : NativeRegisterContextOpenBSD(native_thread,
                                   CreateRegisterInfoInterface(target_arch)),
      m_gpr(), m_fpr(), m_regset_offsets() {
  assert(m_gpr.size() == GetRegisterInfoInterface().GetGPRSize());
#if defined(__x86_64__)
  static_assert(sizeof(struct fpreg) == sizeof(FXSAVE),
                "struct fpreg is not an FXSAVE area");
  const uint32_t first_fpr = lldb_fctrl_x86_64;
#else
  static_assert(sizeof(struct xmmregs) == sizeof(FXSAVE),
                "struct xmmregs is not an FXSAVE area");
  const uint32_t first_fpr = lldb_fctrl_i386;
#endif

  // The RegisterInfo byte offsets describe a UserArea with the FPR area
  // following the GPRs; remember where it starts so that the offsets can be
  // mapped into m_fpr.
  m_regset_offsets[FPRegSet] =
      GetRegisterInfoInterface().GetRegisterInfo()[first_fpr].byte_offset;

  m_xsave_offsets.fill(k_xsave_offset_invalid);

#if defined(PT_GETXSTATE_INFO)
  struct ptrace_xstate_info info;
  const Status error = NativeProcessOpenBSD::PtraceWrapper(
      PT_GETXSTATE_INFO, m_thread.GetID(), &info, sizeof(info));
  if (error.Fail())
    return;

  assert(info.xsave_mask & XFEATURE_X87);
  assert(info.xsave_mask & XFEATURE_SSE);

  if (info.xsave_mask & XFEATURE_AVX) {
    uint32_t eax, ecx, edx;
    __get_cpuid_count(0x0D, 2, &eax, &m_xsave_offsets[YMMRegSet], &ecx, &edx);
  }

  m_xsave.resize(info.xsave_len);
#endif
}

uint32_t NativeRegisterContextOpenBSD_x86::GetUserRegisterCount() const {
  uint32_t count = 0;
  for (const RegisterSet &set : g_reg_sets)
    count += set.num_registers;
  return count;
}

uint32_t NativeRegisterContextOpenBSD_x86::GetRegisterSetCount() const {
  return k_num_register_sets;
}

const RegisterSet *
NativeRegisterContextOpenBSD_x86::GetRegisterSet(uint32_t set_index) const {
  if (set_index >= k_num_register_sets)
    return nullptr;
  return &g_reg_sets[set_index];
}

std::optional<NativeRegisterContextOpenBSD_x86::RegSetKind>
NativeRegisterContextOpenBSD_x86::GetSetForNativeRegNum(
    uint32_t reg_num) const {
#if defined(__x86_64__)
  if (reg_num >= k_first_gpr_x86_64 && reg_num <= k_last_gpr_x86_64)
    return GPRegSet;
  if (reg_num >= k_first_fpr_x86_64 && reg_num <= k_last_fpr_x86_64)
    return FPRegSet;
  if (reg_num >= k_first_avx_x86_64 && reg_num <= k_last_avx_x86_64)
    return YMMRegSet;
#else
  if (reg_num >= k_first_gpr_i386 && reg_num <= k_last_gpr_i386)
    return GPRegSet;
  if (reg_num >= k_first_fpr_i386 && reg_num <= k_last_fpr_i386)
    return FPRegSet;
#endif
  return std::nullopt;
}

Status NativeRegisterContextOpenBSD_x86::DoReadFPR(void *buf) {
#if defined(__x86_64__)
  return NativeRegisterContextOpenBSD::DoReadFPR(buf);
#else
  // On i386 struct fpreg is not an FXSAVE area; PT_GETXMMREGS returns one.
  return NativeProcessOpenBSD::PtraceWrapper(PT_GETXMMREGS, m_thread.GetID(),
                                             buf);
#endif
}

Status NativeRegisterContextOpenBSD_x86::DoWriteFPR(void *buf) {
#if defined(__x86_64__)
  return NativeRegisterContextOpenBSD::DoWriteFPR(buf);
#else
  return NativeProcessOpenBSD::PtraceWrapper(PT_SETXMMREGS, m_thread.GetID(),
                                             buf);
#endif
}

Status NativeRegisterContextOpenBSD_x86::ReadRegisterSet(RegSetKind set) {
  switch (set) {
  case GPRegSet:
    return ReadGPR();
  case FPRegSet:
    return ReadFPR();
  case YMMRegSet:
#if defined(PT_GETXSTATE)
    if (m_xsave.empty())
      return Status::FromErrorString("XSAVE area is not available");
    return NativeProcessOpenBSD::PtraceWrapper(PT_GETXSTATE, m_thread.GetID(),
                                               m_xsave.data(), m_xsave.size());
#else
    return Status::FromErrorString("XSAVE area is not available");
#endif
  }
  llvm_unreachable("NativeRegisterContextOpenBSD_x86::ReadRegisterSet");
}

Status NativeRegisterContextOpenBSD_x86::WriteRegisterSet(RegSetKind set) {
  switch (set) {
  case GPRegSet:
    return WriteGPR();
  case FPRegSet:
    return WriteFPR();
  case YMMRegSet:
#if defined(PT_SETXSTATE)
    // ReadRegisterSet() must always be called before WriteRegisterSet().
    if (m_xsave.empty())
      return Status::FromErrorString("XSAVE area is not available");
    return NativeProcessOpenBSD::PtraceWrapper(PT_SETXSTATE, m_thread.GetID(),
                                               m_xsave.data(), m_xsave.size());
#else
    return Status::FromErrorString("XSAVE area is not available");
#endif
  }
  llvm_unreachable("NativeRegisterContextOpenBSD_x86::WriteRegisterSet");
}

Status
NativeRegisterContextOpenBSD_x86::ReadRegister(const RegisterInfo *reg_info,
                                               RegisterValue &reg_value) {
  if (!reg_info)
    return Status::FromErrorString("reg_info NULL");

  const uint32_t reg = reg_info->kinds[lldb::eRegisterKindLLDB];
  if (reg == LLDB_INVALID_REGNUM) {
    // This is likely an internal register for lldb use only and should not be
    // directly queried.
    return Status::FromErrorStringWithFormat(
        "register \"%s\" is an internal-only lldb register, cannot read "
        "directly",
        reg_info->name);
  }

  std::optional<RegSetKind> set = GetSetForNativeRegNum(reg);
  if (!set)
    return Status::FromErrorStringWithFormat(
        "register \"%s\" is in unrecognized set", reg_info->name);

  Status error = ReadRegisterSet(*set);
  if (error.Fail())
    return error;

  if (*set == YMMRegSet) {
    std::optional<YMMSplitPtr> ymm_reg = GetYMMSplitReg(reg);
    if (!ymm_reg)
      return Status::FromErrorStringWithFormat(
          "register \"%s\" not supported by CPU/kernel", reg_info->name);
    YMMReg ymm = XStateToYMM(ymm_reg->xmm, ymm_reg->ymm_hi);
    reg_value.SetBytes(ymm.bytes, reg_info->byte_size,
                       endian::InlHostByteOrder());
    return error;
  }

  void *data = GetOffsetRegSetData(*set, reg_info->byte_offset);
  FXSAVE *fpr = reinterpret_cast<FXSAVE *>(m_fpr.data());
  if (data == &fpr->ftag) // FXSAVE stores the abridged tag word.
    reg_value.SetUInt16(
        AbridgedToFullTagWord(fpr->ftag, fpr->fstat, fpr->stmm));
  else
    reg_value.SetBytes(data, reg_info->byte_size, endian::InlHostByteOrder());
  return error;
}

Status NativeRegisterContextOpenBSD_x86::WriteRegister(
    const RegisterInfo *reg_info, const RegisterValue &reg_value) {
  if (!reg_info)
    return Status::FromErrorString("reg_info NULL");

  const uint32_t reg = reg_info->kinds[lldb::eRegisterKindLLDB];
  if (reg == LLDB_INVALID_REGNUM) {
    // This is likely an internal register for lldb use only and should not be
    // directly queried.
    return Status::FromErrorStringWithFormat(
        "register \"%s\" is an internal-only lldb register, cannot write "
        "directly",
        reg_info->name);
  }

  std::optional<RegSetKind> set = GetSetForNativeRegNum(reg);
  if (!set)
    return Status::FromErrorStringWithFormat(
        "register \"%s\" is in unrecognized set", reg_info->name);

  Status error = ReadRegisterSet(*set);
  if (error.Fail())
    return error;

  if (*set == YMMRegSet) {
    std::optional<YMMSplitPtr> ymm_reg = GetYMMSplitReg(reg);
    if (!ymm_reg)
      return Status::FromErrorStringWithFormat(
          "register \"%s\" not supported by CPU/kernel", reg_info->name);
    YMMReg ymm;
    ::memcpy(ymm.bytes, reg_value.GetBytes(), reg_value.GetByteSize());
    YMMToXState(ymm, ymm_reg->xmm, ymm_reg->ymm_hi);
  } else {
    void *data = GetOffsetRegSetData(*set, reg_info->byte_offset);
    FXSAVE *fpr = reinterpret_cast<FXSAVE *>(m_fpr.data());
    if (data == &fpr->ftag) // FXSAVE stores the abridged tag word.
      fpr->ftag = FullToAbridgedTagWord(reg_value.GetAsUInt16());
    else
      ::memcpy(data, reg_value.GetBytes(), reg_value.GetByteSize());
  }

  return WriteRegisterSet(*set);
}

Status NativeRegisterContextOpenBSD_x86::ReadAllRegisterValues(
    lldb::WritableDataBufferSP &data_sp) {
  Status error = ReadGPR();
  if (error.Fail())
    return error;

  error = ReadFPR();
  if (error.Fail())
    return error;

  data_sp.reset(new DataBufferHeap(m_gpr.size() + m_fpr.size(), 0));
  uint8_t *dst = data_sp->GetBytes();
  ::memcpy(dst, m_gpr.data(), m_gpr.size());
  dst += m_gpr.size();
  ::memcpy(dst, m_fpr.data(), m_fpr.size());

  return error;
}

Status NativeRegisterContextOpenBSD_x86::WriteAllRegisterValues(
    const lldb::DataBufferSP &data_sp) {
  if (!data_sp)
    return Status::FromErrorStringWithFormat(
        "NativeRegisterContextOpenBSD_x86::%s invalid data_sp provided",
        __FUNCTION__);

  const size_t expected = m_gpr.size() + m_fpr.size();
  if (data_sp->GetByteSize() != expected)
    return Status::FromErrorStringWithFormat(
        "NativeRegisterContextOpenBSD_x86::%s data_sp contained mismatched "
        "data size, expected %zu, actual %" PRIu64,
        __FUNCTION__, expected, data_sp->GetByteSize());

  const uint8_t *src = data_sp->GetBytes();
  if (src == nullptr)
    return Status::FromErrorStringWithFormat(
        "NativeRegisterContextOpenBSD_x86::%s DataBuffer::GetBytes() returned "
        "a null pointer",
        __FUNCTION__);

  ::memcpy(m_gpr.data(), src, m_gpr.size());
  Status error = WriteGPR();
  if (error.Fail())
    return error;
  src += m_gpr.size();

  ::memcpy(m_fpr.data(), src, m_fpr.size());
  return WriteFPR();
}

uint8_t *
NativeRegisterContextOpenBSD_x86::GetOffsetRegSetData(RegSetKind set,
                                                      size_t reg_offset) {
  uint8_t *base;
  switch (set) {
  case GPRegSet:
    base = m_gpr.data();
    break;
  case FPRegSet:
    base = m_fpr.data();
    break;
  case YMMRegSet:
    llvm_unreachable("GetOffsetRegSetData() is unsuitable for this regset.");
  }
  assert(reg_offset >= m_regset_offsets[set]);
  return base + (reg_offset - m_regset_offsets[set]);
}

std::optional<NativeRegisterContextOpenBSD_x86::YMMSplitPtr>
NativeRegisterContextOpenBSD_x86::GetYMMSplitReg(uint32_t reg) {
#if defined(__x86_64__)
  if (m_xsave_offsets[YMMRegSet] == k_xsave_offset_invalid)
    return std::nullopt;

  uint32_t reg_index = reg - lldb_ymm0_x86_64;
  auto *xmm =
      reinterpret_cast<XMMReg *>(m_xsave.data() + k_xsave_offset_legacy_region);
  auto *ymm =
      reinterpret_cast<XMMReg *>(m_xsave.data() + m_xsave_offsets[YMMRegSet]);
  return YMMSplitPtr{&xmm[reg_index], &ymm[reg_index]};
#else
  return std::nullopt;
#endif
}

#endif // defined(__i386__) || defined(__x86_64__)
