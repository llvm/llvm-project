//===-- NativeRegisterContextFreeBSD_riscv64.cpp --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#if defined(__riscv) && __riscv_xlen == 64

#include "NativeRegisterContextFreeBSD_riscv64.h"

#include "lldb/Utility/DataBufferHeap.h"
#include "lldb/Utility/RegisterValue.h"
#include "lldb/Utility/Status.h"

#include "Plugins/Process/FreeBSD/NativeProcessFreeBSD.h"
#include "Plugins/Process/Utility/RegisterInfoPOSIX_riscv64.h"

// clang-format off
#include <sys/param.h>
#include <sys/ptrace.h>
#include <sys/types.h>
// clang-format on

using namespace lldb;
using namespace lldb_private;
using namespace lldb_private::process_freebsd;

// Translation between RegisterInfoPosix_riscv64 and reg.h
// https://cgit.freebsd.org/src/tree/sys/riscv/include/reg.h:
//
// struct reg {
// 	__uint64_t	ra;		/* return address */
// 	__uint64_t	sp;		/* stack pointer */
// 	__uint64_t	gp;		/* global pointer */
// 	__uint64_t	tp;		/* thread pointer */
// 	__uint64_t	t[7];		/* temporaries */
// 	__uint64_t	s[12];		/* saved registers */
// 	__uint64_t	a[8];		/* function arguments */
// 	__uint64_t	sepc;		/* exception program counter */
// 	__uint64_t	sstatus;	/* status register */
// };
//
// struct fpreg {
// 	__uint64_t	fp_x[32][2];	/* Floating point registers */
// 	__uint64_t	fp_fcsr;	/* Floating point control reg */
// };
//
// struct dbreg {
// 	int dummy;
// };

void NativeRegisterContextFreeBSD_riscv64::FreeBSDToPOSIXGPR(
    const struct reg &freebsd_gpr, RegisterInfoPOSIX_riscv64::GPR &posix_gpr) {
  posix_gpr.gpr[gpr_pc_riscv64] = freebsd_gpr.sepc; // x0/pc
  posix_gpr.gpr[gpr_ra_riscv64] = freebsd_gpr.ra;   // x1/ra
  posix_gpr.gpr[gpr_sp_riscv64] = freebsd_gpr.sp;   // x2/sp
  posix_gpr.gpr[gpr_gp_riscv64] = freebsd_gpr.gp;   // x3/gp
  posix_gpr.gpr[gpr_tp_riscv64] = freebsd_gpr.tp;   // x4/tp

  // x5-x7: t0-t2
  posix_gpr.gpr[gpr_t0_riscv64] = freebsd_gpr.t[0];
  posix_gpr.gpr[gpr_t1_riscv64] = freebsd_gpr.t[1];
  posix_gpr.gpr[gpr_t2_riscv64] = freebsd_gpr.t[2];

  // x8-x9: s0-s1 (s0 is also fp)
  posix_gpr.gpr[gpr_s0_riscv64] = freebsd_gpr.s[0];
  posix_gpr.gpr[gpr_s1_riscv64] = freebsd_gpr.s[1];

  // x10-x17: a0-a7
  for (int i = 0; i < 8; i++)
    posix_gpr.gpr[gpr_a0_riscv64 + i] = freebsd_gpr.a[i];

  // x18-x27: s2-s11
  for (int i = 0; i < 10; i++)
    posix_gpr.gpr[gpr_s2_riscv64 + i] = freebsd_gpr.s[2 + i];

  // x28-x31: t3-t6
  posix_gpr.gpr[gpr_t3_riscv64] = freebsd_gpr.t[3];
  posix_gpr.gpr[gpr_t4_riscv64] = freebsd_gpr.t[4];
  posix_gpr.gpr[gpr_t5_riscv64] = freebsd_gpr.t[5];
  posix_gpr.gpr[gpr_t6_riscv64] = freebsd_gpr.t[6];
}

void NativeRegisterContextFreeBSD_riscv64::POSIXToFreeBSDGPR(
    const RegisterInfoPOSIX_riscv64::GPR &posix_gpr, struct reg &freebsd_gpr) {
  freebsd_gpr.sepc = posix_gpr.gpr[gpr_pc_riscv64]; // x0/pc
  freebsd_gpr.ra = posix_gpr.gpr[gpr_ra_riscv64];   // x1/ra
  freebsd_gpr.sp = posix_gpr.gpr[gpr_sp_riscv64];   // x2/sp
  freebsd_gpr.gp = posix_gpr.gpr[gpr_gp_riscv64];   // x3/gp
  freebsd_gpr.tp = posix_gpr.gpr[gpr_tp_riscv64];   // x4/tp

  // x5-x7: t0-t2
  freebsd_gpr.t[0] = posix_gpr.gpr[gpr_t0_riscv64];
  freebsd_gpr.t[1] = posix_gpr.gpr[gpr_t1_riscv64];
  freebsd_gpr.t[2] = posix_gpr.gpr[gpr_t2_riscv64];

  // x8-x9: s0-s1
  freebsd_gpr.s[0] = posix_gpr.gpr[gpr_s0_riscv64];
  freebsd_gpr.s[1] = posix_gpr.gpr[gpr_s1_riscv64];

  // x10-x17: a0-a7
  for (int i = 0; i < 8; i++)
    freebsd_gpr.a[i] = posix_gpr.gpr[gpr_a0_riscv64 + i];

  // x18-x27: s2-s11
  for (int i = 0; i < 10; i++)
    freebsd_gpr.s[2 + i] = posix_gpr.gpr[gpr_s2_riscv64 + i];

  // x28-x31: t3-t6
  freebsd_gpr.t[3] = posix_gpr.gpr[gpr_t3_riscv64];
  freebsd_gpr.t[4] = posix_gpr.gpr[gpr_t4_riscv64];
  freebsd_gpr.t[5] = posix_gpr.gpr[gpr_t5_riscv64];
  freebsd_gpr.t[6] = posix_gpr.gpr[gpr_t6_riscv64];
}

void NativeRegisterContextFreeBSD_riscv64::FreeBSDToPOSIXFPR(
    const struct fpreg &freebsd_fpr,
    RegisterInfoPOSIX_riscv64::FPR &posix_fpr) {
  // FreeBSD stores FP registers as 128-bit (fp_x[32][2])
  // POSIX expects 64-bit (fpr[32])
  // We only use the lower 64 bits (D extension, double precision)
  for (int i = 0; i < 32; i++)
    posix_fpr.fpr[i] = freebsd_fpr.fp_x[i][0];

  // FCSR: FreeBSD has 64-bit, POSIX expects 32-bit
  posix_fpr.fcsr = static_cast<uint32_t>(freebsd_fpr.fp_fcsr);
}

void NativeRegisterContextFreeBSD_riscv64::POSIXToFreeBSDFPR(
    const RegisterInfoPOSIX_riscv64::FPR &posix_fpr,
    struct fpreg &freebsd_fpr) {
  // POSIX has 64-bit FP registers, FreeBSD expects 128-bit
  for (int i = 0; i < 32; i++) {
    freebsd_fpr.fp_x[i][0] = posix_fpr.fpr[i]; // Lower 64 bits
    freebsd_fpr.fp_x[i][1] = 0;                // Upper 64 bits (unused for D)
  }

  // FCSR: POSIX has 32-bit, FreeBSD expects 64-bit
  freebsd_fpr.fp_fcsr = static_cast<uint64_t>(posix_fpr.fcsr);
}

NativeRegisterContextFreeBSD *
NativeRegisterContextFreeBSD::CreateHostNativeRegisterContextFreeBSD(
    const ArchSpec &target_arch, NativeThreadFreeBSD &native_thread) {
  return new NativeRegisterContextFreeBSD_riscv64(target_arch, native_thread);
}

NativeRegisterContextFreeBSD_riscv64::NativeRegisterContextFreeBSD_riscv64(
    const ArchSpec &target_arch, NativeThreadFreeBSD &native_thread)
    : NativeRegisterContextFreeBSD(native_thread), m_gpr(), m_fpr(),
      m_gpr_is_valid(false), m_fpr_is_valid(false) {
  m_register_info_interface_up =
      std::make_unique<RegisterInfoPOSIX_riscv64>(target_arch, 0);

  ::memset(&m_gpr, 0, sizeof(m_gpr));
  ::memset(&m_fpr, 0, sizeof(m_fpr));
}

RegisterInfoPOSIX_riscv64 &
NativeRegisterContextFreeBSD_riscv64::GetRegisterInfo() const {
  return static_cast<RegisterInfoPOSIX_riscv64 &>(
      *m_register_info_interface_up);
}

uint32_t NativeRegisterContextFreeBSD_riscv64::GetRegisterSetCount() const {
  return GetRegisterInfo().GetRegisterSetCount();
}

const RegisterSet *
NativeRegisterContextFreeBSD_riscv64::GetRegisterSet(uint32_t set_index) const {
  return GetRegisterInfo().GetRegisterSet(set_index);
}

uint32_t NativeRegisterContextFreeBSD_riscv64::GetUserRegisterCount() const {
  uint32_t count = 0;
  for (uint32_t set_index = 0; set_index < GetRegisterSetCount(); ++set_index)
    count += GetRegisterSet(set_index)->num_registers;
  return count;
}

void NativeRegisterContextFreeBSD_riscv64::InvalidateAllRegisters() {
  m_gpr_is_valid = false;
  m_fpr_is_valid = false;
}

Status NativeRegisterContextFreeBSD_riscv64::ReadGPR() {
  if (m_gpr_is_valid)
    return Status();

  Status error =
      NativeProcessFreeBSD::PtraceWrapper(PT_GETREGS, m_thread.GetID(), &m_gpr);

  if (error.Success())
    m_gpr_is_valid = true;

  return error;
}

Status NativeRegisterContextFreeBSD_riscv64::WriteGPR() {
  Status error =
      NativeProcessFreeBSD::PtraceWrapper(PT_SETREGS, m_thread.GetID(), &m_gpr);

  if (error.Success())
    m_gpr_is_valid = true;

  return error;
}

Status NativeRegisterContextFreeBSD_riscv64::ReadFPR() {
  if (m_fpr_is_valid)
    return Status();

  Status error = NativeProcessFreeBSD::PtraceWrapper(PT_GETFPREGS,
                                                     m_thread.GetID(), &m_fpr);

  if (error.Success())
    m_fpr_is_valid = true;

  return error;
}

Status NativeRegisterContextFreeBSD_riscv64::WriteFPR() {
  Status error = NativeProcessFreeBSD::PtraceWrapper(PT_SETFPREGS,
                                                     m_thread.GetID(), &m_fpr);

  if (error.Success())
    m_fpr_is_valid = true;

  return error;
}

Status
NativeRegisterContextFreeBSD_riscv64::GetGPRValue(uint32_t reg_index,
                                                  uint64_t &value) const {
  // Map LLDB register index to FreeBSD struct reg field
  switch (reg_index) {
  case gpr_pc_riscv64:
    value = m_gpr.sepc;
    return Status();
  case gpr_ra_riscv64:
    value = m_gpr.ra;
    return Status();
  case gpr_sp_riscv64:
    value = m_gpr.sp;
    return Status();
  case gpr_gp_riscv64:
    value = m_gpr.gp;
    return Status();
  case gpr_tp_riscv64:
    value = m_gpr.tp;
    return Status();

  // t0-t6
  case gpr_t0_riscv64:
  case gpr_t1_riscv64:
  case gpr_t2_riscv64:
    value = m_gpr.t[reg_index - gpr_t0_riscv64];
    return Status();
  case gpr_t3_riscv64:
  case gpr_t4_riscv64:
  case gpr_t5_riscv64:
  case gpr_t6_riscv64:
    value = m_gpr.t[reg_index - gpr_t3_riscv64 + 3];
    return Status();

  // s0-s11
  case gpr_s0_riscv64:
  case gpr_s1_riscv64:
    value = m_gpr.s[reg_index - gpr_s0_riscv64];
    return Status();
  case gpr_s2_riscv64:
  case gpr_s3_riscv64:
  case gpr_s4_riscv64:
  case gpr_s5_riscv64:
  case gpr_s6_riscv64:
  case gpr_s7_riscv64:
  case gpr_s8_riscv64:
  case gpr_s9_riscv64:
  case gpr_s10_riscv64:
  case gpr_s11_riscv64:
    value = m_gpr.s[reg_index - gpr_s2_riscv64 + 2];
    return Status();

  // a0-a7
  case gpr_a0_riscv64:
  case gpr_a1_riscv64:
  case gpr_a2_riscv64:
  case gpr_a3_riscv64:
  case gpr_a4_riscv64:
  case gpr_a5_riscv64:
  case gpr_a6_riscv64:
  case gpr_a7_riscv64:
    value = m_gpr.a[reg_index - gpr_a0_riscv64];
    return Status();

  default:
    return Status::FromErrorStringWithFormat("invalid GPR register index: %u",
                                             reg_index);
  }
}

Status NativeRegisterContextFreeBSD_riscv64::SetGPRValue(uint32_t reg_index,
                                                         uint64_t value) {
  switch (reg_index) {
  case gpr_pc_riscv64:
    m_gpr.sepc = value;
    return Status();
  case gpr_ra_riscv64:
    m_gpr.ra = value;
    return Status();
  case gpr_sp_riscv64:
    m_gpr.sp = value;
    return Status();
  case gpr_gp_riscv64:
    m_gpr.gp = value;
    return Status();
  case gpr_tp_riscv64:
    m_gpr.tp = value;
    return Status();

  case gpr_t0_riscv64:
  case gpr_t1_riscv64:
  case gpr_t2_riscv64:
    m_gpr.t[reg_index - gpr_t0_riscv64] = value;
    return Status();
  case gpr_t3_riscv64:
  case gpr_t4_riscv64:
  case gpr_t5_riscv64:
  case gpr_t6_riscv64:
    m_gpr.t[reg_index - gpr_t3_riscv64 + 3] = value;
    return Status();

  case gpr_s0_riscv64:
  case gpr_s1_riscv64:
    m_gpr.s[reg_index - gpr_s0_riscv64] = value;
    return Status();
  case gpr_s2_riscv64:
  case gpr_s3_riscv64:
  case gpr_s4_riscv64:
  case gpr_s5_riscv64:
  case gpr_s6_riscv64:
  case gpr_s7_riscv64:
  case gpr_s8_riscv64:
  case gpr_s9_riscv64:
  case gpr_s10_riscv64:
  case gpr_s11_riscv64:
    m_gpr.s[reg_index - gpr_s2_riscv64 + 2] = value;
    return Status();

  case gpr_a0_riscv64:
  case gpr_a1_riscv64:
  case gpr_a2_riscv64:
  case gpr_a3_riscv64:
  case gpr_a4_riscv64:
  case gpr_a5_riscv64:
  case gpr_a6_riscv64:
  case gpr_a7_riscv64:
    m_gpr.a[reg_index - gpr_a0_riscv64] = value;
    return Status();

  default:
    return Status::FromErrorStringWithFormat("invalid GPR register index: %u",
                                             reg_index);
  }
}

Status
NativeRegisterContextFreeBSD_riscv64::ReadRegister(const RegisterInfo *reg_info,
                                                   RegisterValue &reg_value) {
  if (!reg_info)
    return Status::FromErrorString("reg_info NULL");

  const uint32_t reg = reg_info->kinds[lldb::eRegisterKindLLDB];

  if (reg == LLDB_INVALID_REGNUM)
    return Status::FromErrorStringWithFormat(
        "no lldb regnum for %s", reg_info->name ? reg_info->name : "<unknown>");

  if (GetRegisterInfo().IsGPR(reg)) {
    Status error = ReadGPR();
    if (error.Fail())
      return error;

    uint64_t value;
    error = GetGPRValue(reg, value);
    if (error.Fail())
      return error;

    reg_value = value;
    return Status();
  }

  if (GetRegisterInfo().IsFPR(reg)) {
    Status error = ReadFPR();
    if (error.Fail())
      return error;

    uint32_t fpr_index =
        reg - GetRegisterInfo().GetRegisterInfo()[reg].kinds[eRegisterKindLLDB];

    if (fpr_index < 32)
      reg_value = m_fpr.fp_x[fpr_index][0]; // Lower 64 bits
    else if (fpr_index == 32)
      reg_value = static_cast<uint32_t>(m_fpr.fp_fcsr);
    else
      return Status::FromErrorString("invalid FPR index");

    return Status();
  }

  return Status::FromErrorStringWithFormat("unsupported register type: %u",
                                           reg);
}

Status NativeRegisterContextFreeBSD_riscv64::WriteRegister(
    const RegisterInfo *reg_info, const RegisterValue &reg_value) {
  if (!reg_info)
    return Status::FromErrorString("reg_info NULL");

  const uint32_t reg = reg_info->kinds[lldb::eRegisterKindLLDB];

  if (reg == LLDB_INVALID_REGNUM)
    return Status::FromErrorStringWithFormat(
        "no lldb regnum for %s", reg_info->name ? reg_info->name : "<unknown>");

  if (GetRegisterInfo().IsGPR(reg)) {
    Status error = ReadGPR(); // Read first to preserve other registers
    if (error.Fail())
      return error;

    error = SetGPRValue(reg, reg_value.GetAsUInt64());
    if (error.Fail())
      return error;

    return WriteGPR();
  }

  if (GetRegisterInfo().IsFPR(reg)) {
    Status error = ReadFPR();
    if (error.Fail())
      return error;

    uint32_t fpr_index =
        reg - GetRegisterInfo().GetRegisterInfo()[reg].kinds[eRegisterKindLLDB];

    if (fpr_index < 32) {
      m_fpr.fp_x[fpr_index][0] = reg_value.GetAsUInt64();
      m_fpr.fp_x[fpr_index][1] = 0;
    } else if (fpr_index == 32) {
      m_fpr.fp_fcsr = reg_value.GetAsUInt32();
    } else {
      return Status::FromErrorString("invalid FPR index");
    }

    return WriteFPR();
  }

  return Status::FromErrorStringWithFormat("unsupported register type: %u",
                                           reg);
}

Status NativeRegisterContextFreeBSD_riscv64::ReadAllRegisterValues(
    lldb::WritableDataBufferSP &data_sp) {
  Status error = ReadGPR();
  if (error.Fail())
    return error;

  error = ReadFPR();
  if (error.Fail())
    return error;

  // Allocate buffer for POSIX format
  const size_t total_size = sizeof(RegisterInfoPOSIX_riscv64::GPR) +
                            sizeof(RegisterInfoPOSIX_riscv64::FPR);
  data_sp = std::make_shared<DataBufferHeap>(total_size, 0);
  if (!data_sp || !data_sp->GetBytes())
    return Status::FromErrorString(
        "failed to allocate data buffer for POSIX-layout register data");

  // Get pointers to GPR and FPR sections of buffer
  auto *gpr_dst =
      reinterpret_cast<RegisterInfoPOSIX_riscv64::GPR *>(data_sp->GetBytes());
  auto *fpr_dst = reinterpret_cast<RegisterInfoPOSIX_riscv64::FPR *>(
      data_sp->GetBytes() + sizeof(RegisterInfoPOSIX_riscv64::GPR));

  // Convert FreeBSD format to POSIX format
  FreeBSDToPOSIXGPR(m_gpr, *gpr_dst);
  FreeBSDToPOSIXFPR(m_fpr, *fpr_dst);

  return Status();
}

Status NativeRegisterContextFreeBSD_riscv64::WriteAllRegisterValues(
    const lldb::DataBufferSP &data_sp) {
  if (!data_sp)
    return Status::FromErrorString("invalid data_sp provided");

  const size_t expected_size = sizeof(RegisterInfoPOSIX_riscv64::GPR) +
                               sizeof(RegisterInfoPOSIX_riscv64::FPR);

  if (data_sp->GetByteSize() != expected_size) {
    return Status::FromErrorStringWithFormat(
        "data_sp size mismatch, expected %zu, actual %" PRIu64, expected_size,
        data_sp->GetByteSize());
  }

  const uint8_t *src = data_sp->GetBytes();
  if (!src)
    return Status::FromErrorString("DataBuffer::GetBytes() returned null");

  // Get pointers to GPR and FPR sections of buffer
  const auto *gpr_src =
      reinterpret_cast<const RegisterInfoPOSIX_riscv64::GPR *>(src);
  const auto *fpr_src =
      reinterpret_cast<const RegisterInfoPOSIX_riscv64::FPR *>(
          src + sizeof(RegisterInfoPOSIX_riscv64::GPR));

  // Convert POSIX format to FreeBSD format
  POSIXToFreeBSDGPR(*gpr_src, m_gpr);
  POSIXToFreeBSDFPR(*fpr_src, m_fpr);

  Status error = WriteGPR();
  if (error.Fail())
    return error;

  return WriteFPR();
}

#endif // defined(__riscv) && __riscv_xlen == 64
