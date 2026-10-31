//===-- NativeRegisterContextOpenBSD_arm64.cpp ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#if defined(__aarch64__)

#include "NativeRegisterContextOpenBSD_arm64.h"

#include "lldb/Utility/DataBufferHeap.h"
#include "lldb/Utility/RegisterValue.h"
#include "lldb/Utility/Status.h"
#include "llvm/Support/ErrorHandling.h"

#include "Plugins/Process/OpenBSD/NativeProcessOpenBSD.h"
#include "Register/Common/RegisterInfo_arm64.h"
#include "Register/Common/RegisterTypeDetector_arm64.h"

// clang-format off
#include <sys/types.h>
#include <sys/ptrace.h>
// clang-format on

#include <cinttypes>
#include <cstddef>
#include <cstring>
#include <mutex>

using namespace lldb;
using namespace lldb_private;
using namespace lldb_private::process_openbsd;

// The registers are accessed by their RegisterInfo byte_offset, which is only
// correct if OpenBSD's structures match LLDB's at the start.
static_assert(offsetof(struct reg, r_reg) ==
              offsetof(RegisterInfo_arm64::GPR, x));
static_assert(offsetof(struct reg, r_reg[29]) ==
              offsetof(RegisterInfo_arm64::GPR, fp));
static_assert(offsetof(struct reg, r_lr) ==
              offsetof(RegisterInfo_arm64::GPR, lr));
static_assert(offsetof(struct reg, r_sp) ==
              offsetof(RegisterInfo_arm64::GPR, sp));
static_assert(offsetof(struct reg, r_pc) ==
              offsetof(RegisterInfo_arm64::GPR, pc));
// cpsr is the low 32 bits of r_spsr.
static_assert(offsetof(struct reg, r_spsr) ==
              offsetof(RegisterInfo_arm64::GPR, cpsr));
static_assert(sizeof(struct reg) >= sizeof(RegisterInfo_arm64::GPR));
static_assert(offsetof(struct fpreg, fp_reg) ==
              offsetof(RegisterInfo_arm64::FPU, v));
static_assert(offsetof(struct fpreg, fp_sr) ==
              offsetof(RegisterInfo_arm64::FPU, fpsr));
static_assert(offsetof(struct fpreg, fp_cr) ==
              offsetof(RegisterInfo_arm64::FPU, fpcr));

// A register context is created for each thread, but every thread has the
// same register fields. The mutex keeps the instances from racing each other
// and lets later ones reuse the fields the first one detected.
static std::mutex g_register_type_detector_mutex;
static Arm64RegisterTypeDetector g_register_type_detector;

std::unique_ptr<NativeRegisterContextOpenBSD>
NativeRegisterContextOpenBSD::CreateHostNativeRegisterContextOpenBSD(
    const ArchSpec &target_arch, NativeThreadProtocol &native_thread) {
  std::lock_guard<std::mutex> lock(g_register_type_detector_mutex);
  if (!g_register_type_detector.HasDetected()) {
    auto &process =
        static_cast<NativeProcessOpenBSD &>(native_thread.GetProcess());
    // OpenBSD uses the same AT_HWCAP number as FreeBSD, and the same hwcap
    // bits as Linux.
    g_register_type_detector.DetectTypes(
        process.GetAuxValue(AuxVector::AUXV_FREEBSD_AT_HWCAP).value_or(0),
        process.GetAuxValue(AuxVector::AUXV_AT_HWCAP2).value_or(0),
        /*hwcap3=*/0);
  }

  return std::make_unique<NativeRegisterContextOpenBSD_arm64>(target_arch,
                                                              native_thread);
}

static RegisterInfoInterface *
CreateRegisterInfoInterface(const ArchSpec &target_arch) {
  Flags opt_regsets =
      RegisterInfo_arm64::eRegsetMaskPAuth | RegisterInfo_arm64::eRegsetMaskTLS;
  return new RegisterInfo_arm64(target_arch, opt_regsets);
}

NativeRegisterContextOpenBSD_arm64::NativeRegisterContextOpenBSD_arm64(
    const ArchSpec &target_arch, NativeThreadProtocol &native_thread)
    : NativeRegisterContextOpenBSD(native_thread,
                                   CreateRegisterInfoInterface(target_arch)),
      m_gpr(), m_fpr(), m_pacmask() {
  g_register_type_detector.UpdateRegisterInfo(
      GetRegisterInfoInterface().GetRegisterInfo(),
      GetRegisterInfoInterface().GetRegisterCount());
}

RegisterInfo_arm64 &
NativeRegisterContextOpenBSD_arm64::GetRegisterInfo() const {
  return static_cast<RegisterInfo_arm64 &>(*m_register_info_interface_up);
}

uint32_t NativeRegisterContextOpenBSD_arm64::GetRegisterSetCount() const {
  return GetRegisterInfo().GetRegisterSetCount();
}

const RegisterSet *
NativeRegisterContextOpenBSD_arm64::GetRegisterSet(uint32_t set_index) const {
  return GetRegisterInfo().GetRegisterSet(set_index);
}

uint32_t NativeRegisterContextOpenBSD_arm64::GetUserRegisterCount() const {
  uint32_t count = 0;
  for (uint32_t set_index = 0; set_index < GetRegisterSetCount(); ++set_index)
    count += GetRegisterSet(set_index)->num_registers;
  return count;
}

std::optional<NativeRegisterContextOpenBSD_arm64::RegSetKind>
NativeRegisterContextOpenBSD_arm64::GetSetForNativeRegNum(
    uint32_t reg_num) const {
  if (GetRegisterInfo().IsPAuthReg(reg_num))
    return PACMaskRegSet;
  if (GetRegisterInfo().IsTLSReg(reg_num))
    return TLSRegSet;

  switch (GetRegisterInfo().GetRegisterSetFromRegisterIndex(reg_num)) {
  case RegisterInfo_arm64::GPRegSet:
    return GPRegSet;
  case RegisterInfo_arm64::FPRegSet:
    return FPRegSet;
  }
  return std::nullopt;
}

Status NativeRegisterContextOpenBSD_arm64::ReadRegisterSet(RegSetKind set) {
  switch (set) {
  case GPRegSet:
  case TLSRegSet:
    return ReadGPR();
  case FPRegSet:
    return ReadFPR();
  case PACMaskRegSet:
#ifdef PT_PACMASK
    return NativeProcessOpenBSD::PtraceWrapper(PT_PACMASK, m_thread.GetID(),
                                               &m_pacmask, sizeof(m_pacmask));
#else
    ::memset(&m_pacmask, 0, sizeof(m_pacmask));
    return Status();
#endif
  }
  llvm_unreachable("NativeRegisterContextOpenBSD_arm64::ReadRegisterSet");
}

Status NativeRegisterContextOpenBSD_arm64::WriteRegisterSet(RegSetKind set) {
  switch (set) {
  case GPRegSet:
  case TLSRegSet:
    return WriteGPR();
  case FPRegSet:
    return WriteFPR();
  case PACMaskRegSet:
    return Status::FromErrorString("the PAC mask registers are read-only");
  }
  llvm_unreachable("NativeRegisterContextOpenBSD_arm64::WriteRegisterSet");
}

uint8_t *
NativeRegisterContextOpenBSD_arm64::GetRegisterData(RegSetKind set,
                                                    const RegisterInfo &info) {
  uint8_t *base = nullptr;
  size_t base_size = 0;
  uint32_t offset = info.byte_offset;

  switch (set) {
  case GPRegSet:
    base = reinterpret_cast<uint8_t *>(&m_gpr);
    base_size = sizeof(m_gpr);
    break;
  case FPRegSet:
    base = reinterpret_cast<uint8_t *>(&m_fpr);
    base_size = sizeof(m_fpr);
    offset -= sizeof(RegisterInfo_arm64::GPR);
    break;
  case PACMaskRegSet:
    base = reinterpret_cast<uint8_t *>(&m_pacmask);
    base_size = sizeof(m_pacmask);
    offset -= GetRegisterInfo().GetPAuthOffset();
    break;
  case TLSRegSet:
    // tpidr is r_tpidr in struct reg.
    base = reinterpret_cast<uint8_t *>(&m_gpr.r_tpidr);
    base_size = sizeof(m_gpr.r_tpidr);
    offset -= GetRegisterInfo().GetTLSOffset();
    break;
  }

  assert(offset + info.byte_size <= base_size);
  UNUSED_IF_ASSERT_DISABLED(base_size);
  return base + offset;
}

Status
NativeRegisterContextOpenBSD_arm64::ReadRegister(const RegisterInfo *reg_info,
                                                 RegisterValue &reg_value) {
  if (!reg_info)
    return Status::FromErrorString("reg_info NULL");

  const uint32_t reg = reg_info->kinds[lldb::eRegisterKindLLDB];
  if (reg == LLDB_INVALID_REGNUM)
    return Status::FromErrorStringWithFormat(
        "no lldb regnum for %s",
        reg_info->name ? reg_info->name : "<unknown register>");

  std::optional<RegSetKind> set = GetSetForNativeRegNum(reg);
  if (!set)
    return Status::FromErrorStringWithFormat(
        "register \"%s\" is in unrecognized set", reg_info->name);

  Status error = ReadRegisterSet(*set);
  if (error.Fail())
    return error;

  reg_value.SetBytes(GetRegisterData(*set, *reg_info), reg_info->byte_size,
                     endian::InlHostByteOrder());
  return error;
}

Status NativeRegisterContextOpenBSD_arm64::WriteRegister(
    const RegisterInfo *reg_info, const RegisterValue &reg_value) {
  if (!reg_info)
    return Status::FromErrorString("reg_info NULL");

  const uint32_t reg = reg_info->kinds[lldb::eRegisterKindLLDB];
  if (reg == LLDB_INVALID_REGNUM)
    return Status::FromErrorStringWithFormat(
        "no lldb regnum for %s",
        reg_info->name ? reg_info->name : "<unknown register>");

  std::optional<RegSetKind> set = GetSetForNativeRegNum(reg);
  if (!set)
    return Status::FromErrorStringWithFormat(
        "register \"%s\" is in unrecognized set", reg_info->name);

  if (*set == PACMaskRegSet)
    return Status::FromErrorStringWithFormat("register \"%s\" is read-only",
                                             reg_info->name);

  Status error = ReadRegisterSet(*set);
  if (error.Fail())
    return error;

  if (reg_value.GetByteSize() < reg_info->byte_size)
    return Status::FromErrorStringWithFormat(
        "value for register \"%s\" is too short", reg_info->name);

  ::memcpy(GetRegisterData(*set, *reg_info), reg_value.GetBytes(),
           reg_info->byte_size);

  return WriteRegisterSet(*set);
}

Status NativeRegisterContextOpenBSD_arm64::ReadAllRegisterValues(
    lldb::WritableDataBufferSP &data_sp) {
  Status error = ReadRegisterSet(GPRegSet);
  if (error.Fail())
    return error;

  error = ReadRegisterSet(FPRegSet);
  if (error.Fail())
    return error;

  data_sp.reset(new DataBufferHeap(sizeof(m_gpr) + sizeof(m_fpr), 0));
  uint8_t *dst = data_sp->GetBytes();
  ::memcpy(dst, &m_gpr, sizeof(m_gpr));
  ::memcpy(dst + sizeof(m_gpr), &m_fpr, sizeof(m_fpr));

  return error;
}

Status NativeRegisterContextOpenBSD_arm64::WriteAllRegisterValues(
    const lldb::DataBufferSP &data_sp) {
  if (!data_sp)
    return Status::FromErrorStringWithFormat(
        "NativeRegisterContextOpenBSD_arm64::%s invalid data_sp provided",
        __FUNCTION__);

  const size_t size = sizeof(m_gpr) + sizeof(m_fpr);
  if (data_sp->GetByteSize() != size)
    return Status::FromErrorStringWithFormat(
        "NativeRegisterContextOpenBSD_arm64::%s data_sp contained mismatched "
        "data size, expected %zu, actual %" PRIu64,
        __FUNCTION__, size, data_sp->GetByteSize());

  const uint8_t *src = data_sp->GetBytes();
  if (src == nullptr)
    return Status::FromErrorStringWithFormat(
        "NativeRegisterContextOpenBSD_arm64::%s "
        "DataBuffer::GetBytes() returned a null pointer",
        __FUNCTION__);

  ::memcpy(&m_gpr, src, sizeof(m_gpr));
  ::memcpy(&m_fpr, src + sizeof(m_gpr), sizeof(m_fpr));

  Status error = WriteRegisterSet(GPRegSet);
  if (error.Fail())
    return error;

  return WriteRegisterSet(FPRegSet);
}

#endif // defined(__aarch64__)
