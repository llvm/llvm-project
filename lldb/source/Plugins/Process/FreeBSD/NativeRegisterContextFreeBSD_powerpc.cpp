//===-- NativeRegisterContextFreeBSD_powerpc.cpp --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#if defined(__powerpc__)

#include "NativeRegisterContextFreeBSD_powerpc.h"

#include "lldb/Host/HostInfo.h"
#include "lldb/Utility/DataBufferHeap.h"
#include "lldb/Register/RegisterValue.h"
#include "lldb/Utility/Status.h"

#include "Plugins/Process/FreeBSD/NativeProcessFreeBSD.h"
#include "lldb/Register/Arch/lldb-powerpc-register-enums.h"

// clang-format off
#include <sys/param.h>
#include <sys/ptrace.h>
#include <sys/types.h>
// clang-format on
#include <optional>


namespace reg_ids = lldb_private::registers::powerpc::id;
using namespace lldb;
using namespace lldb_private;
using namespace lldb_private::process_freebsd;

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

// Number of register sets provided by this context.
enum { k_num_register_sets = 2 };

static const RegisterSet g_reg_sets_powerpc[k_num_register_sets] = {
    {"General Purpose Registers", "gpr", reg_ids::k_num_gpr_registers,
     g_gpr_regnums},
    {"Floating Point Registers", "fpr", reg_ids::k_num_fpr_registers,
     g_fpr_regnums},
};

NativeRegisterContextFreeBSD *
NativeRegisterContextFreeBSD::CreateHostNativeRegisterContextFreeBSD(
    const ArchSpec &target_arch, NativeThreadFreeBSD &native_thread) {
  return new NativeRegisterContextFreeBSD_powerpc(target_arch, native_thread);
}

static RegisterInfoInterface *
CreateRegisterInfoInterface(const ArchSpec &target_arch) {
  if (HostInfo::GetArchitecture().GetAddressByteSize() == 4) {
    return new RegisterInfoFreeBSD_powerpc32(target_arch);
  } else {
    assert((HostInfo::GetArchitecture().GetAddressByteSize() == 8) &&
           "Register setting path assumes this is a 64-bit host");
    return new RegisterInfoFreeBSD_powerpc64(target_arch);
  }
}

NativeRegisterContextFreeBSD_powerpc::NativeRegisterContextFreeBSD_powerpc(
    const ArchSpec &target_arch, NativeThreadFreeBSD &native_thread)
    : NativeRegisterContextRegisterInfo(
          native_thread, CreateRegisterInfoInterface(target_arch)) {}

RegisterInfoFreeBSD_powerpc &
NativeRegisterContextFreeBSD_powerpc::GetRegisterInfo() const {
  return static_cast<RegisterInfoFreeBSD_powerpc &>(
      *m_register_info_interface_up);
}

uint32_t NativeRegisterContextFreeBSD_powerpc::GetRegisterSetCount() const {
  return k_num_register_sets;
}

const RegisterSet *
NativeRegisterContextFreeBSD_powerpc::GetRegisterSet(uint32_t set_index) const {
  switch (GetRegisterInfoInterface().GetTargetArchitecture().GetMachine()) {
  case llvm::Triple::ppc:
    return &g_reg_sets_powerpc[set_index];
  default:
    llvm_unreachable("Unhandled target architecture.");
  }
}

std::optional<NativeRegisterContextFreeBSD_powerpc::RegSetKind>
NativeRegisterContextFreeBSD_powerpc::GetSetForNativeRegNum(
    uint32_t reg_num) const {
  switch (GetRegisterInfoInterface().GetTargetArchitecture().GetMachine()) {
  case llvm::Triple::ppc:
    if (reg_num >= reg_ids::k_first_gpr && reg_num <= reg_ids::k_last_gpr)
      return GPRegSet;
    if (reg_num >= reg_ids::k_first_fpr && reg_num <= reg_ids::k_last_fpr)
      return FPRegSet;
    break;
  default:
    llvm_unreachable("Unhandled target architecture.");
  }

  llvm_unreachable("Register does not belong to any register set");
}

uint32_t NativeRegisterContextFreeBSD_powerpc::GetUserRegisterCount() const {
  uint32_t count = 0;
  for (uint32_t set_index = 0; set_index < GetRegisterSetCount(); ++set_index)
    count += GetRegisterSet(set_index)->num_registers;
  return count;
}

Status NativeRegisterContextFreeBSD_powerpc::ReadRegisterSet(RegSetKind set) {
  switch (set) {
  case GPRegSet:
    return NativeProcessFreeBSD::PtraceWrapper(PT_GETREGS, m_thread.GetID(),
                                               m_reg_data.data());
  case FPRegSet:
    return NativeProcessFreeBSD::PtraceWrapper(PT_GETFPREGS, m_thread.GetID(),
                                               m_reg_data.data() + sizeof(reg));
  }
  llvm_unreachable("NativeRegisterContextFreeBSD_powerpc::ReadRegisterSet");
}

Status NativeRegisterContextFreeBSD_powerpc::WriteRegisterSet(RegSetKind set) {
  switch (set) {
  case GPRegSet:
    return NativeProcessFreeBSD::PtraceWrapper(PT_SETREGS, m_thread.GetID(),
                                               m_reg_data.data());
  case FPRegSet:
    return NativeProcessFreeBSD::PtraceWrapper(PT_SETFPREGS, m_thread.GetID(),
                                               m_reg_data.data() + sizeof(reg));
  }
  llvm_unreachable("NativeRegisterContextFreeBSD_powerpc::WriteRegisterSet");
}

Status
NativeRegisterContextFreeBSD_powerpc::ReadRegister(const RegisterInfo *reg_info,
                                                   RegisterValue &reg_value) {
  Status error;

  if (!reg_info) {
    error = Status::FromErrorString("reg_info NULL");
    return error;
  }

  const uint32_t reg = reg_info->kinds[lldb::eRegisterKindLLDB];

  if (reg == LLDB_INVALID_REGNUM)
    return Status::FromErrorStringWithFormat(
        "no lldb regnum for %s",
        reg_info && reg_info->name ? reg_info->name : "<unknown register>");

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

  assert(reg_info->byte_offset + reg_info->byte_size <= m_reg_data.size());
  reg_value.SetBytes(m_reg_data.data() + reg_info->byte_offset,
                     reg_info->byte_size, endian::InlHostByteOrder());
  return error;
}

Status NativeRegisterContextFreeBSD_powerpc::WriteRegister(
    const RegisterInfo *reg_info, const RegisterValue &reg_value) {
  Status error;

  if (!reg_info)
    return Status::FromErrorString("reg_info NULL");

  const uint32_t reg = reg_info->kinds[lldb::eRegisterKindLLDB];

  if (reg == LLDB_INVALID_REGNUM)
    return Status::FromErrorStringWithFormat(
        "no lldb regnum for %s",
        reg_info && reg_info->name ? reg_info->name : "<unknown register>");

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

  assert(reg_info->byte_offset + reg_info->byte_size <= m_reg_data.size());
  ::memcpy(m_reg_data.data() + reg_info->byte_offset, reg_value.GetBytes(),
           reg_info->byte_size);

  return WriteRegisterSet(set);
}

Status NativeRegisterContextFreeBSD_powerpc::ReadAllRegisterValues(
    lldb::WritableDataBufferSP &data_sp) {
  Status error;

  error = ReadRegisterSet(GPRegSet);
  if (error.Fail())
    return error;

  error = ReadRegisterSet(FPRegSet);
  if (error.Fail())
    return error;

  data_sp.reset(new DataBufferHeap(m_reg_data.size(), 0));
  uint8_t *dst = data_sp->GetBytes();
  ::memcpy(dst, m_reg_data.data(), m_reg_data.size());

  return error;
}

Status NativeRegisterContextFreeBSD_powerpc::WriteAllRegisterValues(
    const lldb::DataBufferSP &data_sp) {
  Status error;

  if (!data_sp) {
    error = Status::FromErrorStringWithFormat(
        "NativeRegisterContextFreeBSD_powerpc::%s invalid data_sp provided",
        __FUNCTION__);
    return error;
  }

  if (data_sp->GetByteSize() != m_reg_data.size()) {
    error = Status::FromErrorStringWithFormat(
        "NativeRegisterContextFreeBSD_powerpc::%s data_sp contained mismatched "
        "data size, expected %zu, actual %" PRIu64,
        __FUNCTION__, m_reg_data.size(), data_sp->GetByteSize());
    return error;
  }

  const uint8_t *src = data_sp->GetBytes();
  if (src == nullptr) {
    error = Status::FromErrorStringWithFormat(
        "NativeRegisterContextFreeBSD_powerpc::%s "
        "DataBuffer::GetBytes() returned a null "
        "pointer",
        __FUNCTION__);
    return error;
  }
  ::memcpy(m_reg_data.data(), src, m_reg_data.size());

  error = WriteRegisterSet(GPRegSet);
  if (error.Fail())
    return error;

  return WriteRegisterSet(FPRegSet);
}

llvm::Error NativeRegisterContextFreeBSD_powerpc::CopyHardwareWatchpointsFrom(
    NativeRegisterContextFreeBSD &source) {
  return llvm::Error::success();
}

#endif // defined (__powerpc__)
