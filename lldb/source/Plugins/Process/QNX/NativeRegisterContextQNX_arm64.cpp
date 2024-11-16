//===-- NativeRegisterContextQNX_arm64.cpp --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#if defined(__aarch64__) && defined(__QNX__)

#include "NativeRegisterContextQNX_arm64.h"

#include <sys/procfs.h>

#include "lldb/Utility/DataBufferHeap.h"
#include "lldb/Utility/RegisterValue.h"

#include "Plugins/Process/POSIX/ProcessPOSIXLog.h"
#include "Plugins/Process/QNX/NativeProcessQNX.h"
#include "Plugins/Process/Utility/RegisterInfoPOSIX_arm64.h"

using namespace lldb;
using namespace lldb_private;
using namespace lldb_private::process_qnx;

NativeRegisterContextQNX *
NativeRegisterContextQNX::CreateHostNativeRegisterContextQNX(
    const ArchSpec &target_arch, NativeThreadProtocol &native_thread) {
  return new NativeRegisterContextQNX_arm64(target_arch, native_thread);
}

NativeRegisterContextQNX_arm64::NativeRegisterContextQNX_arm64(
    const ArchSpec &target_arch, NativeThreadProtocol &native_thread)
    : NativeRegisterContextRegisterInfo(
          native_thread, new RegisterInfoPOSIX_arm64(target_arch, 0)) {}

uint32_t NativeRegisterContextQNX_arm64::GetRegisterSetCount() const {
  return GetRegisterInfo().GetRegisterSetCount();
}

uint32_t NativeRegisterContextQNX_arm64::GetUserRegisterCount() const {
  uint32_t count = 0;
  for (uint32_t set_index = 0; set_index < GetRegisterSetCount(); ++set_index)
    count += GetRegisterSet(set_index)->num_registers;
  return count;
}

const RegisterSet *
NativeRegisterContextQNX_arm64::GetRegisterSet(uint32_t set_index) const {
  return GetRegisterInfo().GetRegisterSet(set_index);
}

Status
NativeRegisterContextQNX_arm64::ReadRegister(const RegisterInfo *reg_info,
                                             RegisterValue &reg_value) {
  Status error;

  if (!reg_info) {
    error.SetErrorString("reg_info is NULL");
    return error;
  }

  const uint32_t reg = reg_info->kinds[lldb::eRegisterKindLLDB];

  if (reg == LLDB_INVALID_REGNUM) {
    error.SetErrorStringWithFormat(
        "no lldb regnum for %s",
        reg_info && reg_info->name ? reg_info->name : "<unknown register>");
    return error;
  }

  uint32_t set = GetRegisterInfo().GetRegisterSetFromRegisterIndex(reg);

  switch (set) {
  // TODO: Read alternate and performance registers.
  case RegisterInfoPOSIX_arm64::GPRegSet:
    if (reg_info->byte_offset + reg_info->byte_size >
        sizeof(AARCH64_CPU_REGISTERS)) {
      error.SetErrorString("reg_info->byte_offset + reg_info->byte_size > "
                           "sizeof(AARCH64_CPU_REGISTERS)");
      return error;
    }
    error = ReadGPR();
    if (error.Fail())
      return error;
    reg_value.SetBytes(reinterpret_cast<_Uint8t *>(&m_cpu_reg_data) +
                           reg_info->byte_offset,
                       reg_info->byte_size, endian::InlHostByteOrder());
    break;

  case RegisterInfoPOSIX_arm64::FPRegSet:
    if (reg_info->byte_offset + reg_info->byte_size >
        sizeof(AARCH64_FPU_REGISTERS)) {
      error.SetErrorString("reg_info->byte_offset + reg_info->byte_size > "
                           "sizeof(AARCH64_FPU_REGISTERS)");
      return error;
    }
    error = ReadFPR();
    if (error.Fail())
      return error;
    reg_value.SetBytes(reinterpret_cast<_Uint8t *>(&m_fpu_reg_data) +
                           reg_info->byte_offset,
                       reg_info->byte_size, endian::InlHostByteOrder());
    break;

  default:
    error.SetErrorString("unrecognized register set");
  }

  return error;
}

Status
NativeRegisterContextQNX_arm64::WriteRegister(const RegisterInfo *reg_info,
                                              const RegisterValue &reg_value) {
  Status error;

  if (!reg_info) {
    error.SetErrorString("reg_info is NULL");
    return error;
  }

  const uint32_t reg = reg_info->kinds[lldb::eRegisterKindLLDB];

  if (reg == LLDB_INVALID_REGNUM) {
    error.SetErrorStringWithFormat(
        "no lldb regnum for %s",
        reg_info && reg_info->name ? reg_info->name : "<unknown register>");
    return error;
  }

  uint32_t set = GetRegisterInfo().GetRegisterSetFromRegisterIndex(reg);

  switch (set) {
  // TODO: Write to alternate and performance registers.
  case RegisterInfoPOSIX_arm64::GPRegSet:
    if (reg_info->byte_offset + reg_info->byte_size >
        sizeof(AARCH64_CPU_REGISTERS)) {
      error.SetErrorString("reg_info->byte_offset + reg_info->byte_size > "
                           "sizeof(AARCH64_CPU_REGISTERS)");
      return error;
    }
    ::memcpy(&m_cpu_reg_data.gpr + reg_info->byte_offset, reg_value.GetBytes(),
             reg_info->byte_size);
    error = WriteGPR();
    if (error.Fail())
      return error;
    break;

  case RegisterInfoPOSIX_arm64::FPRegSet:
    if (reg_info->byte_offset + reg_info->byte_size >
        sizeof(AARCH64_FPU_REGISTERS)) {
      error.SetErrorString("reg_info->byte_offset + reg_info->byte_size > "
                           "sizeof(AARCH64_FPU_REGISTERS)");
      return error;
    }
    ::memcpy(&m_fpu_reg_data.reg + reg_info->byte_offset, reg_value.GetBytes(),
             reg_info->byte_size);
    error = WriteFPR();
    if (error.Fail())
      return error;
    break;

  default:
    error.SetErrorString("unrecognized register set");
  }

  return error;
}

Status NativeRegisterContextQNX_arm64::ReadAllRegisterValues(
    lldb::WritableDataBufferSP &data_sp) {
  // TODO: Read alternate and performance registers.
  Status error;
  uint32_t reg_data_byte_size = sizeof(AARCH64_CPU_REGISTERS);

  error = ReadGPR();
  if (error.Fail())
    return error;

  reg_data_byte_size += sizeof(AARCH64_FPU_REGISTERS);

  error = ReadFPR();
  if (error.Fail())
    return error;

  data_sp.reset(new DataBufferHeap(reg_data_byte_size, 0));
  uint8_t *dst = data_sp->GetBytes();
  ::memcpy(dst, &m_cpu_reg_data.gpr, sizeof(AARCH64_CPU_REGISTERS));
  dst += sizeof(AARCH64_CPU_REGISTERS);
  ::memcpy(dst, &m_fpu_reg_data.reg, sizeof(AARCH64_FPU_REGISTERS));

  return error;
}

Status NativeRegisterContextQNX_arm64::WriteAllRegisterValues(
    const lldb::DataBufferSP &data_sp) {
  Status error;

  if (!data_sp) {
    error.SetErrorString("data_sp is NULL");
    return error;
  }

  if (data_sp->GetByteSize() !=
      (sizeof(AARCH64_CPU_REGISTERS) + sizeof(AARCH64_FPU_REGISTERS))) {
    error.SetErrorStringWithFormat(
        "data_sp->GetByteSize() != "
        "(sizeof(AARCH64_CPU_REGISTERS) + sizeof(AARCH64_FPU_REGISTERS)), "
        "expected %" PRIu64 ", is %" PRIu64,
        sizeof(AARCH64_CPU_REGISTERS) + sizeof(AARCH64_FPU_REGISTERS),
        data_sp->GetByteSize());
    return error;
  }

  const uint8_t *src = data_sp->GetBytes();
  if (src == nullptr) {
    error.SetErrorString("DataBuffer::GetBytes() returned nullptr");
    return error;
  }
  ::memcpy(&m_cpu_reg_data.gpr, src, sizeof(AARCH64_CPU_REGISTERS));
  error = WriteGPR();
  if (error.Fail())
    return error;

  src += sizeof(AARCH64_CPU_REGISTERS);
  ::memcpy(&m_fpu_reg_data.reg, src, sizeof(AARCH64_FPU_REGISTERS));
  error = WriteFPR();
  if (error.Fail())
    return error;

  return error;
}

Status NativeRegisterContextQNX_arm64::ReadGPR() {
  procfs_greg greg;
  memset(&greg, 0x0, sizeof(procfs_greg));

  pthread_t tid = static_cast<pthread_t>(GetThreadID());
  int greg_size;
  Status error = NativeProcessQNX::DevctlWrapper(
      static_cast<NativeProcessQNX &>(GetThread().GetProcess())
          .GetFileDescriptor(),
      DCMD_PROC_CURTHREAD, &tid, sizeof(tid), nullptr);

  if (error.Fail())
    return error;

  error = NativeProcessQNX::DevctlWrapper(
      static_cast<NativeProcessQNX &>(GetThread().GetProcess())
          .GetFileDescriptor(),
      DCMD_PROC_GETGREG, &greg, sizeof(procfs_greg), &greg_size);

  if (error.Fail())
    return error;

  if (greg_size != sizeof(AARCH64_CPU_REGISTERS)) {
    error.SetErrorString("greg_size != sizeof(AARCH64_CPU_REGISTERS)");
    return error;
  }

  memcpy(&m_cpu_reg_data, &greg, sizeof(AARCH64_CPU_REGISTERS));

  return error;
}

Status NativeRegisterContextQNX_arm64::WriteGPR() {
  procfs_greg greg;
  memcpy(&greg, &m_cpu_reg_data, sizeof(AARCH64_CPU_REGISTERS));

  pthread_t tid = static_cast<pthread_t>(GetThreadID());
  Status error = NativeProcessQNX::DevctlWrapper(
      static_cast<NativeProcessQNX &>(GetThread().GetProcess())
          .GetFileDescriptor(),
      DCMD_PROC_CURTHREAD, &tid, sizeof(tid), nullptr);

  if (error.Fail())
    return error;

  return NativeProcessQNX::DevctlWrapper(
      static_cast<NativeProcessQNX &>(GetThread().GetProcess())
          .GetFileDescriptor(),
      DCMD_PROC_SETGREG, &greg, sizeof(procfs_greg), nullptr);
}

Status NativeRegisterContextQNX_arm64::ReadFPR() {
  // If the thread hasn't used any floating-point arithmetic, then the read may
  // fail because an FPU context hasn't been allocated yet.
  procfs_fpreg fpreg;
  memset(&fpreg, 0x0, sizeof(procfs_fpreg));

  pthread_t tid = static_cast<pthread_t>(GetThreadID());
  int fpreg_size;
  Status error = NativeProcessQNX::DevctlWrapper(
      static_cast<NativeProcessQNX &>(GetThread().GetProcess())
          .GetFileDescriptor(),
      DCMD_PROC_CURTHREAD, &tid, sizeof(tid), nullptr);

  if (error.Fail())
    return error;

  error = NativeProcessQNX::DevctlWrapper(
      static_cast<NativeProcessQNX &>(GetThread().GetProcess())
          .GetFileDescriptor(),
      DCMD_PROC_GETFPREG, &fpreg, sizeof(procfs_fpreg), &fpreg_size);

  if (error.Fail())
    return error;

  if (fpreg_size != sizeof(AARCH64_FPU_REGISTERS)) {
    error.SetErrorString("fpreg_size != sizeof(AARCH64_FPU_REGISTERS)");
    return error;
  }

  memcpy(&m_fpu_reg_data, &fpreg, sizeof(AARCH64_FPU_REGISTERS));

  return error;
}

Status NativeRegisterContextQNX_arm64::WriteFPR() {
  procfs_fpreg fpreg;
  memcpy(&fpreg, &m_fpu_reg_data, sizeof(AARCH64_FPU_REGISTERS));

  pthread_t tid = static_cast<pthread_t>(GetThreadID());
  Status error = NativeProcessQNX::DevctlWrapper(
      static_cast<NativeProcessQNX &>(GetThread().GetProcess())
          .GetFileDescriptor(),
      DCMD_PROC_CURTHREAD, &tid, sizeof(tid), nullptr);

  if (error.Fail())
    return error;

  return NativeProcessQNX::DevctlWrapper(
      static_cast<NativeProcessQNX &>(GetThread().GetProcess())
          .GetFileDescriptor(),
      DCMD_PROC_SETFPREG, &fpreg, sizeof(procfs_fpreg), nullptr);
}

RegisterInfoPOSIX_arm64 &
NativeRegisterContextQNX_arm64::GetRegisterInfo() const {
  return static_cast<RegisterInfoPOSIX_arm64 &>(*m_register_info_interface_up);
}

#endif // defined(__aarch64__) && defined(__QNX__)
