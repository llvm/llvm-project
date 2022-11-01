//===-- NativeRegisterContextLinux_loongarch64.cpp ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#if defined(__loongarch__) && __loongarch_grlen == 64

#include "NativeRegisterContextLinux_loongarch64.h"

#include "lldb/Host/HostInfo.h"
#include "lldb/Utility/DataBufferHeap.h"
#include "lldb/Utility/Log.h"
#include "lldb/Utility/RegisterValue.h"
#include "lldb/Utility/Status.h"

#include "Plugins/Process/Linux/NativeProcessLinux.h"
#include "Plugins/Process/Linux/Procfs.h"

using namespace lldb;
using namespace lldb_private;
using namespace lldb_private::process_linux;

std::unique_ptr<NativeRegisterContextLinux>
NativeRegisterContextLinux::CreateHostNativeRegisterContextLinux(
    const ArchSpec &target_arch, NativeThreadLinux &native_thread) {
  switch (target_arch.GetMachine()) {
  case llvm::Triple::loongarch64: {
    Flags opt_regsets;
    auto register_info_up = std::make_unique<RegisterInfoPOSIX_loongarch64>(
        target_arch, opt_regsets);
    return std::make_unique<NativeRegisterContextLinux_loongarch64>(
        target_arch, native_thread, std::move(register_info_up));
  }
  default:
    llvm_unreachable("have no register context for architecture");
  }
}

llvm::Expected<ArchSpec>
NativeRegisterContextLinux::DetermineArchitecture(lldb::tid_t tid) {
  return HostInfo::GetArchitecture();
}

NativeRegisterContextLinux_loongarch64::NativeRegisterContextLinux_loongarch64(
    const ArchSpec &target_arch, NativeThreadProtocol &native_thread,
    std::unique_ptr<RegisterInfoPOSIX_loongarch64> register_info_up)
    : NativeRegisterContextRegisterInfo(native_thread,
                                        register_info_up.release()),
      NativeRegisterContextLinux(native_thread) {
  ::memset(&m_fpr, 0, sizeof(m_fpr));
  ::memset(&m_gpr, 0, sizeof(m_gpr));
}

const RegisterInfoPOSIX_loongarch64 &
NativeRegisterContextLinux_loongarch64::GetRegisterInfo() const {
  return static_cast<const RegisterInfoPOSIX_loongarch64 &>(
      NativeRegisterContextRegisterInfo::GetRegisterInfoInterface());
}

uint32_t NativeRegisterContextLinux_loongarch64::GetRegisterSetCount() const {
  return GetRegisterInfo().GetRegisterSetCount();
}

const RegisterSet *NativeRegisterContextLinux_loongarch64::GetRegisterSet(
    uint32_t set_index) const {
  return GetRegisterInfo().GetRegisterSet(set_index);
}

Status NativeRegisterContextLinux_loongarch64::ReadRegister(
    const RegisterInfo *reg_info, RegisterValue &reg_value) {
  return Status("Failed to read register value");
}

Status NativeRegisterContextLinux_loongarch64::WriteRegister(
    const RegisterInfo *reg_info, const RegisterValue &reg_value) {
  return Status("Failed to write register value");
}

Status NativeRegisterContextLinux_loongarch64::ReadAllRegisterValues(
    lldb::WritableDataBufferSP &data_sp) {
  return Status("Failed to read all register values");
}

Status NativeRegisterContextLinux_loongarch64::WriteAllRegisterValues(
    const lldb::DataBufferSP &data_sp) {
  return Status("Failed to write all register values");
}

#endif // defined(__loongarch__) && __loongarch_grlen == 64
