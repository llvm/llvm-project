//===-- NativeRegisterContextLinux_loongarch64.h ----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#if defined(__loongarch__) && __loongarch_grlen == 64

#ifndef lldb_NativeRegisterContextLinux_loongarch64_h
#define lldb_NativeRegisterContextLinux_loongarch64_h

#include "Plugins/Process/Linux/NativeRegisterContextLinux.h"
#include "Plugins/Register/Native/NativeRegisterContextDBReg_loongarch.h"
#include "lldb/Register/Common/RegisterInfo_loongarch64.h"
#include "lldb/Register/Layouts/Linux/RegisterLayoutLinux_loongarch64.h"

#include <asm/ptrace.h>

namespace lldb_private {
namespace process_linux {

class NativeProcessLinux;

class NativeRegisterContextLinux_loongarch64
    : public NativeRegisterContextLinux,
      public NativeRegisterContextDBReg_loongarch {
public:
  NativeRegisterContextLinux_loongarch64(
      const ArchSpec &target_arch, NativeThreadProtocol &native_thread,
      lldb_private::RegisterLayout register_model);

  llvm::Expected<lldb_private::RegisterValue>
  ReadRegister(lldb_private::RegisterID id) override;

  llvm::Error
  WriteRegister(lldb_private::RegisterID id,
                const lldb_private::RegisterValue &reg_value) override;

  Status ReadAllRegisterValues(lldb::WritableDataBufferSP &data_sp) override;

  Status WriteAllRegisterValues(const lldb::DataBufferSP &data_sp) override;

  void InvalidateAllRegisters() override;

  bool RegisterOffsetIsDynamic() const override { return true; }

protected:
  Status ReadGPR() override;

  Status WriteGPR() override;

  Status ReadFPR() override;

  Status WriteFPR() override;

  Status ReadLSX();

  Status WriteLSX();

  Status ReadLASX();

  Status WriteLASX();

  void *GetGPRBuffer() override { return &m_gpr; }

  void *GetFPRBuffer() override { return &m_fpr; }

  size_t GetGPRSize() const override { return GetBankSize(0); }

  size_t GetFPRSize() override { return GetBankSize(1); }

private:
  bool m_gpr_is_valid;
  bool m_fpu_is_valid;
  bool m_lsx_is_valid;
  bool m_lasx_is_valid;
  bool m_refresh_hwdebug_info;

  linux_loongarch64::GPR m_gpr;
  linux_loongarch64::FPR m_fpr;
  linux_loongarch64::LSX m_lsx;
  linux_loongarch64::LASX m_lasx;

  bool IsGPR(unsigned reg) const;

  bool IsFPR(unsigned reg) const;

  bool IsLSX(unsigned reg) const;

  bool IsLASX(unsigned reg) const;

  llvm::Error ReadHardwareDebugInfo() override;

  llvm::Error WriteHardwareDebugRegs(DREGType hwbType) override;
};

} // namespace process_linux
} // namespace lldb_private

#endif // #ifndef lldb_NativeRegisterContextLinux_loongarch64_h

#endif // defined(__loongarch__) && __loongarch_grlen == 64
