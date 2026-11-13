//===-- NativeRegisterContextLinux_riscv64.h -------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#if defined(__riscv) && __riscv_xlen == 64

#ifndef lldb_NativeRegisterContextLinux_riscv64_h
#define lldb_NativeRegisterContextLinux_riscv64_h

#include "Plugins/Process/Linux/NativeRegisterContextLinux.h"
#include "lldb/Register/Common/RegisterInfo_riscv64.h"
#include "lldb/Register/Layouts/Linux/RegisterLayoutLinux_riscv64.h"

#include <asm/ptrace.h>

namespace lldb_private {
namespace process_linux {

class NativeProcessLinux;

class NativeRegisterContextLinux_riscv64 : public NativeRegisterContextLinux {
public:
  NativeRegisterContextLinux_riscv64(
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

  void *GetGPRBuffer() override { return &m_gpr; }

  void *GetFPRBuffer() override { return &m_fpr; }

  size_t GetGPRSize() const override { return GetBankSize(0); }

  size_t GetFPRSize() override { return GetBankSize(1); }

private:
  bool m_gpr_is_valid;
  bool m_fpu_is_valid;

  linux_riscv64::GPR m_gpr;

  linux_riscv64::FPR m_fpr;

  size_t GetRegContextSize();

  bool IsGPR(unsigned reg) const;

  bool IsFPR(unsigned reg) const;
};

} // namespace process_linux
} // namespace lldb_private

#endif // #ifndef lldb_NativeRegisterContextLinux_riscv64_h

#endif // defined(__riscv) && __riscv_xlen == 64
