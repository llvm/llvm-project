//===-- RegisterContextPOSIX_ppc64.h ----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SOURCE_PLUGINS_PROCESS_UTILITY_REGISTERCONTEXTPOSIX_PPC64_H
#define LLDB_SOURCE_PLUGINS_PROCESS_UTILITY_REGISTERCONTEXTPOSIX_PPC64_H

#include "Plugins/Process/Utility/lldb-ppc64-register-enums.h"
#include "RegisterInfoInterface.h"
#include "Utility/PPC64_DWARF_Registers.h"
#include "lldb/Target/RegisterContext.h"
#include "lldb/Utility/Log.h"

class RegisterContextPOSIX_ppc64 : public lldb_private::RegisterContext {
public:
  RegisterContextPOSIX_ppc64(
      lldb_private::Thread &thread, uint32_t concrete_frame_idx,
      lldb_private::RegisterInfoInterface *register_info);

  void InvalidateAllRegisters() override;

  size_t GetRegisterCount() override;

  virtual size_t GetGPRSize();

  virtual unsigned GetRegisterSize(unsigned reg);

  virtual unsigned GetRegisterOffset(unsigned reg);

  const lldb_private::RegisterInfo *GetRegisterInfoAtIndex(size_t reg) override;

  size_t GetRegisterSetCount() override;

  const lldb_private::RegisterSet *GetRegisterSet(size_t set) override;

  const char *GetRegisterName(unsigned reg);

protected:
  // 64-bit general purpose registers.
  uint64_t m_gpr_ppc64[k_num_gpr_registers_ppc64];

  // floating-point registers including extended register.
  uint64_t m_fpr_ppc64[k_num_fpr_registers_ppc64];

  // VMX registers.
  uint64_t m_vmx_ppc64[k_num_vmx_registers_ppc64 * 2];

  // VSX registers.
  uint64_t m_vsx_ppc64[k_num_vsx_registers_ppc64 * 2];

  std::unique_ptr<lldb_private::RegisterInfoInterface> m_register_info_up;

  // Determines if an extended register set is supported on the processor
  // running the inferior process.
  virtual bool IsRegisterSetAvailable(size_t set_index);

  virtual const lldb_private::RegisterInfo *GetRegisterInfo();

  bool IsGPR(unsigned reg);

  bool IsFPR(unsigned reg);

  bool IsVMX(unsigned reg);

  bool IsVSX(unsigned reg);
};
#endif // LLDB_SOURCE_PLUGINS_PROCESS_UTILITY_REGISTERCONTEXTPOSIX_PPC64_H
