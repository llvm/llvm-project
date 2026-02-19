//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains the declaration of the
/// RegisterContextFreeBSDKernel_riscv64 class, which is used for reading
/// registers from PCB in riscv64 kernel dump.
///
//===----------------------------------------------------------------------===//

#ifndef LLDB_SOURCE_PLUGINS_PROCESS_FREEBSDKERNEL_REGISTERCONTEXTFREEBSDKERNEL_RISCV64_H
#define LLDB_SOURCE_PLUGINS_PROCESS_FREEBSDKERNEL_REGISTERCONTEXTFREEBSDKERNEL_RISCV64_H

#include "Plugins/Process/Utility/RegisterContextPOSIX_riscv64.h"
#include "Plugins/Process/elf-core/RegisterUtilities.h"

class RegisterContextFreeBSDKernel_riscv64
    : public RegisterContextPOSIX_riscv64 {
public:
  RegisterContextFreeBSDKernel_riscv64(
      lldb_private::Thread &thread,
      std::unique_ptr<RegisterInfoPOSIX_riscv64> register_info_up,
      lldb::addr_t pcb_addr);

  bool ReadRegister(const lldb_private::RegisterInfo *reg_info,
                    lldb_private::RegisterValue &value) override;

  bool WriteRegister(const lldb_private::RegisterInfo *reg_info,
                     const lldb_private::RegisterValue &value) override;

protected:
  bool ReadGPR() override;

  bool ReadFPR() override;

  bool WriteGPR() override;

  bool WriteFPR() override;

private:
  lldb::addr_t m_pcb_addr;
};

#endif // LLDB_SOURCE_PLUGINS_PROCESS_FREEBSDKERNEL_REGISTERCONTEXTFREEBSDKERNEL_RISCV64_H
