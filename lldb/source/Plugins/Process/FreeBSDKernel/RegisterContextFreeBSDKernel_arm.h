//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains the declaration of the RegisterContextFreeBSDKernel_arm
/// class, which is used for reading registers from PCB on arm kernel dump.
///
//===----------------------------------------------------------------------===//

#ifndef LLDB_SOURCE_PLUGINS_PROCESS_FREEBSDKERNEL_REGISTERCONTEXTFREEBSDKERNEL_ARM_H
#define LLDB_SOURCE_PLUGINS_PROCESS_FREEBSDKERNEL_REGISTERCONTEXTFREEBSDKERNEL_ARM_H

#include "Plugins/Process/Utility/RegisterContextPOSIX_arm.h"
#include "Plugins/Process/elf-core/RegisterUtilities.h"

class RegisterContextFreeBSDKernel_arm : public RegisterContextPOSIX_arm {
public:
  RegisterContextFreeBSDKernel_arm(
      lldb_private::Thread &thread,
      std::unique_ptr<RegisterInfoPOSIX_arm> register_info_up,
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

#endif // LLDB_SOURCE_PLUGINS_PROCESS_FREEBSDKERNEL_REGISTERCONTEXTFREEBSDKERNEL_ARM_H
