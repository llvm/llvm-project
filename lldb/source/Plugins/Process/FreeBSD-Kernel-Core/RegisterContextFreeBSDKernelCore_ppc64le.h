//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SOURCE_PLUGINS_PROCESS_FREEBSDKERNEL_REGISTERCONTEXTFREEBSDKERNELCORE_PPC64LE_H
#define LLDB_SOURCE_PLUGINS_PROCESS_FREEBSDKERNEL_REGISTERCONTEXTFREEBSDKERNELCORE_PPC64LE_H

#include "lldb/Register/Arch/lldb-ppc64le-dwarf-register-enums.h"
#include "lldb/Register/Arch/lldb-ppc64le-register-enums.h"
#include "lldb/Target/ThreadRegisterContext.h"

class RegisterContextFreeBSDKernelCore_ppc64le
    : public lldb_private::ThreadRegisterContext {
public:
  RegisterContextFreeBSDKernelCore_ppc64le(lldb_private::Thread &thread,
                                           lldb_private::RegisterLayout layout,
                                           lldb::addr_t pcb_addr);

  llvm::Expected<lldb_private::RegisterValue>
  ReadRegister(lldb_private::RegisterID id) override;

  llvm::Error WriteRegister(lldb_private::RegisterID id,
                     const lldb_private::RegisterValue &value) override;

private:
  lldb::addr_t m_pcb_addr;
};

#endif // LLDB_SOURCE_PLUGINS_PROCESS_FREEBSDKERNEL_REGISTERCONTEXTFREEBSDKERNELCORE_PPC64LE_H
