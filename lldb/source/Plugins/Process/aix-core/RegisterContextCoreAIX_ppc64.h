//===-- RegisterContextCoreAIX_ppc64.h ----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SOURCE_PLUGINS_PROCESS_AIX_CORE_REGISTERCONTEXTAIXCORE_PPC64_H
#define LLDB_SOURCE_PLUGINS_PROCESS_AIX_CORE_REGISTERCONTEXTAIXCORE_PPC64_H

#include "Plugins/Process/Utility/RegisterContextPOSIX_ppc64le.h"
#include "lldb/Utility/DataExtractor.h"

class RegisterContextCoreAIX_ppc64 : public RegisterContextPOSIX_ppc64le {
public:
  RegisterContextCoreAIX_ppc64(
      lldb_private::Thread &thread,
      lldb_private::RegisterInfoInterface *register_info,
      const lldb_private::DataExtractor &gpregset);

  bool ReadRegister(const lldb_private::RegisterInfo *reg_info,
                    lldb_private::RegisterValue &value) override;

  bool WriteRegister(const lldb_private::RegisterInfo *reg_info,
                     const lldb_private::RegisterValue &value) override;

protected:
  size_t GetFPRSize() const;

  size_t GetVMXSize() const;

  size_t GetVSXSize() const;

private:
  lldb::DataBufferSP m_gpr_buffer;
  lldb::DataBufferSP m_fpr_buffer;
  lldb::DataBufferSP m_vmx_buffer;
  lldb::DataBufferSP m_vsx_buffer;
  lldb_private::DataExtractor m_gpr;
  lldb_private::DataExtractor m_fpr;
  lldb_private::DataExtractor m_vmx;
  lldb_private::DataExtractor m_vsx;
};

#endif // LLDB_SOURCE_PLUGINS_PROCESS_AIX_CORE_REGISTERCONTEXTAIXCORE_PPC64_H
