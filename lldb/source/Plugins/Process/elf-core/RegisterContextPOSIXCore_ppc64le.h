//===-- RegisterContextPOSIXCore_ppc64le.h ----------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_RegisterContextCorePOSIX_ppc64le_h_
#define liblldb_RegisterContextCorePOSIX_ppc64le_h_

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes
#include "Plugins/Process/Utility/RegisterContextPOSIX_ppc64le.h"
#include "lldb/Utility/DataExtractor.h"
#include "llvm/ADT/DenseMap.h"

class RegisterContextCorePOSIX_ppc64le : public RegisterContextPOSIX_ppc64le {
public:
  RegisterContextCorePOSIX_ppc64le(
      lldb_private::Thread &thread,
      lldb_private::RegisterInfoInterface *register_info,
      const llvm::DenseMap<uint32_t, lldb_private::DataExtractor> &regsets);

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

#endif // liblldb_RegisterContextCorePOSIX_ppc64le_h_
