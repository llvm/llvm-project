//===-- RegisterContextCorePOSIX_arm.h --------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_RegisterContextCorePOSIX_arm_h_
#define liblldb_RegisterContextCorePOSIX_arm_h_

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes
#include "Plugins/Process/Utility/RegisterContextPOSIX_arm.h"
#include "lldb/Core/DataBufferHeap.h"
#include "lldb/Core/DataExtractor.h"

class RegisterContextCorePOSIX_arm :
    public RegisterContextPOSIX_arm
{
public:
    RegisterContextCorePOSIX_arm (lldb_private::Thread &thread,
                                     lldb_private::RegisterInfoInterface *register_info,
                                     const lldb_private::DataExtractor &gpregset,
                                     const lldb_private::DataExtractor &fpregset);

    ~RegisterContextCorePOSIX_arm() override;

    bool
    ReadRegister(const lldb_private::RegisterInfo *reg_info,
                 lldb_private::RegisterValue &value) override;

    bool
    WriteRegister(const lldb_private::RegisterInfo *reg_info,
                  const lldb_private::RegisterValue &value) override;

    bool
    ReadAllRegisterValues(lldb::DataBufferSP &data_sp) override;

    bool
    WriteAllRegisterValues(const lldb::DataBufferSP &data_sp) override;

    bool
    HardwareSingleStep(bool enable) override;

protected:
    bool
    ReadGPR() override;

    bool
    ReadFPR() override;

    bool
    WriteGPR() override;

    bool
    WriteFPR() override;

private:
    lldb::DataBufferSP m_gpr_buffer;
    lldb_private::DataExtractor m_gpr;
};

#endif // liblldb_RegisterContextCorePOSIX_arm_h_
