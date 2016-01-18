//===-- RegisterContextWindowsMiniDump.h --------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_RegisterContextWindowsMiniDump_H_
#define liblldb_RegisterContextWindowsMiniDump_H_

#include "lldb/lldb-forward.h"
#include "lldb/Target/RegisterContext.h"

#include "Plugins/Process/Windows/Common/RegisterContextWindows.h"


namespace lldb_private
{

class Thread;

class RegisterContextWindowsMiniDump : public lldb_private::RegisterContextWindows
{
  public:
    RegisterContextWindowsMiniDump(Thread &thread, uint32_t concrete_frame_idx);

    virtual ~RegisterContextWindowsMiniDump();

    void
    InvalidateAllRegisters() override;

    size_t
    GetRegisterCount() override;

    const RegisterInfo *
    GetRegisterInfoAtIndex(size_t reg) override;

    size_t
    GetRegisterSetCount() override;

    const RegisterSet *
    GetRegisterSet(size_t reg_set) override;

    bool
    ReadRegister(const RegisterInfo *reg_info, RegisterValue &reg_value) override;

    bool
    WriteRegister(const RegisterInfo *reg_info, const RegisterValue &reg_value) override;

    bool ReadAllRegisterValues(lldb::DataBufferSP &data_sp) override;

    bool WriteAllRegisterValues(const lldb::DataBufferSP &data_sp) override;

    uint32_t ConvertRegisterKindToRegisterNumber(lldb::RegisterKind kind, uint32_t num) override;

    uint32_t NumSupportedHardwareBreakpoints() override;

    uint32_t SetHardwareBreakpoint(lldb::addr_t addr, size_t size) override;

    bool ClearHardwareBreakpoint(uint32_t hw_idx) override;

    uint32_t NumSupportedHardwareWatchpoints() override;

    uint32_t SetHardwareWatchpoint(lldb::addr_t addr, size_t size, bool read, bool write) override;

    bool ClearHardwareWatchpoint(uint32_t hw_index) override;

    bool HardwareSingleStep(bool enable) override;
};

}

#endif // #ifndef liblldb_RegisterContextWindowsMiniDump_H_
