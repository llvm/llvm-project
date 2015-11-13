//===-- RegisterContextWindows.h --------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_RegisterContextWindows_H_
#define liblldb_RegisterContextWindows_H_

#include "lldb/lldb-forward.h"
#include "lldb/Target/RegisterContext.h"

namespace lldb_private
{

class Thread;

class RegisterContextWindows : public lldb_private::RegisterContext
{
  public:
    //------------------------------------------------------------------
    // Constructors and Destructors
    //------------------------------------------------------------------
    RegisterContextWindows(Thread &thread, uint32_t concrete_frame_idx);

    virtual ~RegisterContextWindows();

    //------------------------------------------------------------------
    // Subclasses must override these functions
    //------------------------------------------------------------------
    void InvalidateAllRegisters() override;

    bool ReadAllRegisterValues(lldb::DataBufferSP &data_sp) override;

    bool WriteAllRegisterValues(const lldb::DataBufferSP &data_sp) override;

    uint32_t ConvertRegisterKindToRegisterNumber(lldb::RegisterKind kind, uint32_t num) override;

    //------------------------------------------------------------------
    // Subclasses can override these functions if desired
    //------------------------------------------------------------------
    uint32_t NumSupportedHardwareBreakpoints() override;

    uint32_t SetHardwareBreakpoint(lldb::addr_t addr, size_t size) override;

    bool ClearHardwareBreakpoint(uint32_t hw_idx) override;

    uint32_t NumSupportedHardwareWatchpoints() override;

    uint32_t SetHardwareWatchpoint(lldb::addr_t addr, size_t size, bool read, bool write) override;

    bool ClearHardwareWatchpoint(uint32_t hw_index) override;

    bool HardwareSingleStep(bool enable) override;

  protected:
    virtual bool CacheAllRegisterValues();

    CONTEXT m_context;
    bool m_context_stale;
};
}

#endif // #ifndef liblldb_RegisterContextWindows_H_
