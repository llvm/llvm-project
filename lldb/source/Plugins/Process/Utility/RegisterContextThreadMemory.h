//===-- RegisterContextThreadMemory.h ---------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef lldb_RegisterContextThreadMemory_h_
#define lldb_RegisterContextThreadMemory_h_

#include <vector>

#include "lldb/lldb-private.h"
#include "lldb/Target/RegisterContext.h"
#include "lldb/Symbol/SymbolContext.h"

namespace lldb_private {
    
class RegisterContextThreadMemory : public lldb_private::RegisterContext
{
public:
    RegisterContextThreadMemory (Thread &thread,
                                 lldb::addr_t register_data_addr);
    
    virtual ~RegisterContextThreadMemory();
    //------------------------------------------------------------------
    // Subclasses must override these functions
    //------------------------------------------------------------------
    virtual void
    InvalidateAllRegisters ();
    
    virtual size_t
    GetRegisterCount ();
    
    virtual const RegisterInfo *
    GetRegisterInfoAtIndex (size_t reg);
    
    virtual size_t
    GetRegisterSetCount ();
    
    virtual const RegisterSet *
    GetRegisterSet (size_t reg_set);
    
    virtual bool
    ReadRegister (const RegisterInfo *reg_info, RegisterValue &reg_value);
    
    virtual bool
    WriteRegister (const RegisterInfo *reg_info, const RegisterValue &reg_value);
    
    // These two functions are used to implement "push" and "pop" of register states.  They are used primarily
    // for expression evaluation, where we need to push a new state (storing the old one in data_sp) and then
    // restoring the original state by passing the data_sp we got from ReadAllRegisters to WriteAllRegisterValues.
    // ReadAllRegisters will do what is necessary to return a coherent set of register values for this thread, which
    // may mean e.g. interrupting a thread that is sitting in a kernel trap.  That is a somewhat disruptive operation,
    // so these API's should only be used when this behavior is needed.
    
    virtual bool
    ReadAllRegisterValues (lldb::DataBufferSP &data_sp);
    
    virtual bool
    WriteAllRegisterValues (const lldb::DataBufferSP &data_sp);
    
    bool
    CopyFromRegisterContext (lldb::RegisterContextSP context);
    
    virtual uint32_t
    ConvertRegisterKindToRegisterNumber (uint32_t kind, uint32_t num);
    
    //------------------------------------------------------------------
    // Subclasses can override these functions if desired
    //------------------------------------------------------------------
    virtual uint32_t
    NumSupportedHardwareBreakpoints ();
    
    virtual uint32_t
    SetHardwareBreakpoint (lldb::addr_t addr, size_t size);
    
    virtual bool
    ClearHardwareBreakpoint (uint32_t hw_idx);
    
    virtual uint32_t
    NumSupportedHardwareWatchpoints ();
    
    virtual uint32_t
    SetHardwareWatchpoint (lldb::addr_t addr, size_t size, bool read, bool write);
    
    virtual bool
    ClearHardwareWatchpoint (uint32_t hw_index);
    
    virtual bool
    HardwareSingleStep (bool enable);
    
    virtual Error
    ReadRegisterValueFromMemory (const lldb_private::RegisterInfo *reg_info, lldb::addr_t src_addr, uint32_t src_len, RegisterValue &reg_value);
    
    virtual Error
    WriteRegisterValueToMemory (const lldb_private::RegisterInfo *reg_info, lldb::addr_t dst_addr, uint32_t dst_len, const RegisterValue &reg_value);
    
protected:
    void
    UpdateRegisterContext ();
    
    lldb::ThreadWP m_thread_wp;
    lldb::RegisterContextSP m_reg_ctx_sp;
    lldb::addr_t m_register_data_addr;
    uint32_t m_stop_id;
private:
    DISALLOW_COPY_AND_ASSIGN (RegisterContextThreadMemory);
};
} // namespace lldb_private

#endif  // lldb_RegisterContextThreadMemory_h_
