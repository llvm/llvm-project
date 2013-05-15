//===-- RegisterContextMemory.h ---------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef lldb_RegisterContextMemory_h_
#define lldb_RegisterContextMemory_h_

// C Includes
// C++ Includes
#include <vector>

// Other libraries and framework includes
// Project includes
#include "lldb/lldb-private.h"
#include "lldb/Core/DataExtractor.h"
#include "lldb/Target/RegisterContext.h"

class DynamicRegisterInfo;

class RegisterContextMemory : public lldb_private::RegisterContext
{
public:
    //------------------------------------------------------------------
    // Constructors and Destructors
    //------------------------------------------------------------------
    RegisterContextMemory (lldb_private::Thread &thread,
                            uint32_t concrete_frame_idx,
                            DynamicRegisterInfo &reg_info,
                            lldb::addr_t reg_data_addr);

    virtual
    ~RegisterContextMemory ();

    //------------------------------------------------------------------
    // Subclasses must override these functions
    //------------------------------------------------------------------
    virtual void
    InvalidateAllRegisters ();

    virtual size_t
    GetRegisterCount ();

    virtual const lldb_private::RegisterInfo *
    GetRegisterInfoAtIndex (size_t reg);

    virtual size_t
    GetRegisterSetCount ();

    virtual const lldb_private::RegisterSet *
    GetRegisterSet (size_t reg_set);

    virtual uint32_t
    ConvertRegisterKindToRegisterNumber (uint32_t kind, uint32_t num);

    
    //------------------------------------------------------------------
    // If all of the thread register are in a contiguous buffer in 
    // memory, then the default ReadRegister/WriteRegister and
    // ReadAllRegisterValues/WriteAllRegisterValues will work. If thread
    // registers are not contiguous, clients will want to subclass this
    // class and modify the read/write functions as needed.
    //------------------------------------------------------------------

    virtual bool
    ReadRegister (const lldb_private::RegisterInfo *reg_info, 
                  lldb_private::RegisterValue &reg_value);
    
    virtual bool
    WriteRegister (const lldb_private::RegisterInfo *reg_info, 
                   const lldb_private::RegisterValue &reg_value);
    
    virtual bool
    ReadAllRegisterValues (lldb::DataBufferSP &data_sp);
    
    virtual bool
    WriteAllRegisterValues (const lldb::DataBufferSP &data_sp);

    void
    SetAllRegisterData  (const lldb::DataBufferSP &data_sp);
protected:
    
    void
    SetAllRegisterValid (bool b);

    DynamicRegisterInfo &m_reg_infos;
    std::vector<bool> m_reg_valid;
    lldb_private::DataExtractor m_reg_data;
    lldb::addr_t m_reg_data_addr; // If this is valid, then we have a register context that is stored in memmory

private:
    //------------------------------------------------------------------
    // For RegisterContextMemory only
    //------------------------------------------------------------------
    DISALLOW_COPY_AND_ASSIGN (RegisterContextMemory);
};

#endif  // lldb_RegisterContextMemory_h_
