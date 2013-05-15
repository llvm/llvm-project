//===-- RegisterContextMacOSXFrameBackchain.h -------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef lldb_RegisterContextMacOSXFrameBackchain_h_
#define lldb_RegisterContextMacOSXFrameBackchain_h_

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes
#include "lldb/lldb-private.h"
#include "lldb/Target/RegisterContext.h"

#include "UnwindMacOSXFrameBackchain.h"

class RegisterContextMacOSXFrameBackchain : public lldb_private::RegisterContext
{
public:
    //------------------------------------------------------------------
    // Constructors and Destructors
    //------------------------------------------------------------------
    RegisterContextMacOSXFrameBackchain (lldb_private::Thread &thread,
                                         uint32_t concrete_frame_idx,
                                         const UnwindMacOSXFrameBackchain::Cursor &cursor);

    virtual
    ~RegisterContextMacOSXFrameBackchain ();

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

    virtual bool
    ReadRegister (const lldb_private::RegisterInfo *reg_info, lldb_private::RegisterValue &value);

    virtual bool
    WriteRegister (const lldb_private::RegisterInfo *reg_info, const lldb_private::RegisterValue &value);
    
    virtual bool
    ReadAllRegisterValues (lldb::DataBufferSP &data_sp);

    virtual bool
    WriteAllRegisterValues (const lldb::DataBufferSP &data_sp);

    virtual uint32_t
    ConvertRegisterKindToRegisterNumber (uint32_t kind, uint32_t num);
    
private:
    UnwindMacOSXFrameBackchain::Cursor m_cursor;
    bool m_cursor_is_valid;
    //------------------------------------------------------------------
    // For RegisterContextMacOSXFrameBackchain only
    //------------------------------------------------------------------
    DISALLOW_COPY_AND_ASSIGN (RegisterContextMacOSXFrameBackchain);
};

#endif  // lldb_RegisterContextMacOSXFrameBackchain_h_
