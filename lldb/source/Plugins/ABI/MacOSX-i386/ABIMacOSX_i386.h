//===-- ABIMacOSX_i386.h ----------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_ABIMacOSX_i386_h_
#define liblldb_ABIMacOSX_i386_h_

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes
#include "lldb/lldb-private.h"
#include "lldb/Target/ABI.h"
#include "lldb/Core/Value.h"
    
class ABIMacOSX_i386 :
    public lldb_private::ABI
{
public:

    ~ABIMacOSX_i386() { }
    
    virtual size_t 
    GetRedZoneSize () const;
    
    virtual bool
    PrepareTrivialCall (lldb_private::Thread &thread, 
                        lldb::addr_t sp,
                        lldb::addr_t func_addr,
                        lldb::addr_t return_addr, 
                        lldb::addr_t *arg1_ptr = NULL,
                        lldb::addr_t *arg2_ptr = NULL,
                        lldb::addr_t *arg3_ptr = NULL,
                        lldb::addr_t *arg4_ptr = NULL,
                        lldb::addr_t *arg5_ptr = NULL,
                        lldb::addr_t *arg6_ptr = NULL) const;
    
    virtual bool
    PrepareNormalCall (lldb_private::Thread &thread,
                       lldb::addr_t sp,
                       lldb::addr_t func_addr,
                       lldb::addr_t return_addr,
                       lldb_private::ValueList &args) const;
    
    virtual bool
    GetArgumentValues (lldb_private::Thread &thread,
                       lldb_private::ValueList &values) const;
    
    virtual lldb_private::Error
    SetReturnValueObject(lldb::StackFrameSP &frame_sp, lldb::ValueObjectSP &new_value);

protected:
    virtual lldb::ValueObjectSP
    GetReturnValueObjectImpl (lldb_private::Thread &thread,
                    lldb_private::ClangASTType &ast_type) const;

public:

    virtual bool
    CreateFunctionEntryUnwindPlan (lldb_private::UnwindPlan &unwind_plan);
    
    virtual bool
    CreateDefaultUnwindPlan (lldb_private::UnwindPlan &unwind_plan);
    
    virtual bool
    RegisterIsVolatile (const lldb_private::RegisterInfo *reg_info);
    
    virtual bool
    StackUsesFrames ()
    {
        return true;
    }
    
    virtual bool
    CallFrameAddressIsValid (lldb::addr_t cfa)
    {
        // Make sure the stack call frame addresses are are 8 byte aligned
        if (cfa & (8ull - 1ull))
            return false;   // Not 8 byte aligned
        if (cfa == 0)
            return false;   // Zero is not a valid stack address
        return true;
    }

    virtual bool
    CodeAddressIsValid (lldb::addr_t pc)
    {
        // Just make sure the address is a valid 32 bit address. 
        return pc <= UINT32_MAX;
    }

    virtual bool
    FunctionCallsChangeCFA ()
    {
        return true;
    }

    virtual const lldb_private::RegisterInfo *
    GetRegisterInfoArray (uint32_t &count);

    //------------------------------------------------------------------
    // Static Functions
    //------------------------------------------------------------------
    static void
    Initialize();
    
    static void
    Terminate();
    
    static lldb::ABISP
    CreateInstance (const lldb_private::ArchSpec &arch);
    
    //------------------------------------------------------------------
    // PluginInterface protocol
    //------------------------------------------------------------------
    virtual const char *
    GetPluginName();
    
    virtual const char *
    GetShortPluginName();
    
    virtual uint32_t
    GetPluginVersion();
    
protected:
    bool
    RegisterIsCalleeSaved (const lldb_private::RegisterInfo *reg_info);

private:
    ABIMacOSX_i386() : lldb_private::ABI() { } // Call CreateInstance instead.
};


#endif  // liblldb_ABI_h_
