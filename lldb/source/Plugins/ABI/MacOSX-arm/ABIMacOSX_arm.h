//===-- ABIMacOSX_arm.h ----------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_ABIMacOSX_arm_h_
#define liblldb_ABIMacOSX_arm_h_

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes
#include "lldb/lldb-private.h"
#include "lldb/Target/ABI.h"

class ABIMacOSX_arm : public lldb_private::ABI
{
public:
    ~ABIMacOSX_arm() { }
    
    virtual size_t 
    GetRedZoneSize () const;
    
    virtual bool
    PrepareTrivialCall (lldb_private::Thread &thread, 
                        lldb::addr_t sp,
                        lldb::addr_t func_addr,
                        lldb::addr_t returnAddress, 
                        lldb::addr_t *arg1_ptr = NULL,
                        lldb::addr_t *arg2_ptr = NULL,
                        lldb::addr_t *arg3_ptr = NULL,
                        lldb::addr_t *arg4_ptr = NULL,
                        lldb::addr_t *arg5_ptr = NULL,
                        lldb::addr_t *arg6_ptr = NULL) const;
    
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
        // Make sure the stack call frame addresses are are 4 byte aligned
        if (cfa & (4ull - 1ull))
            return false;   // Not 4 byte aligned
        if (cfa == 0)
            return false;   // Zero is not a valid stack address
        return true;
    }
    
    virtual bool
    CodeAddressIsValid (lldb::addr_t pc)
    {
        // Just make sure the address is a valid 32 bit address. Bit zero
        // might be set due to Thumb function calls, so don't enforce 2 byte
        // alignment
        return pc <= UINT32_MAX;
    }
    
    virtual lldb::addr_t
    FixCodeAddress (lldb::addr_t pc)
    {
        // ARM uses bit zero to signify a code address is thumb, so we must
        // strip bit zero in any code addresses.
        return pc & ~(lldb::addr_t)1;
    }

    virtual bool
    FunctionCallsChangeCFA ()
    {
        return false;
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
    
    static lldb_private::ConstString
    GetPluginNameStatic();
    
    //------------------------------------------------------------------
    // PluginInterface protocol
    //------------------------------------------------------------------
    virtual lldb_private::ConstString
    GetPluginName();
    
    virtual uint32_t
    GetPluginVersion();
    
protected:
private:
    ABIMacOSX_arm() : 
        lldb_private::ABI() 
    {
         // Call CreateInstance instead.
    }
};

#endif  // liblldb_ABIMacOSX_arm_h_
