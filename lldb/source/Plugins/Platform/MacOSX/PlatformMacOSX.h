//===-- PlatformMacOSX.h ----------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_PlatformMacOSX_h_
#define liblldb_PlatformMacOSX_h_

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes
#include "PlatformDarwin.h"

class PlatformMacOSX : public PlatformDarwin
{
public:

    //------------------------------------------------------------
    // Class functions
    //------------------------------------------------------------
    static lldb_private::Platform* 
    CreateInstance (bool force, const lldb_private::ArchSpec *arch);

    static void
    Initialize ();

    static void
    Terminate ();
    
    static lldb_private::ConstString
    GetPluginNameStatic (bool is_host);

    static const char *
    GetDescriptionStatic(bool is_host);
    
    //------------------------------------------------------------
    // Class Methods
    //------------------------------------------------------------
    PlatformMacOSX (bool is_host);

    virtual
    ~PlatformMacOSX();

    //------------------------------------------------------------
    // lldb_private::PluginInterface functions
    //------------------------------------------------------------
    virtual lldb_private::ConstString
    GetPluginName()
    {
        return GetPluginNameStatic (IsHost());
    }
    
    virtual uint32_t
    GetPluginVersion()
    {
        return 1;
    }
    
    virtual const char *
    GetDescription ()
    {
        return GetDescriptionStatic (IsHost());
    }

    virtual lldb_private::Error
    GetFile (const lldb_private::FileSpec &platform_file, 
             const lldb_private::UUID *uuid_ptr,
             lldb_private::FileSpec &local_file);

    virtual bool
    GetSupportedArchitectureAtIndex (uint32_t idx, 
                                     lldb_private::ArchSpec &arch);

private:
    DISALLOW_COPY_AND_ASSIGN (PlatformMacOSX);

};

#endif  // liblldb_PlatformMacOSX_h_
