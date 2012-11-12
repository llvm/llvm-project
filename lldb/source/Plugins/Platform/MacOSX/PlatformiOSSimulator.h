//===-- PlatformiOSSimulator.h ----------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_PlatformiOSSimulator_h_
#define liblldb_PlatformiOSSimulator_h_

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes
#include "PlatformDarwin.h"

class PlatformiOSSimulator : public PlatformDarwin
{
public:

    //------------------------------------------------------------
    // Class Functions
    //------------------------------------------------------------
    static lldb_private::Platform* 
    CreateInstance (bool force, const lldb_private::ArchSpec *arch);

    static void
    Initialize ();

    static void
    Terminate ();
    
    static const char *
    GetPluginNameStatic ();

    static const char *
    GetShortPluginNameStatic();

    static const char *
    GetDescriptionStatic();
    
    //------------------------------------------------------------
    // Class Methods
    //------------------------------------------------------------
    PlatformiOSSimulator ();

    virtual
    ~PlatformiOSSimulator();

    //------------------------------------------------------------
    // lldb_private::PluginInterface functions
    //------------------------------------------------------------
    virtual const char *
    GetPluginName()
    {
        return GetPluginNameStatic();
    }
    
    virtual const char *
    GetShortPluginName()
    {
        return GetShortPluginNameStatic();
    }
    
    virtual uint32_t
    GetPluginVersion()
    {
        return 1;
    }

    //------------------------------------------------------------
    // lldb_private::Platform functions
    //------------------------------------------------------------
    virtual lldb_private::Error
    ResolveExecutable (const lldb_private::FileSpec &exe_file,
                       const lldb_private::ArchSpec &arch,
                       lldb::ModuleSP &module_sp,
                       const lldb_private::FileSpecList *module_search_paths_ptr);

    virtual const char *
    GetDescription ()
    {
        return GetDescriptionStatic();
    }

    virtual void
    GetStatus (lldb_private::Stream &strm);

    virtual lldb_private::Error
    GetFile (const lldb_private::FileSpec &platform_file, 
             const lldb_private::UUID *uuid_ptr,
             lldb_private::FileSpec &local_file);

    virtual lldb_private::Error
    GetSharedModule (const lldb_private::ModuleSpec &module_spec,
                     lldb::ModuleSP &module_sp,
                     const lldb_private::FileSpecList *module_search_paths_ptr,
                     lldb::ModuleSP *old_module_sp_ptr,
                     bool *did_create_ptr);

    virtual uint32_t
    FindProcesses (const lldb_private::ProcessInstanceInfoMatch &match_info,
                   lldb_private::ProcessInstanceInfoList &process_infos);

    virtual bool
    GetSupportedArchitectureAtIndex (uint32_t idx, 
                                     lldb_private::ArchSpec &arch);

protected:
    std::string m_sdk_directory;
    std::string m_build_update;
    //std::vector<FileSpec> m_device_support_os_dirs;
    
    const char *
    GetSDKDirectory();

private:
    DISALLOW_COPY_AND_ASSIGN (PlatformiOSSimulator);

};

#endif  // liblldb_PlatformiOSSimulator_h_
