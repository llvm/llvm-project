//===-- PlatformDarwin.h ----------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_PlatformDarwin_h_
#define liblldb_PlatformDarwin_h_

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes
#include "lldb/Target/Platform.h"

class PlatformDarwin : public lldb_private::Platform
{
public:
    PlatformDarwin (bool is_host);

    virtual
    ~PlatformDarwin();
    
    //------------------------------------------------------------
    // lldb_private::Platform functions
    //------------------------------------------------------------
    virtual lldb_private::Error
    ResolveExecutable (const lldb_private::FileSpec &exe_file,
                       const lldb_private::ArchSpec &arch,
                       lldb::ModuleSP &module_sp,
                       const lldb_private::FileSpecList *module_search_paths_ptr);

    virtual lldb_private::Error
    ResolveSymbolFile (lldb_private::Target &target,
                       const lldb_private::ModuleSpec &sym_spec,
                       lldb_private::FileSpec &sym_file);

    lldb_private::FileSpec
    LocateExecutableScriptingResource (const lldb_private::ModuleSpec &module_spec);
    
    virtual lldb_private::Error
    GetSharedModule (const lldb_private::ModuleSpec &module_spec,
                     lldb::ModuleSP &module_sp,
                     const lldb_private::FileSpecList *module_search_paths_ptr,
                     lldb::ModuleSP *old_module_sp_ptr,
                     bool *did_create_ptr);

    virtual size_t
    GetSoftwareBreakpointTrapOpcode (lldb_private::Target &target, 
                                     lldb_private::BreakpointSite *bp_site);

    virtual bool
    GetRemoteOSVersion ();

    virtual bool
    GetRemoteOSBuildString (std::string &s);
    
    virtual bool
    GetRemoteOSKernelDescription (std::string &s);

    // Remote Platform subclasses need to override this function
    virtual lldb_private::ArchSpec
    GetRemoteSystemArchitecture ();

    virtual bool
    IsConnected () const;

    virtual lldb_private::Error
    ConnectRemote (lldb_private::Args& args);

    virtual lldb_private::Error
    DisconnectRemote ();

    virtual const char *
    GetHostname ();

    virtual const char *
    GetUserName (uint32_t uid);
    
    virtual const char *
    GetGroupName (uint32_t gid);

    virtual bool
    GetProcessInfo (lldb::pid_t pid, 
                    lldb_private::ProcessInstanceInfo &proc_info);
    
    virtual lldb::BreakpointSP
    SetThreadCreationBreakpoint (lldb_private::Target &target);

    virtual uint32_t
    FindProcesses (const lldb_private::ProcessInstanceInfoMatch &match_info,
                   lldb_private::ProcessInstanceInfoList &process_infos);
    
    virtual lldb_private::Error
    LaunchProcess (lldb_private::ProcessLaunchInfo &launch_info);

    virtual lldb::ProcessSP
    Attach (lldb_private::ProcessAttachInfo &attach_info,
            lldb_private::Debugger &debugger,
            lldb_private::Target *target,       // Can be NULL, if NULL create a new target, else use existing one
            lldb_private::Listener &listener, 
            lldb_private::Error &error);

    virtual bool
    ModuleIsExcludedForNonModuleSpecificSearches (lldb_private::Target &target, const lldb::ModuleSP &module_sp);
    
    virtual size_t
    GetEnvironment (lldb_private::StringList &environment);

    bool
    ARMGetSupportedArchitectureAtIndex (uint32_t idx, lldb_private::ArchSpec &arch);
    
    bool 
    x86GetSupportedArchitectureAtIndex (uint32_t idx, lldb_private::ArchSpec &arch);

protected:
    lldb::PlatformSP m_remote_platform_sp; // Allow multiple ways to connect to a remote darwin OS
    std::string m_developer_directory;
    
    const char *
    GetDeveloperDirectory();

private:
    DISALLOW_COPY_AND_ASSIGN (PlatformDarwin);

};

#endif  // liblldb_PlatformDarwin_h_
