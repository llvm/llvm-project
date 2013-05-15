//===-- PlatformFreeBSD.h -----------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_PlatformFreeBSD_h_
#define liblldb_PlatformFreeBSD_h_

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes
#include "lldb/Target/Platform.h"

class PlatformFreeBSD : public lldb_private::Platform
{
public:
    // Mostly taken from PlatformDarwin and PlatformMacOSX

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
    GetDescriptionStatic (bool is_host);

    //------------------------------------------------------------
    // Class Methods
    //------------------------------------------------------------
    PlatformFreeBSD (bool is_host);

    virtual
    ~PlatformFreeBSD();

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
        return GetDescriptionStatic(IsHost());
    }

    //------------------------------------------------------------
    // lldb_private::Platform functions
    //------------------------------------------------------------
    virtual lldb_private::Error
    ResolveExecutable (const lldb_private::FileSpec &exe_file,
                       const lldb_private::ArchSpec &arch,
                       lldb::ModuleSP &module_sp,
                       const lldb_private::FileSpecList *module_search_paths_ptr);

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

    virtual uint32_t
    FindProcesses (const lldb_private::ProcessInstanceInfoMatch &match_info,
                   lldb_private::ProcessInstanceInfoList &process_infos);

    virtual lldb_private::Error
    LaunchProcess (lldb_private::ProcessLaunchInfo &launch_info);

    virtual lldb::ProcessSP
    Attach(lldb_private::ProcessAttachInfo &attach_info,
           lldb_private::Debugger &debugger,
           lldb_private::Target *target,
           lldb_private::Listener &listener,
           lldb_private::Error &error);

    // FreeBSD processes can not be launched by spawning and attaching.
    virtual bool
    CanDebugProcess () { return false; }

    // Only on PlatformMacOSX:
    virtual lldb_private::Error
    GetFile (const lldb_private::FileSpec &platform_file,
             const lldb_private::UUID* uuid, lldb_private::FileSpec &local_file);

    lldb_private::Error
    GetSharedModule (const lldb_private::ModuleSpec &module_spec,
                     lldb::ModuleSP &module_sp,
                     const lldb_private::FileSpecList *module_search_paths_ptr,
                     lldb::ModuleSP *old_module_sp_ptr,
                     bool *did_create_ptr);

    virtual bool
    GetSupportedArchitectureAtIndex (uint32_t idx, lldb_private::ArchSpec &arch);

    virtual void
    GetStatus (lldb_private::Stream &strm);

protected:
    lldb::PlatformSP m_remote_platform_sp; // Allow multiple ways to connect to a remote freebsd OS

private:
    DISALLOW_COPY_AND_ASSIGN (PlatformFreeBSD);
};

#endif  // liblldb_PlatformFreeBSD_h_
