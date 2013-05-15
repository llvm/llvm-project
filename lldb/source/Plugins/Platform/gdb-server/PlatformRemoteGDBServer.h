//===-- PlatformRemoteGDBServer.h ----------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_PlatformRemoteGDBServer_h_
#define liblldb_PlatformRemoteGDBServer_h_

// C Includes
// C++ Includes
#include <string>

// Other libraries and framework includes
// Project includes
#include "lldb/Target/Platform.h"
#include "../../Process/gdb-remote/GDBRemoteCommunicationClient.h"

class PlatformRemoteGDBServer : public lldb_private::Platform
{
public:

    static void
    Initialize ();

    static void
    Terminate ();
    
    static lldb_private::Platform* 
    CreateInstance (bool force, const lldb_private::ArchSpec *arch);

    static lldb_private::ConstString
    GetPluginNameStatic();

    static const char *
    GetDescriptionStatic();


    PlatformRemoteGDBServer ();

    virtual
    ~PlatformRemoteGDBServer();

    //------------------------------------------------------------
    // lldb_private::PluginInterface functions
    //------------------------------------------------------------
    virtual lldb_private::ConstString
    GetPluginName()
    {
        return GetPluginNameStatic();
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
    GetDescription ();

    virtual lldb_private::Error
    GetFile (const lldb_private::FileSpec &platform_file, 
             const lldb_private::UUID *uuid_ptr,
             lldb_private::FileSpec &local_file);

    virtual bool
    GetProcessInfo (lldb::pid_t pid, 
                    lldb_private::ProcessInstanceInfo &proc_info);
    
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
    GetSupportedArchitectureAtIndex (uint32_t idx, lldb_private::ArchSpec &arch);

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

    // Remote subclasses should override this and return a valid instance
    // name if connected.
    virtual const char *
    GetHostname ();

    virtual const char *
    GetUserName (uint32_t uid);
    
    virtual const char *
    GetGroupName (uint32_t gid);

    virtual bool
    IsConnected () const;

    virtual lldb_private::Error
    ConnectRemote (lldb_private::Args& args);

    virtual lldb_private::Error
    DisconnectRemote ();

protected:
    GDBRemoteCommunicationClient m_gdb_client;
    std::string m_platform_description; // After we connect we can get a more complete description of what we are connected to

private:
    DISALLOW_COPY_AND_ASSIGN (PlatformRemoteGDBServer);

};

#endif  // liblldb_PlatformRemoteGDBServer_h_
