//===-- PlatformLinux.h -----------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_PlatformLinux_h_
#define liblldb_PlatformLinux_h_

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes
#include "lldb/Target/Platform.h"

namespace lldb_private {

    class PlatformLinux : public Platform
    {
    public:

        static void
        Initialize ();

        static void
        Terminate ();
        
        PlatformLinux (bool is_host);

        virtual
        ~PlatformLinux();

        //------------------------------------------------------------
        // lldb_private::PluginInterface functions
        //------------------------------------------------------------
        static Platform *
        CreateInstance (bool force, const lldb_private::ArchSpec *arch);

        static lldb_private::ConstString
        GetPluginNameStatic (bool is_host);

        static const char *
        GetPluginDescriptionStatic (bool is_host);

        virtual lldb_private::ConstString
        GetPluginName();
        
        virtual uint32_t
        GetPluginVersion()
        {
            return 1;
        }

        //------------------------------------------------------------
        // lldb_private::Platform functions
        //------------------------------------------------------------
        virtual Error
        ResolveExecutable (const FileSpec &exe_file,
                           const ArchSpec &arch,
                           lldb::ModuleSP &module_sp,
                           const FileSpecList *module_search_paths_ptr);

        virtual const char *
        GetDescription ()
        {
            return GetPluginDescriptionStatic(IsHost());
        }

        virtual void
        GetStatus (Stream &strm);

        virtual Error
        GetFile (const FileSpec &platform_file,
                 const UUID* uuid, FileSpec &local_file);

        virtual bool
        GetProcessInfo (lldb::pid_t pid, ProcessInstanceInfo &proc_info);

        virtual bool
        GetSupportedArchitectureAtIndex (uint32_t idx, ArchSpec &arch);

        virtual size_t
        GetSoftwareBreakpointTrapOpcode (Target &target, 
                                         BreakpointSite *bp_site);

        virtual lldb_private::Error
        LaunchProcess (lldb_private::ProcessLaunchInfo &launch_info);

        virtual lldb::ProcessSP
        Attach(ProcessAttachInfo &attach_info, Debugger &debugger,
               Target *target, Listener &listener, Error &error);

        // Linux processes can not be launched by spawning and attaching.
        virtual bool
        CanDebugProcess ()
        {
            return false;
        }

    protected:
        lldb::PlatformSP m_remote_platform_sp; // Allow multiple ways to connect to a remote darwin OS
        
    private:
        DISALLOW_COPY_AND_ASSIGN (PlatformLinux);
    };
} // namespace lldb_private

#endif  // liblldb_PlatformLinux_h_
