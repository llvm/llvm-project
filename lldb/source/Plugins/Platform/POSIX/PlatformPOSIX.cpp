//===-- PlatformPOSIX.cpp ---------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "PlatformPOSIX.h"

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes

#include "lldb/Core/DataBufferHeap.h"
#include "lldb/Core/Debugger.h"
#include "lldb/Core/Log.h"
#include "lldb/Core/Module.h"
#include "lldb/Core/StreamString.h"
#include "lldb/Core/ValueObject.h"
#include "lldb/Expression/UserExpression.h"
#include "lldb/Host/File.h"
#include "lldb/Host/FileCache.h"
#include "lldb/Host/FileSpec.h"
#include "lldb/Host/FileSystem.h"
#include "lldb/Host/Host.h"
#include "lldb/Target/DynamicLoader.h"
#include "lldb/Target/ExecutionContext.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/ProcessLaunchInfo.h"
#include "lldb/Target/Thread.h"

using namespace lldb;
using namespace lldb_private;


//------------------------------------------------------------------
/// Default Constructor
//------------------------------------------------------------------
PlatformPOSIX::PlatformPOSIX (bool is_host) :
Platform(is_host),  // This is the local host platform
m_remote_platform_sp ()
{
}

//------------------------------------------------------------------
/// Destructor.
///
/// The destructor is virtual since this class is designed to be
/// inherited from by the plug-in instance.
//------------------------------------------------------------------
PlatformPOSIX::~PlatformPOSIX()
{
}

bool
PlatformPOSIX::GetModuleSpec (const FileSpec& module_file_spec,
                              const ArchSpec& arch,
                              ModuleSpec &module_spec)
{
    if (m_remote_platform_sp)
        return m_remote_platform_sp->GetModuleSpec (module_file_spec, arch, module_spec);

    return Platform::GetModuleSpec (module_file_spec, arch, module_spec);
}

lldb_private::OptionGroupOptions*
PlatformPOSIX::GetConnectionOptions (lldb_private::CommandInterpreter& interpreter)
{
    if (m_options.get() == NULL)
    {
        m_options.reset(new OptionGroupOptions(interpreter));
        m_options->Append(new OptionGroupPlatformRSync());
        m_options->Append(new OptionGroupPlatformSSH());
        m_options->Append(new OptionGroupPlatformCaching());
    }
    return m_options.get();
}

bool
PlatformPOSIX::IsConnected () const
{
    if (IsHost())
        return true;
    else if (m_remote_platform_sp)
        return m_remote_platform_sp->IsConnected();
    return false;
}

lldb_private::Error
PlatformPOSIX::RunShellCommand(const char *command,           // Shouldn't be NULL
                               const FileSpec &working_dir,   // Pass empty FileSpec to use the current working directory
                               int *status_ptr,               // Pass NULL if you don't want the process exit status
                               int *signo_ptr,                // Pass NULL if you don't want the signal that caused the process to exit
                               std::string *command_output,   // Pass NULL if you don't want the command output
                               uint32_t timeout_sec)          // Timeout in seconds to wait for shell program to finish
{
    if (IsHost())
        return Host::RunShellCommand(command, working_dir, status_ptr, signo_ptr, command_output, timeout_sec);
    else
    {
        if (m_remote_platform_sp)
            return m_remote_platform_sp->RunShellCommand(command, working_dir, status_ptr, signo_ptr, command_output, timeout_sec);
        else
            return Error("unable to run a remote command without a platform");
    }
}

Error
PlatformPOSIX::MakeDirectory(const FileSpec &file_spec, uint32_t file_permissions)
{
    if (m_remote_platform_sp)
        return m_remote_platform_sp->MakeDirectory(file_spec, file_permissions);
    else
        return Platform::MakeDirectory(file_spec ,file_permissions);
}

Error
PlatformPOSIX::GetFilePermissions(const FileSpec &file_spec, uint32_t &file_permissions)
{
    if (m_remote_platform_sp)
        return m_remote_platform_sp->GetFilePermissions(file_spec, file_permissions);
    else
        return Platform::GetFilePermissions(file_spec ,file_permissions);
}

Error
PlatformPOSIX::SetFilePermissions(const FileSpec &file_spec, uint32_t file_permissions)
{
    if (m_remote_platform_sp)
        return m_remote_platform_sp->SetFilePermissions(file_spec, file_permissions);
    else
        return Platform::SetFilePermissions(file_spec, file_permissions);
}

lldb::user_id_t
PlatformPOSIX::OpenFile (const FileSpec& file_spec,
                         uint32_t flags,
                         uint32_t mode,
                         Error &error)
{
    if (IsHost())
        return FileCache::GetInstance().OpenFile(file_spec, flags, mode, error);
    else if (m_remote_platform_sp)
        return m_remote_platform_sp->OpenFile(file_spec, flags, mode, error);
    else
        return Platform::OpenFile(file_spec, flags, mode, error);
}

bool
PlatformPOSIX::CloseFile (lldb::user_id_t fd, Error &error)
{
    if (IsHost())
        return FileCache::GetInstance().CloseFile(fd, error);
    else if (m_remote_platform_sp)
        return m_remote_platform_sp->CloseFile(fd, error);
    else
        return Platform::CloseFile(fd, error);
}

uint64_t
PlatformPOSIX::ReadFile (lldb::user_id_t fd,
                         uint64_t offset,
                         void *dst,
                         uint64_t dst_len,
                         Error &error)
{
    if (IsHost())
        return FileCache::GetInstance().ReadFile(fd, offset, dst, dst_len, error);
    else if (m_remote_platform_sp)
        return m_remote_platform_sp->ReadFile(fd, offset, dst, dst_len, error);
    else
        return Platform::ReadFile(fd, offset, dst, dst_len, error);
}

uint64_t
PlatformPOSIX::WriteFile (lldb::user_id_t fd,
                          uint64_t offset,
                          const void* src,
                          uint64_t src_len,
                          Error &error)
{
    if (IsHost())
        return FileCache::GetInstance().WriteFile(fd, offset, src, src_len, error);
    else if (m_remote_platform_sp)
        return m_remote_platform_sp->WriteFile(fd, offset, src, src_len, error);
    else
        return Platform::WriteFile(fd, offset, src, src_len, error);
}

static uint32_t
chown_file(Platform *platform,
           const char* path,
           uint32_t uid = UINT32_MAX,
           uint32_t gid = UINT32_MAX)
{
    if (!platform || !path || *path == 0)
        return UINT32_MAX;
    
    if (uid == UINT32_MAX && gid == UINT32_MAX)
        return 0;   // pretend I did chown correctly - actually I just didn't care
    
    StreamString command;
    command.PutCString("chown ");
    if (uid != UINT32_MAX)
        command.Printf("%d",uid);
    if (gid != UINT32_MAX)
        command.Printf(":%d",gid);
    command.Printf("%s",path);
    int status;
    platform->RunShellCommand(command.GetData(),
                              NULL,
                              &status,
                              NULL,
                              NULL,
                              10);
    return status;
}

lldb_private::Error
PlatformPOSIX::PutFile (const lldb_private::FileSpec& source,
                         const lldb_private::FileSpec& destination,
                         uint32_t uid,
                         uint32_t gid)
{
    Log *log(GetLogIfAnyCategoriesSet(LIBLLDB_LOG_PLATFORM));

    if (IsHost())
    {
        if (FileSpec::Equal(source, destination, true))
            return Error();
        // cp src dst
        // chown uid:gid dst
        std::string src_path (source.GetPath());
        if (src_path.empty())
            return Error("unable to get file path for source");
        std::string dst_path (destination.GetPath());
        if (dst_path.empty())
            return Error("unable to get file path for destination");
        StreamString command;
        command.Printf("cp %s %s", src_path.c_str(), dst_path.c_str());
        int status;
        RunShellCommand(command.GetData(),
                        NULL,
                        &status,
                        NULL,
                        NULL,
                        10);
        if (status != 0)
            return Error("unable to perform copy");
        if (uid == UINT32_MAX && gid == UINT32_MAX)
            return Error();
        if (chown_file(this,dst_path.c_str(),uid,gid) != 0)
            return Error("unable to perform chown");
        return Error();
    }
    else if (m_remote_platform_sp)
    {
        if (GetSupportsRSync())
        {
            std::string src_path (source.GetPath());
            if (src_path.empty())
                return Error("unable to get file path for source");
            std::string dst_path (destination.GetPath());
            if (dst_path.empty())
                return Error("unable to get file path for destination");
            StreamString command;
            if (GetIgnoresRemoteHostname())
            {
                if (!GetRSyncPrefix())
                    command.Printf("rsync %s %s %s",
                                   GetRSyncOpts(),
                                   src_path.c_str(),
                                   dst_path.c_str());
                else
                    command.Printf("rsync %s %s %s%s",
                                   GetRSyncOpts(),
                                   src_path.c_str(),
                                   GetRSyncPrefix(),
                                   dst_path.c_str());
            }
            else
                command.Printf("rsync %s %s %s:%s",
                               GetRSyncOpts(),
                               src_path.c_str(),
                               GetHostname(),
                               dst_path.c_str());
            if (log)
                log->Printf("[PutFile] Running command: %s\n", command.GetData());
            int retcode;
            Host::RunShellCommand(command.GetData(),
                                  NULL,
                                  &retcode,
                                  NULL,
                                  NULL,
                                  60);
            if (retcode == 0)
            {
                // Don't chown a local file for a remote system
//                if (chown_file(this,dst_path.c_str(),uid,gid) != 0)
//                    return Error("unable to perform chown");
                return Error();
            }
            // if we are still here rsync has failed - let's try the slow way before giving up
        }
    }
    return Platform::PutFile(source,destination,uid,gid);
}

lldb::user_id_t
PlatformPOSIX::GetFileSize (const FileSpec& file_spec)
{
    if (IsHost())
        return FileSystem::GetFileSize(file_spec);
    else if (m_remote_platform_sp)
        return m_remote_platform_sp->GetFileSize(file_spec);
    else
        return Platform::GetFileSize(file_spec);
}

Error
PlatformPOSIX::CreateSymlink(const FileSpec &src, const FileSpec &dst)
{
    if (IsHost())
        return FileSystem::Symlink(src, dst);
    else if (m_remote_platform_sp)
        return m_remote_platform_sp->CreateSymlink(src, dst);
    else
        return Platform::CreateSymlink(src, dst);
}

bool
PlatformPOSIX::GetFileExists (const FileSpec& file_spec)
{
    if (IsHost())
        return file_spec.Exists();
    else if (m_remote_platform_sp)
        return m_remote_platform_sp->GetFileExists(file_spec);
    else
        return Platform::GetFileExists(file_spec);
}

Error
PlatformPOSIX::Unlink(const FileSpec &file_spec)
{
    if (IsHost())
        return FileSystem::Unlink(file_spec);
    else if (m_remote_platform_sp)
        return m_remote_platform_sp->Unlink(file_spec);
    else
        return Platform::Unlink(file_spec);
}

lldb_private::Error
PlatformPOSIX::GetFile(const lldb_private::FileSpec &source,      // remote file path
                       const lldb_private::FileSpec &destination) // local file path
{
    Log *log(GetLogIfAnyCategoriesSet(LIBLLDB_LOG_PLATFORM));

    // Check the args, first.
    std::string src_path (source.GetPath());
    if (src_path.empty())
        return Error("unable to get file path for source");
    std::string dst_path (destination.GetPath());
    if (dst_path.empty())
        return Error("unable to get file path for destination");
    if (IsHost())
    {
        if (FileSpec::Equal(source, destination, true))
            return Error("local scenario->source and destination are the same file path: no operation performed");
        // cp src dst
        StreamString cp_command;
        cp_command.Printf("cp %s %s", src_path.c_str(), dst_path.c_str());
        int status;
        RunShellCommand(cp_command.GetData(),
                        NULL,
                        &status,
                        NULL,
                        NULL,
                        10);
        if (status != 0)
            return Error("unable to perform copy");
        return Error();
    }
    else if (m_remote_platform_sp)
    {
        if (GetSupportsRSync())
        {
            StreamString command;
            if (GetIgnoresRemoteHostname())
            {
                if (!GetRSyncPrefix())
                    command.Printf("rsync %s %s %s",
                                   GetRSyncOpts(),
                                   src_path.c_str(),
                                   dst_path.c_str());
                else
                    command.Printf("rsync %s %s%s %s",
                                   GetRSyncOpts(),
                                   GetRSyncPrefix(),
                                   src_path.c_str(),
                                   dst_path.c_str());
            }
            else
                command.Printf("rsync %s %s:%s %s",
                               GetRSyncOpts(),
                               m_remote_platform_sp->GetHostname(),
                               src_path.c_str(),
                               dst_path.c_str());
            if (log)
                log->Printf("[GetFile] Running command: %s\n", command.GetData());
            int retcode;
            Host::RunShellCommand(command.GetData(),
                                  NULL,
                                  &retcode,
                                  NULL,
                                  NULL,
                                  60);
            if (retcode == 0)
                return Error();
            // If we are here, rsync has failed - let's try the slow way before giving up
        }
        // open src and dst
        // read/write, read/write, read/write, ...
        // close src
        // close dst
        if (log)
            log->Printf("[GetFile] Using block by block transfer....\n");
        Error error;
        user_id_t fd_src = OpenFile (source,
                                     File::eOpenOptionRead,
                                     lldb::eFilePermissionsFileDefault,
                                     error);

        if (fd_src == UINT64_MAX)
            return Error("unable to open source file");

        uint32_t permissions = 0;
        error = GetFilePermissions(source, permissions);

        if (permissions == 0)
            permissions = lldb::eFilePermissionsFileDefault;

        user_id_t fd_dst = FileCache::GetInstance().OpenFile(
            destination, File::eOpenOptionCanCreate | File::eOpenOptionWrite | File::eOpenOptionTruncate, permissions,
            error);

        if (fd_dst == UINT64_MAX)
        {
            if (error.Success())
                error.SetErrorString("unable to open destination file");
        }

        if (error.Success())
        {
            lldb::DataBufferSP buffer_sp(new DataBufferHeap(1024, 0));
            uint64_t offset = 0;
            error.Clear();
            while (error.Success())
            {
                const uint64_t n_read = ReadFile (fd_src,
                                                  offset,
                                                  buffer_sp->GetBytes(),
                                                  buffer_sp->GetByteSize(),
                                                  error);
                if (error.Fail())
                    break;
                if (n_read == 0)
                    break;
                if (FileCache::GetInstance().WriteFile(fd_dst, offset, buffer_sp->GetBytes(), n_read, error) != n_read)
                {
                    if (!error.Fail())
                        error.SetErrorString("unable to write to destination file");
                    break;
                }
                offset += n_read;
            }
        }
        // Ignore the close error of src.
        if (fd_src != UINT64_MAX)
            CloseFile(fd_src, error);
        // And close the dst file descriptot.
        if (fd_dst != UINT64_MAX && !FileCache::GetInstance().CloseFile(fd_dst, error))
        {
            if (!error.Fail())
                error.SetErrorString("unable to close destination file");

        }
        return error;
    }
    return Platform::GetFile(source,destination);
}

std::string
PlatformPOSIX::GetPlatformSpecificConnectionInformation()
{
    StreamString stream;
    if (GetSupportsRSync())
    {
        stream.PutCString("rsync");
        if ( (GetRSyncOpts() && *GetRSyncOpts()) ||
             (GetRSyncPrefix() && *GetRSyncPrefix()) ||
             GetIgnoresRemoteHostname())
        {
            stream.Printf(", options: ");
            if (GetRSyncOpts() && *GetRSyncOpts())
                stream.Printf("'%s' ",GetRSyncOpts());
            stream.Printf(", prefix: ");
            if (GetRSyncPrefix() && *GetRSyncPrefix())
                stream.Printf("'%s' ",GetRSyncPrefix());
            if (GetIgnoresRemoteHostname())
                stream.Printf("ignore remote-hostname ");
        }
    }
    if (GetSupportsSSH())
    {
        stream.PutCString("ssh");
        if (GetSSHOpts() && *GetSSHOpts())
            stream.Printf(", options: '%s' ",GetSSHOpts());
    }
    if (GetLocalCacheDirectory() && *GetLocalCacheDirectory())
        stream.Printf("cache dir: %s",GetLocalCacheDirectory());
    if (stream.GetSize())
        return stream.GetData();
    else
        return "";
}

bool
PlatformPOSIX::CalculateMD5 (const FileSpec& file_spec,
                            uint64_t &low,
                            uint64_t &high)
{
    if (IsHost())
        return Platform::CalculateMD5 (file_spec, low, high);
    if (m_remote_platform_sp)
        return m_remote_platform_sp->CalculateMD5(file_spec, low, high);
    return false;
}

const lldb::UnixSignalsSP &
PlatformPOSIX::GetRemoteUnixSignals() {
    if (IsRemote() && m_remote_platform_sp)
        return m_remote_platform_sp->GetRemoteUnixSignals();
    return Platform::GetRemoteUnixSignals();
}


FileSpec
PlatformPOSIX::GetRemoteWorkingDirectory()
{
    if (IsRemote() && m_remote_platform_sp)
        return m_remote_platform_sp->GetRemoteWorkingDirectory();
    else
        return Platform::GetRemoteWorkingDirectory();
}

bool
PlatformPOSIX::SetRemoteWorkingDirectory(const FileSpec &working_dir)
{
    if (IsRemote() && m_remote_platform_sp)
        return m_remote_platform_sp->SetRemoteWorkingDirectory(working_dir);
    else
        return Platform::SetRemoteWorkingDirectory(working_dir);
}

bool
PlatformPOSIX::GetRemoteOSVersion ()
{
    if (m_remote_platform_sp)
        return m_remote_platform_sp->GetOSVersion (m_major_os_version,
                                                   m_minor_os_version,
                                                   m_update_os_version);
    return false;
}

bool
PlatformPOSIX::GetRemoteOSBuildString (std::string &s)
{
    if (m_remote_platform_sp)
        return m_remote_platform_sp->GetRemoteOSBuildString (s);
    s.clear();
    return false;
}

size_t
PlatformPOSIX::GetEnvironment (StringList &env)
{
    if (IsRemote())
    {
        if (m_remote_platform_sp)
            return m_remote_platform_sp->GetEnvironment(env);
        return 0;
    }
    return Host::GetEnvironment(env);
}

bool
PlatformPOSIX::GetRemoteOSKernelDescription (std::string &s)
{
    if (m_remote_platform_sp)
        return m_remote_platform_sp->GetRemoteOSKernelDescription (s);
    s.clear();
    return false;
}

// Remote Platform subclasses need to override this function
ArchSpec
PlatformPOSIX::GetRemoteSystemArchitecture ()
{
    if (m_remote_platform_sp)
        return m_remote_platform_sp->GetRemoteSystemArchitecture ();
    return ArchSpec();
}

const char *
PlatformPOSIX::GetHostname ()
{
    if (IsHost())
        return Platform::GetHostname();

    if (m_remote_platform_sp)
        return m_remote_platform_sp->GetHostname ();
    return NULL;
}

const char *
PlatformPOSIX::GetUserName (uint32_t uid)
{
    // Check the cache in Platform in case we have already looked this uid up
    const char *user_name = Platform::GetUserName(uid);
    if (user_name)
        return user_name;

    if (IsRemote() && m_remote_platform_sp)
        return m_remote_platform_sp->GetUserName(uid);
    return NULL;
}

const char *
PlatformPOSIX::GetGroupName (uint32_t gid)
{
    const char *group_name = Platform::GetGroupName(gid);
    if (group_name)
        return group_name;

    if (IsRemote() && m_remote_platform_sp)
        return m_remote_platform_sp->GetGroupName(gid);
    return NULL;
}

Error
PlatformPOSIX::ConnectRemote (Args& args)
{
    Error error;
    if (IsHost())
    {
        error.SetErrorStringWithFormat ("can't connect to the host platform '%s', always connected", GetPluginName().GetCString());
    }
    else
    {
        if (!m_remote_platform_sp)
            m_remote_platform_sp = Platform::Create (ConstString("remote-gdb-server"), error);

        if (m_remote_platform_sp && error.Success())
            error = m_remote_platform_sp->ConnectRemote (args);
        else
            error.SetErrorString ("failed to create a 'remote-gdb-server' platform");

        if (error.Fail())
            m_remote_platform_sp.reset();
    }

    if (error.Success() && m_remote_platform_sp)
    {
        if (m_options.get())
        {
            OptionGroupOptions* options = m_options.get();
            const OptionGroupPlatformRSync *m_rsync_options =
                static_cast<const OptionGroupPlatformRSync *>(options->GetGroupWithOption('r'));
            const OptionGroupPlatformSSH *m_ssh_options =
                static_cast<const OptionGroupPlatformSSH *>(options->GetGroupWithOption('s'));
            const OptionGroupPlatformCaching *m_cache_options =
                static_cast<const OptionGroupPlatformCaching *>(options->GetGroupWithOption('c'));

            if (m_rsync_options->m_rsync)
            {
                SetSupportsRSync(true);
                SetRSyncOpts(m_rsync_options->m_rsync_opts.c_str());
                SetRSyncPrefix(m_rsync_options->m_rsync_prefix.c_str());
                SetIgnoresRemoteHostname(m_rsync_options->m_ignores_remote_hostname);
            }
            if (m_ssh_options->m_ssh)
            {
                SetSupportsSSH(true);
                SetSSHOpts(m_ssh_options->m_ssh_opts.c_str());
            }
            SetLocalCacheDirectory(m_cache_options->m_cache_dir.c_str());
        }
    }

    return error;
}

Error
PlatformPOSIX::DisconnectRemote ()
{
    Error error;

    if (IsHost())
    {
        error.SetErrorStringWithFormat ("can't disconnect from the host platform '%s', always connected", GetPluginName().GetCString());
    }
    else
    {
        if (m_remote_platform_sp)
            error = m_remote_platform_sp->DisconnectRemote ();
        else
            error.SetErrorString ("the platform is not currently connected");
    }
    return error;
}

Error
PlatformPOSIX::LaunchProcess (ProcessLaunchInfo &launch_info)
{
    Error error;

    if (IsHost())
    {
        error = Platform::LaunchProcess (launch_info);
    }
    else
    {
        if (m_remote_platform_sp)
            error = m_remote_platform_sp->LaunchProcess (launch_info);
        else
            error.SetErrorString ("the platform is not currently connected");
    }
    return error;
}

lldb_private::Error
PlatformPOSIX::KillProcess (const lldb::pid_t pid)
{
    if (IsHost())
        return Platform::KillProcess (pid);

    if (m_remote_platform_sp)
        return m_remote_platform_sp->KillProcess (pid);

    return Error ("the platform is not currently connected");
}

lldb::ProcessSP
PlatformPOSIX::Attach (ProcessAttachInfo &attach_info,
                       Debugger &debugger,
                       Target *target,
                       Error &error)
{
    lldb::ProcessSP process_sp;
    Log *log(GetLogIfAnyCategoriesSet(LIBLLDB_LOG_PLATFORM));

    if (IsHost())
    {
        if (target == NULL)
        {
            TargetSP new_target_sp;

            error = debugger.GetTargetList().CreateTarget (debugger,
                                                           NULL,
                                                           NULL,
                                                           false,
                                                           NULL,
                                                           new_target_sp);
            target = new_target_sp.get();
            if (log)
                log->Printf ("PlatformPOSIX::%s created new target", __FUNCTION__);
        }
        else
        {
            error.Clear();
            if (log)
                log->Printf ("PlatformPOSIX::%s target already existed, setting target", __FUNCTION__);
        }

        if (target && error.Success())
        {
            debugger.GetTargetList().SetSelectedTarget(target);
            if (log)
            {
                ModuleSP exe_module_sp = target->GetExecutableModule ();
                log->Printf("PlatformPOSIX::%s set selected target to %p %s", __FUNCTION__, (void *)target,
                            exe_module_sp ? exe_module_sp->GetFileSpec().GetPath().c_str() : "<null>");
            }


            process_sp = target->CreateProcess (attach_info.GetListenerForProcess(debugger), attach_info.GetProcessPluginName(), NULL);

            if (process_sp)
            {
                auto listener_sp = attach_info.GetHijackListener();
                if (listener_sp == nullptr)
                {
                    listener_sp.reset(new Listener("lldb.PlatformPOSIX.attach.hijack"));
                    attach_info.SetHijackListener(listener_sp);
                }
                process_sp->HijackProcessEvents(listener_sp.get());
                error = process_sp->Attach (attach_info);
            }
        }
    }
    else
    {
        if (m_remote_platform_sp)
            process_sp = m_remote_platform_sp->Attach (attach_info, debugger, target, error);
        else
            error.SetErrorString ("the platform is not currently connected");
    }
    return process_sp;
}

lldb::ProcessSP
PlatformPOSIX::DebugProcess (ProcessLaunchInfo &launch_info,
                              Debugger &debugger,
                              Target *target,       // Can be NULL, if NULL create a new target, else use existing one
                              Error &error)
{
    ProcessSP process_sp;

    if (IsHost())
    {
        // We are going to hand this process off to debugserver which will be in charge of setting the exit status.
        // We still need to reap it from lldb but if we let the monitor thread also set the exit status, we set up a
        // race between debugserver & us for who will find out about the debugged process's death.
        launch_info.GetFlags().Set(eLaunchFlagDontSetExitStatus);
        process_sp = Platform::DebugProcess (launch_info, debugger, target, error);
    }
    else
    {
        if (m_remote_platform_sp)
            process_sp = m_remote_platform_sp->DebugProcess (launch_info, debugger, target, error);
        else
            error.SetErrorString ("the platform is not currently connected");
    }
    return process_sp;

}

void
PlatformPOSIX::CalculateTrapHandlerSymbolNames ()
{
    m_trap_handlers.push_back (ConstString ("_sigtramp"));
}

Error
PlatformPOSIX::EvaluateLibdlExpression(lldb_private::Process* process,
                                       const char* expr_cstr,
                                       const char* expr_prefix,
                                       lldb::ValueObjectSP& result_valobj_sp)
{
    DynamicLoader *loader = process->GetDynamicLoader();
    if (loader)
    {
        Error error = loader->CanLoadImage();
        if (error.Fail())
            return error;
    }

    ThreadSP thread_sp(process->GetThreadList().GetSelectedThread());
    if (!thread_sp)
        return Error("Selected thread isn't valid");

    StackFrameSP frame_sp(thread_sp->GetStackFrameAtIndex(0));
    if (!frame_sp)
        return Error("Frame 0 isn't valid");

    ExecutionContext exe_ctx;
    frame_sp->CalculateExecutionContext(exe_ctx);
    EvaluateExpressionOptions expr_options;
    expr_options.SetUnwindOnError(true);
    expr_options.SetIgnoreBreakpoints(true);
    expr_options.SetExecutionPolicy(eExecutionPolicyAlways);
    expr_options.SetLanguage(eLanguageTypeC_plus_plus);

    Error expr_error;
    UserExpression::Evaluate(exe_ctx,
                             expr_options,
                             expr_cstr,
                             expr_prefix,
                             result_valobj_sp,
                             expr_error);
    if (result_valobj_sp->GetError().Fail())
        return result_valobj_sp->GetError();
    return Error();
}

uint32_t
PlatformPOSIX::LoadImage(lldb_private::Process* process, const FileSpec& image_spec, Error& error)
{
    char path[PATH_MAX];
    image_spec.GetPath(path, sizeof(path));

    StreamString expr;
    expr.Printf(R"(
                   struct __lldb_dlopen_result { void *image_ptr; const char *error_str; } the_result;
                   the_result.image_ptr = dlopen ("%s", 2);
                   if (the_result.image_ptr == (void *) 0x0)
                   {
                       the_result.error_str = dlerror();
                   }
                   else
                   {
                       the_result.error_str = (const char *) 0x0;
                   }
                   the_result;
                  )",
                  path);
    const char *prefix = R"(
                            extern "C" void* dlopen (const char *path, int mode);
                            extern "C" const char *dlerror (void);
                            )";
    lldb::ValueObjectSP result_valobj_sp;
    error = EvaluateLibdlExpression(process, expr.GetData(), prefix, result_valobj_sp);
    if (error.Fail())
        return LLDB_INVALID_IMAGE_TOKEN;

    error = result_valobj_sp->GetError();
    if (error.Fail())
        return LLDB_INVALID_IMAGE_TOKEN;

    Scalar scalar;
    ValueObjectSP image_ptr_sp = result_valobj_sp->GetChildAtIndex(0, true);
    if (!image_ptr_sp || !image_ptr_sp->ResolveValue(scalar))
    {
        error.SetErrorStringWithFormat("unable to load '%s'", path);
        return LLDB_INVALID_IMAGE_TOKEN;
    }

    addr_t image_ptr = scalar.ULongLong(LLDB_INVALID_ADDRESS);
    if (image_ptr != 0 && image_ptr != LLDB_INVALID_ADDRESS)
        return process->AddImageToken(image_ptr);

    if (image_ptr == 0)
    {
        ValueObjectSP error_str_sp = result_valobj_sp->GetChildAtIndex(1, true);
        if (error_str_sp && error_str_sp->IsCStringContainer(true))
        {
            DataBufferSP buffer_sp(new DataBufferHeap(10240,0));
            size_t num_chars = error_str_sp->ReadPointedString (buffer_sp, error, 10240).first;
            if (error.Success() && num_chars > 0)
                error.SetErrorStringWithFormat("dlopen error: %s", buffer_sp->GetBytes());
            else
                error.SetErrorStringWithFormat("dlopen failed for unknown reasons.");
            return LLDB_INVALID_IMAGE_TOKEN;
        }
    }
    error.SetErrorStringWithFormat("unable to load '%s'", path);
    return LLDB_INVALID_IMAGE_TOKEN;
}

Error
PlatformPOSIX::UnloadImage (lldb_private::Process* process, uint32_t image_token)
{
    const addr_t image_addr = process->GetImagePtrFromToken(image_token);
    if (image_addr == LLDB_INVALID_ADDRESS)
        return Error("Invalid image token");

    StreamString expr;
    expr.Printf("dlclose((void *)0x%" PRIx64 ")", image_addr);
    const char *prefix = "extern \"C\" int dlclose(void* handle);\n";
    lldb::ValueObjectSP result_valobj_sp;
    Error error = EvaluateLibdlExpression(process, expr.GetData(), prefix, result_valobj_sp);
    if (error.Fail())
        return error;

    if (result_valobj_sp->GetError().Fail())
        return result_valobj_sp->GetError();

    Scalar scalar;
    if (result_valobj_sp->ResolveValue(scalar))
    {
        if (scalar.UInt(1))
            return Error("expression failed: \"%s\"", expr.GetData());
        process->ResetImageToken(image_token);
    }
    return Error();
}
