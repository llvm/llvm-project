//===-- PlatformRemoteGDBServer.cpp ---------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PlatformRemoteGDBServer.h"
#include "lldb/Host/Config.h"

#include "lldb/Breakpoint/BreakpointLocation.h"
#include "lldb/Core/Debugger.h"
#include "lldb/Core/Module.h"
#include "lldb/Core/ModuleList.h"
#include "lldb/Core/ModuleSpec.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/Core/StreamFile.h"
#include "lldb/Host/ConnectionFileDescriptor.h"
#include "lldb/Host/Host.h"
#include "lldb/Host/HostInfo.h"
#include "lldb/Host/PosixApi.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/Target.h"
#include "lldb/Utility/FileSpec.h"
#include "lldb/Utility/Log.h"
#include "lldb/Utility/ProcessInfo.h"
#include "lldb/Utility/Status.h"
#include "lldb/Utility/StreamString.h"
#include "lldb/Utility/UriParser.h"

#include "Plugins/Process/Utility/GDBRemoteSignals.h"
#include "Plugins/Process/gdb-remote/ProcessGDBRemote.h"

using namespace lldb;
using namespace lldb_private;
using namespace lldb_private::platform_gdb_server;

LLDB_PLUGIN_DEFINE_ADV(PlatformRemoteGDBServer, PlatformGDB)

static bool g_initialized = false;

void PlatformRemoteGDBServer::Initialize() {
  Platform::Initialize();

  if (!g_initialized) {
    g_initialized = true;
    PluginManager::RegisterPlugin(
        PlatformRemoteGDBServer::GetPluginNameStatic(),
        PlatformRemoteGDBServer::GetDescriptionStatic(),
        PlatformRemoteGDBServer::CreateInstance);
  }
}

void PlatformRemoteGDBServer::Terminate() {
  if (g_initialized) {
    g_initialized = false;
    PluginManager::UnregisterPlugin(PlatformRemoteGDBServer::CreateInstance);
  }

  Platform::Terminate();
}

PlatformSP PlatformRemoteGDBServer::CreateInstance(bool force,
                                                   const ArchSpec *arch) {
  bool create = force;
  if (!create) {
    create = !arch->TripleVendorWasSpecified() && !arch->TripleOSWasSpecified();
  }
  if (create)
    return PlatformSP(new PlatformRemoteGDBServer());
  return PlatformSP();
}

ConstString PlatformRemoteGDBServer::GetPluginNameStatic() {
  static ConstString g_name("remote-gdb-server");
  return g_name;
}

const char *PlatformRemoteGDBServer::GetDescriptionStatic() {
  return "A platform that uses the GDB remote protocol as the communication "
         "transport.";
}

const char *PlatformRemoteGDBServer::GetDescription() {
  if (m_platform_description.empty()) {
    if (IsConnected()) {
      // Send the get description packet
    }
  }

  if (!m_platform_description.empty())
    return m_platform_description.c_str();
  return GetDescriptionStatic();
}

Status PlatformRemoteGDBServer::ResolveExecutable(
    const ModuleSpec &module_spec, lldb::ModuleSP &exe_module_sp,
    const FileSpecList *module_search_paths_ptr) {
  // copied from PlatformRemoteiOS

  Status error;
  // Nothing special to do here, just use the actual file and architecture

  ModuleSpec resolved_module_spec(module_spec);

  // Resolve any executable within an apk on Android?
  // Host::ResolveExecutableInBundle (resolved_module_spec.GetFileSpec());

  if (FileSystem::Instance().Exists(resolved_module_spec.GetFileSpec()) ||
      module_spec.GetUUID().IsValid()) {
    if (resolved_module_spec.GetArchitecture().IsValid() ||
        resolved_module_spec.GetUUID().IsValid()) {
      error = ModuleList::GetSharedModule(resolved_module_spec, exe_module_sp,
                                          module_search_paths_ptr, nullptr,
                                          nullptr);

      if (exe_module_sp && exe_module_sp->GetObjectFile())
        return error;
      exe_module_sp.reset();
    }
    // No valid architecture was specified or the exact arch wasn't found so
    // ask the platform for the architectures that we should be using (in the
    // correct order) and see if we can find a match that way
    StreamString arch_names;
    for (uint32_t idx = 0; GetSupportedArchitectureAtIndex(
             idx, resolved_module_spec.GetArchitecture());
         ++idx) {
      error = ModuleList::GetSharedModule(resolved_module_spec, exe_module_sp,
                                          module_search_paths_ptr, nullptr,
                                          nullptr);
      // Did we find an executable using one of the
      if (error.Success()) {
        if (exe_module_sp && exe_module_sp->GetObjectFile())
          break;
        else
          error.SetErrorToGenericError();
      }

      if (idx > 0)
        arch_names.PutCString(", ");
      arch_names.PutCString(
          resolved_module_spec.GetArchitecture().GetArchitectureName());
    }

    if (error.Fail() || !exe_module_sp) {
      if (FileSystem::Instance().Readable(resolved_module_spec.GetFileSpec())) {
        error.SetErrorStringWithFormat(
            "'%s' doesn't contain any '%s' platform architectures: %s",
            resolved_module_spec.GetFileSpec().GetPath().c_str(),
            GetPluginName().GetCString(), arch_names.GetData());
      } else {
        error.SetErrorStringWithFormat(
            "'%s' is not readable",
            resolved_module_spec.GetFileSpec().GetPath().c_str());
      }
    }
  } else {
    error.SetErrorStringWithFormat(
        "'%s' does not exist",
        resolved_module_spec.GetFileSpec().GetPath().c_str());
  }

  return error;
}

bool PlatformRemoteGDBServer::GetModuleSpec(const FileSpec &module_file_spec,
                                            const ArchSpec &arch,
                                            ModuleSpec &module_spec) {
  Log *log = GetLogIfAnyCategoriesSet(LIBLLDB_LOG_PLATFORM);

  const auto module_path = module_file_spec.GetPath(false);

  if (!m_gdb_client.GetModuleInfo(module_file_spec, arch, module_spec)) {
    LLDB_LOGF(
        log,
        "PlatformRemoteGDBServer::%s - failed to get module info for %s:%s",
        __FUNCTION__, module_path.c_str(),
        arch.GetTriple().getTriple().c_str());
    return false;
  }

  if (log) {
    StreamString stream;
    module_spec.Dump(stream);
    LLDB_LOGF(log,
              "PlatformRemoteGDBServer::%s - got module info for (%s:%s) : %s",
              __FUNCTION__, module_path.c_str(),
              arch.GetTriple().getTriple().c_str(), stream.GetData());
  }

  return true;
}

Status PlatformRemoteGDBServer::GetFileWithUUID(const FileSpec &platform_file,
                                                const UUID *uuid_ptr,
                                                FileSpec &local_file) {
  // Default to the local case
  local_file = platform_file;
  return Status();
}

/// Default Constructor
PlatformRemoteGDBServer::PlatformRemoteGDBServer()
    : Platform(false), // This is a remote platform
      m_gdb_client() {
  m_gdb_client.SetPacketTimeout(
      process_gdb_remote::ProcessGDBRemote::GetPacketTimeout());
}

/// Destructor.
///
/// The destructor is virtual since this class is designed to be
/// inherited from by the plug-in instance.
PlatformRemoteGDBServer::~PlatformRemoteGDBServer() = default;

bool PlatformRemoteGDBServer::GetSupportedArchitectureAtIndex(uint32_t idx,
                                                              ArchSpec &arch) {
  ArchSpec remote_arch = m_gdb_client.GetSystemArchitecture();

  if (idx == 0) {
    arch = remote_arch;
    return arch.IsValid();
  } else if (idx == 1 && remote_arch.IsValid() &&
             remote_arch.GetTriple().isArch64Bit()) {
    arch.SetTriple(remote_arch.GetTriple().get32BitArchVariant());
    return arch.IsValid();
  }
  return false;
}

size_t PlatformRemoteGDBServer::GetSoftwareBreakpointTrapOpcode(
    Target &target, BreakpointSite *bp_site) {
  // This isn't needed if the z/Z packets are supported in the GDB remote
  // server. But we might need a packet to detect this.
  return 0;
}

bool PlatformRemoteGDBServer::GetRemoteOSVersion() {
  m_os_version = m_gdb_client.GetOSVersion();
  return !m_os_version.empty();
}

bool PlatformRemoteGDBServer::GetRemoteOSBuildString(std::string &s) {
  return m_gdb_client.GetOSBuildString(s);
}

bool PlatformRemoteGDBServer::GetRemoteOSKernelDescription(std::string &s) {
  return m_gdb_client.GetOSKernelDescription(s);
}

// Remote Platform subclasses need to override this function
ArchSpec PlatformRemoteGDBServer::GetRemoteSystemArchitecture() {
  return m_gdb_client.GetSystemArchitecture();
}

FileSpec PlatformRemoteGDBServer::GetRemoteWorkingDirectory() {
  if (IsConnected()) {
    Log *log = GetLogIfAnyCategoriesSet(LIBLLDB_LOG_PLATFORM);
    FileSpec working_dir;
    if (m_gdb_client.GetWorkingDir(working_dir) && log)
      LLDB_LOGF(log,
                "PlatformRemoteGDBServer::GetRemoteWorkingDirectory() -> '%s'",
                working_dir.GetCString());
    return working_dir;
  } else {
    return Platform::GetRemoteWorkingDirectory();
  }
}

bool PlatformRemoteGDBServer::SetRemoteWorkingDirectory(
    const FileSpec &working_dir) {
  if (IsConnected()) {
    // Clear the working directory it case it doesn't get set correctly. This
    // will for use to re-read it
    Log *log = GetLogIfAnyCategoriesSet(LIBLLDB_LOG_PLATFORM);
    LLDB_LOGF(log, "PlatformRemoteGDBServer::SetRemoteWorkingDirectory('%s')",
              working_dir.GetCString());
    return m_gdb_client.SetWorkingDir(working_dir) == 0;
  } else
    return Platform::SetRemoteWorkingDirectory(working_dir);
}

bool PlatformRemoteGDBServer::IsConnected() const {
  return m_gdb_client.IsConnected();
}

Status PlatformRemoteGDBServer::ConnectRemote(Args &args) {
  Status error;
  if (IsConnected()) {
    error.SetErrorStringWithFormat("the platform is already connected to '%s', "
                                   "execute 'platform disconnect' to close the "
                                   "current connection",
                                   GetHostname());
    return error;
  }

  if (args.GetArgumentCount() != 1) {
    error.SetErrorString(
        "\"platform connect\" takes a single argument: <connect-url>");
    return error;
  }

  const char *url = args.GetArgumentAtIndex(0);
  if (!url)
    return Status("URL is null.");

  int port;
  llvm::StringRef scheme, hostname, pathname;
  if (!UriParser::Parse(url, scheme, hostname, port, pathname))
    return Status("Invalid URL: %s", url);

  // We're going to reuse the hostname when we connect to the debugserver.
  m_platform_scheme = std::string(scheme);
  m_platform_hostname = std::string(hostname);

  m_gdb_client.SetConnection(std::make_unique<ConnectionFileDescriptor>());
  if (repro::Reproducer::Instance().IsReplaying()) {
    error = m_gdb_replay_server.Connect(m_gdb_client);
    if (error.Success())
      m_gdb_replay_server.StartAsyncThread();
  } else {
    if (repro::Generator *g = repro::Reproducer::Instance().GetGenerator()) {
      repro::GDBRemoteProvider &provider =
          g->GetOrCreate<repro::GDBRemoteProvider>();
      m_gdb_client.SetPacketRecorder(provider.GetNewPacketRecorder());
    }
    m_gdb_client.Connect(url, &error);
  }

  if (error.Fail())
    return error;

  if (m_gdb_client.HandshakeWithServer(&error)) {
    m_gdb_client.GetHostInfo();
    // If a working directory was set prior to connecting, send it down
    // now.
    if (m_working_dir)
      m_gdb_client.SetWorkingDir(m_working_dir);
  } else {
    m_gdb_client.Disconnect();
    if (error.Success())
      error.SetErrorString("handshake failed");
  }
  return error;
}

Status PlatformRemoteGDBServer::DisconnectRemote() {
  Status error;
  m_gdb_client.Disconnect(&error);
  m_remote_signals_sp.reset();
  return error;
}

const char *PlatformRemoteGDBServer::GetHostname() {
  m_gdb_client.GetHostname(m_name);
  if (m_name.empty())
    return nullptr;
  return m_name.c_str();
}

llvm::Optional<std::string>
PlatformRemoteGDBServer::DoGetUserName(UserIDResolver::id_t uid) {
  std::string name;
  if (m_gdb_client.GetUserName(uid, name))
    return std::move(name);
  return llvm::None;
}

llvm::Optional<std::string>
PlatformRemoteGDBServer::DoGetGroupName(UserIDResolver::id_t gid) {
  std::string name;
  if (m_gdb_client.GetGroupName(gid, name))
    return std::move(name);
  return llvm::None;
}

uint32_t PlatformRemoteGDBServer::FindProcesses(
    const ProcessInstanceInfoMatch &match_info,
    ProcessInstanceInfoList &process_infos) {
  return m_gdb_client.FindProcesses(match_info, process_infos);
}

bool PlatformRemoteGDBServer::GetProcessInfo(
    lldb::pid_t pid, ProcessInstanceInfo &process_info) {
  return m_gdb_client.GetProcessInfo(pid, process_info);
}

Status PlatformRemoteGDBServer::LaunchProcess(ProcessLaunchInfo &launch_info) {
  Log *log(GetLogIfAllCategoriesSet(LIBLLDB_LOG_PLATFORM));
  Status error;

  LLDB_LOGF(log, "PlatformRemoteGDBServer::%s() called", __FUNCTION__);

  auto num_file_actions = launch_info.GetNumFileActions();
  for (decltype(num_file_actions) i = 0; i < num_file_actions; ++i) {
    const auto file_action = launch_info.GetFileActionAtIndex(i);
    if (file_action->GetAction() != FileAction::eFileActionOpen)
      continue;
    switch (file_action->GetFD()) {
    case STDIN_FILENO:
      m_gdb_client.SetSTDIN(file_action->GetFileSpec());
      break;
    case STDOUT_FILENO:
      m_gdb_client.SetSTDOUT(file_action->GetFileSpec());
      break;
    case STDERR_FILENO:
      m_gdb_client.SetSTDERR(file_action->GetFileSpec());
      break;
    }
  }

  m_gdb_client.SetDisableASLR(
      launch_info.GetFlags().Test(eLaunchFlagDisableASLR));
  m_gdb_client.SetDetachOnError(
      launch_info.GetFlags().Test(eLaunchFlagDetachOnError));

  FileSpec working_dir = launch_info.GetWorkingDirectory();
  if (working_dir) {
    m_gdb_client.SetWorkingDir(working_dir);
  }

  // Send the environment and the program + arguments after we connect
  m_gdb_client.SendEnvironment(launch_info.GetEnvironment());

  ArchSpec arch_spec = launch_info.GetArchitecture();
  const char *arch_triple = arch_spec.GetTriple().str().c_str();

  m_gdb_client.SendLaunchArchPacket(arch_triple);
  LLDB_LOGF(
      log,
      "PlatformRemoteGDBServer::%s() set launch architecture triple to '%s'",
      __FUNCTION__, arch_triple ? arch_triple : "<NULL>");

  int arg_packet_err;
  {
    // Scope for the scoped timeout object
    process_gdb_remote::GDBRemoteCommunication::ScopedTimeout timeout(
        m_gdb_client, std::chrono::seconds(5));
    arg_packet_err = m_gdb_client.SendArgumentsPacket(launch_info);
  }

  if (arg_packet_err == 0) {
    std::string error_str;
    if (m_gdb_client.GetLaunchSuccess(error_str)) {
      const auto pid = m_gdb_client.GetCurrentProcessID(false);
      if (pid != LLDB_INVALID_PROCESS_ID) {
        launch_info.SetProcessID(pid);
        LLDB_LOGF(log,
                  "PlatformRemoteGDBServer::%s() pid %" PRIu64
                  " launched successfully",
                  __FUNCTION__, pid);
      } else {
        LLDB_LOGF(log,
                  "PlatformRemoteGDBServer::%s() launch succeeded but we "
                  "didn't get a valid process id back!",
                  __FUNCTION__);
        error.SetErrorString("failed to get PID");
      }
    } else {
      error.SetErrorString(error_str.c_str());
      LLDB_LOGF(log, "PlatformRemoteGDBServer::%s() launch failed: %s",
                __FUNCTION__, error.AsCString());
    }
  } else {
    error.SetErrorStringWithFormat("'A' packet returned an error: %i",
                                   arg_packet_err);
  }
  return error;
}

Status PlatformRemoteGDBServer::KillProcess(const lldb::pid_t pid) {
  if (!KillSpawnedProcess(pid))
    return Status("failed to kill remote spawned process");
  return Status();
}

lldb::ProcessSP PlatformRemoteGDBServer::DebugProcess(
    ProcessLaunchInfo &launch_info, Debugger &debugger,
    Target *target, // Can be NULL, if NULL create a new target, else use
                    // existing one
    Status &error) {
  lldb::ProcessSP process_sp;
  if (IsRemote()) {
    if (IsConnected()) {
      lldb::pid_t debugserver_pid = LLDB_INVALID_PROCESS_ID;
      std::string connect_url;
      if (!LaunchGDBServer(debugserver_pid, connect_url)) {
        error.SetErrorStringWithFormat("unable to launch a GDB server on '%s'",
                                       GetHostname());
      } else {
        if (target == nullptr) {
          TargetSP new_target_sp;

          error = debugger.GetTargetList().CreateTarget(
              debugger, "", "", eLoadDependentsNo, nullptr, new_target_sp);
          target = new_target_sp.get();
        } else
          error.Clear();

        if (target && error.Success()) {
          // The darwin always currently uses the GDB remote debugger plug-in
          // so even when debugging locally we are debugging remotely!
          process_sp = target->CreateProcess(launch_info.GetListener(),
                                             "gdb-remote", nullptr, true);

          if (process_sp) {
            error = process_sp->ConnectRemote(connect_url.c_str());
            // Retry the connect remote one time...
            if (error.Fail())
              error = process_sp->ConnectRemote(connect_url.c_str());
            if (error.Success())
              error = process_sp->Launch(launch_info);
            else if (debugserver_pid != LLDB_INVALID_PROCESS_ID) {
              printf("error: connect remote failed (%s)\n", error.AsCString());
              KillSpawnedProcess(debugserver_pid);
            }
          }
        }
      }
    } else {
      error.SetErrorString("not connected to remote gdb server");
    }
  }
  return process_sp;
}

bool PlatformRemoteGDBServer::LaunchGDBServer(lldb::pid_t &pid,
                                              std::string &connect_url) {
  ArchSpec remote_arch = GetRemoteSystemArchitecture();
  llvm::Triple &remote_triple = remote_arch.GetTriple();

  uint16_t port = 0;
  std::string socket_name;
  bool launch_result = false;
  if (remote_triple.getVendor() == llvm::Triple::Apple &&
      remote_triple.getOS() == llvm::Triple::IOS) {
    // When remote debugging to iOS, we use a USB mux that always talks to
    // localhost, so we will need the remote debugserver to accept connections
    // only from localhost, no matter what our current hostname is
    launch_result =
        m_gdb_client.LaunchGDBServer("127.0.0.1", pid, port, socket_name);
  } else {
    // All other hosts should use their actual hostname
    launch_result =
        m_gdb_client.LaunchGDBServer(nullptr, pid, port, socket_name);
  }

  if (!launch_result)
    return false;

  connect_url =
      MakeGdbServerUrl(m_platform_scheme, m_platform_hostname, port,
                       (socket_name.empty()) ? nullptr : socket_name.c_str());
  return true;
}

bool PlatformRemoteGDBServer::KillSpawnedProcess(lldb::pid_t pid) {
  return m_gdb_client.KillSpawnedProcess(pid);
}

lldb::ProcessSP PlatformRemoteGDBServer::Attach(
    ProcessAttachInfo &attach_info, Debugger &debugger,
    Target *target, // Can be NULL, if NULL create a new target, else use
                    // existing one
    Status &error) {
  lldb::ProcessSP process_sp;
  if (IsRemote()) {
    if (IsConnected()) {
      lldb::pid_t debugserver_pid = LLDB_INVALID_PROCESS_ID;
      std::string connect_url;
      if (!LaunchGDBServer(debugserver_pid, connect_url)) {
        error.SetErrorStringWithFormat("unable to launch a GDB server on '%s'",
                                       GetHostname());
      } else {
        if (target == nullptr) {
          TargetSP new_target_sp;

          error = debugger.GetTargetList().CreateTarget(
              debugger, "", "", eLoadDependentsNo, nullptr, new_target_sp);
          target = new_target_sp.get();
        } else
          error.Clear();

        if (target && error.Success()) {
          // The darwin always currently uses the GDB remote debugger plug-in
          // so even when debugging locally we are debugging remotely!
          process_sp =
              target->CreateProcess(attach_info.GetListenerForProcess(debugger),
                                    "gdb-remote", nullptr, true);
          if (process_sp) {
            error = process_sp->ConnectRemote(connect_url.c_str());
            if (error.Success()) {
              ListenerSP listener_sp = attach_info.GetHijackListener();
              if (listener_sp)
                process_sp->HijackProcessEvents(listener_sp);
              error = process_sp->Attach(attach_info);
            }

            if (error.Fail() && debugserver_pid != LLDB_INVALID_PROCESS_ID) {
              KillSpawnedProcess(debugserver_pid);
            }
          }
        }
      }
    } else {
      error.SetErrorString("not connected to remote gdb server");
    }
  }
  return process_sp;
}

Status PlatformRemoteGDBServer::MakeDirectory(const FileSpec &file_spec,
                                              uint32_t mode) {
  Status error = m_gdb_client.MakeDirectory(file_spec, mode);
  Log *log = GetLogIfAnyCategoriesSet(LIBLLDB_LOG_PLATFORM);
  LLDB_LOGF(log,
            "PlatformRemoteGDBServer::MakeDirectory(path='%s', mode=%o) "
            "error = %u (%s)",
            file_spec.GetCString(), mode, error.GetError(), error.AsCString());
  return error;
}

Status PlatformRemoteGDBServer::GetFilePermissions(const FileSpec &file_spec,
                                                   uint32_t &file_permissions) {
  Status error = m_gdb_client.GetFilePermissions(file_spec, file_permissions);
  Log *log = GetLogIfAnyCategoriesSet(LIBLLDB_LOG_PLATFORM);
  LLDB_LOGF(log,
            "PlatformRemoteGDBServer::GetFilePermissions(path='%s', "
            "file_permissions=%o) error = %u (%s)",
            file_spec.GetCString(), file_permissions, error.GetError(),
            error.AsCString());
  return error;
}

Status PlatformRemoteGDBServer::SetFilePermissions(const FileSpec &file_spec,
                                                   uint32_t file_permissions) {
  Status error = m_gdb_client.SetFilePermissions(file_spec, file_permissions);
  Log *log = GetLogIfAnyCategoriesSet(LIBLLDB_LOG_PLATFORM);
  LLDB_LOGF(log,
            "PlatformRemoteGDBServer::SetFilePermissions(path='%s', "
            "file_permissions=%o) error = %u (%s)",
            file_spec.GetCString(), file_permissions, error.GetError(),
            error.AsCString());
  return error;
}

lldb::user_id_t PlatformRemoteGDBServer::OpenFile(const FileSpec &file_spec,
                                                  File::OpenOptions flags,
                                                  uint32_t mode,
                                                  Status &error) {
  return m_gdb_client.OpenFile(file_spec, flags, mode, error);
}

bool PlatformRemoteGDBServer::CloseFile(lldb::user_id_t fd, Status &error) {
  return m_gdb_client.CloseFile(fd, error);
}

lldb::user_id_t
PlatformRemoteGDBServer::GetFileSize(const FileSpec &file_spec) {
  return m_gdb_client.GetFileSize(file_spec);
}

void PlatformRemoteGDBServer::AutoCompleteDiskFileOrDirectory(
    CompletionRequest &request, bool only_dir) {
  m_gdb_client.AutoCompleteDiskFileOrDirectory(request, only_dir);
}

uint64_t PlatformRemoteGDBServer::ReadFile(lldb::user_id_t fd, uint64_t offset,
                                           void *dst, uint64_t dst_len,
                                           Status &error) {
  return m_gdb_client.ReadFile(fd, offset, dst, dst_len, error);
}

uint64_t PlatformRemoteGDBServer::WriteFile(lldb::user_id_t fd, uint64_t offset,
                                            const void *src, uint64_t src_len,
                                            Status &error) {
  return m_gdb_client.WriteFile(fd, offset, src, src_len, error);
}

Status PlatformRemoteGDBServer::PutFile(const FileSpec &source,
                                        const FileSpec &destination,
                                        uint32_t uid, uint32_t gid) {
  return Platform::PutFile(source, destination, uid, gid);
}

Status PlatformRemoteGDBServer::CreateSymlink(
    const FileSpec &src, // The name of the link is in src
    const FileSpec &dst) // The symlink points to dst
{
  Status error = m_gdb_client.CreateSymlink(src, dst);
  Log *log = GetLogIfAnyCategoriesSet(LIBLLDB_LOG_PLATFORM);
  LLDB_LOGF(log,
            "PlatformRemoteGDBServer::CreateSymlink(src='%s', dst='%s') "
            "error = %u (%s)",
            src.GetCString(), dst.GetCString(), error.GetError(),
            error.AsCString());
  return error;
}

Status PlatformRemoteGDBServer::Unlink(const FileSpec &file_spec) {
  Status error = m_gdb_client.Unlink(file_spec);
  Log *log = GetLogIfAnyCategoriesSet(LIBLLDB_LOG_PLATFORM);
  LLDB_LOGF(log, "PlatformRemoteGDBServer::Unlink(path='%s') error = %u (%s)",
            file_spec.GetCString(), error.GetError(), error.AsCString());
  return error;
}

bool PlatformRemoteGDBServer::GetFileExists(const FileSpec &file_spec) {
  return m_gdb_client.GetFileExists(file_spec);
}

Status PlatformRemoteGDBServer::RunShellCommand(
    llvm::StringRef shell, llvm::StringRef command,
    const FileSpec &
        working_dir, // Pass empty FileSpec to use the current working directory
    int *status_ptr, // Pass NULL if you don't want the process exit status
    int *signo_ptr,  // Pass NULL if you don't want the signal that caused the
                     // process to exit
    std::string
        *command_output, // Pass NULL if you don't want the command output
    const Timeout<std::micro> &timeout) {
  return m_gdb_client.RunShellCommand(command, working_dir, status_ptr,
                                      signo_ptr, command_output, timeout);
}

void PlatformRemoteGDBServer::CalculateTrapHandlerSymbolNames() {
  m_trap_handlers.push_back(ConstString("_sigtramp"));
}

const UnixSignalsSP &PlatformRemoteGDBServer::GetRemoteUnixSignals() {
  if (!IsConnected())
    return Platform::GetRemoteUnixSignals();

  if (m_remote_signals_sp)
    return m_remote_signals_sp;

  // If packet not implemented or JSON failed to parse, we'll guess the signal
  // set based on the remote architecture.
  m_remote_signals_sp = UnixSignals::Create(GetRemoteSystemArchitecture());

  StringExtractorGDBRemote response;
  auto result = m_gdb_client.SendPacketAndWaitForResponse("jSignalsInfo",
                                                          response, false);

  if (result != decltype(result)::Success ||
      response.GetResponseType() != response.eResponse)
    return m_remote_signals_sp;

  auto object_sp =
      StructuredData::ParseJSON(std::string(response.GetStringRef()));
  if (!object_sp || !object_sp->IsValid())
    return m_remote_signals_sp;

  auto array_sp = object_sp->GetAsArray();
  if (!array_sp || !array_sp->IsValid())
    return m_remote_signals_sp;

  auto remote_signals_sp = std::make_shared<lldb_private::GDBRemoteSignals>();

  bool done = array_sp->ForEach(
      [&remote_signals_sp](StructuredData::Object *object) -> bool {
        if (!object || !object->IsValid())
          return false;

        auto dict = object->GetAsDictionary();
        if (!dict || !dict->IsValid())
          return false;

        // Signal number and signal name are required.
        int signo;
        if (!dict->GetValueForKeyAsInteger("signo", signo))
          return false;

        llvm::StringRef name;
        if (!dict->GetValueForKeyAsString("name", name))
          return false;

        // We can live without short_name, description, etc.
        bool suppress{false};
        auto object_sp = dict->GetValueForKey("suppress");
        if (object_sp && object_sp->IsValid())
          suppress = object_sp->GetBooleanValue();

        bool stop{false};
        object_sp = dict->GetValueForKey("stop");
        if (object_sp && object_sp->IsValid())
          stop = object_sp->GetBooleanValue();

        bool notify{false};
        object_sp = dict->GetValueForKey("notify");
        if (object_sp && object_sp->IsValid())
          notify = object_sp->GetBooleanValue();

        std::string description{""};
        object_sp = dict->GetValueForKey("description");
        if (object_sp && object_sp->IsValid())
          description = std::string(object_sp->GetStringValue());

        remote_signals_sp->AddSignal(signo, name.str().c_str(), suppress, stop,
                                     notify, description.c_str());
        return true;
      });

  if (done)
    m_remote_signals_sp = std::move(remote_signals_sp);

  return m_remote_signals_sp;
}

std::string PlatformRemoteGDBServer::MakeGdbServerUrl(
    const std::string &platform_scheme, const std::string &platform_hostname,
    uint16_t port, const char *socket_name) {
  const char *override_scheme =
      getenv("LLDB_PLATFORM_REMOTE_GDB_SERVER_SCHEME");
  const char *override_hostname =
      getenv("LLDB_PLATFORM_REMOTE_GDB_SERVER_HOSTNAME");
  const char *port_offset_c_str =
      getenv("LLDB_PLATFORM_REMOTE_GDB_SERVER_PORT_OFFSET");
  int port_offset = port_offset_c_str ? ::atoi(port_offset_c_str) : 0;

  return MakeUrl(override_scheme ? override_scheme : platform_scheme.c_str(),
                 override_hostname ? override_hostname
                                   : platform_hostname.c_str(),
                 port + port_offset, socket_name);
}

std::string PlatformRemoteGDBServer::MakeUrl(const char *scheme,
                                             const char *hostname,
                                             uint16_t port, const char *path) {
  StreamString result;
  result.Printf("%s://[%s]", scheme, hostname);
  if (port != 0)
    result.Printf(":%u", port);
  if (path)
    result.Write(path, strlen(path));
  return std::string(result.GetString());
}

size_t PlatformRemoteGDBServer::ConnectToWaitingProcesses(Debugger &debugger,
                                                          Status &error) {
  std::vector<std::string> connection_urls;
  GetPendingGdbServerList(connection_urls);

  for (size_t i = 0; i < connection_urls.size(); ++i) {
    ConnectProcess(connection_urls[i].c_str(), "gdb-remote", debugger, nullptr, error);
    if (error.Fail())
      return i; // We already connected to i process succsessfully
  }
  return connection_urls.size();
}

size_t PlatformRemoteGDBServer::GetPendingGdbServerList(
    std::vector<std::string> &connection_urls) {
  std::vector<std::pair<uint16_t, std::string>> remote_servers;
  m_gdb_client.QueryGDBServer(remote_servers);
  for (const auto &gdbserver : remote_servers) {
    const char *socket_name_cstr =
        gdbserver.second.empty() ? nullptr : gdbserver.second.c_str();
    connection_urls.emplace_back(
        MakeGdbServerUrl(m_platform_scheme, m_platform_hostname,
                         gdbserver.first, socket_name_cstr));
  }
  return connection_urls.size();
}
