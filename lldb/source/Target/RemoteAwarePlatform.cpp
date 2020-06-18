//===-- RemoteAwarePlatform.cpp -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Target/RemoteAwarePlatform.h"
#include "lldb/Core/Module.h"
#include "lldb/Core/ModuleList.h"
#include "lldb/Core/ModuleSpec.h"
#include "lldb/Host/FileCache.h"
#include "lldb/Host/FileSystem.h"
#include "lldb/Host/Host.h"
#include "lldb/Host/HostInfo.h"
#include "lldb/Utility/StreamString.h"

using namespace lldb_private;
using namespace lldb;

bool RemoteAwarePlatform::GetModuleSpec(const FileSpec &module_file_spec,
                                        const ArchSpec &arch,
                                        ModuleSpec &module_spec) {
  if (m_remote_platform_sp)
    return m_remote_platform_sp->GetModuleSpec(module_file_spec, arch,
                                               module_spec);

  return false;
}

Status RemoteAwarePlatform::ResolveExecutable(
    const ModuleSpec &module_spec, ModuleSP &exe_module_sp,
    const FileSpecList *module_search_paths_ptr) {
  Status error;
  // Nothing special to do here, just use the actual file and architecture

  char exe_path[PATH_MAX];
  ModuleSpec resolved_module_spec(module_spec);

  if (IsHost()) {
    // If we have "ls" as the exe_file, resolve the executable location based
    // on the current path variables
    if (!FileSystem::Instance().Exists(resolved_module_spec.GetFileSpec())) {
      resolved_module_spec.GetFileSpec().GetPath(exe_path, sizeof(exe_path));
      resolved_module_spec.GetFileSpec().SetFile(exe_path,
                                                 FileSpec::Style::native);
      FileSystem::Instance().Resolve(resolved_module_spec.GetFileSpec());
    }

    if (!FileSystem::Instance().Exists(resolved_module_spec.GetFileSpec()))
      FileSystem::Instance().ResolveExecutableLocation(
          resolved_module_spec.GetFileSpec());

    // Resolve any executable within a bundle on MacOSX
    Host::ResolveExecutableInBundle(resolved_module_spec.GetFileSpec());

    if (FileSystem::Instance().Exists(resolved_module_spec.GetFileSpec()))
      error.Clear();
    else {
      const uint32_t permissions = FileSystem::Instance().GetPermissions(
          resolved_module_spec.GetFileSpec());
      if (permissions && (permissions & eFilePermissionsEveryoneR) == 0)
        error.SetErrorStringWithFormat(
            "executable '%s' is not readable",
            resolved_module_spec.GetFileSpec().GetPath().c_str());
      else
        error.SetErrorStringWithFormat(
            "unable to find executable for '%s'",
            resolved_module_spec.GetFileSpec().GetPath().c_str());
    }
  } else {
    if (m_remote_platform_sp) {
      return GetCachedExecutable(resolved_module_spec, exe_module_sp,
                                 module_search_paths_ptr,
                                 *m_remote_platform_sp);
    }

    // We may connect to a process and use the provided executable (Don't use
    // local $PATH).

    // Resolve any executable within a bundle on MacOSX
    Host::ResolveExecutableInBundle(resolved_module_spec.GetFileSpec());

    if (FileSystem::Instance().Exists(resolved_module_spec.GetFileSpec()))
      error.Clear();
    else
      error.SetErrorStringWithFormat("the platform is not currently "
                                     "connected, and '%s' doesn't exist in "
                                     "the system root.",
                                     exe_path);
  }

  if (error.Success()) {
    if (resolved_module_spec.GetArchitecture().IsValid()) {
      error = ModuleList::GetSharedModule(resolved_module_spec, exe_module_sp,
                                          module_search_paths_ptr, nullptr, nullptr);
      if (error.Fail()) {
        // If we failed, it may be because the vendor and os aren't known. If
	// that is the case, try setting them to the host architecture and give
	// it another try.
        llvm::Triple &module_triple =
            resolved_module_spec.GetArchitecture().GetTriple();
        bool is_vendor_specified =
            (module_triple.getVendor() != llvm::Triple::UnknownVendor);
        bool is_os_specified =
            (module_triple.getOS() != llvm::Triple::UnknownOS);
        if (!is_vendor_specified || !is_os_specified) {
          const llvm::Triple &host_triple =
              HostInfo::GetArchitecture(HostInfo::eArchKindDefault).GetTriple();

          if (!is_vendor_specified)
            module_triple.setVendorName(host_triple.getVendorName());
          if (!is_os_specified)
            module_triple.setOSName(host_triple.getOSName());

          error = ModuleList::GetSharedModule(resolved_module_spec,
                                              exe_module_sp, module_search_paths_ptr, nullptr, nullptr);
        }
      }

      // TODO find out why exe_module_sp might be NULL
      if (error.Fail() || !exe_module_sp || !exe_module_sp->GetObjectFile()) {
        exe_module_sp.reset();
        error.SetErrorStringWithFormat(
            "'%s' doesn't contain the architecture %s",
            resolved_module_spec.GetFileSpec().GetPath().c_str(),
            resolved_module_spec.GetArchitecture().GetArchitectureName());
      }
    } else {
      // No valid architecture was specified, ask the platform for the
      // architectures that we should be using (in the correct order) and see
      // if we can find a match that way
      StreamString arch_names;
      for (uint32_t idx = 0; GetSupportedArchitectureAtIndex(
               idx, resolved_module_spec.GetArchitecture());
           ++idx) {
        error = ModuleList::GetSharedModule(resolved_module_spec, exe_module_sp,
                                            module_search_paths_ptr, nullptr, nullptr);
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
        if (FileSystem::Instance().Readable(
                resolved_module_spec.GetFileSpec())) {
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
    }
  }

  return error;
}

Status RemoteAwarePlatform::RunShellCommand(
    const char *command, const FileSpec &working_dir, int *status_ptr,
    int *signo_ptr, std::string *command_output,
    const Timeout<std::micro> &timeout) {
  if (IsHost())
    return Host::RunShellCommand(command, working_dir, status_ptr, signo_ptr,
                                 command_output, timeout);
  if (m_remote_platform_sp)
    return m_remote_platform_sp->RunShellCommand(
        command, working_dir, status_ptr, signo_ptr, command_output, timeout);
  return Status("unable to run a remote command without a platform");
}

Status RemoteAwarePlatform::MakeDirectory(const FileSpec &file_spec,
                                          uint32_t file_permissions) {
  if (m_remote_platform_sp)
    return m_remote_platform_sp->MakeDirectory(file_spec, file_permissions);
  return Platform::MakeDirectory(file_spec, file_permissions);
}

Status RemoteAwarePlatform::GetFilePermissions(const FileSpec &file_spec,
                                               uint32_t &file_permissions) {
  if (m_remote_platform_sp)
    return m_remote_platform_sp->GetFilePermissions(file_spec,
                                                    file_permissions);
  return Platform::GetFilePermissions(file_spec, file_permissions);
}

Status RemoteAwarePlatform::SetFilePermissions(const FileSpec &file_spec,
                                               uint32_t file_permissions) {
  if (m_remote_platform_sp)
    return m_remote_platform_sp->SetFilePermissions(file_spec,
                                                    file_permissions);
  return Platform::SetFilePermissions(file_spec, file_permissions);
}

lldb::user_id_t RemoteAwarePlatform::OpenFile(const FileSpec &file_spec,
                                              File::OpenOptions flags,
                                              uint32_t mode, Status &error) {
  if (IsHost())
    return FileCache::GetInstance().OpenFile(file_spec, flags, mode, error);
  if (m_remote_platform_sp)
    return m_remote_platform_sp->OpenFile(file_spec, flags, mode, error);
  return Platform::OpenFile(file_spec, flags, mode, error);
}

bool RemoteAwarePlatform::CloseFile(lldb::user_id_t fd, Status &error) {
  if (IsHost())
    return FileCache::GetInstance().CloseFile(fd, error);
  if (m_remote_platform_sp)
    return m_remote_platform_sp->CloseFile(fd, error);
  return Platform::CloseFile(fd, error);
}

uint64_t RemoteAwarePlatform::ReadFile(lldb::user_id_t fd, uint64_t offset,
                                       void *dst, uint64_t dst_len,
                                       Status &error) {
  if (IsHost())
    return FileCache::GetInstance().ReadFile(fd, offset, dst, dst_len, error);
  if (m_remote_platform_sp)
    return m_remote_platform_sp->ReadFile(fd, offset, dst, dst_len, error);
  return Platform::ReadFile(fd, offset, dst, dst_len, error);
}

uint64_t RemoteAwarePlatform::WriteFile(lldb::user_id_t fd, uint64_t offset,
                                        const void *src, uint64_t src_len,
                                        Status &error) {
  if (IsHost())
    return FileCache::GetInstance().WriteFile(fd, offset, src, src_len, error);
  if (m_remote_platform_sp)
    return m_remote_platform_sp->WriteFile(fd, offset, src, src_len, error);
  return Platform::WriteFile(fd, offset, src, src_len, error);
}

lldb::user_id_t RemoteAwarePlatform::GetFileSize(const FileSpec &file_spec) {
  if (IsHost()) {
    uint64_t Size;
    if (llvm::sys::fs::file_size(file_spec.GetPath(), Size))
      return 0;
    return Size;
  }
  if (m_remote_platform_sp)
    return m_remote_platform_sp->GetFileSize(file_spec);
  return Platform::GetFileSize(file_spec);
}

Status RemoteAwarePlatform::CreateSymlink(const FileSpec &src,
                                          const FileSpec &dst) {
  if (IsHost())
    return FileSystem::Instance().Symlink(src, dst);
  if (m_remote_platform_sp)
    return m_remote_platform_sp->CreateSymlink(src, dst);
  return Platform::CreateSymlink(src, dst);
}

bool RemoteAwarePlatform::GetFileExists(const FileSpec &file_spec) {
  if (IsHost())
    return FileSystem::Instance().Exists(file_spec);
  if (m_remote_platform_sp)
    return m_remote_platform_sp->GetFileExists(file_spec);
  return Platform::GetFileExists(file_spec);
}

Status RemoteAwarePlatform::Unlink(const FileSpec &file_spec) {
  if (IsHost())
    return llvm::sys::fs::remove(file_spec.GetPath());
  if (m_remote_platform_sp)
    return m_remote_platform_sp->Unlink(file_spec);
  return Platform::Unlink(file_spec);
}

bool RemoteAwarePlatform::CalculateMD5(const FileSpec &file_spec, uint64_t &low,
                                       uint64_t &high) {
  if (IsHost())
    return Platform::CalculateMD5(file_spec, low, high);
  if (m_remote_platform_sp)
    return m_remote_platform_sp->CalculateMD5(file_spec, low, high);
  return false;
}

FileSpec RemoteAwarePlatform::GetRemoteWorkingDirectory() {
  if (IsRemote() && m_remote_platform_sp)
    return m_remote_platform_sp->GetRemoteWorkingDirectory();
  return Platform::GetRemoteWorkingDirectory();
}

bool RemoteAwarePlatform::SetRemoteWorkingDirectory(
    const FileSpec &working_dir) {
  if (IsRemote() && m_remote_platform_sp)
    return m_remote_platform_sp->SetRemoteWorkingDirectory(working_dir);
  return Platform::SetRemoteWorkingDirectory(working_dir);
}

Status RemoteAwarePlatform::GetFileWithUUID(const FileSpec &platform_file,
                                            const UUID *uuid_ptr,
                                            FileSpec &local_file) {
  if (IsRemote() && m_remote_platform_sp)
    return m_remote_platform_sp->GetFileWithUUID(platform_file, uuid_ptr,
                                                 local_file);

  // Default to the local case
  local_file = platform_file;
  return Status();
}

bool RemoteAwarePlatform::GetRemoteOSVersion() {
  if (m_remote_platform_sp) {
    m_os_version = m_remote_platform_sp->GetOSVersion();
    return !m_os_version.empty();
  }
  return false;
}

bool RemoteAwarePlatform::GetRemoteOSBuildString(std::string &s) {
  if (m_remote_platform_sp)
    return m_remote_platform_sp->GetRemoteOSBuildString(s);
  s.clear();
  return false;
}

bool RemoteAwarePlatform::GetRemoteOSKernelDescription(std::string &s) {
  if (m_remote_platform_sp)
    return m_remote_platform_sp->GetRemoteOSKernelDescription(s);
  s.clear();
  return false;
}

ArchSpec RemoteAwarePlatform::GetRemoteSystemArchitecture() {
  if (m_remote_platform_sp)
    return m_remote_platform_sp->GetRemoteSystemArchitecture();
  return ArchSpec();
}

const char *RemoteAwarePlatform::GetHostname() {
  if (IsHost())
    return Platform::GetHostname();
  if (m_remote_platform_sp)
    return m_remote_platform_sp->GetHostname();
  return nullptr;
}

UserIDResolver &RemoteAwarePlatform::GetUserIDResolver() {
  if (IsHost())
    return HostInfo::GetUserIDResolver();
  if (m_remote_platform_sp)
    return m_remote_platform_sp->GetUserIDResolver();
  return UserIDResolver::GetNoopResolver();
}

Environment RemoteAwarePlatform::GetEnvironment() {
  if (IsRemote()) {
    if (m_remote_platform_sp)
      return m_remote_platform_sp->GetEnvironment();
    return Environment();
  }
  return Host::GetEnvironment();
}

bool RemoteAwarePlatform::IsConnected() const {
  if (IsHost())
    return true;
  else if (m_remote_platform_sp)
    return m_remote_platform_sp->IsConnected();
  return false;
}

bool RemoteAwarePlatform::GetProcessInfo(lldb::pid_t pid,
                                         ProcessInstanceInfo &process_info) {
  if (IsHost())
    return Platform::GetProcessInfo(pid, process_info);
  if (m_remote_platform_sp)
    return m_remote_platform_sp->GetProcessInfo(pid, process_info);
  return false;
}

uint32_t
RemoteAwarePlatform::FindProcesses(const ProcessInstanceInfoMatch &match_info,
                                   ProcessInstanceInfoList &process_infos) {
  if (IsHost())
    return Platform::FindProcesses(match_info, process_infos);
  if (m_remote_platform_sp)
    return m_remote_platform_sp->FindProcesses(match_info, process_infos);
  return 0;
}

lldb::ProcessSP RemoteAwarePlatform::ConnectProcess(llvm::StringRef connect_url,
                                                    llvm::StringRef plugin_name,
                                                    Debugger &debugger,
                                                    Target *target,
                                                    Status &error) {
  if (m_remote_platform_sp)
    return m_remote_platform_sp->ConnectProcess(connect_url, plugin_name,
                                                debugger, target, error);
  return Platform::ConnectProcess(connect_url, plugin_name, debugger, target,
                                  error);
}

Status RemoteAwarePlatform::LaunchProcess(ProcessLaunchInfo &launch_info) {
  Status error;

  if (IsHost()) {
    error = Platform::LaunchProcess(launch_info);
  } else {
    if (m_remote_platform_sp)
      error = m_remote_platform_sp->LaunchProcess(launch_info);
    else
      error.SetErrorString("the platform is not currently connected");
  }
  return error;
}

Status RemoteAwarePlatform::KillProcess(const lldb::pid_t pid) {
  if (IsHost())
    return Platform::KillProcess(pid);
  if (m_remote_platform_sp)
    return m_remote_platform_sp->KillProcess(pid);
  return Status("the platform is not currently connected");
}
