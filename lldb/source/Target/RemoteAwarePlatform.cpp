//===-- RemoteAwarePlatform.cpp ---------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Target/RemoteAwarePlatform.h"
#include "lldb/Host/Host.h"

using namespace lldb_private;

bool RemoteAwarePlatform::GetModuleSpec(const FileSpec &module_file_spec,
                                        const ArchSpec &arch,
                                        ModuleSpec &module_spec) {
  if (m_remote_platform_sp)
    return m_remote_platform_sp->GetModuleSpec(module_file_spec, arch,
                                               module_spec);

  return Platform::GetModuleSpec(module_file_spec, arch, module_spec);
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

const char *RemoteAwarePlatform::GetUserName(uint32_t uid) {
  // Check the cache in Platform in case we have already looked this uid up
  const char *user_name = Platform::GetUserName(uid);
  if (user_name)
    return user_name;

  if (IsRemote() && m_remote_platform_sp)
    return m_remote_platform_sp->GetUserName(uid);
  return nullptr;
}

const char *RemoteAwarePlatform::GetGroupName(uint32_t gid) {
  const char *group_name = Platform::GetGroupName(gid);
  if (group_name)
    return group_name;

  if (IsRemote() && m_remote_platform_sp)
    return m_remote_platform_sp->GetGroupName(gid);
  return nullptr;
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
