//===-- HostInfoPosix.cpp ---------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Host/posix/HostInfoPosix.h"
#include "lldb/Utility/Log.h"

#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

#include <grp.h>
#include <limits.h>
#include <mutex>
#include <netdb.h>
#include <pwd.h>
#include <stdlib.h>
#include <sys/types.h>
#include <unistd.h>

using namespace lldb_private;

size_t HostInfoPosix::GetPageSize() { return ::getpagesize(); }

bool HostInfoPosix::GetHostname(std::string &s) {
  char hostname[PATH_MAX];
  hostname[sizeof(hostname) - 1] = '\0';
  if (::gethostname(hostname, sizeof(hostname) - 1) == 0) {
    struct hostent *h = ::gethostbyname(hostname);
    if (h)
      s.assign(h->h_name);
    else
      s.assign(hostname);
    return true;
  }
  return false;
}

#ifdef __ANDROID__
#include <android/api-level.h>
#endif
#if defined(__ANDROID_API__) && __ANDROID_API__ < 21
#define USE_GETPWUID
#endif

#ifdef USE_GETPWUID
static std::mutex s_getpwuid_lock;
#endif

const char *HostInfoPosix::LookupUserName(uint32_t uid,
                                          std::string &user_name) {
#ifdef USE_GETPWUID
  // getpwuid_r is missing from android-9
  // make getpwuid thread safe with a mutex
  std::lock_guard<std::mutex> lock(s_getpwuid_lock);
  struct passwd *user_info_ptr = ::getpwuid(uid);
  if (user_info_ptr) {
    user_name.assign(user_info_ptr->pw_name);
    return user_name.c_str();
  }
#else
  struct passwd user_info;
  struct passwd *user_info_ptr = &user_info;
  char user_buffer[PATH_MAX];
  size_t user_buffer_size = sizeof(user_buffer);
  if (::getpwuid_r(uid, &user_info, user_buffer, user_buffer_size,
                   &user_info_ptr) == 0) {
    if (user_info_ptr) {
      user_name.assign(user_info_ptr->pw_name);
      return user_name.c_str();
    }
  }
#endif
  user_name.clear();
  return nullptr;
}

const char *HostInfoPosix::LookupGroupName(uint32_t gid,
                                           std::string &group_name) {
#ifndef __ANDROID__
  char group_buffer[PATH_MAX];
  size_t group_buffer_size = sizeof(group_buffer);
  struct group group_info;
  struct group *group_info_ptr = &group_info;
  // Try the threadsafe version first
  if (::getgrgid_r(gid, &group_info, group_buffer, group_buffer_size,
                   &group_info_ptr) == 0) {
    if (group_info_ptr) {
      group_name.assign(group_info_ptr->gr_name);
      return group_name.c_str();
    }
  } else {
    // The threadsafe version isn't currently working for me on darwin, but the
    // non-threadsafe version is, so I am calling it below.
    group_info_ptr = ::getgrgid(gid);
    if (group_info_ptr) {
      group_name.assign(group_info_ptr->gr_name);
      return group_name.c_str();
    }
  }
  group_name.clear();
#else
  assert(false && "getgrgid_r() not supported on Android");
#endif
  return NULL;
}

uint32_t HostInfoPosix::GetUserID() { return getuid(); }

uint32_t HostInfoPosix::GetGroupID() { return getgid(); }

uint32_t HostInfoPosix::GetEffectiveUserID() { return geteuid(); }

uint32_t HostInfoPosix::GetEffectiveGroupID() { return getegid(); }

FileSpec HostInfoPosix::GetDefaultShell() { return FileSpec("/bin/sh"); }

bool HostInfoPosix::ComputeSupportFileDirectory(FileSpec &file_spec) {
  FileSpec temp_file_spec;

  if (FileSpec temp_file_spec = GetShlibDir()) {
    temp_file_spec.AppendPathComponent("lldb");
    file_spec = temp_file_spec;
    return true;
  }

  return false;
}

bool HostInfoPosix::ComputeSupportExeDirectory(FileSpec &file_spec) {
  return ComputePathRelativeToLibrary(file_spec, "/bin");
}

bool HostInfoPosix::ComputeHeaderDirectory(FileSpec &file_spec) {
  FileSpec temp_file("/opt/local/include/lldb");
  file_spec.GetDirectory().SetCString(temp_file.GetPath().c_str());
  return true;
}

bool HostInfoPosix::ComputeSwiftDirectory(FileSpec &file_spec) {
  if (FileSpec lldb_file_spec = GetShlibDir()) {
    lldb_file_spec.AppendPathComponent("swift");
    file_spec = lldb_file_spec;
    return true;
  }
  return false;
}

bool HostInfoPosix::GetEnvironmentVar(const std::string &var_name,
                                      std::string &var) {
  if (const char *pvar = ::getenv(var_name.c_str())) {
    var = std::string(pvar);
    return true;
  }
  return false;
}
