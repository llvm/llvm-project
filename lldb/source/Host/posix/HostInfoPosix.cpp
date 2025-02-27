//===-- HostInfoPosix.cpp -------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Host/posix/HostInfoPosix.h"
#include "lldb/Host/Config.h"
#include "lldb/Host/FileSystem.h"
#include "lldb/Host/HostInfo.h"
#include "lldb/Utility/Log.h"
#include "lldb/Utility/UserIDResolver.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <grp.h>
#include <mutex>
#include <optional>
#include <pwd.h>
#include <sys/types.h>
#include <sys/utsname.h>
#include <unistd.h>

using namespace lldb_private;

namespace {
struct HostInfoPosixFields {
  llvm::once_flag m_os_version_once_flag;
  llvm::VersionTuple m_os_version;
};
} // namespace

llvm::VersionTuple HostInfoPosix::GetOSVersion() {
  static HostInfoPosixFields *g_fields = new HostInfoPosixFields();
  assert(g_fields && "Missing call to Initialize?");
  llvm::call_once(g_fields->m_os_version_once_flag, []() {
    struct utsname un;
    if (uname(&un) != 0)
      return;

    llvm::StringRef release = un.release;
    // The Linux kernel release string can include a lot of stuff (e.g.
    // 4.9.0-6-amd64). We're only interested in the numbered prefix.
    release = release.substr(0, release.find_first_not_of("0123456789."));
    g_fields->m_os_version.tryParse(release);
  });

  return g_fields->m_os_version;
}

size_t HostInfoPosix::GetPageSize() { return ::getpagesize(); }

bool HostInfoPosix::GetHostname(std::string &s) {
  char hostname[PATH_MAX];
  hostname[sizeof(hostname) - 1] = '\0';
  if (::gethostname(hostname, sizeof(hostname) - 1) == 0) {
    s.assign(hostname);
    return true;
  }
  return false;
}

std::optional<std::string> HostInfoPosix::GetOSKernelDescription() {
  struct utsname un;
  if (uname(&un) < 0)
    return std::nullopt;

  return std::string(un.version);
}

std::optional<std::string> HostInfoPosix::GetOSBuildString() {
  struct utsname un;
  ::memset(&un, 0, sizeof(utsname));

  if (uname(&un) < 0)
    return std::nullopt;

  return std::string(un.release);
}

namespace {
class PosixUserIDResolver : public UserIDResolver {
protected:
  std::optional<std::string> DoGetUserName(id_t uid) override;
  std::optional<std::string> DoGetGroupName(id_t gid) override;
};
} // namespace

struct PasswdEntry {
  std::string username;
  std::string shell;
};

static std::optional<PasswdEntry> GetPassword(id_t uid) {
  struct passwd user_info;
  struct passwd *user_info_ptr = &user_info;
  char user_buffer[PATH_MAX];
  size_t user_buffer_size = sizeof(user_buffer);
  if (::getpwuid_r(uid, &user_info, user_buffer, user_buffer_size,
                   &user_info_ptr) == 0 &&
      user_info_ptr) {
    return PasswdEntry{user_info_ptr->pw_name, user_info_ptr->pw_shell};
  }
  return std::nullopt;
}

std::optional<std::string> PosixUserIDResolver::DoGetUserName(id_t uid) {
  if (std::optional<PasswdEntry> password = GetPassword(uid))
    return password->username;
  return std::nullopt;
}

std::optional<std::string> PosixUserIDResolver::DoGetGroupName(id_t gid) {
#if !defined(__ANDROID__) || __ANDROID_API__ >= 24
  char group_buffer[PATH_MAX];
  size_t group_buffer_size = sizeof(group_buffer);
  struct group group_info;
  struct group *group_info_ptr = &group_info;
  // Try the threadsafe version first
  if (::getgrgid_r(gid, &group_info, group_buffer, group_buffer_size,
                   &group_info_ptr) == 0) {
    if (group_info_ptr)
      return std::string(group_info_ptr->gr_name);
  } else {
    // The threadsafe version isn't currently working for me on darwin, but the
    // non-threadsafe version is, so I am calling it below.
    group_info_ptr = ::getgrgid(gid);
    if (group_info_ptr)
      return std::string(group_info_ptr->gr_name);
  }
#endif
  return std::nullopt;
}

static llvm::ManagedStatic<PosixUserIDResolver> g_user_id_resolver;

UserIDResolver &HostInfoPosix::GetUserIDResolver() {
  return *g_user_id_resolver;
}

uint32_t HostInfoPosix::GetUserID() { return getuid(); }

uint32_t HostInfoPosix::GetGroupID() { return getgid(); }

uint32_t HostInfoPosix::GetEffectiveUserID() { return geteuid(); }

uint32_t HostInfoPosix::GetEffectiveGroupID() { return getegid(); }

FileSpec HostInfoPosix::GetDefaultShell() {
  if (const char *v = ::getenv("SHELL"))
    return FileSpec(v);
  if (std::optional<PasswdEntry> password = GetPassword(::geteuid()))
    return FileSpec(password->shell);
  return FileSpec("/bin/sh");
}

bool HostInfoPosix::ComputeSupportExeDirectory(FileSpec &file_spec) {
  if (ComputePathRelativeToLibrary(file_spec, "/bin") &&
      file_spec.IsAbsolute() && FileSystem::Instance().Exists(file_spec))
    return true;
  file_spec.SetDirectory(HostInfo::GetProgramFileSpec().GetDirectory());
  return !file_spec.GetDirectory().IsEmpty();
}

bool HostInfoPosix::ComputeSystemPluginsDirectory(FileSpec &file_spec) {
  FileSpec temp_file("/usr/" LLDB_INSTALL_LIBDIR_BASENAME "/lldb/plugins");
  FileSystem::Instance().Resolve(temp_file);
  file_spec.SetDirectory(temp_file.GetPath());
  return true;
}

bool HostInfoPosix::ComputeUserPluginsDirectory(FileSpec &file_spec) {
  // XDG Base Directory Specification
  // http://standards.freedesktop.org/basedir-spec/basedir-spec-latest.html If
  // XDG_DATA_HOME exists, use that, otherwise use ~/.local/share/lldb.
  const char *xdg_data_home = getenv("XDG_DATA_HOME");
  if (xdg_data_home && xdg_data_home[0]) {
    std::string user_plugin_dir(xdg_data_home);
    user_plugin_dir += "/lldb";
    file_spec.SetDirectory(user_plugin_dir.c_str());
  } else
    file_spec.SetDirectory("~/.local/share/lldb");
  return true;
}

bool HostInfoPosix::ComputeHeaderDirectory(FileSpec &file_spec) {
  FileSpec temp_file("/opt/local/include/lldb");
  file_spec.SetDirectory(temp_file.GetPath());
  return true;
}

bool HostInfoPosix::GetEnvironmentVar(const std::string &var_name,
                                      std::string &var) {
  if (const char *pvar = ::getenv(var_name.c_str())) {
    var = std::string(pvar);
    return true;
  }
  return false;
}
