//===-- HostInfoNetBSD.cpp -------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Host/netbsd/HostInfoNetBSD.h"

#include <inttypes.h>
#include <limits.h>
#include <pthread.h>
#include <stdio.h>
#include <string.h>
#include <sys/sysctl.h>
#include <sys/types.h>
#include <sys/utsname.h>
#include <unistd.h>

using namespace lldb_private;

uint32_t HostInfoNetBSD::GetMaxThreadNameLength() {
  return PTHREAD_MAX_NAMELEN_NP;
}

bool HostInfoNetBSD::GetOSVersion(uint32_t &major, uint32_t &minor,
                                  uint32_t &update) {
  struct utsname un;

  ::memset(&un, 0, sizeof(un));
  if (::uname(&un) < 0)
    return false;

  /* Accept versions like 7.99.21 and 6.1_STABLE */
  int status = ::sscanf(un.release, "%" PRIu32 ".%" PRIu32 ".%" PRIu32, &major,
                        &minor, &update);
  switch (status) {
  case 0:
    return false;
  case 1:
    minor = 0;
  /* FALLTHROUGH */
  case 2:
    update = 0;
  /* FALLTHROUGH */
  case 3:
  default:
    return true;
  }
}

bool HostInfoNetBSD::GetOSBuildString(std::string &s) {
  int mib[2] = {CTL_KERN, KERN_OSREV};
  char osrev_str[12];
  int osrev = 0;
  size_t osrev_len = sizeof(osrev);

  if (::sysctl(mib, 2, &osrev, &osrev_len, NULL, 0) == 0) {
    ::snprintf(osrev_str, sizeof(osrev_str), "%-10.10d", osrev);
    s.assign(osrev_str);
    return true;
  }

  s.clear();
  return false;
}

bool HostInfoNetBSD::GetOSKernelDescription(std::string &s) {
  struct utsname un;

  ::memset(&un, 0, sizeof(un));
  s.clear();

  if (::uname(&un) < 0)
    return false;

  s.assign(un.version);

  return true;
}

FileSpec HostInfoNetBSD::GetProgramFileSpec() {
  static FileSpec g_program_filespec;

  if (!g_program_filespec) {
    ssize_t len;
    static char buf[PATH_MAX];
    char name[PATH_MAX];

    ::snprintf(name, PATH_MAX, "/proc/%d/exe", ::getpid());
    len = ::readlink(name, buf, PATH_MAX - 1);
    if (len != -1) {
      buf[len] = '\0';
      g_program_filespec.SetFile(buf, false);
    }
  }
  return g_program_filespec;
}
