//===-- PlatformNetBSD.cpp ------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PlatformNetBSD.h"
#include "lldb/Host/Config.h"

#include <stdio.h>
#if LLDB_ENABLE_POSIX
#include <sys/utsname.h>
#endif

#include "lldb/Core/Debugger.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/Host/HostInfo.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/Target.h"
#include "lldb/Utility/FileSpec.h"
#include "lldb/Utility/Log.h"
#include "lldb/Utility/State.h"
#include "lldb/Utility/Status.h"
#include "lldb/Utility/StreamString.h"

// Define these constants from NetBSD mman.h for use when targeting remote
// netbsd systems even when host has different values.
#define MAP_PRIVATE 0x0002
#define MAP_ANON 0x1000

using namespace lldb;
using namespace lldb_private;
using namespace lldb_private::platform_netbsd;

LLDB_PLUGIN_DEFINE(PlatformNetBSD)

static uint32_t g_initialize_count = 0;


PlatformSP PlatformNetBSD::CreateInstance(bool force, const ArchSpec *arch) {
  Log *log(GetLogIfAllCategoriesSet(LIBLLDB_LOG_PLATFORM));
  LLDB_LOG(log, "force = {0}, arch=({1}, {2})", force,
           arch ? arch->GetArchitectureName() : "<null>",
           arch ? arch->GetTriple().getTriple() : "<null>");

  bool create = force;
  if (!create && arch && arch->IsValid()) {
    const llvm::Triple &triple = arch->GetTriple();
    switch (triple.getOS()) {
    case llvm::Triple::NetBSD:
      create = true;
      break;

    default:
      break;
    }
  }

  LLDB_LOG(log, "create = {0}", create);
  if (create) {
    return PlatformSP(new PlatformNetBSD(false));
  }
  return PlatformSP();
}

ConstString PlatformNetBSD::GetPluginNameStatic(bool is_host) {
  if (is_host) {
    static ConstString g_host_name(Platform::GetHostPlatformName());
    return g_host_name;
  } else {
    static ConstString g_remote_name("remote-netbsd");
    return g_remote_name;
  }
}

const char *PlatformNetBSD::GetPluginDescriptionStatic(bool is_host) {
  if (is_host)
    return "Local NetBSD user platform plug-in.";
  else
    return "Remote NetBSD user platform plug-in.";
}

ConstString PlatformNetBSD::GetPluginName() {
  return GetPluginNameStatic(IsHost());
}

void PlatformNetBSD::Initialize() {
  PlatformPOSIX::Initialize();

  if (g_initialize_count++ == 0) {
#if defined(__NetBSD__)
    PlatformSP default_platform_sp(new PlatformNetBSD(true));
    default_platform_sp->SetSystemArchitecture(HostInfo::GetArchitecture());
    Platform::SetHostPlatform(default_platform_sp);
#endif
    PluginManager::RegisterPlugin(
        PlatformNetBSD::GetPluginNameStatic(false),
        PlatformNetBSD::GetPluginDescriptionStatic(false),
        PlatformNetBSD::CreateInstance, nullptr);
  }
}

void PlatformNetBSD::Terminate() {
  if (g_initialize_count > 0) {
    if (--g_initialize_count == 0) {
      PluginManager::UnregisterPlugin(PlatformNetBSD::CreateInstance);
    }
  }

  PlatformPOSIX::Terminate();
}

/// Default Constructor
PlatformNetBSD::PlatformNetBSD(bool is_host)
    : PlatformPOSIX(is_host) // This is the local host platform
{}

bool PlatformNetBSD::GetSupportedArchitectureAtIndex(uint32_t idx,
                                                     ArchSpec &arch) {
  if (IsHost()) {
    ArchSpec hostArch = HostInfo::GetArchitecture(HostInfo::eArchKindDefault);
    if (hostArch.GetTriple().isOSNetBSD()) {
      if (idx == 0) {
        arch = hostArch;
        return arch.IsValid();
      } else if (idx == 1) {
        // If the default host architecture is 64-bit, look for a 32-bit
        // variant
        if (hostArch.IsValid() && hostArch.GetTriple().isArch64Bit()) {
          arch = HostInfo::GetArchitecture(HostInfo::eArchKind32);
          return arch.IsValid();
        }
      }
    }
  } else {
    if (m_remote_platform_sp)
      return m_remote_platform_sp->GetSupportedArchitectureAtIndex(idx, arch);

    llvm::Triple triple;
    // Set the OS to NetBSD
    triple.setOS(llvm::Triple::NetBSD);
    // Set the architecture
    switch (idx) {
    case 0:
      triple.setArchName("x86_64");
      break;
    case 1:
      triple.setArchName("i386");
      break;
    default:
      return false;
    }
    // Leave the vendor as "llvm::Triple:UnknownVendor" and don't specify the
    // vendor by calling triple.SetVendorName("unknown") so that it is a
    // "unspecified unknown". This means when someone calls
    // triple.GetVendorName() it will return an empty string which indicates
    // that the vendor can be set when two architectures are merged

    // Now set the triple into "arch" and return true
    arch.SetTriple(triple);
    return true;
  }
  return false;
}

void PlatformNetBSD::GetStatus(Stream &strm) {
  Platform::GetStatus(strm);

#if LLDB_ENABLE_POSIX
  // Display local kernel information only when we are running in host mode.
  // Otherwise, we would end up printing non-NetBSD information (when running
  // on Mac OS for example).
  if (IsHost()) {
    struct utsname un;

    if (uname(&un))
      return;

    strm.Printf("    Kernel: %s\n", un.sysname);
    strm.Printf("   Release: %s\n", un.release);
    strm.Printf("   Version: %s\n", un.version);
  }
#endif
}

int32_t
PlatformNetBSD::GetResumeCountForLaunchInfo(ProcessLaunchInfo &launch_info) {
  int32_t resume_count = 0;

  // Always resume past the initial stop when we use eLaunchFlagDebug
  if (launch_info.GetFlags().Test(eLaunchFlagDebug)) {
    // Resume past the stop for the final exec into the true inferior.
    ++resume_count;
  }

  // If we're not launching a shell, we're done.
  const FileSpec &shell = launch_info.GetShell();
  if (!shell)
    return resume_count;

  std::string shell_string = shell.GetPath();
  // We're in a shell, so for sure we have to resume past the shell exec.
  ++resume_count;

  // Figure out what shell we're planning on using.
  const char *shell_name = strrchr(shell_string.c_str(), '/');
  if (shell_name == nullptr)
    shell_name = shell_string.c_str();
  else
    shell_name++;

  if (strcmp(shell_name, "csh") == 0 || strcmp(shell_name, "tcsh") == 0 ||
      strcmp(shell_name, "zsh") == 0 || strcmp(shell_name, "sh") == 0) {
    // These shells seem to re-exec themselves.  Add another resume.
    ++resume_count;
  }

  return resume_count;
}

bool PlatformNetBSD::CanDebugProcess() {
  if (IsHost()) {
    return true;
  } else {
    // If we're connected, we can debug.
    return IsConnected();
  }
}

void PlatformNetBSD::CalculateTrapHandlerSymbolNames() {
  m_trap_handlers.push_back(ConstString("_sigtramp"));
}

MmapArgList PlatformNetBSD::GetMmapArgumentList(const ArchSpec &arch,
                                                addr_t addr, addr_t length,
                                                unsigned prot, unsigned flags,
                                                addr_t fd, addr_t offset) {
  uint64_t flags_platform = 0;

  if (flags & eMmapFlagsPrivate)
    flags_platform |= MAP_PRIVATE;
  if (flags & eMmapFlagsAnon)
    flags_platform |= MAP_ANON;

  MmapArgList args({addr, length, prot, flags_platform, fd, offset});
  return args;
}
