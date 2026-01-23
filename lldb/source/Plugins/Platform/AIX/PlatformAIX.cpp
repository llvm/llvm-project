//===-- PlatformAIX.cpp -------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PlatformAIX.h"
#include "lldb/Host/Config.h"
#include <cstdio>
#if LLDB_ENABLE_POSIX
#include <sys/utsname.h>
#endif
#include "Utility/ARM64_DWARF_Registers.h"
#include "lldb/Core/Debugger.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/Host/HostInfo.h"
#include "lldb/Symbol/UnwindPlan.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/Target.h"
#include "lldb/Utility/FileSpec.h"
#include "lldb/Utility/LLDBLog.h"
#include "lldb/Utility/Log.h"
#include "lldb/Utility/State.h"
#include "lldb/Utility/Status.h"
#include "lldb/Utility/StreamString.h"

// Use defined constants from AIX mman.h for use when targeting remote aix
// systems even when host has different values.

// For remotely cross debugging aix
constexpr int MapVariable = 0x0;
constexpr int MapPrivate = 0x2;
constexpr int MapAnonymous = 0x10;
#if defined(_AIX)
#include <sys/mman.h>
static_assert(MapVariable == MAP_VARIABLE);
static_assert(MapPrivate == MAP_PRIVATE);
static_assert(MapAnonymous == MAP_ANONYMOUS);
#endif

using namespace lldb;
using namespace lldb_private;
using namespace lldb_private::platform_aix;

LLDB_PLUGIN_DEFINE(PlatformAIX)

static uint32_t g_initialize_count = 0;

PlatformSP PlatformAIX::CreateInstance(bool force, const ArchSpec *arch) {
  Log *log = GetLog(LLDBLog::Platform);
  LLDB_LOG(log, "force = {0}, arch=({1}, {2})", force,
           arch ? arch->GetArchitectureName() : "<null>",
           arch ? arch->GetTriple().getTriple() : "<null>");

  bool create = force || (arch && arch->IsValid() &&
                          arch->GetTriple().getOS() == llvm::Triple::AIX);
  LLDB_LOG(log, "create = {0}", create);
  if (create) {
    return PlatformSP(new PlatformAIX(false));
  }
  return PlatformSP();
}

llvm::StringRef PlatformAIX::GetPluginDescriptionStatic(bool is_host) {
  if (is_host)
    return "Local AIX user platform plug-in.";
  return "Remote AIX user platform plug-in.";
}

void PlatformAIX::Initialize() {
  PlatformPOSIX::Initialize();

  if (g_initialize_count++ == 0) {
#ifdef _AIX
    PlatformSP default_platform_sp(new PlatformAIX(true));
    default_platform_sp->SetSystemArchitecture(HostInfo::GetArchitecture());
    Platform::SetHostPlatform(default_platform_sp);
#endif
    PluginManager::RegisterPlugin(
        PlatformAIX::GetPluginNameStatic(false),
        PlatformAIX::GetPluginDescriptionStatic(false),
        PlatformAIX::CreateInstance, nullptr);
  }
}

void PlatformAIX::Terminate() {
  if (g_initialize_count > 0)
    if (--g_initialize_count == 0)
      PluginManager::UnregisterPlugin(PlatformAIX::CreateInstance);

  PlatformPOSIX::Terminate();
}

PlatformAIX::PlatformAIX(bool is_host) : PlatformPOSIX(is_host) {
  if (is_host) {
    ArchSpec hostArch = HostInfo::GetArchitecture(HostInfo::eArchKindDefault);
    m_supported_architectures.push_back(hostArch);
  } else {
    m_supported_architectures =
        CreateArchList({llvm::Triple::ppc64}, llvm::Triple::AIX);
  }
}

std::vector<ArchSpec>
PlatformAIX::GetSupportedArchitectures(const ArchSpec &process_host_arch) {
  if (m_remote_platform_sp)
    return m_remote_platform_sp->GetSupportedArchitectures(process_host_arch);
  return m_supported_architectures;
}

void PlatformAIX::GetStatus(Stream &strm) {
  Platform::GetStatus(strm);

#if LLDB_ENABLE_POSIX
  // Display local kernel information only when we are running in host mode.
  // Otherwise, we would end up printing non-AIX information (when running on
  // Mac OS for example).
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

void PlatformAIX::CalculateTrapHandlerSymbolNames() {}

lldb::UnwindPlanSP PlatformAIX::GetTrapHandlerUnwindPlan(const ArchSpec &arch,
                                                         ConstString name) {
  return {};
}

MmapArgList PlatformAIX::GetMmapArgumentList(const ArchSpec &arch, addr_t addr,
                                             addr_t length, unsigned prot,
                                             unsigned flags, addr_t fd,
                                             addr_t offset) {
  unsigned flags_platform = MapVariable;

  if (flags & eMmapFlagsPrivate)
    flags_platform |= MapPrivate;
  if (flags & eMmapFlagsAnon)
    flags_platform |= MapAnonymous;

  MmapArgList args({addr, length, prot, flags_platform, fd, offset});
  return args;
}

CompilerType PlatformAIX::GetSiginfoType(const llvm::Triple &triple) {
  return CompilerType();
}
