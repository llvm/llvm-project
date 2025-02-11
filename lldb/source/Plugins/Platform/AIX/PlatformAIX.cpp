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

#if defined(_AIX)
#include <sys/mman.h>
#else // For remotely cross debugging aix
#define MAP_VARIABLE 0x0
#define MAP_PRIVATE 0x2
#define MAP_ANONYMOUS 0x10
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

  bool create = force;
  if (!create && arch && arch->IsValid()) {
    const llvm::Triple &triple = arch->GetTriple();
    switch (triple.getOS()) {
    case llvm::Triple::AIX:
      create = true;
      break;

    default:
      break;
    }
  }

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
  if (g_initialize_count > 0) {
    if (--g_initialize_count == 0) {
      PluginManager::UnregisterPlugin(PlatformAIX::CreateInstance);
    }
  }

  PlatformPOSIX::Terminate();
}

/// Default Constructor
PlatformAIX::PlatformAIX(bool is_host)
    : PlatformPOSIX(is_host) // This is the local host platform
{
  if (is_host) {
    ArchSpec hostArch = HostInfo::GetArchitecture(HostInfo::eArchKindDefault);
    m_supported_architectures.push_back(hostArch);
    if (hostArch.GetTriple().isArch64Bit()) {
      m_supported_architectures.push_back(
          HostInfo::GetArchitecture(HostInfo::eArchKind32));
    }
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

bool PlatformAIX::CanDebugProcess() { return IsHost() ? true : IsConnected(); }

void PlatformAIX::CalculateTrapHandlerSymbolNames() {}

lldb::UnwindPlanSP
PlatformAIX::GetTrapHandlerUnwindPlan(const llvm::Triple &triple,
                                      ConstString name) {
  return {};
}

MmapArgList PlatformAIX::GetMmapArgumentList(const ArchSpec &arch, addr_t addr,
                                             addr_t length, unsigned prot,
                                             unsigned flags, addr_t fd,
                                             addr_t offset) {
#if defined(_AIX)
  unsigned flags_platform = MAP_VARIABLE | MAP_PRIVATE | MAP_ANONYMOUS;
#else
  unsigned flags_platform = 0;
#endif
  MmapArgList args({addr, length, prot, flags_platform, fd, offset});
  return args;
}

CompilerType PlatformAIX::GetSiginfoType(const llvm::Triple &triple) {
  if (!m_type_system_up)
    m_type_system_up.reset(new TypeSystemClang("siginfo", triple));
  TypeSystemClang *ast = m_type_system_up.get();

  // generic types
  CompilerType int_type = ast->GetBasicType(eBasicTypeInt);
  CompilerType uint_type = ast->GetBasicType(eBasicTypeUnsignedInt);
  CompilerType short_type = ast->GetBasicType(eBasicTypeShort);
  CompilerType long_type = ast->GetBasicType(eBasicTypeLong);
  CompilerType voidp_type = ast->GetBasicType(eBasicTypeVoid).GetPointerType();

  // siginfo_t
  CompilerType siginfo_type = ast->CreateRecordType(
      nullptr, OptionalClangModuleID(), lldb::eAccessPublic, "__lldb_siginfo_t",
      llvm::to_underlying(clang::TagTypeKind::Struct), lldb::eLanguageTypeC);
  ast->StartTagDeclarationDefinition(siginfo_type);

  ast->CompleteTagDeclarationDefinition(siginfo_type);
  return siginfo_type;
}
