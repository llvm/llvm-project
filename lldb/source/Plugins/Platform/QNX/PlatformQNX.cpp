//===-- PlatformQNX.cpp ---------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PlatformQNX.h"
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

// Define these constants from QNX mman.h for use when targeting remote QNX
// systems even when host has different values.
#define MAP_PRIVATE 0x00000002
#define MAP_ANON 0x00080000

using namespace lldb;
using namespace lldb_private;
using namespace lldb_private::platform_qnx;

LLDB_PLUGIN_DEFINE(PlatformQNX)

static uint32_t g_initialize_count = 0;

PlatformSP PlatformQNX::CreateInstance(bool force, const ArchSpec *arch) {
  Log *log = GetLog(LLDBLog::Platform);
  LLDB_LOG(log, "force = {0}, arch=({1}, {2})", force,
           arch ? arch->GetArchitectureName() : "<null>",
           arch ? arch->GetTriple().getTriple() : "<null>");

  bool create = force;
  if (!create && arch && arch->IsValid()) {
    const llvm::Triple &triple = arch->GetTriple();
    switch (triple.getOS()) {
    case llvm::Triple::QNX:
      create = true;
      break;
    default:
      break;
    }
  }

  LLDB_LOG(log, "create = {0}", create);
  if (create) {
    return PlatformSP(new PlatformQNX(false));
  }
  return PlatformSP();
}

llvm::StringRef PlatformQNX::GetPluginDescriptionStatic(bool is_host) {
  if (is_host)
    return "Local QNX user platform plug-in.";
  return "Remote QNX user platform plug-in.";
}

void PlatformQNX::Initialize() {
  PlatformPOSIX::Initialize();

  if (g_initialize_count++ == 0) {
#if defined(__QNX__)
    PlatformSP default_platform_sp(new PlatformQNX(true));
    default_platform_sp->SetSystemArchitecture(HostInfo::GetArchitecture());
    Platform::SetHostPlatform(default_platform_sp);
#endif
    PluginManager::RegisterPlugin(
        PlatformQNX::GetPluginNameStatic(false),
        PlatformQNX::GetPluginDescriptionStatic(false),
        PlatformQNX::CreateInstance, nullptr);
  }
}

void PlatformQNX::Terminate() {
  if (g_initialize_count > 0) {
    if (--g_initialize_count == 0) {
      PluginManager::UnregisterPlugin(PlatformQNX::CreateInstance);
    }
  }

  PlatformPOSIX::Terminate();
}

/// Default Constructor
PlatformQNX::PlatformQNX(bool is_host)
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
    m_supported_architectures = CreateArchList(
        {llvm::Triple::x86_64, llvm::Triple::arm, llvm::Triple::aarch64},
        llvm::Triple::QNX);
  }
}

std::vector<ArchSpec>
PlatformQNX::GetSupportedArchitectures(const ArchSpec &process_host_arch) {
  if (m_remote_platform_sp)
    return m_remote_platform_sp->GetSupportedArchitectures(process_host_arch);
  return m_supported_architectures;
}

void PlatformQNX::GetStatus(Stream &strm) {
  Platform::GetStatus(strm);

#if LLDB_ENABLE_POSIX
  // Display local kernel information only when we are running in host mode.
  // Otherwise, we would end up printing non-QNX information (when running on
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

uint32_t
PlatformQNX::GetResumeCountForLaunchInfo(ProcessLaunchInfo &launch_info) {
  uint32_t resume_count = 0;

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

bool PlatformQNX::CanDebugProcess() {
  if (IsHost()) {
    return true;
  } else {
    // If we're connected, we can debug.
    return IsConnected();
  }
}

void PlatformQNX::CalculateTrapHandlerSymbolNames() {
  m_trap_handlers.push_back(ConstString("_sigtramp"));
}

MmapArgList PlatformQNX::GetMmapArgumentList(const ArchSpec &arch, addr_t addr,
                                             addr_t length, unsigned prot,
                                             unsigned flags, addr_t fd,
                                             addr_t offset) {
  uint64_t flags_platform = 0;

  if (flags & eMmapFlagsPrivate)
    flags_platform |= MAP_PRIVATE;
  if (flags & eMmapFlagsAnon)
    flags_platform |= MAP_ANON;

  MmapArgList args({addr, length, prot, flags_platform, fd, offset});
  return args;
}

CompilerType PlatformQNX::GetSiginfoType(const llvm::Triple &triple) {
  {
    std::lock_guard<std::mutex> guard(m_mutex);
    if (!m_type_system)
      m_type_system = std::make_shared<TypeSystemClang>("siginfo", triple);
  }
  TypeSystemClang *ast = m_type_system.get();

  // generic types
  CompilerType int_type = ast->GetBasicType(eBasicTypeInt);
  CompilerType uint_type = ast->GetBasicType(eBasicTypeUnsignedInt);
  CompilerType short_type = ast->GetBasicType(eBasicTypeShort);
  CompilerType long_type = ast->GetBasicType(eBasicTypeLong);
  CompilerType voidp_type = ast->GetBasicType(eBasicTypeVoid).GetPointerType();

  // platform-specific types
  CompilerType &pid_type = int_type;
  CompilerType &uid_type = uint_type;
  CompilerType &clock_type = long_type;

  CompilerType sigval_type = ast->CreateRecordType(
      nullptr, OptionalClangModuleID(), lldb::eAccessPublic, "sigval",
      llvm::to_underlying(clang::TagTypeKind::Union), lldb::eLanguageTypeC);
  ast->StartTagDeclarationDefinition(sigval_type);
  ast->AddFieldToRecordType(sigval_type, "sival_int", int_type,
                            lldb::eAccessPublic, 0);
  ast->AddFieldToRecordType(sigval_type, "sival_ptr", voidp_type,
                            lldb::eAccessPublic, 0);
  ast->CompleteTagDeclarationDefinition(sigval_type);

  // siginfo_t
  CompilerType siginfo_type = ast->CreateRecordType(
      nullptr, OptionalClangModuleID(), lldb::eAccessPublic, "__lldb_siginfo_t",
      llvm::to_underlying(clang::TagTypeKind::Struct), lldb::eLanguageTypeC);
  ast->StartTagDeclarationDefinition(siginfo_type);
  ast->AddFieldToRecordType(siginfo_type, "si_signo", int_type,
                            lldb::eAccessPublic, 0);
  ast->AddFieldToRecordType(siginfo_type, "si_code", int_type,
                            lldb::eAccessPublic, 0);
  ast->AddFieldToRecordType(siginfo_type, "si_errno", int_type,
                            lldb::eAccessPublic, 0);

  // union used to hold the signal data
  CompilerType union_type = ast->CreateRecordType(
      nullptr, OptionalClangModuleID(), lldb::eAccessPublic, "",
      llvm::to_underlying(clang::TagTypeKind::Union), lldb::eLanguageTypeC);
  ast->StartTagDeclarationDefinition(union_type);

  ast->AddFieldToRecordType(union_type, "__pad", int_type.GetArrayType(7),
                            lldb::eAccessPublic, 0);

  // union used to hold __kill or __chld
  CompilerType pdata_union_type = ast->CreateRecordType(
      nullptr, OptionalClangModuleID(), lldb::eAccessPublic, "",
      llvm::to_underlying(clang::TagTypeKind::Union), lldb::eLanguageTypeC);
  ast->StartTagDeclarationDefinition(pdata_union_type);

  ast->AddFieldToRecordType(
      pdata_union_type, "__kill",
      ast->CreateStructForIdentifier(llvm::StringRef(),
                                     {
                                         {"__uid", uid_type},
                                         {"__value", sigval_type},
                                     }),
      lldb::eAccessPublic, 0);

  ast->AddFieldToRecordType(
      pdata_union_type, "__chld",
      ast->CreateStructForIdentifier(llvm::StringRef(),
                                     {
                                         {"__utime", clock_type},
                                         {"__status", int_type},
                                         {"__stime", clock_type},
                                     }),
      lldb::eAccessPublic, 0);

  ast->CompleteTagDeclarationDefinition(pdata_union_type);

  ast->AddFieldToRecordType(
      union_type, "__proc",
      ast->CreateStructForIdentifier(llvm::StringRef(),
                                     {
                                         {"__pid", pid_type},
                                         {"__pdata", pdata_union_type},
                                     }),
      lldb::eAccessPublic, 0);

  ast->AddFieldToRecordType(
      union_type, "__fault",
      ast->CreateStructForIdentifier(llvm::StringRef(),
                                     {
                                         {"__fltno", int_type},
                                         {"__fltip", voidp_type},
                                         {"__addr", voidp_type},
                                         {"__bdslot", int_type},
                                     }),
      lldb::eAccessPublic, 0);

  ast->CompleteTagDeclarationDefinition(union_type);
  ast->AddFieldToRecordType(siginfo_type, "__data", union_type,
                            lldb::eAccessPublic, 0);

  ast->CompleteTagDeclarationDefinition(siginfo_type);
  return siginfo_type;
}
