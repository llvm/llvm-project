//===-- PlatformLinux.cpp -------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PlatformLinux.h"
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

// Define these constants from Linux mman.h for use when targeting remote linux
// systems even when host has different values.
#define MAP_PRIVATE 2
#define MAP_ANON 0x20

using namespace lldb;
using namespace lldb_private;
using namespace lldb_private::platform_linux;

LLDB_PLUGIN_DEFINE(PlatformLinux)

static uint32_t g_initialize_count = 0;


PlatformSP PlatformLinux::CreateInstance(bool force, const ArchSpec *arch) {
  Log *log(GetLogIfAllCategoriesSet(LIBLLDB_LOG_PLATFORM));
  LLDB_LOG(log, "force = {0}, arch=({1}, {2})", force,
           arch ? arch->GetArchitectureName() : "<null>",
           arch ? arch->GetTriple().getTriple() : "<null>");

  bool create = force;
  if (!create && arch && arch->IsValid()) {
    const llvm::Triple &triple = arch->GetTriple();
    switch (triple.getOS()) {
    case llvm::Triple::Linux:
      create = true;
      break;

#if defined(__linux__)
    // Only accept "unknown" for the OS if the host is linux and it "unknown"
    // wasn't specified (it was just returned because it was NOT specified)
    case llvm::Triple::OSType::UnknownOS:
      create = !arch->TripleOSWasSpecified();
      break;
#endif
    default:
      break;
    }
  }

  LLDB_LOG(log, "create = {0}", create);
  if (create) {
    return PlatformSP(new PlatformLinux(false));
  }
  return PlatformSP();
}

ConstString PlatformLinux::GetPluginNameStatic(bool is_host) {
  if (is_host) {
    static ConstString g_host_name(Platform::GetHostPlatformName());
    return g_host_name;
  } else {
    static ConstString g_remote_name("remote-linux");
    return g_remote_name;
  }
}

const char *PlatformLinux::GetPluginDescriptionStatic(bool is_host) {
  if (is_host)
    return "Local Linux user platform plug-in.";
  else
    return "Remote Linux user platform plug-in.";
}

ConstString PlatformLinux::GetPluginName() {
  return GetPluginNameStatic(IsHost());
}

void PlatformLinux::Initialize() {
  PlatformPOSIX::Initialize();

  if (g_initialize_count++ == 0) {
#if defined(__linux__) && !defined(__ANDROID__)
    PlatformSP default_platform_sp(new PlatformLinux(true));
    default_platform_sp->SetSystemArchitecture(HostInfo::GetArchitecture());
    Platform::SetHostPlatform(default_platform_sp);
#endif
    PluginManager::RegisterPlugin(
        PlatformLinux::GetPluginNameStatic(false),
        PlatformLinux::GetPluginDescriptionStatic(false),
        PlatformLinux::CreateInstance, nullptr);
  }
}

void PlatformLinux::Terminate() {
  if (g_initialize_count > 0) {
    if (--g_initialize_count == 0) {
      PluginManager::UnregisterPlugin(PlatformLinux::CreateInstance);
    }
  }

  PlatformPOSIX::Terminate();
}

/// Default Constructor
PlatformLinux::PlatformLinux(bool is_host)
    : PlatformPOSIX(is_host) // This is the local host platform
{}

PlatformLinux::~PlatformLinux() = default;

bool PlatformLinux::GetSupportedArchitectureAtIndex(uint32_t idx,
                                                    ArchSpec &arch) {
  if (IsHost()) {
    ArchSpec hostArch = HostInfo::GetArchitecture(HostInfo::eArchKindDefault);
    if (hostArch.GetTriple().isOSLinux()) {
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
    // Set the OS to linux
    triple.setOS(llvm::Triple::Linux);
    // Set the architecture
    switch (idx) {
    case 0:
      triple.setArchName("x86_64");
      break;
    case 1:
      triple.setArchName("i386");
      break;
    case 2:
      triple.setArchName("arm");
      break;
    case 3:
      triple.setArchName("aarch64");
      break;
    case 4:
      triple.setArchName("mips64");
      break;
    case 5:
      triple.setArchName("hexagon");
      break;
    case 6:
      triple.setArchName("mips");
      break;
    case 7:
      triple.setArchName("mips64el");
      break;
    case 8:
      triple.setArchName("mipsel");
      break;
    case 9:
      triple.setArchName("s390x");
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

void PlatformLinux::GetStatus(Stream &strm) {
  Platform::GetStatus(strm);

#if LLDB_ENABLE_POSIX
  // Display local kernel information only when we are running in host mode.
  // Otherwise, we would end up printing non-Linux information (when running on
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

int32_t
PlatformLinux::GetResumeCountForLaunchInfo(ProcessLaunchInfo &launch_info) {
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

bool PlatformLinux::CanDebugProcess() {
  if (IsHost()) {
    return true;
  } else {
    // If we're connected, we can debug.
    return IsConnected();
  }
}

// For local debugging, Linux will override the debug logic to use llgs-launch
// rather than lldb-launch, llgs-attach.  This differs from current lldb-
// launch, debugserver-attach approach on MacOSX.
lldb::ProcessSP
PlatformLinux::DebugProcess(ProcessLaunchInfo &launch_info, Debugger &debugger,
                            Target *target, // Can be NULL, if NULL create a new
                                            // target, else use existing one
                            Status &error) {
  Log *log(GetLogIfAllCategoriesSet(LIBLLDB_LOG_PLATFORM));
  LLDB_LOG(log, "target {0}", target);

  // If we're a remote host, use standard behavior from parent class.
  if (!IsHost())
    return PlatformPOSIX::DebugProcess(launch_info, debugger, target, error);

  //
  // For local debugging, we'll insist on having ProcessGDBRemote create the
  // process.
  //

  ProcessSP process_sp;

  // Make sure we stop at the entry point
  launch_info.GetFlags().Set(eLaunchFlagDebug);

  // We always launch the process we are going to debug in a separate process
  // group, since then we can handle ^C interrupts ourselves w/o having to
  // worry about the target getting them as well.
  launch_info.SetLaunchInSeparateProcessGroup(true);

  // Ensure we have a target.
  if (target == nullptr) {
    LLDB_LOG(log, "creating new target");
    TargetSP new_target_sp;
    error = debugger.GetTargetList().CreateTarget(
        debugger, "", "", eLoadDependentsNo, nullptr, new_target_sp);
    if (error.Fail()) {
      LLDB_LOG(log, "failed to create new target: {0}", error);
      return process_sp;
    }

    target = new_target_sp.get();
    if (!target) {
      error.SetErrorString("CreateTarget() returned nullptr");
      LLDB_LOG(log, "error: {0}", error);
      return process_sp;
    }
  }

  // Mark target as currently selected target.
  debugger.GetTargetList().SetSelectedTarget(target);

  // Now create the gdb-remote process.
  LLDB_LOG(log, "having target create process with gdb-remote plugin");
  process_sp =
      target->CreateProcess(launch_info.GetListener(), "gdb-remote", nullptr);

  if (!process_sp) {
    error.SetErrorString("CreateProcess() failed for gdb-remote process");
    LLDB_LOG(log, "error: {0}", error);
    return process_sp;
  }

  LLDB_LOG(log, "successfully created process");
  // Adjust launch for a hijacker.
  ListenerSP listener_sp;
  if (!launch_info.GetHijackListener()) {
    LLDB_LOG(log, "setting up hijacker");
    listener_sp =
        Listener::MakeListener("lldb.PlatformLinux.DebugProcess.hijack");
    launch_info.SetHijackListener(listener_sp);
    process_sp->HijackProcessEvents(listener_sp);
  }

  // Log file actions.
  if (log) {
    LLDB_LOG(log, "launching process with the following file actions:");
    StreamString stream;
    size_t i = 0;
    const FileAction *file_action;
    while ((file_action = launch_info.GetFileActionAtIndex(i++)) != nullptr) {
      file_action->Dump(stream);
      LLDB_LOG(log, "{0}", stream.GetData());
      stream.Clear();
    }
  }

  // Do the launch.
  error = process_sp->Launch(launch_info);
  if (error.Success()) {
    // Handle the hijacking of process events.
    if (listener_sp) {
      const StateType state = process_sp->WaitForProcessToStop(
          llvm::None, nullptr, false, listener_sp);

      LLDB_LOG(log, "pid {0} state {0}", process_sp->GetID(), state);
    }

    // Hook up process PTY if we have one (which we should for local debugging
    // with llgs).
    int pty_fd = launch_info.GetPTY().ReleasePrimaryFileDescriptor();
    if (pty_fd != PseudoTerminal::invalid_fd) {
      process_sp->SetSTDIOFileDescriptor(pty_fd);
      LLDB_LOG(log, "hooked up STDIO pty to process");
    } else
      LLDB_LOG(log, "not using process STDIO pty");
  } else {
    LLDB_LOG(log, "{0}", error);
    // FIXME figure out appropriate cleanup here.  Do we delete the target? Do
    // we delete the process?  Does our caller do that?
  }

  return process_sp;
}

void PlatformLinux::CalculateTrapHandlerSymbolNames() {
  m_trap_handlers.push_back(ConstString("_sigtramp"));
  m_trap_handlers.push_back(ConstString("__kernel_rt_sigreturn"));
  m_trap_handlers.push_back(ConstString("__restore_rt"));
}

MmapArgList PlatformLinux::GetMmapArgumentList(const ArchSpec &arch,
                                               addr_t addr, addr_t length,
                                               unsigned prot, unsigned flags,
                                               addr_t fd, addr_t offset) {
  uint64_t flags_platform = 0;
  uint64_t map_anon = arch.IsMIPS() ? 0x800 : MAP_ANON;

  if (flags & eMmapFlagsPrivate)
    flags_platform |= MAP_PRIVATE;
  if (flags & eMmapFlagsAnon)
    flags_platform |= map_anon;

  MmapArgList args({addr, length, prot, flags_platform, fd, offset});
  return args;
}

