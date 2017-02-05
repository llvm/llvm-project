//===-- PlatformNetBSD.cpp -------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "PlatformNetBSD.h"
#include "lldb/Host/Config.h"

// C Includes
#include <stdio.h>
#ifndef LLDB_DISABLE_POSIX
#include <sys/utsname.h>
#endif

// C++ Includes
// Other libraries and framework includes
// Project includes
#include "lldb/Core/Debugger.h"
#include "lldb/Core/Log.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/Core/State.h"
#include "lldb/Host/FileSpec.h"
#include "lldb/Host/HostInfo.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/Target.h"
#include "lldb/Utility/Error.h"
#include "lldb/Utility/StreamString.h"

// Define these constants from NetBSD mman.h for use when targeting
// remote netbsd systems even when host has different values.
#define MAP_PRIVATE 0x0002
#define MAP_ANON 0x1000

using namespace lldb;
using namespace lldb_private;
using namespace lldb_private::platform_netbsd;

static uint32_t g_initialize_count = 0;

//------------------------------------------------------------------

PlatformSP PlatformNetBSD::CreateInstance(bool force, const ArchSpec *arch) {
  Log *log(GetLogIfAllCategoriesSet(LIBLLDB_LOG_PLATFORM));
  if (log) {
    const char *arch_name;
    if (arch && arch->GetArchitectureName())
      arch_name = arch->GetArchitectureName();
    else
      arch_name = "<null>";

    const char *triple_cstr =
        arch ? arch->GetTriple().getTriple().c_str() : "<null>";

    log->Printf("PlatformNetBSD::%s(force=%s, arch={%s,%s})", __FUNCTION__,
                force ? "true" : "false", arch_name, triple_cstr);
  }

  bool create = force;
  if (create == false && arch && arch->IsValid()) {
    const llvm::Triple &triple = arch->GetTriple();
    switch (triple.getOS()) {
    case llvm::Triple::NetBSD:
      create = true;
      break;

    default:
      break;
    }
  }

  if (create) {
    if (log)
      log->Printf("PlatformNetBSD::%s() creating remote-netbsd platform",
                  __FUNCTION__);
    return PlatformSP(new PlatformNetBSD(false));
  }

  if (log)
    log->Printf(
        "PlatformNetBSD::%s() aborting creation of remote-netbsd platform",
        __FUNCTION__);

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

//------------------------------------------------------------------
/// Default Constructor
//------------------------------------------------------------------
PlatformNetBSD::PlatformNetBSD(bool is_host)
    : PlatformPOSIX(is_host) // This is the local host platform
{}

PlatformNetBSD::~PlatformNetBSD() = default;

bool PlatformNetBSD::GetSupportedArchitectureAtIndex(uint32_t idx,
                                                     ArchSpec &arch) {
  if (IsHost()) {
    ArchSpec hostArch = HostInfo::GetArchitecture(HostInfo::eArchKindDefault);
    if (hostArch.GetTriple().isOSNetBSD()) {
      if (idx == 0) {
        arch = hostArch;
        return arch.IsValid();
      } else if (idx == 1) {
        // If the default host architecture is 64-bit, look for a 32-bit variant
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
    // vendor by
    // calling triple.SetVendorName("unknown") so that it is a "unspecified
    // unknown".
    // This means when someone calls triple.GetVendorName() it will return an
    // empty string
    // which indicates that the vendor can be set when two architectures are
    // merged

    // Now set the triple into "arch" and return true
    arch.SetTriple(triple);
    return true;
  }
  return false;
}

void PlatformNetBSD::GetStatus(Stream &strm) {
  Platform::GetStatus(strm);

#ifndef LLDB_DISABLE_POSIX
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
  if (shell_name == NULL)
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

// For local debugging, NetBSD will override the debug logic to use llgs-launch
// rather than
// lldb-launch, llgs-attach.  This differs from current lldb-launch,
// debugserver-attach
// approach on MacOSX.
lldb::ProcessSP
PlatformNetBSD::DebugProcess(ProcessLaunchInfo &launch_info, Debugger &debugger,
                            Target *target, // Can be NULL, if NULL create a new
                                            // target, else use existing one
                            Error &error) {
  Log *log(GetLogIfAllCategoriesSet(LIBLLDB_LOG_PLATFORM));
  if (log)
    log->Printf("PlatformNetBSD::%s entered (target %p)", __FUNCTION__,
                static_cast<void *>(target));

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
  // group, since then we can handle ^C interrupts ourselves w/o having to worry
  // about the target getting them as well.
  launch_info.SetLaunchInSeparateProcessGroup(true);

  // Ensure we have a target.
  if (target == nullptr) {
    if (log)
      log->Printf("PlatformNetBSD::%s creating new target", __FUNCTION__);

    TargetSP new_target_sp;
    error = debugger.GetTargetList().CreateTarget(debugger, "", "", false,
                                                  nullptr, new_target_sp);
    if (error.Fail()) {
      if (log)
        log->Printf("PlatformNetBSD::%s failed to create new target: %s",
                    __FUNCTION__, error.AsCString());
      return process_sp;
    }

    target = new_target_sp.get();
    if (!target) {
      error.SetErrorString("CreateTarget() returned nullptr");
      if (log)
        log->Printf("PlatformNetBSD::%s failed: %s", __FUNCTION__,
                    error.AsCString());
      return process_sp;
    }
  } else {
    if (log)
      log->Printf("PlatformNetBSD::%s using provided target", __FUNCTION__);
  }

  // Mark target as currently selected target.
  debugger.GetTargetList().SetSelectedTarget(target);

  // Now create the gdb-remote process.
  if (log)
    log->Printf(
        "PlatformNetBSD::%s having target create process with gdb-remote plugin",
        __FUNCTION__);
  process_sp = target->CreateProcess(
      launch_info.GetListenerForProcess(debugger), "gdb-remote", nullptr);

  if (!process_sp) {
    error.SetErrorString("CreateProcess() failed for gdb-remote process");
    if (log)
      log->Printf("PlatformNetBSD::%s failed: %s", __FUNCTION__,
                  error.AsCString());
    return process_sp;
  } else {
    if (log)
      log->Printf("PlatformNetBSD::%s successfully created process",
                  __FUNCTION__);
  }

  // Adjust launch for a hijacker.
  ListenerSP listener_sp;
  if (!launch_info.GetHijackListener()) {
    if (log)
      log->Printf("PlatformNetBSD::%s setting up hijacker", __FUNCTION__);

    listener_sp =
        Listener::MakeListener("lldb.PlatformNetBSD.DebugProcess.hijack");
    launch_info.SetHijackListener(listener_sp);
    process_sp->HijackProcessEvents(listener_sp);
  }

  // Log file actions.
  if (log) {
    log->Printf(
        "PlatformNetBSD::%s launching process with the following file actions:",
        __FUNCTION__);

    StreamString stream;
    size_t i = 0;
    const FileAction *file_action;
    while ((file_action = launch_info.GetFileActionAtIndex(i++)) != nullptr) {
      file_action->Dump(stream);
      log->PutCString(stream.GetData());
      stream.Clear();
    }
  }

  // Do the launch.
  error = process_sp->Launch(launch_info);
  if (error.Success()) {
    // Handle the hijacking of process events.
    if (listener_sp) {
      const StateType state = process_sp->WaitForProcessToStop(
          llvm::None, NULL, false, listener_sp);

      if (state == eStateStopped) {
        if (log)
          log->Printf("PlatformNetBSD::%s pid %" PRIu64 " state %s\n",
                      __FUNCTION__, process_sp->GetID(), StateAsCString(state));
      } else {
        if (log)
          log->Printf("PlatformNetBSD::%s pid %" PRIu64
                      " state is not stopped - %s\n",
                      __FUNCTION__, process_sp->GetID(), StateAsCString(state));
      }
    }

    // Hook up process PTY if we have one (which we should for local debugging
    // with llgs).
    int pty_fd = launch_info.GetPTY().ReleaseMasterFileDescriptor();
    if (pty_fd != lldb_utility::PseudoTerminal::invalid_fd) {
      process_sp->SetSTDIOFileDescriptor(pty_fd);
      if (log)
        log->Printf("PlatformNetBSD::%s pid %" PRIu64
                    " hooked up STDIO pty to process",
                    __FUNCTION__, process_sp->GetID());
    } else {
      if (log)
        log->Printf("PlatformNetBSD::%s pid %" PRIu64
                    " not using process STDIO pty",
                    __FUNCTION__, process_sp->GetID());
    }
  } else {
    if (log)
      log->Printf("PlatformNetBSD::%s process launch failed: %s", __FUNCTION__,
                  error.AsCString());
    // FIXME figure out appropriate cleanup here.  Do we delete the target? Do
    // we delete the process?  Does our caller do that?
  }

  return process_sp;
}

void PlatformNetBSD::CalculateTrapHandlerSymbolNames() {
  m_trap_handlers.push_back(ConstString("_sigtramp"));
}

uint64_t PlatformNetBSD::ConvertMmapFlagsToPlatform(const ArchSpec &arch,
                                                   unsigned flags) {
  uint64_t flags_platform = 0;

  if (flags & eMmapFlagsPrivate)
    flags_platform |= MAP_PRIVATE;
  if (flags & eMmapFlagsAnon)
    flags_platform |= MAP_ANON;
  return flags_platform;
}
