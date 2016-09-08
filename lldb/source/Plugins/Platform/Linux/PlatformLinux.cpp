//===-- PlatformLinux.cpp ---------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "PlatformLinux.h"
#include "lldb/Host/Config.h"

// C Includes
#include <stdio.h>
#ifndef LLDB_DISABLE_POSIX
#include <sys/utsname.h>
#endif

// C++ Includes
// Other libraries and framework includes
// Project includes
#include "lldb/Breakpoint/BreakpointLocation.h"
#include "lldb/Core/Debugger.h"
#include "lldb/Core/Error.h"
#include "lldb/Core/Log.h"
#include "lldb/Core/Module.h"
#include "lldb/Core/ModuleList.h"
#include "lldb/Core/ModuleSpec.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/Core/State.h"
#include "lldb/Core/StreamString.h"
#include "lldb/Host/FileSpec.h"
#include "lldb/Host/HostInfo.h"
#include "lldb/Interpreter/OptionValueProperties.h"
#include "lldb/Interpreter/Property.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/Target.h"

// Define these constants from Linux mman.h for use when targeting
// remote linux systems even when host has different values.
#define MAP_PRIVATE 2
#define MAP_ANON 0x20

using namespace lldb;
using namespace lldb_private;
using namespace lldb_private::platform_linux;

static uint32_t g_initialize_count = 0;

//------------------------------------------------------------------
/// Code to handle the PlatformLinux settings
//------------------------------------------------------------------

namespace {
class PlatformLinuxProperties : public Properties {
public:
  PlatformLinuxProperties();

  ~PlatformLinuxProperties() override = default;

  static ConstString &GetSettingName();

private:
  static const PropertyDefinition *GetStaticPropertyDefinitions();
};

typedef std::shared_ptr<PlatformLinuxProperties> PlatformLinuxPropertiesSP;

} // anonymous namespace

PlatformLinuxProperties::PlatformLinuxProperties() : Properties() {
  m_collection_sp.reset(new OptionValueProperties(GetSettingName()));
  m_collection_sp->Initialize(GetStaticPropertyDefinitions());
}

ConstString &PlatformLinuxProperties::GetSettingName() {
  static ConstString g_setting_name("linux");
  return g_setting_name;
}

const PropertyDefinition *
PlatformLinuxProperties::GetStaticPropertyDefinitions() {
  static PropertyDefinition g_properties[] = {
      {NULL, OptionValue::eTypeInvalid, false, 0, NULL, NULL, NULL}};

  return g_properties;
}

static const PlatformLinuxPropertiesSP &GetGlobalProperties() {
  static PlatformLinuxPropertiesSP g_settings_sp;
  if (!g_settings_sp)
    g_settings_sp.reset(new PlatformLinuxProperties());
  return g_settings_sp;
}

void PlatformLinux::DebuggerInitialize(Debugger &debugger) {
  if (!PluginManager::GetSettingForPlatformPlugin(
          debugger, PlatformLinuxProperties::GetSettingName())) {
    const bool is_global_setting = true;
    PluginManager::CreateSettingForPlatformPlugin(
        debugger, GetGlobalProperties()->GetValueProperties(),
        ConstString("Properties for the PlatformLinux plug-in."),
        is_global_setting);
  }
}

//------------------------------------------------------------------

PlatformSP PlatformLinux::CreateInstance(bool force, const ArchSpec *arch) {
  Log *log(GetLogIfAllCategoriesSet(LIBLLDB_LOG_PLATFORM));
  if (log) {
    const char *arch_name;
    if (arch && arch->GetArchitectureName())
      arch_name = arch->GetArchitectureName();
    else
      arch_name = "<null>";

    const char *triple_cstr =
        arch ? arch->GetTriple().getTriple().c_str() : "<null>";

    log->Printf("PlatformLinux::%s(force=%s, arch={%s,%s})", __FUNCTION__,
                force ? "true" : "false", arch_name, triple_cstr);
  }

  bool create = force;
  if (create == false && arch && arch->IsValid()) {
    const llvm::Triple &triple = arch->GetTriple();
    switch (triple.getOS()) {
    case llvm::Triple::Linux:
      create = true;
      break;

#if defined(__linux__)
    // Only accept "unknown" for the OS if the host is linux and
    // it "unknown" wasn't specified (it was just returned because it
    // was NOT specified)
    case llvm::Triple::OSType::UnknownOS:
      create = !arch->TripleOSWasSpecified();
      break;
#endif
    default:
      break;
    }
  }

  if (create) {
    if (log)
      log->Printf("PlatformLinux::%s() creating remote-linux platform",
                  __FUNCTION__);
    return PlatformSP(new PlatformLinux(false));
  }

  if (log)
    log->Printf(
        "PlatformLinux::%s() aborting creation of remote-linux platform",
        __FUNCTION__);

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
        PlatformLinux::CreateInstance, PlatformLinux::DebuggerInitialize);
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

Error PlatformLinux::ResolveExecutable(
    const ModuleSpec &ms, lldb::ModuleSP &exe_module_sp,
    const FileSpecList *module_search_paths_ptr) {
  Error error;
  // Nothing special to do here, just use the actual file and architecture

  char exe_path[PATH_MAX];
  ModuleSpec resolved_module_spec(ms);

  if (IsHost()) {
    // If we have "ls" as the exe_file, resolve the executable location based on
    // the current path variables
    if (!resolved_module_spec.GetFileSpec().Exists()) {
      resolved_module_spec.GetFileSpec().GetPath(exe_path, sizeof(exe_path));
      resolved_module_spec.GetFileSpec().SetFile(exe_path, true);
    }

    if (!resolved_module_spec.GetFileSpec().Exists())
      resolved_module_spec.GetFileSpec().ResolveExecutableLocation();

    if (resolved_module_spec.GetFileSpec().Exists())
      error.Clear();
    else {
      error.SetErrorStringWithFormat(
          "unable to find executable for '%s'",
          resolved_module_spec.GetFileSpec().GetPath().c_str());
    }
  } else {
    if (m_remote_platform_sp) {
      error =
          GetCachedExecutable(resolved_module_spec, exe_module_sp,
                              module_search_paths_ptr, *m_remote_platform_sp);
    } else {
      // We may connect to a process and use the provided executable (Don't use
      // local $PATH).

      if (resolved_module_spec.GetFileSpec().Exists())
        error.Clear();
      else
        error.SetErrorStringWithFormat("the platform is not currently "
                                       "connected, and '%s' doesn't exist in "
                                       "the system root.",
                                       exe_path);
    }
  }

  if (error.Success()) {
    if (resolved_module_spec.GetArchitecture().IsValid()) {
      error = ModuleList::GetSharedModule(resolved_module_spec, exe_module_sp,
                                          NULL, NULL, NULL);
      if (error.Fail()) {
        // If we failed, it may be because the vendor and os aren't known. If
        // that is the
        // case, try setting them to the host architecture and give it another
        // try.
        llvm::Triple &module_triple =
            resolved_module_spec.GetArchitecture().GetTriple();
        bool is_vendor_specified =
            (module_triple.getVendor() != llvm::Triple::UnknownVendor);
        bool is_os_specified =
            (module_triple.getOS() != llvm::Triple::UnknownOS);
        if (!is_vendor_specified || !is_os_specified) {
          const llvm::Triple &host_triple =
              HostInfo::GetArchitecture(HostInfo::eArchKindDefault).GetTriple();

          if (!is_vendor_specified)
            module_triple.setVendorName(host_triple.getVendorName());
          if (!is_os_specified)
            module_triple.setOSName(host_triple.getOSName());

          error = ModuleList::GetSharedModule(resolved_module_spec,
                                              exe_module_sp, NULL, NULL, NULL);
        }
      }

      // TODO find out why exe_module_sp might be NULL
      if (!exe_module_sp || exe_module_sp->GetObjectFile() == NULL) {
        exe_module_sp.reset();
        error.SetErrorStringWithFormat(
            "'%s' doesn't contain the architecture %s",
            resolved_module_spec.GetFileSpec().GetPath().c_str(),
            resolved_module_spec.GetArchitecture().GetArchitectureName());
      }
    } else {
      // No valid architecture was specified, ask the platform for
      // the architectures that we should be using (in the correct order)
      // and see if we can find a match that way
      StreamString arch_names;
      for (uint32_t idx = 0; GetSupportedArchitectureAtIndex(
               idx, resolved_module_spec.GetArchitecture());
           ++idx) {
        error = ModuleList::GetSharedModule(resolved_module_spec, exe_module_sp,
                                            NULL, NULL, NULL);
        // Did we find an executable using one of the
        if (error.Success()) {
          if (exe_module_sp && exe_module_sp->GetObjectFile())
            break;
          else
            error.SetErrorToGenericError();
        }

        if (idx > 0)
          arch_names.PutCString(", ");
        arch_names.PutCString(
            resolved_module_spec.GetArchitecture().GetArchitectureName());
      }

      if (error.Fail() || !exe_module_sp) {
        if (resolved_module_spec.GetFileSpec().Readable()) {
          error.SetErrorStringWithFormat(
              "'%s' doesn't contain any '%s' platform architectures: %s",
              resolved_module_spec.GetFileSpec().GetPath().c_str(),
              GetPluginName().GetCString(), arch_names.GetString().c_str());
        } else {
          error.SetErrorStringWithFormat(
              "'%s' is not readable",
              resolved_module_spec.GetFileSpec().GetPath().c_str());
        }
      }
    }
  }

  return error;
}

Error PlatformLinux::GetFileWithUUID(const FileSpec &platform_file,
                                     const UUID *uuid_ptr,
                                     FileSpec &local_file) {
  if (IsRemote()) {
    if (m_remote_platform_sp)
      return m_remote_platform_sp->GetFileWithUUID(platform_file, uuid_ptr,
                                                   local_file);
  }

  // Default to the local case
  local_file = platform_file;
  return Error();
}

//------------------------------------------------------------------
/// Default Constructor
//------------------------------------------------------------------
PlatformLinux::PlatformLinux(bool is_host)
    : PlatformPOSIX(is_host) // This is the local host platform
{}

//------------------------------------------------------------------
/// Destructor.
///
/// The destructor is virtual since this class is designed to be
/// inherited from by the plug-in instance.
//------------------------------------------------------------------
PlatformLinux::~PlatformLinux() = default;

bool PlatformLinux::GetProcessInfo(lldb::pid_t pid,
                                   ProcessInstanceInfo &process_info) {
  bool success = false;
  if (IsHost()) {
    success = Platform::GetProcessInfo(pid, process_info);
  } else {
    if (m_remote_platform_sp)
      success = m_remote_platform_sp->GetProcessInfo(pid, process_info);
  }
  return success;
}

uint32_t
PlatformLinux::FindProcesses(const ProcessInstanceInfoMatch &match_info,
                             ProcessInstanceInfoList &process_infos) {
  uint32_t match_count = 0;
  if (IsHost()) {
    // Let the base class figure out the host details
    match_count = Platform::FindProcesses(match_info, process_infos);
  } else {
    // If we are remote, we can only return results if we are connected
    if (m_remote_platform_sp)
      match_count =
          m_remote_platform_sp->FindProcesses(match_info, process_infos);
  }
  return match_count;
}

bool PlatformLinux::GetSupportedArchitectureAtIndex(uint32_t idx,
                                                    ArchSpec &arch) {
  if (IsHost()) {
    ArchSpec hostArch = HostInfo::GetArchitecture(HostInfo::eArchKindDefault);
    if (hostArch.GetTriple().isOSLinux()) {
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

void PlatformLinux::GetStatus(Stream &strm) {
  Platform::GetStatus(strm);

#ifndef LLDB_DISABLE_POSIX
  // Display local kernel information only when we are running in host mode.
  // Otherwise, we would end up printing non-Linux information (when running
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

bool PlatformLinux::CanDebugProcess() {
  if (IsHost()) {
    return true;
  } else {
    // If we're connected, we can debug.
    return IsConnected();
  }
}

// For local debugging, Linux will override the debug logic to use llgs-launch
// rather than
// lldb-launch, llgs-attach.  This differs from current lldb-launch,
// debugserver-attach
// approach on MacOSX.
lldb::ProcessSP
PlatformLinux::DebugProcess(ProcessLaunchInfo &launch_info, Debugger &debugger,
                            Target *target, // Can be NULL, if NULL create a new
                                            // target, else use existing one
                            Error &error) {
  Log *log(GetLogIfAllCategoriesSet(LIBLLDB_LOG_PLATFORM));
  if (log)
    log->Printf("PlatformLinux::%s entered (target %p)", __FUNCTION__,
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
      log->Printf("PlatformLinux::%s creating new target", __FUNCTION__);

    TargetSP new_target_sp;
    error = debugger.GetTargetList().CreateTarget(
        debugger, nullptr, nullptr, false, nullptr, new_target_sp);
    if (error.Fail()) {
      if (log)
        log->Printf("PlatformLinux::%s failed to create new target: %s",
                    __FUNCTION__, error.AsCString());
      return process_sp;
    }

    target = new_target_sp.get();
    if (!target) {
      error.SetErrorString("CreateTarget() returned nullptr");
      if (log)
        log->Printf("PlatformLinux::%s failed: %s", __FUNCTION__,
                    error.AsCString());
      return process_sp;
    }
  } else {
    if (log)
      log->Printf("PlatformLinux::%s using provided target", __FUNCTION__);
  }

  // Mark target as currently selected target.
  debugger.GetTargetList().SetSelectedTarget(target);

  // Now create the gdb-remote process.
  if (log)
    log->Printf(
        "PlatformLinux::%s having target create process with gdb-remote plugin",
        __FUNCTION__);
  process_sp = target->CreateProcess(
      launch_info.GetListenerForProcess(debugger), "gdb-remote", nullptr);

  if (!process_sp) {
    error.SetErrorString("CreateProcess() failed for gdb-remote process");
    if (log)
      log->Printf("PlatformLinux::%s failed: %s", __FUNCTION__,
                  error.AsCString());
    return process_sp;
  } else {
    if (log)
      log->Printf("PlatformLinux::%s successfully created process",
                  __FUNCTION__);
  }

  // Adjust launch for a hijacker.
  ListenerSP listener_sp;
  if (!launch_info.GetHijackListener()) {
    if (log)
      log->Printf("PlatformLinux::%s setting up hijacker", __FUNCTION__);

    listener_sp =
        Listener::MakeListener("lldb.PlatformLinux.DebugProcess.hijack");
    launch_info.SetHijackListener(listener_sp);
    process_sp->HijackProcessEvents(listener_sp);
  }

  // Log file actions.
  if (log) {
    log->Printf(
        "PlatformLinux::%s launching process with the following file actions:",
        __FUNCTION__);

    StreamString stream;
    size_t i = 0;
    const FileAction *file_action;
    while ((file_action = launch_info.GetFileActionAtIndex(i++)) != nullptr) {
      file_action->Dump(stream);
      log->PutCString(stream.GetString().c_str());
      stream.Clear();
    }
  }

  // Do the launch.
  error = process_sp->Launch(launch_info);
  if (error.Success()) {
    // Handle the hijacking of process events.
    if (listener_sp) {
      const StateType state =
          process_sp->WaitForProcessToStop(NULL, NULL, false, listener_sp);

      if (state == eStateStopped) {
        if (log)
          log->Printf("PlatformLinux::%s pid %" PRIu64 " state %s\n",
                      __FUNCTION__, process_sp->GetID(), StateAsCString(state));
      } else {
        if (log)
          log->Printf("PlatformLinux::%s pid %" PRIu64
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
        log->Printf("PlatformLinux::%s pid %" PRIu64
                    " hooked up STDIO pty to process",
                    __FUNCTION__, process_sp->GetID());
    } else {
      if (log)
        log->Printf("PlatformLinux::%s pid %" PRIu64
                    " not using process STDIO pty",
                    __FUNCTION__, process_sp->GetID());
    }
  } else {
    if (log)
      log->Printf("PlatformLinux::%s process launch failed: %s", __FUNCTION__,
                  error.AsCString());
    // FIXME figure out appropriate cleanup here.  Do we delete the target? Do
    // we delete the process?  Does our caller do that?
  }

  return process_sp;
}

void PlatformLinux::CalculateTrapHandlerSymbolNames() {
  m_trap_handlers.push_back(ConstString("_sigtramp"));
}

uint64_t PlatformLinux::ConvertMmapFlagsToPlatform(const ArchSpec &arch,
                                                   unsigned flags) {
  uint64_t flags_platform = 0;
  uint64_t map_anon = MAP_ANON;

  // To get correct flags for MIPS Architecture
  if (arch.GetTriple().getArch() == llvm::Triple::mips64 ||
      arch.GetTriple().getArch() == llvm::Triple::mips64el ||
      arch.GetTriple().getArch() == llvm::Triple::mips ||
      arch.GetTriple().getArch() == llvm::Triple::mipsel)
    map_anon = 0x800;

  if (flags & eMmapFlagsPrivate)
    flags_platform |= MAP_PRIVATE;
  if (flags & eMmapFlagsAnon)
    flags_platform |= map_anon;
  return flags_platform;
}

ConstString PlatformLinux::GetFullNameForDylib(ConstString basename) {
  if (basename.IsEmpty())
    return basename;

  StreamString stream;
  stream.Printf("lib%s.so", basename.GetCString());
  return ConstString(stream.GetData());
}
