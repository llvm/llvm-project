//===-- PlatformKalimba.cpp ---------------------------------------*- C++
//-*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "PlatformKalimba.h"
#include "lldb/Host/Config.h"

// C++ Includes
// Other libraries and framework includes
// Project includes
#include "lldb/Core/Debugger.h"
#include "lldb/Core/Error.h"
#include "lldb/Core/Module.h"
#include "lldb/Core/ModuleList.h"
#include "lldb/Core/ModuleSpec.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/Core/StreamString.h"
#include "lldb/Host/FileSpec.h"
#include "lldb/Host/HostInfo.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/Target.h"

using namespace lldb;
using namespace lldb_private;

static uint32_t g_initialize_count = 0;

PlatformSP PlatformKalimba::CreateInstance(bool force, const ArchSpec *arch) {
  bool create = force;
  if (create == false && arch && arch->IsValid()) {
    const llvm::Triple &triple = arch->GetTriple();
    switch (triple.getVendor()) {
    case llvm::Triple::CSR:
      create = true;
      break;

    default:
      break;
    }
  }
  if (create)
    return PlatformSP(new PlatformKalimba(false));
  return PlatformSP();
}

lldb_private::ConstString
PlatformKalimba::GetPluginNameStatic(bool /*is_host*/) {
  static ConstString g_remote_name("kalimba");
  return g_remote_name;
}

const char *PlatformKalimba::GetPluginDescriptionStatic(bool /*is_host*/) {
  return "Kalimba user platform plug-in.";
}

lldb_private::ConstString PlatformKalimba::GetPluginName() {
  return GetPluginNameStatic(false);
}

void PlatformKalimba::Initialize() {
  Platform::Initialize();

  if (g_initialize_count++ == 0) {
    PluginManager::RegisterPlugin(
        PlatformKalimba::GetPluginNameStatic(false),
        PlatformKalimba::GetPluginDescriptionStatic(false),
        PlatformKalimba::CreateInstance);
  }
}

void PlatformKalimba::Terminate() {
  if (g_initialize_count > 0) {
    if (--g_initialize_count == 0) {
      PluginManager::UnregisterPlugin(PlatformKalimba::CreateInstance);
    }
  }

  Platform::Terminate();
}

Error PlatformKalimba::ResolveExecutable(
    const ModuleSpec &ms, lldb::ModuleSP &exe_module_sp,
    const FileSpecList *module_search_paths_ptr) {
  Error error;
  char exe_path[PATH_MAX];
  ModuleSpec resolved_module_spec(ms);

  if (!resolved_module_spec.GetFileSpec().Exists()) {
    resolved_module_spec.GetFileSpec().GetPath(exe_path, sizeof(exe_path));
    error.SetErrorStringWithFormat("unable to find executable for '%s'",
                                   exe_path);
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

Error PlatformKalimba::GetFileWithUUID(const FileSpec & /*platform_file*/,
                                       const UUID * /*uuid_ptr*/,
                                       FileSpec & /*local_file*/) {
  return Error();
}

//------------------------------------------------------------------
/// Default Constructor
//------------------------------------------------------------------
PlatformKalimba::PlatformKalimba(bool is_host)
    : Platform(is_host), // This is the local host platform
      m_remote_platform_sp() {}

//------------------------------------------------------------------
/// Destructor.
///
/// The destructor is virtual since this class is designed to be
/// inherited from by the plug-in instance.
//------------------------------------------------------------------
PlatformKalimba::~PlatformKalimba() {}

bool PlatformKalimba::GetProcessInfo(lldb::pid_t pid,
                                     ProcessInstanceInfo &process_info) {
  bool success = false;
  if (IsHost()) {
    success = false;
  } else {
    if (m_remote_platform_sp)
      success = m_remote_platform_sp->GetProcessInfo(pid, process_info);
  }
  return success;
}

bool PlatformKalimba::GetSupportedArchitectureAtIndex(uint32_t idx,
                                                      ArchSpec &arch) {
  if (idx == 0) {
    arch = ArchSpec("kalimba3-csr-unknown");
    return true;
  }
  if (idx == 1) {
    arch = ArchSpec("kalimba4-csr-unknown");
    return true;
  }
  if (idx == 2) {
    arch = ArchSpec("kalimba5-csr-unknown");
    return true;
  }
  return false;
}

void PlatformKalimba::GetStatus(Stream &strm) { Platform::GetStatus(strm); }

size_t
PlatformKalimba::GetSoftwareBreakpointTrapOpcode(Target & /*target*/,
                                                 BreakpointSite * /*bp_site*/) {
  // the target hardware does not support software breakpoints
  return 0;
}

Error PlatformKalimba::LaunchProcess(ProcessLaunchInfo &launch_info) {
  Error error;

  if (IsHost()) {
    error.SetErrorString("native execution is not possible");
  } else {
    error.SetErrorString("the platform is not currently connected");
  }
  return error;
}

lldb::ProcessSP PlatformKalimba::Attach(ProcessAttachInfo &attach_info,
                                        Debugger &debugger, Target *target,
                                        Error &error) {
  lldb::ProcessSP process_sp;
  if (IsHost()) {
    error.SetErrorString("native execution is not possible");
  } else {
    if (m_remote_platform_sp)
      process_sp =
          m_remote_platform_sp->Attach(attach_info, debugger, target, error);
    else
      error.SetErrorString("the platform is not currently connected");
  }
  return process_sp;
}

void PlatformKalimba::CalculateTrapHandlerSymbolNames() {
  // TODO Research this sometime.
}
