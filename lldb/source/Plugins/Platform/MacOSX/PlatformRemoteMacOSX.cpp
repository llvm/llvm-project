//===-- PlatformRemoteMacOSX.cpp -------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <memory>
#include <string>
#include <vector>

#include "PlatformRemoteMacOSX.h"

#include "lldb/Breakpoint/BreakpointLocation.h"
#include "lldb/Core/Module.h"
#include "lldb/Core/ModuleList.h"
#include "lldb/Core/ModuleSpec.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/Host/Host.h"
#include "lldb/Host/HostInfo.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/Target.h"
#include "lldb/Utility/ArchSpec.h"
#include "lldb/Utility/FileSpec.h"
#include "lldb/Utility/Log.h"
#include "lldb/Utility/StreamString.h"

using namespace lldb;
using namespace lldb_private;

/// Default Constructor
PlatformRemoteMacOSX::PlatformRemoteMacOSX() : PlatformRemoteDarwinDevice() {}

// Static Variables
static uint32_t g_initialize_count = 0;

// Static Functions
void PlatformRemoteMacOSX::Initialize() {
  PlatformDarwin::Initialize();

  if (g_initialize_count++ == 0) {
    PluginManager::RegisterPlugin(PlatformRemoteMacOSX::GetPluginNameStatic(),
                                  PlatformRemoteMacOSX::GetDescriptionStatic(),
                                  PlatformRemoteMacOSX::CreateInstance);
  }
}

void PlatformRemoteMacOSX::Terminate() {
  if (g_initialize_count > 0) {
    if (--g_initialize_count == 0) {
      PluginManager::UnregisterPlugin(PlatformRemoteMacOSX::CreateInstance);
    }
  }

  PlatformDarwin::Terminate();
}

PlatformSP PlatformRemoteMacOSX::CreateInstance(bool force,
                                                const ArchSpec *arch) {
  Log *log(GetLogIfAllCategoriesSet(LIBLLDB_LOG_PLATFORM));
  if (log) {
    const char *arch_name;
    if (arch && arch->GetArchitectureName())
      arch_name = arch->GetArchitectureName();
    else
      arch_name = "<null>";

    const char *triple_cstr =
        arch ? arch->GetTriple().getTriple().c_str() : "<null>";

    LLDB_LOGF(log, "PlatformMacOSX::%s(force=%s, arch={%s,%s})", __FUNCTION__,
              force ? "true" : "false", arch_name, triple_cstr);
  }

  bool create = force;
  if (!create && arch && arch->IsValid()) {
    const llvm::Triple &triple = arch->GetTriple();
    switch (triple.getVendor()) {
    case llvm::Triple::Apple:
      create = true;
      break;

#if defined(__APPLE__)
    // Only accept "unknown" for vendor if the host is Apple and it "unknown"
    // wasn't specified (it was just returned because it was NOT specified)
    case llvm::Triple::UnknownVendor:
      create = !arch->TripleVendorWasSpecified();
      break;
#endif
    default:
      break;
    }

    if (create) {
      switch (triple.getOS()) {
      case llvm::Triple::Darwin: // Deprecated, but still support Darwin for
                                 // historical reasons
      case llvm::Triple::MacOSX:
        break;
#if defined(__APPLE__)
      // Only accept "vendor" for vendor if the host is Apple and it "unknown"
      // wasn't specified (it was just returned because it was NOT specified)
      case llvm::Triple::UnknownOS:
        create = !arch->TripleOSWasSpecified();
        break;
#endif
      default:
        create = false;
        break;
      }
    }
  }

  if (create) {
    LLDB_LOGF(log, "PlatformRemoteMacOSX::%s() creating platform",
              __FUNCTION__);
    return std::make_shared<PlatformRemoteMacOSX>();
  }

  LLDB_LOGF(log, "PlatformRemoteMacOSX::%s() aborting creation of platform",
            __FUNCTION__);

  return PlatformSP();
}

bool PlatformRemoteMacOSX::GetSupportedArchitectureAtIndex(uint32_t idx,
                                                           ArchSpec &arch) {
  // macOS for ARM64 support both native and translated x86_64 processes
  if (!m_num_arm_arches || idx < m_num_arm_arches) {
    bool res = ARMGetSupportedArchitectureAtIndex(idx, arch);
    if (res)
      return true;
    if (!m_num_arm_arches)
      m_num_arm_arches = idx;
  }

  // We can't use x86GetSupportedArchitectureAtIndex() because it uses
  // the system architecture for some of its return values and also
  // has a 32bits variant.
  if (idx == m_num_arm_arches) {
    arch.SetTriple("x86_64-apple-macosx");
    return true;
  } else if (idx == m_num_arm_arches + 1) {
    arch.SetTriple("x86_64-apple-ios-macabi");
    return true;
  } else if (idx == m_num_arm_arches + 2) {
    arch.SetTriple("arm64-apple-ios");
    return true;
  } else if (idx == m_num_arm_arches + 3) {
    arch.SetTriple("arm64e-apple-ios");
    return true;
  }

  return false;
}

lldb_private::Status PlatformRemoteMacOSX::GetFileWithUUID(
    const lldb_private::FileSpec &platform_file,
    const lldb_private::UUID *uuid_ptr, lldb_private::FileSpec &local_file) {
  if (m_remote_platform_sp) {
    std::string local_os_build;
#if !defined(__linux__)
    local_os_build = HostInfo::GetOSBuildString().getValueOr("");
#endif
    std::string remote_os_build;
    m_remote_platform_sp->GetOSBuildString(remote_os_build);
    if (local_os_build == remote_os_build) {
      // same OS version: the local file is good enough
      local_file = platform_file;
      return Status();
    } else {
      // try to find the file in the cache
      std::string cache_path(GetLocalCacheDirectory());
      std::string module_path(platform_file.GetPath());
      cache_path.append(module_path);
      FileSpec module_cache_spec(cache_path);
      if (FileSystem::Instance().Exists(module_cache_spec)) {
        local_file = module_cache_spec;
        return Status();
      }
      // bring in the remote module file
      FileSpec module_cache_folder =
          module_cache_spec.CopyByRemovingLastPathComponent();
      // try to make the local directory first
      Status err(
          llvm::sys::fs::create_directory(module_cache_folder.GetPath()));
      if (err.Fail())
        return err;
      err = GetFile(platform_file, module_cache_spec);
      if (err.Fail())
        return err;
      if (FileSystem::Instance().Exists(module_cache_spec)) {
        local_file = module_cache_spec;
        return Status();
      } else
        return Status("unable to obtain valid module file");
    }
  }
  local_file = platform_file;
  return Status();
}

lldb_private::ConstString PlatformRemoteMacOSX::GetPluginNameStatic() {
  static ConstString g_name("remote-macosx");
  return g_name;
}

const char *PlatformRemoteMacOSX::GetDescriptionStatic() {
  return "Remote Mac OS X user platform plug-in.";
}

llvm::StringRef PlatformRemoteMacOSX::GetDeviceSupportDirectoryName() {
  return "macOS DeviceSupport";
}

llvm::StringRef PlatformRemoteMacOSX::GetPlatformName() {
  return "MacOSX.platform";
}
