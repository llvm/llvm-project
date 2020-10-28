//===-- PlatformMacOSX.cpp ------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PlatformMacOSX.h"
#include "PlatformRemoteiOS.h"
#if defined(__APPLE__)
#include "PlatformAppleTVSimulator.h"
#include "PlatformAppleWatchSimulator.h"
#include "PlatformDarwinKernel.h"
#include "PlatformRemoteAppleBridge.h"
#include "PlatformRemoteAppleTV.h"
#include "PlatformRemoteAppleWatch.h"
#include "PlatformiOSSimulator.h"
#endif
#include "lldb/Breakpoint/BreakpointLocation.h"
#include "lldb/Core/Module.h"
#include "lldb/Core/ModuleList.h"
#include "lldb/Core/ModuleSpec.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/Host/Config.h"
#include "lldb/Host/Host.h"
#include "lldb/Host/HostInfo.h"
#include "lldb/Symbol/ObjectFile.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/Target.h"
#include "lldb/Utility/DataBufferHeap.h"
#include "lldb/Utility/FileSpec.h"
#include "lldb/Utility/Log.h"
#include "lldb/Utility/Status.h"
#include "lldb/Utility/StreamString.h"

#include <sstream>

using namespace lldb;
using namespace lldb_private;

LLDB_PLUGIN_DEFINE(PlatformMacOSX)

static uint32_t g_initialize_count = 0;

void PlatformMacOSX::Initialize() {
  PlatformDarwin::Initialize();
  PlatformRemoteiOS::Initialize();
#if defined(__APPLE__)
  PlatformiOSSimulator::Initialize();
  PlatformDarwinKernel::Initialize();
  PlatformAppleTVSimulator::Initialize();
  PlatformAppleWatchSimulator::Initialize();
  PlatformRemoteAppleTV::Initialize();
  PlatformRemoteAppleWatch::Initialize();
  PlatformRemoteAppleBridge::Initialize();
#endif

  if (g_initialize_count++ == 0) {
#if defined(__APPLE__)
    PlatformSP default_platform_sp(new PlatformMacOSX(true));
    default_platform_sp->SetSystemArchitecture(HostInfo::GetArchitecture());
    Platform::SetHostPlatform(default_platform_sp);
#endif
    PluginManager::RegisterPlugin(PlatformMacOSX::GetPluginNameStatic(false),
                                  PlatformMacOSX::GetDescriptionStatic(false),
                                  PlatformMacOSX::CreateInstance);
  }
}

void PlatformMacOSX::Terminate() {
  if (g_initialize_count > 0) {
    if (--g_initialize_count == 0) {
      PluginManager::UnregisterPlugin(PlatformMacOSX::CreateInstance);
    }
  }

#if defined(__APPLE__)
  PlatformRemoteAppleBridge::Terminate();
  PlatformRemoteAppleWatch::Terminate();
  PlatformRemoteAppleTV::Terminate();
  PlatformAppleWatchSimulator::Terminate();
  PlatformAppleTVSimulator::Terminate();
  PlatformDarwinKernel::Terminate();
  PlatformiOSSimulator::Terminate();
#endif
  PlatformRemoteiOS::Terminate();
  PlatformDarwin::Terminate();
}

PlatformSP PlatformMacOSX::CreateInstance(bool force, const ArchSpec *arch) {
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

  // The only time we create an instance is when we are creating a remote
  // macosx platform
  const bool is_host = false;

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
    LLDB_LOGF(log, "PlatformMacOSX::%s() creating platform", __FUNCTION__);
    return PlatformSP(new PlatformMacOSX(is_host));
  }

  LLDB_LOGF(log, "PlatformMacOSX::%s() aborting creation of platform",
            __FUNCTION__);

  return PlatformSP();
}

lldb_private::ConstString PlatformMacOSX::GetPluginNameStatic(bool is_host) {
  if (is_host) {
    static ConstString g_host_name(Platform::GetHostPlatformName());
    return g_host_name;
  } else {
    static ConstString g_remote_name("remote-macosx");
    return g_remote_name;
  }
}

const char *PlatformMacOSX::GetDescriptionStatic(bool is_host) {
  if (is_host)
    return "Local Mac OS X user platform plug-in.";
  else
    return "Remote Mac OS X user platform plug-in.";
}

/// Default Constructor
PlatformMacOSX::PlatformMacOSX(bool is_host) : PlatformDarwin(is_host) {}

/// Destructor.
///
/// The destructor is virtual since this class is designed to be
/// inherited from by the plug-in instance.
PlatformMacOSX::~PlatformMacOSX() {}

ConstString PlatformMacOSX::GetSDKDirectory(lldb_private::Target &target) {
  ModuleSP exe_module_sp(target.GetExecutableModule());
  if (!exe_module_sp)
    return {};

  ObjectFile *objfile = exe_module_sp->GetObjectFile();
  if (!objfile)
    return {};

  llvm::VersionTuple version = objfile->GetSDKVersion();
  if (version.empty())
    return {};

  // First try to find an SDK that matches the given SDK version.
  if (FileSpec fspec = HostInfo::GetXcodeContentsDirectory()) {
    StreamString sdk_path;
    sdk_path.Printf("%s/Developer/Platforms/MacOSX.platform/Developer/"
                    "SDKs/MacOSX%u.%u.sdk",
                    fspec.GetPath().c_str(), version.getMajor(),
                    version.getMinor().getValue());
    if (FileSystem::Instance().Exists(fspec))
      return ConstString(sdk_path.GetString());
  }

  // Use the default SDK as a fallback.
  FileSpec fspec(
      HostInfo::GetXcodeSDKPath(lldb_private::XcodeSDK::GetAnyMacOS()));
  if (fspec) {
    if (FileSystem::Instance().Exists(fspec))
      return ConstString(fspec.GetPath());
  }

  return {};
}

Status PlatformMacOSX::GetSymbolFile(const FileSpec &platform_file,
                                     const UUID *uuid_ptr,
                                     FileSpec &local_file) {
  if (IsRemote()) {
    if (m_remote_platform_sp)
      return m_remote_platform_sp->GetFileWithUUID(platform_file, uuid_ptr,
                                                   local_file);
  }

  // Default to the local case
  local_file = platform_file;
  return Status();
}

lldb_private::Status
PlatformMacOSX::GetFileWithUUID(const lldb_private::FileSpec &platform_file,
                                const lldb_private::UUID *uuid_ptr,
                                lldb_private::FileSpec &local_file) {
  if (IsRemote() && m_remote_platform_sp) {
    std::string local_os_build;
#if !defined(__linux__)
    HostInfo::GetOSBuildString(local_os_build);
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

bool PlatformMacOSX::GetSupportedArchitectureAtIndex(uint32_t idx,
                                                     ArchSpec &arch) {
#if defined(__arm__) || defined(__arm64__) || defined(__aarch64__)
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
#else
  return x86GetSupportedArchitectureAtIndex(idx, arch);
#endif
}

lldb_private::Status PlatformMacOSX::GetSharedModule(
    const lldb_private::ModuleSpec &module_spec, Process *process,
    lldb::ModuleSP &module_sp,
    const lldb_private::FileSpecList *module_search_paths_ptr,
    lldb::ModuleSP *old_module_sp_ptr, bool *did_create_ptr) {
  Status error = GetSharedModuleWithLocalCache(
      module_spec, module_sp, module_search_paths_ptr, old_module_sp_ptr,
      did_create_ptr);

  if (module_sp) {
    if (module_spec.GetArchitecture().GetCore() ==
        ArchSpec::eCore_x86_64_x86_64h) {
      ObjectFile *objfile = module_sp->GetObjectFile();
      if (objfile == nullptr) {
        // We didn't find an x86_64h slice, fall back to a x86_64 slice
        ModuleSpec module_spec_x86_64(module_spec);
        module_spec_x86_64.GetArchitecture() = ArchSpec("x86_64-apple-macosx");
        lldb::ModuleSP x86_64_module_sp;
        lldb::ModuleSP old_x86_64_module_sp;
        bool did_create = false;
        Status x86_64_error = GetSharedModuleWithLocalCache(
            module_spec_x86_64, x86_64_module_sp, module_search_paths_ptr,
            &old_x86_64_module_sp, &did_create);
        if (x86_64_module_sp && x86_64_module_sp->GetObjectFile()) {
          module_sp = x86_64_module_sp;
          if (old_module_sp_ptr)
            *old_module_sp_ptr = old_x86_64_module_sp;
          if (did_create_ptr)
            *did_create_ptr = did_create;
          return x86_64_error;
        }
      }
    }
  }

  if (!module_sp) {
      error = FindBundleBinaryInExecSearchPaths (module_spec, process, module_sp, module_search_paths_ptr, old_module_sp_ptr, did_create_ptr);
  }
  return error;
}
