//===-- PlatformMacOSX.cpp ------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PlatformMacOSX.h"
#include "PlatformRemoteMacOSX.h"
#include "PlatformRemoteiOS.h"
#if defined(__APPLE__)
#include "PlatformAppleSimulator.h"
#include "PlatformDarwinKernel.h"
#include "PlatformRemoteAppleBridge.h"
#include "PlatformRemoteAppleTV.h"
#include "PlatformRemoteAppleWatch.h"
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
  PlatformRemoteMacOSX::Initialize();
#if defined(__APPLE__)
  PlatformAppleSimulator::Initialize();
  PlatformDarwinKernel::Initialize();
  PlatformRemoteAppleTV::Initialize();
  PlatformRemoteAppleWatch::Initialize();
  PlatformRemoteAppleBridge::Initialize();
#endif

  if (g_initialize_count++ == 0) {
#if defined(__APPLE__)
    PlatformSP default_platform_sp(new PlatformMacOSX());
    default_platform_sp->SetSystemArchitecture(HostInfo::GetArchitecture());
    Platform::SetHostPlatform(default_platform_sp);
#endif
    PluginManager::RegisterPlugin(PlatformMacOSX::GetPluginNameStatic(),
                                  PlatformMacOSX::GetDescriptionStatic(),
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
  PlatformDarwinKernel::Terminate();
  PlatformAppleSimulator::Terminate();
#endif
  PlatformRemoteMacOSX::Initialize();
  PlatformRemoteiOS::Terminate();
  PlatformDarwin::Terminate();
}

lldb_private::ConstString PlatformMacOSX::GetPluginNameStatic() {
  static ConstString g_host_name(Platform::GetHostPlatformName());
  return g_host_name;
}

const char *PlatformMacOSX::GetDescriptionStatic() {
  return "Local Mac OS X user platform plug-in.";
}

PlatformSP PlatformMacOSX::CreateInstance(bool force, const ArchSpec *arch) {
  // The only time we create an instance is when we are creating a remote
  // macosx platform which is handled by PlatformRemoteMacOSX.
  return PlatformSP();
}

/// Default Constructor
PlatformMacOSX::PlatformMacOSX() : PlatformDarwin(true) {}

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

bool PlatformMacOSX::GetSupportedArchitectureAtIndex(uint32_t idx,
                                                     ArchSpec &arch) {
#if defined(__arm__) || defined(__arm64__) || defined(__aarch64__)
  return ARMGetSupportedArchitectureAtIndex(idx, arch);
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
