//===-- PlatformMacOSX.cpp --------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "PlatformMacOSX.h"
#include "lldb/Host/Config.h"

// C++ Includes

#include <sstream>

// Other libraries and framework includes
// Project includes
#include "lldb/Breakpoint/BreakpointLocation.h"
#include "lldb/Core/Module.h"
#include "lldb/Core/ModuleList.h"
#include "lldb/Core/ModuleSpec.h"
#include "lldb/Core/PluginManager.h"
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

using namespace lldb;
using namespace lldb_private;

static uint32_t g_initialize_count = 0;

void PlatformMacOSX::Initialize() {
  PlatformDarwin::Initialize();

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

    log->Printf("PlatformMacOSX::%s(force=%s, arch={%s,%s})", __FUNCTION__,
                force ? "true" : "false", arch_name, triple_cstr);
  }

  // The only time we create an instance is when we are creating a remote
  // macosx platform
  const bool is_host = false;

  bool create = force;
  if (create == false && arch && arch->IsValid()) {
    const llvm::Triple &triple = arch->GetTriple();
    switch (triple.getVendor()) {
    case llvm::Triple::Apple:
      create = true;
      break;

#if defined(__APPLE__)
    // Only accept "unknown" for vendor if the host is Apple and it "unknown"
    // wasn't specified (it was just returned because it was NOT specified)
    case llvm::Triple::UnknownArch:
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
    if (log)
      log->Printf("PlatformMacOSX::%s() creating platform", __FUNCTION__);
    return PlatformSP(new PlatformMacOSX(is_host));
  }

  if (log)
    log->Printf("PlatformMacOSX::%s() aborting creation of platform",
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

//------------------------------------------------------------------
/// Default Constructor
//------------------------------------------------------------------
PlatformMacOSX::PlatformMacOSX(bool is_host) : PlatformDarwin(is_host) {}

//------------------------------------------------------------------
/// Destructor.
///
/// The destructor is virtual since this class is designed to be
/// inherited from by the plug-in instance.
//------------------------------------------------------------------
PlatformMacOSX::~PlatformMacOSX() {}

ConstString PlatformMacOSX::GetSDKDirectory(lldb_private::Target &target) {
  ModuleSP exe_module_sp(target.GetExecutableModule());
  if (exe_module_sp) {
    ObjectFile *objfile = exe_module_sp->GetObjectFile();
    if (objfile) {
      std::string xcode_contents_path;
      std::string default_xcode_sdk;
      FileSpec fspec;
      uint32_t versions[2];
      if (objfile->GetSDKVersion(versions, sizeof(versions))) {
        if (HostInfo::GetLLDBPath(ePathTypeLLDBShlibDir, fspec)) {
          std::string path;
          xcode_contents_path = fspec.GetPath();
          size_t pos = xcode_contents_path.find("/Xcode.app/Contents/");
          if (pos != std::string::npos) {
            // LLDB.framework is inside an Xcode app bundle, we can locate the
            // SDK from here
            xcode_contents_path.erase(pos + strlen("/Xcode.app/Contents/"));
          } else {
            xcode_contents_path.clear();
            // Use the selected Xcode
            int status = 0;
            int signo = 0;
            std::string output;
            const char *command = "xcrun -sdk macosx --show-sdk-path";
            lldb_private::Status error = RunShellCommand(
                command, // shell command to run
                NULL,    // current working directory
                &status, // Put the exit status of the process in here
                &signo,  // Put the signal that caused the process to exit in
                         // here
                &output, // Get the output from the command and place it in this
                         // string
                3); // Timeout in seconds to wait for shell program to finish
            if (status == 0 && !output.empty()) {
              size_t first_non_newline = output.find_last_not_of("\r\n");
              if (first_non_newline != std::string::npos)
                output.erase(first_non_newline + 1);
              default_xcode_sdk = output;

              pos = default_xcode_sdk.find("/Xcode.app/Contents/");
              if (pos != std::string::npos)
                xcode_contents_path = default_xcode_sdk.substr(
                    0, pos + strlen("/Xcode.app/Contents/"));
            }
          }
        }

        if (!xcode_contents_path.empty()) {
          StreamString sdk_path;
          sdk_path.Printf("%sDeveloper/Platforms/MacOSX.platform/Developer/"
                          "SDKs/MacOSX%u.%u.sdk",
                          xcode_contents_path.c_str(), versions[0],
                          versions[1]);
          fspec.SetFile(sdk_path.GetString(), false);
          if (fspec.Exists())
            return ConstString(sdk_path.GetString());
        }

        if (!default_xcode_sdk.empty()) {
          fspec.SetFile(default_xcode_sdk, false);
          if (fspec.Exists())
            return ConstString(default_xcode_sdk);
        }
      }
    }
  }
  return ConstString();
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
    if (local_os_build.compare(remote_os_build) == 0) {
      // same OS version: the local file is good enough
      local_file = platform_file;
      return Status();
    } else {
      // try to find the file in the cache
      std::string cache_path(GetLocalCacheDirectory());
      std::string module_path(platform_file.GetPath());
      cache_path.append(module_path);
      FileSpec module_cache_spec(cache_path, false);
      if (module_cache_spec.Exists()) {
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
      if (module_cache_spec.Exists()) {
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
      if (objfile == NULL) {
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
