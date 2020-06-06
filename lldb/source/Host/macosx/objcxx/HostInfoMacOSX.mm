//===-- HostInfoMacOSX.mm ---------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Host/macosx/HostInfoMacOSX.h"
#include "lldb/Host/FileSystem.h"
#include "lldb/Host/Host.h"
#include "lldb/Host/HostInfo.h"
#include "lldb/Utility/Args.h"
#include "lldb/Utility/Log.h"

#include "llvm/ADT/SmallString.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

// C++ Includes
#include <string>

// C inclues
#include <stdlib.h>
#include <sys/sysctl.h>
#include <sys/syslimits.h>
#include <sys/types.h>

// Objective-C/C++ includes
#include <CoreFoundation/CoreFoundation.h>
#include <Foundation/Foundation.h>
#include <mach-o/dyld.h>
#include <objc/objc-auto.h>

// These are needed when compiling on systems
// that do not yet have these definitions
#include <AvailabilityMacros.h>
#ifndef CPU_SUBTYPE_X86_64_H
#define CPU_SUBTYPE_X86_64_H ((cpu_subtype_t)8)
#endif
#ifndef CPU_TYPE_ARM64
#define CPU_TYPE_ARM64 (CPU_TYPE_ARM | CPU_ARCH_ABI64)
#endif

#ifndef CPU_TYPE_ARM64_32
#define CPU_ARCH_ABI64_32 0x02000000
#define CPU_TYPE_ARM64_32 (CPU_TYPE_ARM | CPU_ARCH_ABI64_32)
#endif

#include <TargetConditionals.h> // for TARGET_OS_TV, TARGET_OS_WATCH

using namespace lldb_private;

bool HostInfoMacOSX::GetOSBuildString(std::string &s) {
  int mib[2] = {CTL_KERN, KERN_OSVERSION};
  char cstr[PATH_MAX];
  size_t cstr_len = sizeof(cstr);
  if (::sysctl(mib, 2, cstr, &cstr_len, NULL, 0) == 0) {
    s.assign(cstr, cstr_len);
    return true;
  }

  s.clear();
  return false;
}

bool HostInfoMacOSX::GetOSKernelDescription(std::string &s) {
  int mib[2] = {CTL_KERN, KERN_VERSION};
  char cstr[PATH_MAX];
  size_t cstr_len = sizeof(cstr);
  if (::sysctl(mib, 2, cstr, &cstr_len, NULL, 0) == 0) {
    s.assign(cstr, cstr_len);
    return true;
  }
  s.clear();
  return false;
}

static void ParseOSVersion(llvm::VersionTuple &version, NSString *Key) {
  @autoreleasepool {
    NSDictionary *version_info =
      [NSDictionary dictionaryWithContentsOfFile:
       @"/System/Library/CoreServices/SystemVersion.plist"];
    NSString *version_value = [version_info objectForKey: Key];
    const char *version_str = [version_value UTF8String];
    version.tryParse(version_str);
  }
}

llvm::VersionTuple HostInfoMacOSX::GetOSVersion() {
  static llvm::VersionTuple g_version;
  if (g_version.empty())
    ParseOSVersion(g_version, @"ProductVersion");
  return g_version;
}

llvm::VersionTuple HostInfoMacOSX::GetMacCatalystVersion() {
  static llvm::VersionTuple g_version;
  if (g_version.empty())
    ParseOSVersion(g_version, @"iOSSupportVersion");
  return g_version;
}


FileSpec HostInfoMacOSX::GetProgramFileSpec() {
  static FileSpec g_program_filespec;
  if (!g_program_filespec) {
    char program_fullpath[PATH_MAX];
    // If DST is NULL, then return the number of bytes needed.
    uint32_t len = sizeof(program_fullpath);
    int err = _NSGetExecutablePath(program_fullpath, &len);
    if (err == 0)
      g_program_filespec.SetFile(program_fullpath, FileSpec::Style::native);
    else if (err == -1) {
      char *large_program_fullpath = (char *)::malloc(len + 1);

      err = _NSGetExecutablePath(large_program_fullpath, &len);
      if (err == 0)
        g_program_filespec.SetFile(large_program_fullpath,
                                   FileSpec::Style::native);

      ::free(large_program_fullpath);
    }
  }
  return g_program_filespec;
}

bool HostInfoMacOSX::ComputeSupportExeDirectory(FileSpec &file_spec) {
  FileSpec lldb_file_spec = GetShlibDir();
  if (!lldb_file_spec)
    return false;

  std::string raw_path = lldb_file_spec.GetPath();

  size_t framework_pos = raw_path.find("LLDB.framework");
  if (framework_pos != std::string::npos) {
    framework_pos += strlen("LLDB.framework");
#if defined(__arm__) || defined(__arm64__) || defined(__aarch64__)
    // Shallow bundle
    raw_path.resize(framework_pos);
#else
    // Normal bundle
    raw_path.resize(framework_pos);
    raw_path.append("/Resources");
#endif
  } else {
    // Find the bin path relative to the lib path where the cmake-based
    // OS X .dylib lives.  This is not going to work if the bin and lib
    // dir are not both in the same dir.
    //
    // It is not going to work to do it by the executable path either,
    // as in the case of a python script, the executable is python, not
    // the lldb driver.
    raw_path.append("/../bin");
    FileSpec support_dir_spec(raw_path);
    FileSystem::Instance().Resolve(support_dir_spec);
    if (!FileSystem::Instance().IsDirectory(support_dir_spec)) {
      Log *log = lldb_private::GetLogIfAllCategoriesSet(LIBLLDB_LOG_HOST);
      LLDB_LOGF(log, "HostInfoMacOSX::%s(): failed to find support directory",
                __FUNCTION__);
      return false;
    }

    // Get normalization from support_dir_spec.  Note the FileSpec resolve
    // does not remove '..' in the path.
    char *const dir_realpath =
        realpath(support_dir_spec.GetPath().c_str(), NULL);
    if (dir_realpath) {
      raw_path = dir_realpath;
      free(dir_realpath);
    } else {
      raw_path = support_dir_spec.GetPath();
    }
  }

  file_spec.GetDirectory().SetString(
      llvm::StringRef(raw_path.c_str(), raw_path.size()));
  return (bool)file_spec.GetDirectory();
}

bool HostInfoMacOSX::ComputeHeaderDirectory(FileSpec &file_spec) {
  FileSpec lldb_file_spec = GetShlibDir();
  if (!lldb_file_spec)
    return false;

  std::string raw_path = lldb_file_spec.GetPath();

  size_t framework_pos = raw_path.find("LLDB.framework");
  if (framework_pos != std::string::npos) {
    framework_pos += strlen("LLDB.framework");
    raw_path.resize(framework_pos);
    raw_path.append("/Headers");
  }
  file_spec.GetDirectory().SetString(
      llvm::StringRef(raw_path.c_str(), raw_path.size()));
  return true;
}

bool HostInfoMacOSX::ComputeSystemPluginsDirectory(FileSpec &file_spec) {
  FileSpec lldb_file_spec = GetShlibDir();
  if (!lldb_file_spec)
    return false;

  std::string raw_path = lldb_file_spec.GetPath();

  size_t framework_pos = raw_path.find("LLDB.framework");
  if (framework_pos == std::string::npos)
    return false;

  framework_pos += strlen("LLDB.framework");
  raw_path.resize(framework_pos);
  raw_path.append("/Resources/PlugIns");
  file_spec.GetDirectory().SetString(
      llvm::StringRef(raw_path.c_str(), raw_path.size()));
  return true;
}

bool HostInfoMacOSX::ComputeUserPluginsDirectory(FileSpec &file_spec) {
  FileSpec temp_file("~/Library/Application Support/LLDB/PlugIns");
  FileSystem::Instance().Resolve(temp_file);
  file_spec.GetDirectory().SetCString(temp_file.GetPath().c_str());
  return true;
}

void HostInfoMacOSX::ComputeHostArchitectureSupport(ArchSpec &arch_32,
                                                    ArchSpec &arch_64) {
  // All apple systems support 32 bit execution.
  uint32_t cputype, cpusubtype;
  uint32_t is_64_bit_capable = false;
  size_t len = sizeof(cputype);
  ArchSpec host_arch;
  // These will tell us about the kernel architecture, which even on a 64
  // bit machine can be 32 bit...
  if (::sysctlbyname("hw.cputype", &cputype, &len, NULL, 0) == 0) {
    len = sizeof(cpusubtype);
    if (::sysctlbyname("hw.cpusubtype", &cpusubtype, &len, NULL, 0) != 0)
      cpusubtype = CPU_TYPE_ANY;

    len = sizeof(is_64_bit_capable);
    ::sysctlbyname("hw.cpu64bit_capable", &is_64_bit_capable, &len, NULL, 0);

    if (is_64_bit_capable) {
      if (cputype & CPU_ARCH_ABI64) {
        // We have a 64 bit kernel on a 64 bit system
        arch_64.SetArchitecture(eArchTypeMachO, cputype, cpusubtype);
      } else {
        // We have a 64 bit kernel that is returning a 32 bit cputype, the
        // cpusubtype will be correct as if it were for a 64 bit architecture
        arch_64.SetArchitecture(eArchTypeMachO, cputype | CPU_ARCH_ABI64,
                                cpusubtype);
      }

      // Now we need modify the cpusubtype for the 32 bit slices.
      uint32_t cpusubtype32 = cpusubtype;
#if defined(__i386__) || defined(__x86_64__)
      if (cpusubtype == CPU_SUBTYPE_486 || cpusubtype == CPU_SUBTYPE_X86_64_H)
        cpusubtype32 = CPU_SUBTYPE_I386_ALL;
#elif defined(__arm__) || defined(__arm64__) || defined(__aarch64__)
      if (cputype == CPU_TYPE_ARM || cputype == CPU_TYPE_ARM64)
        cpusubtype32 = CPU_SUBTYPE_ARM_V7S;
#endif
      arch_32.SetArchitecture(eArchTypeMachO, cputype & ~(CPU_ARCH_MASK),
                              cpusubtype32);

      if (cputype == CPU_TYPE_ARM || 
          cputype == CPU_TYPE_ARM64 || 
          cputype == CPU_TYPE_ARM64_32) {
// When running on a watch or tv, report the host os correctly
#if defined(TARGET_OS_TV) && TARGET_OS_TV == 1
        arch_32.GetTriple().setOS(llvm::Triple::TvOS);
        arch_64.GetTriple().setOS(llvm::Triple::TvOS);
#elif defined(TARGET_OS_BRIDGE) && TARGET_OS_BRIDGE == 1
        arch_32.GetTriple().setOS(llvm::Triple::BridgeOS);
        arch_64.GetTriple().setOS(llvm::Triple::BridgeOS);
#elif defined(TARGET_OS_WATCHOS) && TARGET_OS_WATCHOS == 1
        arch_32.GetTriple().setOS(llvm::Triple::WatchOS);
        arch_64.GetTriple().setOS(llvm::Triple::WatchOS);
#else
        arch_32.GetTriple().setOS(llvm::Triple::IOS);
        arch_64.GetTriple().setOS(llvm::Triple::IOS);
#endif
      } else {
        arch_32.GetTriple().setOS(llvm::Triple::MacOSX);
        arch_64.GetTriple().setOS(llvm::Triple::MacOSX);
      }
    } else {
      // We have a 32 bit kernel on a 32 bit system
      arch_32.SetArchitecture(eArchTypeMachO, cputype, cpusubtype);
#if defined(TARGET_OS_WATCH) && TARGET_OS_WATCH == 1
      arch_32.GetTriple().setOS(llvm::Triple::WatchOS);
#else
      arch_32.GetTriple().setOS(llvm::Triple::IOS);
#endif
      arch_64.Clear();
    }
  }
}

/// Return and cache $DEVELOPER_DIR if it is set and exists.
static std::string GetEnvDeveloperDir() {
  static std::string g_env_developer_dir;
  static std::once_flag g_once_flag;
  std::call_once(g_once_flag, [&]() {
    if (const char *developer_dir_env_var = getenv("DEVELOPER_DIR")) {
      FileSpec fspec(developer_dir_env_var);
      if (FileSystem::Instance().Exists(fspec))
        g_env_developer_dir = fspec.GetPath();
    }});
  return g_env_developer_dir;
}

FileSpec HostInfoMacOSX::GetXcodeContentsDirectory() {
  static FileSpec g_xcode_contents_path;
  static std::once_flag g_once_flag;
  std::call_once(g_once_flag, [&]() {
    // Try the shlib dir first.
    if (FileSpec fspec = HostInfo::GetShlibDir()) {
      if (FileSystem::Instance().Exists(fspec)) {
        std::string xcode_contents_dir =
            XcodeSDK::FindXcodeContentsDirectoryInPath(fspec.GetPath());
        if (!xcode_contents_dir.empty()) {
          g_xcode_contents_path = FileSpec(xcode_contents_dir);
          return;
        }
      }
    }

    std::string env_developer_dir = GetEnvDeveloperDir();
    if (!env_developer_dir.empty()) {
      // FIXME: This looks like it couldn't possibly work!
      std::string xcode_contents_dir =
          XcodeSDK::FindXcodeContentsDirectoryInPath(env_developer_dir);
      if (!xcode_contents_dir.empty()) {
        g_xcode_contents_path = FileSpec(xcode_contents_dir);
        return;
      }
    }

    FileSpec fspec(HostInfo::GetXcodeSDKPath(XcodeSDK::GetAnyMacOS()));
    if (fspec) {
      if (FileSystem::Instance().Exists(fspec)) {
        std::string xcode_contents_dir =
            XcodeSDK::FindXcodeContentsDirectoryInPath(fspec.GetPath());
        if (!xcode_contents_dir.empty()) {
          g_xcode_contents_path = FileSpec(xcode_contents_dir);
          return;
        }
      }
    }
  });
  return g_xcode_contents_path;
}

lldb_private::FileSpec HostInfoMacOSX::GetXcodeDeveloperDirectory() {
  static lldb_private::FileSpec g_developer_directory;
  static llvm::once_flag g_once_flag;
  llvm::call_once(g_once_flag, []() {
    if (FileSpec fspec = GetXcodeContentsDirectory()) {
      fspec.AppendPathComponent("Developer");
      if (FileSystem::Instance().Exists(fspec))
        g_developer_directory = fspec;
    }
  });
  return g_developer_directory;
}

static std::string GetXcodeSDK(XcodeSDK sdk) {
  XcodeSDK::Info info = sdk.Parse();
  std::string sdk_name = XcodeSDK::GetCanonicalName(info);
  auto find_sdk = [](std::string sdk_name) -> std::string {
    std::string xcrun_cmd;
    std::string developer_dir = GetEnvDeveloperDir();
    if (developer_dir.empty())
      if (FileSpec fspec = HostInfo::GetShlibDir())
        if (FileSystem::Instance().Exists(fspec)) {
          FileSpec path(
              XcodeSDK::FindXcodeContentsDirectoryInPath(fspec.GetPath()));
          if (path.RemoveLastPathComponent())
            developer_dir = path.GetPath();
        }
    if (!developer_dir.empty())
      xcrun_cmd = "/usr/bin/env DEVELOPER_DIR=\"" + developer_dir + "\" ";
    xcrun_cmd += "xcrun --show-sdk-path --sdk " + sdk_name;

    int status = 0;
    int signo = 0;
    std::string output_str;
    lldb_private::Status error =
        Host::RunShellCommand(xcrun_cmd.c_str(), FileSpec(), &status, &signo,
                              &output_str, std::chrono::seconds(15));

    // Check that xcrun return something useful.
    if (status != 0 || output_str.empty())
      return {};

    // Convert to a StringRef so we can manipulate the string without modifying
    // the underlying data.
    llvm::StringRef output(output_str);

    // Remove any trailing newline characters.
    output = output.rtrim();

    // Strip any leading newline characters and everything before them.
    const size_t last_newline = output.rfind('\n');
    if (last_newline != llvm::StringRef::npos)
      output = output.substr(last_newline + 1);

    return output.str();
  };

  std::string path = find_sdk(sdk_name);
  while (path.empty()) {
    // Try an alternate spelling of the name ("macosx10.9internal").
    if (info.type == XcodeSDK::Type::MacOSX && !info.version.empty() &&
        info.internal) {
      llvm::StringRef fixed(sdk_name);
      if (fixed.consume_back(".internal"))
        sdk_name = fixed.str() + "internal";
      path = find_sdk(sdk_name);
      if (!path.empty())
        break;
    }
    Log *log = lldb_private::GetLogIfAllCategoriesSet(LIBLLDB_LOG_HOST);
    LLDB_LOGF(log, "Couldn't find SDK %s on host", sdk_name.c_str());

    // Try without the version.
    if (!info.version.empty()) {
      info.version = {};
      sdk_name = XcodeSDK::GetCanonicalName(info);
      path = find_sdk(sdk_name);
      if (!path.empty())
        break;
    }

    LLDB_LOGF(log, "Couldn't find any matching SDK on host");
    return {};
  }

  // Whatever is left in output should be a valid path.
  if (!FileSystem::Instance().Exists(path))
    return {};
  return path;
}

llvm::StringRef HostInfoMacOSX::GetXcodeSDKPath(XcodeSDK sdk) {
  static llvm::StringMap<std::string> g_sdk_path;
  static std::mutex g_sdk_path_mutex;

  std::lock_guard<std::mutex> guard(g_sdk_path_mutex);
  auto it = g_sdk_path.find(sdk.GetString());
  if (it != g_sdk_path.end())
    return it->second;
  auto it_new = g_sdk_path.insert({sdk.GetString(), GetXcodeSDK(sdk)});
  return it_new.first->second;
}
