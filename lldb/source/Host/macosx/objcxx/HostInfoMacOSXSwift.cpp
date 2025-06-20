//===-- HostInfoMacOSSwift.cpp --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "lldb/Host/HostInfo.h"

#include "Plugins/Platform/MacOSX/PlatformDarwin.h"
#include "lldb/Host/Config.h"
#include "lldb/Host/FileSystem.h"
#include "lldb/Host/HostInfo.h"
#include "lldb/Host/posix/HostInfoPosix.h"
#include "lldb/Host/common/HostInfoSwift.h"
#include "lldb/Utility/FileSpec.h"
#include "lldb/Utility/LLDBLog.h"
#include "lldb/Utility/Log.h"
#include <regex>
#include <string>

using namespace lldb_private;

#ifdef LLDB_ENABLE_SWIFT

bool HostInfoMacOSX::ComputeSwiftResourceDirectory(FileSpec &lldb_shlib_spec,
                                                   FileSpec &file_spec,
                                                   bool verify) {
  if (!lldb_shlib_spec)
    return false;

  std::string raw_path = lldb_shlib_spec.GetPath();
  size_t framework_pos = raw_path.find("LLDB.framework");
  if (framework_pos == std::string::npos)
    return HostInfoPosix::ComputeSwiftResourceDirectory(lldb_shlib_spec,
                                                           file_spec, verify);

  framework_pos += strlen("LLDB.framework");
  raw_path.resize(framework_pos);
  raw_path.append("/Resources/Swift");
  if (!verify || VerifySwiftPath(raw_path)) {
    file_spec.SetDirectory(raw_path);
    FileSystem::Instance().Resolve(file_spec);
    return true;
  }
  return true;
}

FileSpec HostInfoMacOSX::GetSwiftResourceDir() {
  static std::once_flag g_once_flag;
  static FileSpec g_swift_resource_dir;
  std::call_once(g_once_flag, []() {
    FileSpec lldb_file_spec = HostInfo::GetShlibDir();
    ComputeSwiftResourceDirectory(lldb_file_spec, g_swift_resource_dir, true);
    Log *log = GetLog(LLDBLog::Host);
    LLDB_LOG(log, "swift dir -> '{0}'", g_swift_resource_dir);
  });
  return g_swift_resource_dir;
}

/// Return the Xcode sdk type for the target triple, if that makes sense.
/// Otherwise, return the unknown sdk type.
static XcodeSDK::Type GetSDKType(const llvm::Triple &target,
                                 const llvm::Triple &host) {
  // Only Darwin platforms know the concept of an SDK.
  auto host_os = host.getOS();
  if (host_os != llvm::Triple::OSType::MacOSX)
    return XcodeSDK::Type::unknown;

  auto is_simulator = [&]() -> bool {
    return target.getEnvironment() == llvm::Triple::Simulator ||
           !target.getArchName().starts_with("arm");
  };

  switch (target.getOS()) {
  case llvm::Triple::OSType::MacOSX:
  case llvm::Triple::OSType::Darwin:
    return XcodeSDK::Type::MacOSX;
  case llvm::Triple::OSType::IOS:
    if (is_simulator())
      return XcodeSDK::Type::iPhoneSimulator;
    return XcodeSDK::Type::iPhoneOS;
  case llvm::Triple::OSType::TvOS:
    if (is_simulator())
      return XcodeSDK::Type::AppleTVSimulator;
    return XcodeSDK::Type::AppleTVOS;
  case llvm::Triple::OSType::WatchOS:
    if (is_simulator())
      return XcodeSDK::Type::WatchSimulator;
    return XcodeSDK::Type::watchOS;
  case llvm::Triple::OSType::XROS:
    if (is_simulator())
      return XcodeSDK::Type::XRSimulator;
    return XcodeSDK::Type::XROS;
  default:
    return XcodeSDK::Type::unknown;
  }
}

std::string HostInfoMacOSX::GetSwiftStdlibOSDir(llvm::Triple target,
                                                llvm::Triple host) {
  XcodeSDK::Info sdk_info;
  sdk_info.type = GetSDKType(target, host);
  std::string sdk_name = XcodeSDK::GetCanonicalName(sdk_info);
  if (!sdk_name.empty())
    return sdk_name;
  return target.getOSName().str();
}

static bool IsDirectory(const FileSpec &spec) {
  return llvm::sys::fs::is_directory(spec.GetPath());
}

std::string HostInfoMacOSX::DetectSwiftResourceDir(
    llvm::StringRef platform_sdk_path, llvm::StringRef swift_stdlib_os_dir,
    std::string swift_dir, std::string xcode_contents_path,
    std::string toolchain_path, std::string cl_tools_path) {
  llvm::SmallString<16> m_description("SwiftASTContext");
  // First, check if there's something in our bundle.
  {
    FileSpec swift_dir_spec(swift_dir);
    if (swift_dir_spec) {
      LLDB_LOGF(GetLog(LLDBLog::Types), "trying ePathTypeSwiftDir: %s",
                swift_dir_spec.GetPath().c_str());
      // We can't just check for the Swift directory, because that
      // always exists.  We have to look for "clang" inside that.
      FileSpec swift_clang_dir_spec = swift_dir_spec;
      swift_clang_dir_spec.AppendPathComponent("clang");

      if (IsDirectory(swift_clang_dir_spec)) {
        LLDB_LOGF(GetLog(LLDBLog::Types),
                  "found Swift resource dir via ePathTypeSwiftDir': %s",
                  swift_dir_spec.GetPath().c_str());
        return swift_dir_spec.GetPath();
      }
    }
  }

  // Nothing in our bundle. Are we in a toolchain that has its own Swift
  // compiler resource dir?

  {
    llvm::SmallString<256> path(toolchain_path);
    LLDB_LOGF(GetLog(LLDBLog::Types), "trying toolchain path: %s",
               path.c_str());

    if (!path.empty()) {
      llvm::sys::path::append(path, "usr/lib/swift");
      LLDB_LOGF(GetLog(LLDBLog::Types), "trying toolchain-based lib path: %s",
                 path.c_str());

      if (IsDirectory(FileSpec(path))) {
        LLDB_LOGF(GetLog(LLDBLog::Types),
                   "found Swift resource dir via "
                   "toolchain path + 'usr/lib/swift': %s",
                   path.c_str());
        return std::string(path);
      }
    }
  }

  // We're not in a toolchain that has one. Use the Xcode default toolchain.

  {
    llvm::SmallString<256> path(xcode_contents_path);
    LLDB_LOGF(GetLog(LLDBLog::Types), "trying Xcode path: %s", path.c_str());

    if (!path.empty()) {
      llvm::sys::path::append(path, "Developer",
                              "Toolchains/XcodeDefault.xctoolchain",
                              "usr/lib/swift");
      LLDB_LOGF(GetLog(LLDBLog::Types), "trying Xcode-based lib path: %s",
                 path.c_str());

      if (IsDirectory(FileSpec(path))) {
        llvm::StringRef resource_dir = path;
        llvm::sys::path::append(path, swift_stdlib_os_dir);
        std::string s(path);
        if (IsDirectory(FileSpec(path))) {
          LLDB_LOGF(GetLog(LLDBLog::Types),
                     "found Swift resource dir via "
                     "Xcode contents path + default toolchain "
                     "relative dir: %s",
                     resource_dir.str().c_str());
          return resource_dir.str();
        } else {
          // Search the SDK for a matching cross-SDK.
          path = platform_sdk_path;
          llvm::sys::path::append(path, "usr/lib/swift");
          llvm::StringRef resource_dir = path;
          llvm::sys::path::append(path, swift_stdlib_os_dir);
          if (IsDirectory(FileSpec(path))) {
            LLDB_LOGF(GetLog(LLDBLog::Types),
                       "found Swift resource dir via "
                       "Xcode contents path + cross-compilation SDK "
                       "relative dir: %s",
                       resource_dir.str().c_str());
            return resource_dir.str();
          }
        }
      }
    }
  }

  // We're not in Xcode. We might be in the command-line tools.

  {
    llvm::SmallString<256> path(cl_tools_path);
    LLDB_LOGF(GetLog(LLDBLog::Types), "trying command-line tools path: %s",
               path.c_str());

    if (!path.empty()) {
      llvm::sys::path::append(path, "usr/lib/swift");
      LLDB_LOGF(GetLog(LLDBLog::Types),
                 "trying command-line tools-based lib path: %s", path.c_str());

      if (IsDirectory(FileSpec(path))) {
        LLDB_LOGF(GetLog(LLDBLog::Types),
                   "found Swift resource dir via command-line tools "
                   "path + usr/lib/swift: %s",
                   path.c_str());
        return std::string(path);
      }
    }
  }

  // We might be in the build-dir configuration for a
  // build-script-driven LLDB build, which has the Swift build dir as
  // a sibling directory to the lldb build dir.  This looks much
  // different than the install- dir layout that the previous checks
  // would try.
  {
    FileSpec faux_swift_dir_spec(swift_dir);
    if (faux_swift_dir_spec) {
      // Let's try to regex this.
      // We're looking for /some/path/lldb-{os}-{arch}, and want to
      // build the following:
      //    /some/path/swift-{os}-{arch}/lib/swift/{os}/{arch}
      // In a match, these are the following assignments for
      // backrefs:
      //   $1 - first part of path before swift build dir
      //   $2 - the host OS path separator character
      //   $3 - all the stuff that should come after changing
      //        lldb to swift for the lib dir.
      auto match_regex =
          std::regex("^(.+([/\\\\]))lldb-(.+)$");
      const std::string replace_format = "$1swift-$3";
      const std::string faux_swift_dir = faux_swift_dir_spec.GetPath();
      const std::string build_tree_resource_dir =
          std::regex_replace(faux_swift_dir, match_regex,
                             replace_format);
      LLDB_LOGF(GetLog(LLDBLog::Types),
                 "trying ePathTypeSwiftDir regex-based build dir: %s",
                 build_tree_resource_dir.c_str());
      FileSpec swift_resource_dir_spec(build_tree_resource_dir.c_str());
      if (IsDirectory(swift_resource_dir_spec)) {
        LLDB_LOGF(GetLog(LLDBLog::Types),
                  "found Swift resource dir via "
                  "ePathTypeSwiftDir + inferred build-tree dir: %s",
                  swift_resource_dir_spec.GetPath().c_str());
        return swift_resource_dir_spec.GetPath();
      }
    }
  }

  // We failed to find a reasonable Swift resource dir.
  LLDB_LOGF(GetLog(LLDBLog::Types), "failed to find a Swift resource dir");

  return {};
}

std::string
HostInfoMacOSX::GetSwiftResourceDir(llvm::Triple triple,
                                    llvm::StringRef platform_sdk_path) {
  static std::mutex g_mutex;
  std::lock_guard<std::mutex> locker(g_mutex);
  std::string swift_stdlib_os_dir =
      GetSwiftStdlibOSDir(triple, HostInfo::GetArchitecture().GetTriple());

  // The resource dir depends on the SDK path and the expected OS name.
  llvm::SmallString<128> key(platform_sdk_path);
  key.append(swift_stdlib_os_dir);
  static llvm::StringMap<std::string> g_resource_dir_cache;
  auto it = g_resource_dir_cache.find(key);
  if (it != g_resource_dir_cache.end())
    return it->getValue();

  auto value = DetectSwiftResourceDir(
      platform_sdk_path, swift_stdlib_os_dir,
      HostInfo::GetSwiftResourceDir().GetPath(),
      HostInfo::GetXcodeContentsDirectory().GetPath(),
      PlatformDarwin::GetCurrentToolchainDirectory().GetPath(),
      PlatformDarwin::GetCurrentCommandLineToolsDirectory().GetPath());
  g_resource_dir_cache.insert({key, value});
  return g_resource_dir_cache[key];
}

#endif
