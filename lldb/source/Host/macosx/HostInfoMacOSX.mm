//===-- HostInfoMacOSX.mm ---------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#if !defined(LLDB_DISABLE_PYTHON)
#include "Plugins/ScriptInterpreter/Python/lldb-python.h"
#endif

#include "lldb/Host/HostInfo.h"
#include "lldb/Host/macosx/HostInfoMacOSX.h"
#include "lldb/Interpreter/Args.h"
#include "lldb/Utility/Log.h"
#include "lldb/Utility/SafeMachO.h"

#include "llvm/ADT/SmallString.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"

// C++ Includes
#include <string>

// C inclues
#include <stdlib.h>
#include <sys/sysctl.h>
#include <sys/syslimits.h>
#include <sys/types.h>

// Objective C/C++ includes
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

bool HostInfoMacOSX::GetOSVersion(uint32_t &major, uint32_t &minor,
                                  uint32_t &update) {
  static uint32_t g_major = 0;
  static uint32_t g_minor = 0;
  static uint32_t g_update = 0;

  if (g_major == 0) {
    @autoreleasepool {
      NSDictionary *version_info = [NSDictionary
          dictionaryWithContentsOfFile:
              @"/System/Library/CoreServices/SystemVersion.plist"];
      NSString *version_value = [version_info objectForKey:@"ProductVersion"];
      const char *version_str = [version_value UTF8String];
      if (version_str)
        Args::StringToVersion(llvm::StringRef(version_str), g_major, g_minor,
                              g_update);
    }
  }

  if (g_major != 0) {
    major = g_major;
    minor = g_minor;
    update = g_update;
    return true;
  }
  return false;
}

FileSpec HostInfoMacOSX::GetProgramFileSpec() {
  static FileSpec g_program_filespec;
  if (!g_program_filespec) {
    char program_fullpath[PATH_MAX];
    // If DST is NULL, then return the number of bytes needed.
    uint32_t len = sizeof(program_fullpath);
    int err = _NSGetExecutablePath(program_fullpath, &len);
    if (err == 0)
      g_program_filespec.SetFile(program_fullpath, false);
    else if (err == -1) {
      char *large_program_fullpath = (char *)::malloc(len + 1);

      err = _NSGetExecutablePath(large_program_fullpath, &len);
      if (err == 0)
        g_program_filespec.SetFile(large_program_fullpath, false);

      ::free(large_program_fullpath);
    }
  }
  return g_program_filespec;
}

bool HostInfoMacOSX::ComputeSupportExeDirectory(FileSpec &file_spec) {
  FileSpec lldb_file_spec;
  if (!GetLLDBPath(lldb::ePathTypeLLDBShlibDir, lldb_file_spec))
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
    FileSpec support_dir_spec(raw_path, true);
    if (!llvm::sys::fs::is_directory(support_dir_spec.GetPath())) {
      Log *log = lldb_private::GetLogIfAllCategoriesSet(LIBLLDB_LOG_HOST);
      if (log)
        log->Printf("HostInfoMacOSX::%s(): failed to find support directory",
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
  FileSpec lldb_file_spec;
  if (!HostInfo::GetLLDBPath(lldb::ePathTypeLLDBShlibDir, lldb_file_spec))
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

bool HostInfoMacOSX::ComputePythonDirectory(FileSpec &file_spec) {
#ifndef LLDB_DISABLE_PYTHON
  FileSpec lldb_file_spec;
  if (!GetLLDBPath(lldb::ePathTypeLLDBShlibDir, lldb_file_spec))
    return false;

  std::string raw_path = lldb_file_spec.GetPath();

  size_t framework_pos = raw_path.find("LLDB.framework");
  if (framework_pos != std::string::npos) {
    framework_pos += strlen("LLDB.framework");
    raw_path.resize(framework_pos);
    raw_path.append("/Resources/Python");
  } else {
    llvm::SmallString<256> python_version_dir;
    llvm::raw_svector_ostream os(python_version_dir);
    os << "/python" << PY_MAJOR_VERSION << '.' << PY_MINOR_VERSION
       << "/site-packages";

    // We may get our string truncated. Should we protect this with an assert?
    raw_path.append(python_version_dir.c_str());
  }
  file_spec.GetDirectory().SetString(
      llvm::StringRef(raw_path.c_str(), raw_path.size()));
  return true;
#else
  return false;
#endif
}

bool HostInfoMacOSX::ComputeClangDirectory(FileSpec &file_spec) {
  FileSpec lldb_file_spec;
  if (!GetLLDBPath(lldb::ePathTypeLLDBShlibDir, lldb_file_spec))
    return false;

  std::string raw_path = lldb_file_spec.GetPath();

  size_t framework_pos = raw_path.find("LLDB.framework");
  if (framework_pos == std::string::npos)
    return HostInfoPosix::ComputeClangDirectory(file_spec);
  
  framework_pos += strlen("LLDB.framework");
  raw_path.resize(framework_pos);
  raw_path.append("/Resources/Clang");
  
  file_spec.SetFile(raw_path.c_str(), true);
  return true;
}

bool HostInfoMacOSX::ComputeSystemPluginsDirectory(FileSpec &file_spec) {
  FileSpec lldb_file_spec;
  if (!GetLLDBPath(lldb::ePathTypeLLDBShlibDir, lldb_file_spec))
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
  FileSpec temp_file("~/Library/Application Support/LLDB/PlugIns", true);
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

      if (cputype == CPU_TYPE_ARM || cputype == CPU_TYPE_ARM64) {
// When running on a watch or tv, report the host os correctly
#if defined(TARGET_OS_TV) && TARGET_OS_TV == 1
        arch_32.GetTriple().setOS(llvm::Triple::TvOS);
        arch_64.GetTriple().setOS(llvm::Triple::TvOS);
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
