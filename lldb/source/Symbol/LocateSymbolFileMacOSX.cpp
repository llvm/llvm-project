//===-- LocateSymbolFileMacOSX.cpp ----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Symbol/LocateSymbolFile.h"

#include <dirent.h>
#include <dlfcn.h>
#include <pwd.h>

#include <CoreFoundation/CoreFoundation.h>

#include "Host/macosx/cfcpp/CFCBundle.h"
#include "Host/macosx/cfcpp/CFCData.h"
#include "Host/macosx/cfcpp/CFCReleaser.h"
#include "Host/macosx/cfcpp/CFCString.h"
#include "lldb/Core/Module.h"
#include "lldb/Core/ModuleList.h"
#include "lldb/Core/ModuleSpec.h"
#include "lldb/Host/Host.h"
#include "lldb/Host/HostInfo.h"
#include "lldb/Symbol/ObjectFile.h"
#include "lldb/Utility/ArchSpec.h"
#include "lldb/Utility/DataBuffer.h"
#include "lldb/Utility/DataExtractor.h"
#include "lldb/Utility/Endian.h"
#include "lldb/Utility/LLDBLog.h"
#include "lldb/Utility/Log.h"
#include "lldb/Utility/StreamString.h"
#include "lldb/Utility/Timer.h"
#include "lldb/Utility/UUID.h"
#include "mach/machine.h"

#include "llvm/ADT/ScopeExit.h"
#include "llvm/Support/FileSystem.h"

using namespace lldb;
using namespace lldb_private;

static CFURLRef (*g_dlsym_DBGCopyFullDSYMURLForUUID)(
    CFUUIDRef uuid, CFURLRef exec_url) = nullptr;
static CFDictionaryRef (*g_dlsym_DBGCopyDSYMPropertyLists)(CFURLRef dsym_url) =
    nullptr;

int LocateMacOSXFilesUsingDebugSymbols(const ModuleSpec &module_spec,
                                       ModuleSpec &return_module_spec) {
  Log *log = GetLog(LLDBLog::Host);
  if (!ModuleList::GetGlobalModuleListProperties().GetEnableExternalLookup()) {
    LLDB_LOGF(log, "Spotlight lookup for .dSYM bundles is disabled.");
    return 0;
  }

  return_module_spec = module_spec;
  return_module_spec.GetFileSpec().Clear();
  return_module_spec.GetSymbolFileSpec().Clear();

  const UUID *uuid = module_spec.GetUUIDPtr();
  const ArchSpec *arch = module_spec.GetArchitecturePtr();

  int items_found = 0;

  if (g_dlsym_DBGCopyFullDSYMURLForUUID == nullptr ||
      g_dlsym_DBGCopyDSYMPropertyLists == nullptr) {
    void *handle = dlopen(
        "/System/Library/PrivateFrameworks/DebugSymbols.framework/DebugSymbols",
        RTLD_LAZY | RTLD_LOCAL);
    if (handle) {
      g_dlsym_DBGCopyFullDSYMURLForUUID =
          (CFURLRef(*)(CFUUIDRef, CFURLRef))dlsym(handle,
                                                  "DBGCopyFullDSYMURLForUUID");
      g_dlsym_DBGCopyDSYMPropertyLists = (CFDictionaryRef(*)(CFURLRef))dlsym(
          handle, "DBGCopyDSYMPropertyLists");
    }
  }

  if (g_dlsym_DBGCopyFullDSYMURLForUUID == nullptr ||
      g_dlsym_DBGCopyDSYMPropertyLists == nullptr) {
    return items_found;
  }

  if (uuid && uuid->IsValid()) {
    // Try and locate the dSYM file using DebugSymbols first
    llvm::ArrayRef<uint8_t> module_uuid = uuid->GetBytes();
    if (module_uuid.size() == 16) {
      CFCReleaser<CFUUIDRef> module_uuid_ref(::CFUUIDCreateWithBytes(
          NULL, module_uuid[0], module_uuid[1], module_uuid[2], module_uuid[3],
          module_uuid[4], module_uuid[5], module_uuid[6], module_uuid[7],
          module_uuid[8], module_uuid[9], module_uuid[10], module_uuid[11],
          module_uuid[12], module_uuid[13], module_uuid[14], module_uuid[15]));

      if (module_uuid_ref.get()) {
        CFCReleaser<CFURLRef> exec_url;
        const FileSpec *exec_fspec = module_spec.GetFileSpecPtr();
        if (exec_fspec) {
          char exec_cf_path[PATH_MAX];
          if (exec_fspec->GetPath(exec_cf_path, sizeof(exec_cf_path)))
            exec_url.reset(::CFURLCreateFromFileSystemRepresentation(
                NULL, (const UInt8 *)exec_cf_path, strlen(exec_cf_path),
                FALSE));
        }

        CFCReleaser<CFURLRef> dsym_url(g_dlsym_DBGCopyFullDSYMURLForUUID(
            module_uuid_ref.get(), exec_url.get()));
        char path[PATH_MAX];

        if (dsym_url.get()) {
          if (::CFURLGetFileSystemRepresentation(
                  dsym_url.get(), true, (UInt8 *)path, sizeof(path) - 1)) {
            LLDB_LOGF(log,
                      "DebugSymbols framework returned dSYM path of %s for "
                      "UUID %s -- looking for the dSYM",
                      path, uuid->GetAsString().c_str());
            FileSpec dsym_filespec(path);
            if (path[0] == '~')
              FileSystem::Instance().Resolve(dsym_filespec);

            if (FileSystem::Instance().IsDirectory(dsym_filespec)) {
              dsym_filespec =
                  Symbols::FindSymbolFileInBundle(dsym_filespec, uuid, arch);
              ++items_found;
            } else {
              ++items_found;
            }
            return_module_spec.GetSymbolFileSpec() = dsym_filespec;
          }

          bool success = false;
          if (log) {
            if (::CFURLGetFileSystemRepresentation(
                    dsym_url.get(), true, (UInt8 *)path, sizeof(path) - 1)) {
              LLDB_LOGF(log,
                        "DebugSymbols framework returned dSYM path of %s for "
                        "UUID %s -- looking for an exec file",
                        path, uuid->GetAsString().c_str());
            }
          }

          CFCReleaser<CFDictionaryRef> dict(
              g_dlsym_DBGCopyDSYMPropertyLists(dsym_url.get()));
          CFDictionaryRef uuid_dict = NULL;
          if (dict.get()) {
            CFCString uuid_cfstr(uuid->GetAsString().c_str());
            uuid_dict = static_cast<CFDictionaryRef>(
                ::CFDictionaryGetValue(dict.get(), uuid_cfstr.get()));
          }

          // Check to see if we have the file on the local filesystem.
          if (FileSystem::Instance().Exists(module_spec.GetFileSpec())) {
            ModuleSpec exe_spec;
            exe_spec.GetFileSpec() = module_spec.GetFileSpec();
            exe_spec.GetUUID() = module_spec.GetUUID();
            ModuleSP module_sp;
            module_sp.reset(new Module(exe_spec));
            if (module_sp && module_sp->GetObjectFile() &&
                module_sp->MatchesModuleSpec(exe_spec)) {
              success = true;
              return_module_spec.GetFileSpec() = module_spec.GetFileSpec();
              LLDB_LOGF(log, "using original binary filepath %s for UUID %s",
                        module_spec.GetFileSpec().GetPath().c_str(),
                        uuid->GetAsString().c_str());
              ++items_found;
            }
          }

          // Check if the requested image is in our shared cache.
          if (!success) {
            SharedCacheImageInfo image_info = HostInfo::GetSharedCacheImageInfo(
                module_spec.GetFileSpec().GetPath());

            // If we found it and it has the correct UUID, let's proceed with
            // creating a module from the memory contents.
            if (image_info.uuid && (!module_spec.GetUUID() ||
                                    module_spec.GetUUID() == image_info.uuid)) {
              success = true;
              return_module_spec.GetFileSpec() = module_spec.GetFileSpec();
              LLDB_LOGF(log,
                        "using binary from shared cache for filepath %s for "
                        "UUID %s",
                        module_spec.GetFileSpec().GetPath().c_str(),
                        uuid->GetAsString().c_str());
              ++items_found;
            }
          }

          // Use the DBGSymbolRichExecutable filepath if present
          if (!success && uuid_dict) {
            CFStringRef exec_cf_path =
                static_cast<CFStringRef>(::CFDictionaryGetValue(
                    uuid_dict, CFSTR("DBGSymbolRichExecutable")));
            if (exec_cf_path && ::CFStringGetFileSystemRepresentation(
                                    exec_cf_path, path, sizeof(path))) {
              LLDB_LOGF(log, "plist bundle has exec path of %s for UUID %s",
                        path, uuid->GetAsString().c_str());
              ++items_found;
              FileSpec exec_filespec(path);
              if (path[0] == '~')
                FileSystem::Instance().Resolve(exec_filespec);
              if (FileSystem::Instance().Exists(exec_filespec)) {
                success = true;
                return_module_spec.GetFileSpec() = exec_filespec;
              }
            }
          }

          // Look next to the dSYM for the binary file.
          if (!success) {
            if (::CFURLGetFileSystemRepresentation(
                    dsym_url.get(), true, (UInt8 *)path, sizeof(path) - 1)) {
              char *dsym_extension_pos = ::strstr(path, ".dSYM");
              if (dsym_extension_pos) {
                *dsym_extension_pos = '\0';
                LLDB_LOGF(log,
                          "Looking for executable binary next to dSYM "
                          "bundle with name with name %s",
                          path);
                FileSpec file_spec(path);
                FileSystem::Instance().Resolve(file_spec);
                ModuleSpecList module_specs;
                ModuleSpec matched_module_spec;
                using namespace llvm::sys::fs;
                switch (get_file_type(file_spec.GetPath())) {

                case file_type::directory_file: // Bundle directory?
                {
                  CFCBundle bundle(path);
                  CFCReleaser<CFURLRef> bundle_exe_url(
                      bundle.CopyExecutableURL());
                  if (bundle_exe_url.get()) {
                    if (::CFURLGetFileSystemRepresentation(bundle_exe_url.get(),
                                                           true, (UInt8 *)path,
                                                           sizeof(path) - 1)) {
                      FileSpec bundle_exe_file_spec(path);
                      FileSystem::Instance().Resolve(bundle_exe_file_spec);
                      if (ObjectFile::GetModuleSpecifications(
                              bundle_exe_file_spec, 0, 0, module_specs) &&
                          module_specs.FindMatchingModuleSpec(
                              module_spec, matched_module_spec))

                      {
                        ++items_found;
                        return_module_spec.GetFileSpec() = bundle_exe_file_spec;
                        LLDB_LOGF(log,
                                  "Executable binary %s next to dSYM is "
                                  "compatible; using",
                                  path);
                      }
                    }
                  }
                } break;

                case file_type::fifo_file:      // Forget pipes
                case file_type::socket_file:    // We can't process socket files
                case file_type::file_not_found: // File doesn't exist...
                case file_type::status_error:
                  break;

                case file_type::type_unknown:
                case file_type::regular_file:
                case file_type::symlink_file:
                case file_type::block_file:
                case file_type::character_file:
                  if (ObjectFile::GetModuleSpecifications(file_spec, 0, 0,
                                                          module_specs) &&
                      module_specs.FindMatchingModuleSpec(module_spec,
                                                          matched_module_spec))

                  {
                    ++items_found;
                    return_module_spec.GetFileSpec() = file_spec;
                    LLDB_LOGF(log,
                              "Executable binary %s next to dSYM is "
                              "compatible; using",
                              path);
                  }
                  break;
                }
              }
            }
          }
        }
      }
    }
  }

  return items_found;
}

FileSpec Symbols::FindSymbolFileInBundle(const FileSpec &dsym_bundle_fspec,
                                         const lldb_private::UUID *uuid,
                                         const ArchSpec *arch) {
  std::string dsym_bundle_path = dsym_bundle_fspec.GetPath();
  llvm::SmallString<128> buffer(dsym_bundle_path);
  llvm::sys::path::append(buffer, "Contents", "Resources", "DWARF");

  std::error_code EC;
  llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem> vfs =
      FileSystem::Instance().GetVirtualFileSystem();
  llvm::vfs::recursive_directory_iterator Iter(*vfs, buffer.str(), EC);
  llvm::vfs::recursive_directory_iterator End;
  for (; Iter != End && !EC; Iter.increment(EC)) {
    llvm::ErrorOr<llvm::vfs::Status> Status = vfs->status(Iter->path());
    if (Status->isDirectory())
      continue;

    FileSpec dsym_fspec(Iter->path());
    ModuleSpecList module_specs;
    if (ObjectFile::GetModuleSpecifications(dsym_fspec, 0, 0, module_specs)) {
      ModuleSpec spec;
      for (size_t i = 0; i < module_specs.GetSize(); ++i) {
        bool got_spec = module_specs.GetModuleSpecAtIndex(i, spec);
        assert(got_spec); // The call has side-effects so can't be inlined.
        UNUSED_IF_ASSERT_DISABLED(got_spec);
        if ((uuid == nullptr ||
             (spec.GetUUIDPtr() && spec.GetUUID() == *uuid)) &&
            (arch == nullptr ||
             (spec.GetArchitecturePtr() &&
              spec.GetArchitecture().IsCompatibleMatch(*arch)))) {
          return dsym_fspec;
        }
      }
    }
  }

  return {};
}

static bool GetModuleSpecInfoFromUUIDDictionary(CFDictionaryRef uuid_dict,
                                                ModuleSpec &module_spec,
                                                Status &error) {
  Log *log = GetLog(LLDBLog::Host);
  bool success = false;
  if (uuid_dict != NULL && CFGetTypeID(uuid_dict) == CFDictionaryGetTypeID()) {
    std::string str;
    CFStringRef cf_str;
    CFDictionaryRef cf_dict;

    cf_str = (CFStringRef)CFDictionaryGetValue((CFDictionaryRef)uuid_dict,
                                               CFSTR("DBGError"));
    if (cf_str && CFGetTypeID(cf_str) == CFStringGetTypeID()) {
      if (CFCString::FileSystemRepresentation(cf_str, str)) {
        error.SetErrorString(str);
      }
    }

    cf_str = (CFStringRef)CFDictionaryGetValue(
        (CFDictionaryRef)uuid_dict, CFSTR("DBGSymbolRichExecutable"));
    if (cf_str && CFGetTypeID(cf_str) == CFStringGetTypeID()) {
      if (CFCString::FileSystemRepresentation(cf_str, str)) {
        module_spec.GetFileSpec().SetFile(str.c_str(), FileSpec::Style::native);
        FileSystem::Instance().Resolve(module_spec.GetFileSpec());
        LLDB_LOGF(log,
                  "From dsymForUUID plist: Symbol rich executable is at '%s'",
                  str.c_str());
      }
    }

    cf_str = (CFStringRef)CFDictionaryGetValue((CFDictionaryRef)uuid_dict,
                                               CFSTR("DBGDSYMPath"));
    if (cf_str && CFGetTypeID(cf_str) == CFStringGetTypeID()) {
      if (CFCString::FileSystemRepresentation(cf_str, str)) {
        module_spec.GetSymbolFileSpec().SetFile(str.c_str(),
                                                FileSpec::Style::native);
        FileSystem::Instance().Resolve(module_spec.GetFileSpec());
        success = true;
        LLDB_LOGF(log, "From dsymForUUID plist: dSYM is at '%s'", str.c_str());
      }
    }

    std::string DBGBuildSourcePath;
    std::string DBGSourcePath;

    // If DBGVersion 1 or DBGVersion missing, ignore DBGSourcePathRemapping.
    // If DBGVersion 2, strip last two components of path remappings from
    //                  entries to fix an issue with a specific set of
    //                  DBGSourcePathRemapping entries that lldb worked
    //                  with.
    // If DBGVersion 3, trust & use the source path remappings as-is.
    //
    cf_dict = (CFDictionaryRef)CFDictionaryGetValue(
        (CFDictionaryRef)uuid_dict, CFSTR("DBGSourcePathRemapping"));
    if (cf_dict && CFGetTypeID(cf_dict) == CFDictionaryGetTypeID()) {
      // If we see DBGVersion with a value of 2 or higher, this is a new style
      // DBGSourcePathRemapping dictionary
      bool new_style_source_remapping_dictionary = false;
      bool do_truncate_remapping_names = false;
      std::string original_DBGSourcePath_value = DBGSourcePath;
      cf_str = (CFStringRef)CFDictionaryGetValue((CFDictionaryRef)uuid_dict,
                                                 CFSTR("DBGVersion"));
      if (cf_str && CFGetTypeID(cf_str) == CFStringGetTypeID()) {
        std::string version;
        CFCString::FileSystemRepresentation(cf_str, version);
        if (!version.empty() && isdigit(version[0])) {
          int version_number = atoi(version.c_str());
          if (version_number > 1) {
            new_style_source_remapping_dictionary = true;
          }
          if (version_number == 2) {
            do_truncate_remapping_names = true;
          }
        }
      }

      CFIndex kv_pair_count = CFDictionaryGetCount((CFDictionaryRef)uuid_dict);
      if (kv_pair_count > 0) {
        CFStringRef *keys =
            (CFStringRef *)malloc(kv_pair_count * sizeof(CFStringRef));
        CFStringRef *values =
            (CFStringRef *)malloc(kv_pair_count * sizeof(CFStringRef));
        if (keys != nullptr && values != nullptr) {
          CFDictionaryGetKeysAndValues((CFDictionaryRef)uuid_dict,
                                       (const void **)keys,
                                       (const void **)values);
        }
        for (CFIndex i = 0; i < kv_pair_count; i++) {
          DBGBuildSourcePath.clear();
          DBGSourcePath.clear();
          if (keys[i] && CFGetTypeID(keys[i]) == CFStringGetTypeID()) {
            CFCString::FileSystemRepresentation(keys[i], DBGBuildSourcePath);
          }
          if (values[i] && CFGetTypeID(values[i]) == CFStringGetTypeID()) {
            CFCString::FileSystemRepresentation(values[i], DBGSourcePath);
          }
          if (!DBGBuildSourcePath.empty() && !DBGSourcePath.empty()) {
            // In the "old style" DBGSourcePathRemapping dictionary, the
            // DBGSourcePath values (the "values" half of key-value path pairs)
            // were wrong.  Ignore them and use the universal DBGSourcePath
            // string from earlier.
            if (new_style_source_remapping_dictionary &&
                !original_DBGSourcePath_value.empty()) {
              DBGSourcePath = original_DBGSourcePath_value;
            }
            if (DBGSourcePath[0] == '~') {
              FileSpec resolved_source_path(DBGSourcePath.c_str());
              FileSystem::Instance().Resolve(resolved_source_path);
              DBGSourcePath = resolved_source_path.GetPath();
            }
            // With version 2 of DBGSourcePathRemapping, we can chop off the
            // last two filename parts from the source remapping and get a more
            // general source remapping that still works. Add this as another
            // option in addition to the full source path remap.
            module_spec.GetSourceMappingList().Append(DBGBuildSourcePath,
                                                      DBGSourcePath, true);
            if (do_truncate_remapping_names) {
              FileSpec build_path(DBGBuildSourcePath.c_str());
              FileSpec source_path(DBGSourcePath.c_str());
              build_path.RemoveLastPathComponent();
              build_path.RemoveLastPathComponent();
              source_path.RemoveLastPathComponent();
              source_path.RemoveLastPathComponent();
              module_spec.GetSourceMappingList().Append(
                  build_path.GetPath(), source_path.GetPath(), true);
            }
          }
        }
        if (keys)
          free(keys);
        if (values)
          free(values);
      }
    }

    // If we have a DBGBuildSourcePath + DBGSourcePath pair, append them to the
    // source remappings list.

    cf_str = (CFStringRef)CFDictionaryGetValue((CFDictionaryRef)uuid_dict,
                                               CFSTR("DBGBuildSourcePath"));
    if (cf_str && CFGetTypeID(cf_str) == CFStringGetTypeID()) {
      CFCString::FileSystemRepresentation(cf_str, DBGBuildSourcePath);
    }

    cf_str = (CFStringRef)CFDictionaryGetValue((CFDictionaryRef)uuid_dict,
                                               CFSTR("DBGSourcePath"));
    if (cf_str && CFGetTypeID(cf_str) == CFStringGetTypeID()) {
      CFCString::FileSystemRepresentation(cf_str, DBGSourcePath);
    }

    if (!DBGBuildSourcePath.empty() && !DBGSourcePath.empty()) {
      if (DBGSourcePath[0] == '~') {
        FileSpec resolved_source_path(DBGSourcePath.c_str());
        FileSystem::Instance().Resolve(resolved_source_path);
        DBGSourcePath = resolved_source_path.GetPath();
      }
      module_spec.GetSourceMappingList().Append(DBGBuildSourcePath,
                                                DBGSourcePath, true);
    }
  }
  return success;
}

/// It's expensive to check for the DBGShellCommands defaults setting. Only do
/// it once per lldb run and cache the result.
static llvm::StringRef GetDbgShellCommand() {
  static std::once_flag g_once_flag;
  static std::string g_dbgshell_command;
  std::call_once(g_once_flag, [&]() {
    CFTypeRef defaults_setting = CFPreferencesCopyAppValue(
        CFSTR("DBGShellCommands"), CFSTR("com.apple.DebugSymbols"));
    if (defaults_setting &&
        CFGetTypeID(defaults_setting) == CFStringGetTypeID()) {
      char buffer[PATH_MAX];
      if (CFStringGetCString((CFStringRef)defaults_setting, buffer,
                             sizeof(buffer), kCFStringEncodingUTF8)) {
        g_dbgshell_command = buffer;
      }
    }
    if (defaults_setting) {
      CFRelease(defaults_setting);
    }
  });
  return g_dbgshell_command;
}

/// Get the dsymForUUID executable and cache the result so we don't end up
/// stat'ing the binary over and over.
static FileSpec GetDsymForUUIDExecutable() {
  // The LLDB_APPLE_DSYMFORUUID_EXECUTABLE environment variable is used by the
  // test suite to override the dsymForUUID location. Because we must be able
  // to change the value within a single test, don't bother caching it.
  if (const char *dsymForUUID_env =
          getenv("LLDB_APPLE_DSYMFORUUID_EXECUTABLE")) {
    FileSpec dsymForUUID_executable(dsymForUUID_env);
    FileSystem::Instance().Resolve(dsymForUUID_executable);
    if (FileSystem::Instance().Exists(dsymForUUID_executable))
      return dsymForUUID_executable;
  }

  static std::once_flag g_once_flag;
  static FileSpec g_dsymForUUID_executable;
  std::call_once(g_once_flag, [&]() {
    // Try the DBGShellCommand.
    llvm::StringRef dbgshell_command = GetDbgShellCommand();
    if (!dbgshell_command.empty()) {
      g_dsymForUUID_executable = FileSpec(dbgshell_command);
      FileSystem::Instance().Resolve(g_dsymForUUID_executable);
      if (FileSystem::Instance().Exists(g_dsymForUUID_executable))
        return;
    }

    // Try dsymForUUID in /usr/local/bin
    {
      g_dsymForUUID_executable = FileSpec("/usr/local/bin/dsymForUUID");
      if (FileSystem::Instance().Exists(g_dsymForUUID_executable))
        return;
    }

    // We couldn't find the dsymForUUID binary.
    g_dsymForUUID_executable = {};
  });
  return g_dsymForUUID_executable;
}

bool Symbols::DownloadObjectAndSymbolFile(ModuleSpec &module_spec,
                                          Status &error, bool force_lookup,
                                          bool copy_executable) {
  const UUID *uuid_ptr = module_spec.GetUUIDPtr();
  const FileSpec *file_spec_ptr = module_spec.GetFileSpecPtr();

  // If \a dbgshell_command is set, the user has specified
  // forced symbol lookup via that command.  We'll get the
  // path back from GetDsymForUUIDExecutable() later.
  llvm::StringRef dbgshell_command = GetDbgShellCommand();

  // If forced lookup isn't set, by the user's \a dbgshell_command or
  // by the \a force_lookup argument, exit this method.
  if (!force_lookup && dbgshell_command.empty())
    return false;

  // We need a UUID or valid existing FileSpec.
  if (!uuid_ptr &&
      (!file_spec_ptr || !FileSystem::Instance().Exists(*file_spec_ptr)))
    return false;

  // We need a dsymForUUID binary or an equivalent executable/script.
  FileSpec dsymForUUID_exe_spec = GetDsymForUUIDExecutable();
  if (!dsymForUUID_exe_spec)
    return false;

  const std::string dsymForUUID_exe_path = dsymForUUID_exe_spec.GetPath();
  const std::string uuid_str = uuid_ptr ? uuid_ptr->GetAsString() : "";
  const std::string file_path_str =
      file_spec_ptr ? file_spec_ptr->GetPath() : "";

  Log *log = GetLog(LLDBLog::Host);

  // Create the dsymForUUID command.
  StreamString command;
  const char *copy_executable_arg = copy_executable ? "--copyExecutable " : "";
  if (!uuid_str.empty()) {
    command.Printf("%s --ignoreNegativeCache %s%s",
                   dsymForUUID_exe_path.c_str(), copy_executable_arg,
                   uuid_str.c_str());
    LLDB_LOGF(log, "Calling %s with UUID %s to find dSYM: %s",
              dsymForUUID_exe_path.c_str(), uuid_str.c_str(),
              command.GetString().data());
  } else if (!file_path_str.empty()) {
    command.Printf("%s --ignoreNegativeCache %s%s",
                   dsymForUUID_exe_path.c_str(), copy_executable_arg,
                   file_path_str.c_str());
    LLDB_LOGF(log, "Calling %s with file %s to find dSYM: %s",
              dsymForUUID_exe_path.c_str(), file_path_str.c_str(),
              command.GetString().data());
  } else {
    return false;
  }

  // Invoke dsymForUUID.
  int exit_status = -1;
  int signo = -1;
  std::string command_output;
  error = Host::RunShellCommand(
      command.GetData(),
      FileSpec(),      // current working directory
      &exit_status,    // Exit status
      &signo,          // Signal int *
      &command_output, // Command output
      std::chrono::seconds(
          640), // Large timeout to allow for long dsym download times
      false);   // Don't run in a shell (we don't need shell expansion)

  if (error.Fail() || exit_status != 0 || command_output.empty()) {
    LLDB_LOGF(log, "'%s' failed (exit status: %d, error: '%s', output: '%s')",
              command.GetData(), exit_status, error.AsCString(),
              command_output.c_str());
    return false;
  }

  CFCData data(
      CFDataCreateWithBytesNoCopy(NULL, (const UInt8 *)command_output.data(),
                                  command_output.size(), kCFAllocatorNull));

  CFCReleaser<CFDictionaryRef> plist(
      (CFDictionaryRef)::CFPropertyListCreateWithData(
          NULL, data.get(), kCFPropertyListImmutable, NULL, NULL));

  if (!plist.get()) {
    LLDB_LOGF(log, "'%s' failed: output is not a valid plist",
              command.GetData());
    return false;
  }

  if (CFGetTypeID(plist.get()) != CFDictionaryGetTypeID()) {
    LLDB_LOGF(log, "'%s' failed: output plist is not a valid CFDictionary",
              command.GetData());
    return false;
  }

  if (!uuid_str.empty()) {
    CFCString uuid_cfstr(uuid_str.c_str());
    CFDictionaryRef uuid_dict =
        (CFDictionaryRef)CFDictionaryGetValue(plist.get(), uuid_cfstr.get());
    return GetModuleSpecInfoFromUUIDDictionary(uuid_dict, module_spec, error);
  }

  if (const CFIndex num_values = ::CFDictionaryGetCount(plist.get())) {
    std::vector<CFStringRef> keys(num_values, NULL);
    std::vector<CFDictionaryRef> values(num_values, NULL);
    ::CFDictionaryGetKeysAndValues(plist.get(), NULL,
                                   (const void **)&values[0]);
    if (num_values == 1) {
      return GetModuleSpecInfoFromUUIDDictionary(values[0], module_spec, error);
    }

    for (CFIndex i = 0; i < num_values; ++i) {
      ModuleSpec curr_module_spec;
      if (GetModuleSpecInfoFromUUIDDictionary(values[i], curr_module_spec,
                                              error)) {
        if (module_spec.GetArchitecture().IsCompatibleMatch(
                curr_module_spec.GetArchitecture())) {
          module_spec = curr_module_spec;
          return true;
        }
      }
    }
  }

  return false;
}
