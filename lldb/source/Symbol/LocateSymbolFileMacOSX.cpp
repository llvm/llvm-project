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
#include "lldb/Core/ModuleList.h"
#include "lldb/Core/ModuleSpec.h"
#include "lldb/Host/Host.h"
#include "lldb/Symbol/ObjectFile.h"
#include "lldb/Utility/ArchSpec.h"
#include "lldb/Utility/DataBuffer.h"
#include "lldb/Utility/DataExtractor.h"
#include "lldb/Utility/Endian.h"
#include "lldb/Utility/Log.h"
#include "lldb/Utility/ReproducerProvider.h"
#include "lldb/Utility/StreamString.h"
#include "lldb/Utility/Timer.h"
#include "lldb/Utility/UUID.h"
#include "mach/machine.h"

#include "llvm/ADT/ScopeExit.h"
#include "llvm/Support/FileSystem.h"

using namespace lldb;
using namespace lldb_private;

static CFURLRef (*g_dlsym_DBGCopyFullDSYMURLForUUID)(CFUUIDRef uuid, CFURLRef exec_url) = nullptr;
static CFDictionaryRef (*g_dlsym_DBGCopyDSYMPropertyLists)(CFURLRef dsym_url) = nullptr;

int LocateMacOSXFilesUsingDebugSymbols(const ModuleSpec &module_spec,
                                       ModuleSpec &return_module_spec) {
  Log *log = lldb_private::GetLogIfAllCategoriesSet(LIBLLDB_LOG_HOST);
  if (!ModuleList::GetGlobalModuleListProperties().GetEnableExternalLookup()) {
    LLDB_LOGF(log, "Spotlight lookup for .dSYM bundles is disabled.");
    return 0;
  }

  return_module_spec = module_spec;
  return_module_spec.GetFileSpec().Clear();
  return_module_spec.GetSymbolFileSpec().Clear();

  const UUID *uuid = module_spec.GetUUIDPtr();
  const ArchSpec *arch = module_spec.GetArchitecturePtr();

  if (repro::Loader *l = repro::Reproducer::Instance().GetLoader()) {
    static repro::SymbolFileLoader symbol_file_loader(l);
    std::pair<FileSpec, FileSpec> paths = symbol_file_loader.GetPaths(uuid);
    return_module_spec.GetFileSpec() = paths.first;
    return_module_spec.GetSymbolFileSpec() = paths.second;
    return 1;
  }

  int items_found = 0;

  if (g_dlsym_DBGCopyFullDSYMURLForUUID == nullptr ||
      g_dlsym_DBGCopyDSYMPropertyLists == nullptr) {
    void *handle = dlopen ("/System/Library/PrivateFrameworks/DebugSymbols.framework/DebugSymbols", RTLD_LAZY | RTLD_LOCAL);
    if (handle) {
      g_dlsym_DBGCopyFullDSYMURLForUUID = (CFURLRef (*)(CFUUIDRef, CFURLRef)) dlsym (handle, "DBGCopyFullDSYMURLForUUID");
      g_dlsym_DBGCopyDSYMPropertyLists = (CFDictionaryRef (*)(CFURLRef)) dlsym (handle, "DBGCopyDSYMPropertyLists");
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

        CFCReleaser<CFURLRef> dsym_url(
            g_dlsym_DBGCopyFullDSYMURLForUUID(module_uuid_ref.get(), exec_url.get()));
        char path[PATH_MAX];

        if (dsym_url.get()) {
          if (::CFURLGetFileSystemRepresentation(
                  dsym_url.get(), true, (UInt8 *)path, sizeof(path) - 1)) {
            if (log) {
              LLDB_LOGF(log,
                        "DebugSymbols framework returned dSYM path of %s for "
                        "UUID %s -- looking for the dSYM",
                        path, uuid->GetAsString().c_str());
            }
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
          if (uuid_dict) {
            CFStringRef exec_cf_path =
                static_cast<CFStringRef>(::CFDictionaryGetValue(
                    uuid_dict, CFSTR("DBGSymbolRichExecutable")));
            if (exec_cf_path && ::CFStringGetFileSystemRepresentation(
                                    exec_cf_path, path, sizeof(path))) {
              if (log) {
                LLDB_LOGF(log, "plist bundle has exec path of %s for UUID %s",
                          path, uuid->GetAsString().c_str());
              }
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

          if (!success) {
            // No dictionary, check near the dSYM bundle for an executable that
            // matches...
            if (::CFURLGetFileSystemRepresentation(
                    dsym_url.get(), true, (UInt8 *)path, sizeof(path) - 1)) {
              char *dsym_extension_pos = ::strstr(path, ".dSYM");
              if (dsym_extension_pos) {
                *dsym_extension_pos = '\0';
                if (log) {
                  LLDB_LOGF(log,
                            "Looking for executable binary next to dSYM "
                            "bundle with name with name %s",
                            path);
                }
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
                        if (log) {
                          LLDB_LOGF(log,
                                    "Executable binary %s next to dSYM is "
                                    "compatible; using",
                                    path);
                        }
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
                    if (log) {
                      LLDB_LOGF(log,
                                "Executable binary %s next to dSYM is "
                                "compatible; using",
                                path);
                    }
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

  if (repro::Generator *g = repro::Reproducer::Instance().GetGenerator()) {
    g->GetOrCreate<repro::SymbolFileProvider>().AddSymbolFile(
        uuid, return_module_spec.GetFileSpec(),
        return_module_spec.GetSymbolFileSpec());
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
                                                ModuleSpec &module_spec) {
  Log *log = lldb_private::GetLogIfAllCategoriesSet(LIBLLDB_LOG_HOST);
  bool success = false;
  if (uuid_dict != NULL && CFGetTypeID(uuid_dict) == CFDictionaryGetTypeID()) {
    std::string str;
    CFStringRef cf_str;
    CFDictionaryRef cf_dict;

    cf_str = (CFStringRef)CFDictionaryGetValue(
        (CFDictionaryRef)uuid_dict, CFSTR("DBGSymbolRichExecutable"));
    if (cf_str && CFGetTypeID(cf_str) == CFStringGetTypeID()) {
      if (CFCString::FileSystemRepresentation(cf_str, str)) {
        module_spec.GetFileSpec().SetFile(str.c_str(), FileSpec::Style::native);
        FileSystem::Instance().Resolve(module_spec.GetFileSpec());
        if (log) {
          LLDB_LOGF(log,
                    "From dsymForUUID plist: Symbol rich executable is at '%s'",
                    str.c_str());
        }
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
        if (log) {
          LLDB_LOGF(log, "From dsymForUUID plist: dSYM is at '%s'",
                    str.c_str());
        }
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

bool Symbols::DownloadObjectAndSymbolFile(ModuleSpec &module_spec,
                                          bool force_lookup) {
  bool success = false;
  const UUID *uuid_ptr = module_spec.GetUUIDPtr();
  const FileSpec *file_spec_ptr = module_spec.GetFileSpecPtr();

  if (repro::Loader *l = repro::Reproducer::Instance().GetLoader()) {
    static repro::SymbolFileLoader symbol_file_loader(l);
    std::pair<FileSpec, FileSpec> paths = symbol_file_loader.GetPaths(uuid_ptr);
    if (paths.first)
      module_spec.GetFileSpec() = paths.first;
    if (paths.second)
      module_spec.GetSymbolFileSpec() = paths.second;
    return true;
  }

  // Lambda to capture the state of module_spec before returning from this
  // function.
  auto RecordResult = [&]() {
    if (repro::Generator *g = repro::Reproducer::Instance().GetGenerator()) {
      g->GetOrCreate<repro::SymbolFileProvider>().AddSymbolFile(
          uuid_ptr, module_spec.GetFileSpec(), module_spec.GetSymbolFileSpec());
    }
  };

  // It's expensive to check for the DBGShellCommands defaults setting, only do
  // it once per lldb run and cache the result.
  static bool g_have_checked_for_dbgshell_command = false;
  static const char *g_dbgshell_command = NULL;
  if (!g_have_checked_for_dbgshell_command) {
    g_have_checked_for_dbgshell_command = true;
    CFTypeRef defaults_setting = CFPreferencesCopyAppValue(
        CFSTR("DBGShellCommands"), CFSTR("com.apple.DebugSymbols"));
    if (defaults_setting &&
        CFGetTypeID(defaults_setting) == CFStringGetTypeID()) {
      char cstr_buf[PATH_MAX];
      if (CFStringGetCString((CFStringRef)defaults_setting, cstr_buf,
                             sizeof(cstr_buf), kCFStringEncodingUTF8)) {
        g_dbgshell_command =
            strdup(cstr_buf); // this malloc'ed memory will never be freed
      }
    }
    if (defaults_setting) {
      CFRelease(defaults_setting);
    }
  }

  // When g_dbgshell_command is NULL, the user has not enabled the use of an
  // external program to find the symbols, don't run it for them.
  if (!force_lookup && g_dbgshell_command == NULL) {
    RecordResult();
    return false;
  }

  if (uuid_ptr ||
      (file_spec_ptr && FileSystem::Instance().Exists(*file_spec_ptr))) {
    static bool g_located_dsym_for_uuid_exe = false;
    static bool g_dsym_for_uuid_exe_exists = false;
    static char g_dsym_for_uuid_exe_path[PATH_MAX];
    if (!g_located_dsym_for_uuid_exe) {
      g_located_dsym_for_uuid_exe = true;
      const char *dsym_for_uuid_exe_path_cstr =
          getenv("LLDB_APPLE_DSYMFORUUID_EXECUTABLE");
      FileSpec dsym_for_uuid_exe_spec;
      if (dsym_for_uuid_exe_path_cstr) {
        dsym_for_uuid_exe_spec.SetFile(dsym_for_uuid_exe_path_cstr,
                                       FileSpec::Style::native);
        FileSystem::Instance().Resolve(dsym_for_uuid_exe_spec);
        g_dsym_for_uuid_exe_exists =
            FileSystem::Instance().Exists(dsym_for_uuid_exe_spec);
      }

      if (!g_dsym_for_uuid_exe_exists) {
        dsym_for_uuid_exe_spec.SetFile("/usr/local/bin/dsymForUUID",
                                       FileSpec::Style::native);
        g_dsym_for_uuid_exe_exists =
            FileSystem::Instance().Exists(dsym_for_uuid_exe_spec);
        if (!g_dsym_for_uuid_exe_exists) {
          long bufsize;
          if ((bufsize = sysconf(_SC_GETPW_R_SIZE_MAX)) != -1) {
            char buffer[bufsize];
            struct passwd pwd;
            struct passwd *tilde_rc = NULL;
            // we are a library so we need to use the reentrant version of
            // getpwnam()
            if (getpwnam_r("rc", &pwd, buffer, bufsize, &tilde_rc) == 0 &&
                tilde_rc && tilde_rc->pw_dir) {
              std::string dsymforuuid_path(tilde_rc->pw_dir);
              dsymforuuid_path += "/bin/dsymForUUID";
              dsym_for_uuid_exe_spec.SetFile(dsymforuuid_path.c_str(),
                                             FileSpec::Style::native);
              g_dsym_for_uuid_exe_exists =
                  FileSystem::Instance().Exists(dsym_for_uuid_exe_spec);
            }
          }
        }
      }
      if (!g_dsym_for_uuid_exe_exists && g_dbgshell_command != NULL) {
        dsym_for_uuid_exe_spec.SetFile(g_dbgshell_command,
                                       FileSpec::Style::native);
        FileSystem::Instance().Resolve(dsym_for_uuid_exe_spec);
        g_dsym_for_uuid_exe_exists =
            FileSystem::Instance().Exists(dsym_for_uuid_exe_spec);
      }

      if (g_dsym_for_uuid_exe_exists)
        dsym_for_uuid_exe_spec.GetPath(g_dsym_for_uuid_exe_path,
                                       sizeof(g_dsym_for_uuid_exe_path));
    }
    if (g_dsym_for_uuid_exe_exists) {
      std::string uuid_str;
      char file_path[PATH_MAX];
      file_path[0] = '\0';

      if (uuid_ptr)
        uuid_str = uuid_ptr->GetAsString();

      if (file_spec_ptr)
        file_spec_ptr->GetPath(file_path, sizeof(file_path));

      StreamString command;
      if (!uuid_str.empty())
        command.Printf("%s --ignoreNegativeCache --copyExecutable %s",
                       g_dsym_for_uuid_exe_path, uuid_str.c_str());
      else if (file_path[0] != '\0')
        command.Printf("%s --ignoreNegativeCache --copyExecutable %s",
                       g_dsym_for_uuid_exe_path, file_path);

      if (!command.GetString().empty()) {
        Log *log = lldb_private::GetLogIfAllCategoriesSet(LIBLLDB_LOG_HOST);
        int exit_status = -1;
        int signo = -1;
        std::string command_output;
        if (log) {
          if (!uuid_str.empty())
            LLDB_LOGF(log, "Calling %s with UUID %s to find dSYM",
                      g_dsym_for_uuid_exe_path, uuid_str.c_str());
          else if (file_path[0] != '\0')
            LLDB_LOGF(log, "Calling %s with file %s to find dSYM",
                      g_dsym_for_uuid_exe_path, file_path);
        }
        Status error = Host::RunShellCommand(
            command.GetData(),
            FileSpec(),      // current working directory
            &exit_status,    // Exit status
            &signo,          // Signal int *
            &command_output, // Command output
            std::chrono::seconds(
               640), // Large timeout to allow for long dsym download times
            false);  // Don't run in a shell (we don't need shell expansion)
        if (error.Success() && exit_status == 0 && !command_output.empty()) {
          CFCData data(CFDataCreateWithBytesNoCopy(
              NULL, (const UInt8 *)command_output.data(), command_output.size(),
              kCFAllocatorNull));

          CFCReleaser<CFDictionaryRef> plist(
              (CFDictionaryRef)::CFPropertyListCreateFromXMLData(
                  NULL, data.get(), kCFPropertyListImmutable, NULL));

          if (plist.get() &&
              CFGetTypeID(plist.get()) == CFDictionaryGetTypeID()) {
            if (!uuid_str.empty()) {
              CFCString uuid_cfstr(uuid_str.c_str());
              CFDictionaryRef uuid_dict = (CFDictionaryRef)CFDictionaryGetValue(
                  plist.get(), uuid_cfstr.get());
              success =
                  GetModuleSpecInfoFromUUIDDictionary(uuid_dict, module_spec);
            } else {
              const CFIndex num_values = ::CFDictionaryGetCount(plist.get());
              if (num_values > 0) {
                std::vector<CFStringRef> keys(num_values, NULL);
                std::vector<CFDictionaryRef> values(num_values, NULL);
                ::CFDictionaryGetKeysAndValues(plist.get(), NULL,
                                               (const void **)&values[0]);
                if (num_values == 1) {
                  success = GetModuleSpecInfoFromUUIDDictionary(values[0],
                                                                module_spec);
                  RecordResult();
                  return success;
                } else {
                  for (CFIndex i = 0; i < num_values; ++i) {
                    ModuleSpec curr_module_spec;
                    if (GetModuleSpecInfoFromUUIDDictionary(values[i],
                                                            curr_module_spec)) {
                      if (module_spec.GetArchitecture().IsCompatibleMatch(
                              curr_module_spec.GetArchitecture())) {
                        module_spec = curr_module_spec;
                        RecordResult();
                        return true;
                      }
                    }
                  }
                }
              }
            }
          }
        } else {
          if (log) {
            if (!uuid_str.empty())
              LLDB_LOGF(log, "Called %s on %s, no matches",
                        g_dsym_for_uuid_exe_path, uuid_str.c_str());
            else if (file_path[0] != '\0')
              LLDB_LOGF(log, "Called %s on %s, no matches",
                        g_dsym_for_uuid_exe_path, file_path);
          }
        }
      }
    }
  }
  RecordResult();
  return success;
}
