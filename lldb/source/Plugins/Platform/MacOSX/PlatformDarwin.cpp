//===-- PlatformDarwin.cpp --------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "PlatformDarwin.h"

// C Includes
#include <string.h>

// C++ Includes
#include <algorithm>
#include <mutex>

// Other libraries and framework includes
#include "clang/Basic/VersionTuple.h"
// Project includes
#include "lldb/Breakpoint/BreakpointLocation.h"
#include "lldb/Breakpoint/BreakpointSite.h"
#include "lldb/Core/Debugger.h"
#include "lldb/Core/Error.h"
#include "lldb/Core/Log.h"
#include "lldb/Core/Module.h"
#include "lldb/Core/ModuleSpec.h"
#include "lldb/Core/Timer.h"
#include "lldb/Host/FileSystem.h"
#include "lldb/Host/Host.h"
#include "lldb/Host/HostInfo.h"
#include "lldb/Host/StringConvert.h"
#include "lldb/Host/Symbols.h"
#include "lldb/Host/XML.h"
#include "lldb/Interpreter/CommandInterpreter.h"
#include "lldb/Symbol/ObjectFile.h"
#include "lldb/Symbol/SymbolFile.h"
#include "lldb/Symbol/SymbolVendor.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/Target.h"
#include "llvm/ADT/STLExtras.h"

#if defined(__APPLE__)
#include <TargetConditionals.h> // for TARGET_OS_TV, TARGET_OS_WATCH
#endif

using namespace lldb;
using namespace lldb_private;

//------------------------------------------------------------------
/// Default Constructor
//------------------------------------------------------------------
PlatformDarwin::PlatformDarwin(bool is_host)
    : PlatformPOSIX(is_host), // This is the local host platform
      m_developer_directory() {}

//------------------------------------------------------------------
/// Destructor.
///
/// The destructor is virtual since this class is designed to be
/// inherited from by the plug-in instance.
//------------------------------------------------------------------
PlatformDarwin::~PlatformDarwin() {}

FileSpecList PlatformDarwin::LocateExecutableScriptingResources(
    Target *target, Module &module, Stream *feedback_stream) {
  FileSpecList file_list;
  if (target &&
      target->GetDebugger().GetScriptLanguage() == eScriptLanguagePython) {
    // NB some extensions might be meaningful and should not be stripped -
    // "this.binary.file"
    // should not lose ".file" but GetFileNameStrippingExtension() will do
    // precisely that.
    // Ideally, we should have a per-platform list of extensions (".exe",
    // ".app", ".dSYM", ".framework")
    // which should be stripped while leaving "this.binary.file" as-is.
    ScriptInterpreter *script_interpreter =
        target->GetDebugger().GetCommandInterpreter().GetScriptInterpreter();

    FileSpec module_spec = module.GetFileSpec();

    if (module_spec) {
      SymbolVendor *symbols = module.GetSymbolVendor();
      if (symbols) {
        SymbolFile *symfile = symbols->GetSymbolFile();
        if (symfile) {
          ObjectFile *objfile = symfile->GetObjectFile();
          if (objfile) {
            FileSpec symfile_spec(objfile->GetFileSpec());
            if (symfile_spec && symfile_spec.Exists()) {
              while (module_spec.GetFilename()) {
                std::string module_basename(
                    module_spec.GetFilename().GetCString());
                std::string original_module_basename(module_basename);

                bool was_keyword = false;

                // FIXME: for Python, we cannot allow certain characters in
                // module
                // filenames we import. Theoretically, different scripting
                // languages may
                // have different sets of forbidden tokens in filenames, and
                // that should
                // be dealt with by each ScriptInterpreter. For now, we just
                // replace dots
                // with underscores, but if we ever support anything other than
                // Python
                // we will need to rework this
                std::replace(module_basename.begin(), module_basename.end(),
                             '.', '_');
                std::replace(module_basename.begin(), module_basename.end(),
                             ' ', '_');
                std::replace(module_basename.begin(), module_basename.end(),
                             '-', '_');
                if (script_interpreter &&
                    script_interpreter->IsReservedWord(
                        module_basename.c_str())) {
                  module_basename.insert(module_basename.begin(), '_');
                  was_keyword = true;
                }

                StreamString path_string;
                StreamString original_path_string;
                // for OSX we are going to be in
                // .dSYM/Contents/Resources/DWARF/<basename>
                // let us go to .dSYM/Contents/Resources/Python/<basename>.py
                // and see if the file exists
                path_string.Printf("%s/../Python/%s.py",
                                   symfile_spec.GetDirectory().GetCString(),
                                   module_basename.c_str());
                original_path_string.Printf(
                    "%s/../Python/%s.py",
                    symfile_spec.GetDirectory().GetCString(),
                    original_module_basename.c_str());
                FileSpec script_fspec(path_string.GetData(), true);
                FileSpec orig_script_fspec(original_path_string.GetData(),
                                           true);

                // if we did some replacements of reserved characters, and a
                // file with the untampered name
                // exists, then warn the user that the file as-is shall not be
                // loaded
                if (feedback_stream) {
                  if (module_basename != original_module_basename &&
                      orig_script_fspec.Exists()) {
                    const char *reason_for_complaint =
                        was_keyword ? "conflicts with a keyword"
                                    : "contains reserved characters";
                    if (script_fspec.Exists())
                      feedback_stream->Printf(
                          "warning: the symbol file '%s' contains a debug "
                          "script. However, its name"
                          " '%s' %s and as such cannot be loaded. LLDB will"
                          " load '%s' instead. Consider removing the file with "
                          "the malformed name to"
                          " eliminate this warning.\n",
                          symfile_spec.GetPath().c_str(),
                          original_path_string.GetData(), reason_for_complaint,
                          path_string.GetData());
                    else
                      feedback_stream->Printf(
                          "warning: the symbol file '%s' contains a debug "
                          "script. However, its name"
                          " %s and as such cannot be loaded. If you intend"
                          " to have this script loaded, please rename '%s' to "
                          "'%s' and retry.\n",
                          symfile_spec.GetPath().c_str(), reason_for_complaint,
                          original_path_string.GetData(),
                          path_string.GetData());
                  }
                }

                if (script_fspec.Exists()) {
                  file_list.Append(script_fspec);
                  break;
                }

                // If we didn't find the python file, then keep
                // stripping the extensions and try again
                ConstString filename_no_extension(
                    module_spec.GetFileNameStrippingExtension());
                if (module_spec.GetFilename() == filename_no_extension)
                  break;

                module_spec.GetFilename() = filename_no_extension;
              }
            }
          }
        }
      }
    }
  }
  return file_list;
}

Error PlatformDarwin::ResolveExecutable(
    const ModuleSpec &module_spec, lldb::ModuleSP &exe_module_sp,
    const FileSpecList *module_search_paths_ptr) {
  Error error;
  // Nothing special to do here, just use the actual file and architecture

  char exe_path[PATH_MAX];
  ModuleSpec resolved_module_spec(module_spec);

  if (IsHost()) {
    // If we have "ls" as the exe_file, resolve the executable loation based on
    // the current path variables
    if (!resolved_module_spec.GetFileSpec().Exists()) {
      module_spec.GetFileSpec().GetPath(exe_path, sizeof(exe_path));
      resolved_module_spec.GetFileSpec().SetFile(exe_path, true);
    }

    if (!resolved_module_spec.GetFileSpec().Exists())
      resolved_module_spec.GetFileSpec().ResolveExecutableLocation();

    // Resolve any executable within a bundle on MacOSX
    Host::ResolveExecutableInBundle(resolved_module_spec.GetFileSpec());

    if (resolved_module_spec.GetFileSpec().Exists())
      error.Clear();
    else {
      const uint32_t permissions =
          resolved_module_spec.GetFileSpec().GetPermissions();
      if (permissions && (permissions & eFilePermissionsEveryoneR) == 0)
        error.SetErrorStringWithFormat(
            "executable '%s' is not readable",
            resolved_module_spec.GetFileSpec().GetPath().c_str());
      else
        error.SetErrorStringWithFormat(
            "unable to find executable for '%s'",
            resolved_module_spec.GetFileSpec().GetPath().c_str());
    }
  } else {
    if (m_remote_platform_sp) {
      error =
          GetCachedExecutable(resolved_module_spec, exe_module_sp,
                              module_search_paths_ptr, *m_remote_platform_sp);
    } else {
      // We may connect to a process and use the provided executable (Don't use
      // local $PATH).

      // Resolve any executable within a bundle on MacOSX
      Host::ResolveExecutableInBundle(resolved_module_spec.GetFileSpec());

      if (resolved_module_spec.GetFileSpec().Exists())
        error.Clear();
      else
        error.SetErrorStringWithFormat(
            "the platform is not currently connected, and '%s' doesn't exist "
            "in the system root.",
            resolved_module_spec.GetFileSpec().GetFilename().AsCString(""));
    }
  }

  if (error.Success()) {
    if (resolved_module_spec.GetArchitecture().IsValid()) {
      error = ModuleList::GetSharedModule(resolved_module_spec, exe_module_sp,
                                          module_search_paths_ptr, NULL, NULL);

      if (error.Fail() || exe_module_sp.get() == NULL ||
          exe_module_sp->GetObjectFile() == NULL) {
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
        error = GetSharedModule(resolved_module_spec, NULL, exe_module_sp,
                                module_search_paths_ptr, NULL, NULL);
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

Error PlatformDarwin::ResolveSymbolFile(Target &target,
                                        const ModuleSpec &sym_spec,
                                        FileSpec &sym_file) {
  Error error;
  sym_file = sym_spec.GetSymbolFileSpec();
  if (sym_file.Exists()) {
    if (sym_file.GetFileType() == FileSpec::eFileTypeDirectory) {
      sym_file = Symbols::FindSymbolFileInBundle(
          sym_file, sym_spec.GetUUIDPtr(), sym_spec.GetArchitecturePtr());
    }
  } else {
    if (sym_spec.GetUUID().IsValid()) {
    }
  }
  return error;
}

static lldb_private::Error
MakeCacheFolderForFile(const FileSpec &module_cache_spec) {
  FileSpec module_cache_folder =
      module_cache_spec.CopyByRemovingLastPathComponent();
  return FileSystem::MakeDirectory(module_cache_folder,
                                   eFilePermissionsDirectoryDefault);
}

static lldb_private::Error
BringInRemoteFile(Platform *platform,
                  const lldb_private::ModuleSpec &module_spec,
                  const FileSpec &module_cache_spec) {
  MakeCacheFolderForFile(module_cache_spec);
  Error err = platform->GetFile(module_spec.GetFileSpec(), module_cache_spec);
  return err;
}

lldb_private::Error PlatformDarwin::GetSharedModuleWithLocalCache(
    const lldb_private::ModuleSpec &module_spec, lldb::ModuleSP &module_sp,
    const lldb_private::FileSpecList *module_search_paths_ptr,
    lldb::ModuleSP *old_module_sp_ptr, bool *did_create_ptr) {

  Log *log(GetLogIfAnyCategoriesSet(LIBLLDB_LOG_PLATFORM));
  if (log)
    log->Printf("[%s] Trying to find module %s/%s - platform path %s/%s symbol "
                "path %s/%s",
                (IsHost() ? "host" : "remote"),
                module_spec.GetFileSpec().GetDirectory().AsCString(),
                module_spec.GetFileSpec().GetFilename().AsCString(),
                module_spec.GetPlatformFileSpec().GetDirectory().AsCString(),
                module_spec.GetPlatformFileSpec().GetFilename().AsCString(),
                module_spec.GetSymbolFileSpec().GetDirectory().AsCString(),
                module_spec.GetSymbolFileSpec().GetFilename().AsCString());

  Error err;

  err = ModuleList::GetSharedModule(module_spec, module_sp,
                                    module_search_paths_ptr, old_module_sp_ptr,
                                    did_create_ptr);
  if (module_sp)
    return err;

  if (!IsHost()) {
    std::string cache_path(GetLocalCacheDirectory());
    // Only search for a locally cached file if we have a valid cache path
    if (!cache_path.empty()) {
      std::string module_path(module_spec.GetFileSpec().GetPath());
      cache_path.append(module_path);
      FileSpec module_cache_spec(cache_path.c_str(), false);

      // if rsync is supported, always bring in the file - rsync will be very
      // efficient
      // when files are the same on the local and remote end of the connection
      if (this->GetSupportsRSync()) {
        err = BringInRemoteFile(this, module_spec, module_cache_spec);
        if (err.Fail())
          return err;
        if (module_cache_spec.Exists()) {
          Log *log(GetLogIfAnyCategoriesSet(LIBLLDB_LOG_PLATFORM));
          if (log)
            log->Printf("[%s] module %s/%s was rsynced and is now there",
                        (IsHost() ? "host" : "remote"),
                        module_spec.GetFileSpec().GetDirectory().AsCString(),
                        module_spec.GetFileSpec().GetFilename().AsCString());
          ModuleSpec local_spec(module_cache_spec,
                                module_spec.GetArchitecture());
          module_sp.reset(new Module(local_spec));
          module_sp->SetPlatformFileSpec(module_spec.GetFileSpec());
          return Error();
        }
      }

      // try to find the module in the cache
      if (module_cache_spec.Exists()) {
        // get the local and remote MD5 and compare
        if (m_remote_platform_sp) {
          // when going over the *slow* GDB remote transfer mechanism we first
          // check
          // the hashes of the files - and only do the actual transfer if they
          // differ
          uint64_t high_local, high_remote, low_local, low_remote;
          FileSystem::CalculateMD5(module_cache_spec, low_local, high_local);
          m_remote_platform_sp->CalculateMD5(module_spec.GetFileSpec(),
                                             low_remote, high_remote);
          if (low_local != low_remote || high_local != high_remote) {
            // bring in the remote file
            Log *log(GetLogIfAnyCategoriesSet(LIBLLDB_LOG_PLATFORM));
            if (log)
              log->Printf(
                  "[%s] module %s/%s needs to be replaced from remote copy",
                  (IsHost() ? "host" : "remote"),
                  module_spec.GetFileSpec().GetDirectory().AsCString(),
                  module_spec.GetFileSpec().GetFilename().AsCString());
            Error err = BringInRemoteFile(this, module_spec, module_cache_spec);
            if (err.Fail())
              return err;
          }
        }

        ModuleSpec local_spec(module_cache_spec, module_spec.GetArchitecture());
        module_sp.reset(new Module(local_spec));
        module_sp->SetPlatformFileSpec(module_spec.GetFileSpec());
        Log *log(GetLogIfAnyCategoriesSet(LIBLLDB_LOG_PLATFORM));
        if (log)
          log->Printf("[%s] module %s/%s was found in the cache",
                      (IsHost() ? "host" : "remote"),
                      module_spec.GetFileSpec().GetDirectory().AsCString(),
                      module_spec.GetFileSpec().GetFilename().AsCString());
        return Error();
      }

      // bring in the remote module file
      if (log)
        log->Printf("[%s] module %s/%s needs to come in remotely",
                    (IsHost() ? "host" : "remote"),
                    module_spec.GetFileSpec().GetDirectory().AsCString(),
                    module_spec.GetFileSpec().GetFilename().AsCString());
      Error err = BringInRemoteFile(this, module_spec, module_cache_spec);
      if (err.Fail())
        return err;
      if (module_cache_spec.Exists()) {
        Log *log(GetLogIfAnyCategoriesSet(LIBLLDB_LOG_PLATFORM));
        if (log)
          log->Printf("[%s] module %s/%s is now cached and fine",
                      (IsHost() ? "host" : "remote"),
                      module_spec.GetFileSpec().GetDirectory().AsCString(),
                      module_spec.GetFileSpec().GetFilename().AsCString());
        ModuleSpec local_spec(module_cache_spec, module_spec.GetArchitecture());
        module_sp.reset(new Module(local_spec));
        module_sp->SetPlatformFileSpec(module_spec.GetFileSpec());
        return Error();
      } else
        return Error("unable to obtain valid module file");
    } else
      return Error("no cache path");
  } else
    return Error("unable to resolve module");
}

Error PlatformDarwin::GetSharedModule(
    const ModuleSpec &module_spec, Process *process, ModuleSP &module_sp,
    const FileSpecList *module_search_paths_ptr, ModuleSP *old_module_sp_ptr,
    bool *did_create_ptr) {
  Error error;
  module_sp.reset();

  if (IsRemote()) {
    // If we have a remote platform always, let it try and locate
    // the shared module first.
    if (m_remote_platform_sp) {
      error = m_remote_platform_sp->GetSharedModule(
          module_spec, process, module_sp, module_search_paths_ptr,
          old_module_sp_ptr, did_create_ptr);
    }
  }

  if (!module_sp) {
    // Fall back to the local platform and find the file locally
    error = Platform::GetSharedModule(module_spec, process, module_sp,
                                      module_search_paths_ptr,
                                      old_module_sp_ptr, did_create_ptr);

    const FileSpec &platform_file = module_spec.GetFileSpec();
    if (!module_sp && module_search_paths_ptr && platform_file) {
      // We can try to pull off part of the file path up to the bundle
      // directory level and try any module search paths...
      FileSpec bundle_directory;
      if (Host::GetBundleDirectory(platform_file, bundle_directory)) {
        if (platform_file == bundle_directory) {
          ModuleSpec new_module_spec(module_spec);
          new_module_spec.GetFileSpec() = bundle_directory;
          if (Host::ResolveExecutableInBundle(new_module_spec.GetFileSpec())) {
            Error new_error(Platform::GetSharedModule(
                new_module_spec, process, module_sp, NULL, old_module_sp_ptr,
                did_create_ptr));

            if (module_sp)
              return new_error;
          }
        } else {
          char platform_path[PATH_MAX];
          char bundle_dir[PATH_MAX];
          platform_file.GetPath(platform_path, sizeof(platform_path));
          const size_t bundle_directory_len =
              bundle_directory.GetPath(bundle_dir, sizeof(bundle_dir));
          char new_path[PATH_MAX];
          size_t num_module_search_paths = module_search_paths_ptr->GetSize();
          for (size_t i = 0; i < num_module_search_paths; ++i) {
            const size_t search_path_len =
                module_search_paths_ptr->GetFileSpecAtIndex(i).GetPath(
                    new_path, sizeof(new_path));
            if (search_path_len < sizeof(new_path)) {
              snprintf(new_path + search_path_len,
                       sizeof(new_path) - search_path_len, "/%s",
                       platform_path + bundle_directory_len);
              FileSpec new_file_spec(new_path, false);
              if (new_file_spec.Exists()) {
                ModuleSpec new_module_spec(module_spec);
                new_module_spec.GetFileSpec() = new_file_spec;
                Error new_error(Platform::GetSharedModule(
                    new_module_spec, process, module_sp, NULL,
                    old_module_sp_ptr, did_create_ptr));

                if (module_sp) {
                  module_sp->SetPlatformFileSpec(new_file_spec);
                  return new_error;
                }
              }
            }
          }
        }
      }
    }
  }
  if (module_sp)
    module_sp->SetPlatformFileSpec(module_spec.GetFileSpec());
  return error;
}

size_t
PlatformDarwin::GetSoftwareBreakpointTrapOpcode(Target &target,
                                                BreakpointSite *bp_site) {
  const uint8_t *trap_opcode = nullptr;
  uint32_t trap_opcode_size = 0;
  bool bp_is_thumb = false;

  llvm::Triple::ArchType machine = target.GetArchitecture().GetMachine();
  switch (machine) {
  case llvm::Triple::aarch64: {
    // TODO: fix this with actual darwin breakpoint opcode for arm64.
    // right now debugging uses the Z packets with GDB remote so this
    // is not needed, but the size needs to be correct...
    static const uint8_t g_arm64_breakpoint_opcode[] = {0xFE, 0xDE, 0xFF, 0xE7};
    trap_opcode = g_arm64_breakpoint_opcode;
    trap_opcode_size = sizeof(g_arm64_breakpoint_opcode);
  } break;

  case llvm::Triple::thumb:
    bp_is_thumb = true;
    LLVM_FALLTHROUGH;
  case llvm::Triple::arm: {
    static const uint8_t g_arm_breakpoint_opcode[] = {0xFE, 0xDE, 0xFF, 0xE7};
    static const uint8_t g_thumb_breakpooint_opcode[] = {0xFE, 0xDE};

    // Auto detect arm/thumb if it wasn't explicitly specified
    if (!bp_is_thumb) {
      lldb::BreakpointLocationSP bp_loc_sp(bp_site->GetOwnerAtIndex(0));
      if (bp_loc_sp)
        bp_is_thumb = bp_loc_sp->GetAddress().GetAddressClass() ==
                      eAddressClassCodeAlternateISA;
    }
    if (bp_is_thumb) {
      trap_opcode = g_thumb_breakpooint_opcode;
      trap_opcode_size = sizeof(g_thumb_breakpooint_opcode);
      break;
    }
    trap_opcode = g_arm_breakpoint_opcode;
    trap_opcode_size = sizeof(g_arm_breakpoint_opcode);
  } break;

  case llvm::Triple::ppc:
  case llvm::Triple::ppc64: {
    static const uint8_t g_ppc_breakpoint_opcode[] = {0x7F, 0xC0, 0x00, 0x08};
    trap_opcode = g_ppc_breakpoint_opcode;
    trap_opcode_size = sizeof(g_ppc_breakpoint_opcode);
  } break;

  default:
    return Platform::GetSoftwareBreakpointTrapOpcode(target, bp_site);
  }

  if (trap_opcode && trap_opcode_size) {
    if (bp_site->SetTrapOpcode(trap_opcode, trap_opcode_size))
      return trap_opcode_size;
  }
  return 0;
}

bool PlatformDarwin::GetProcessInfo(lldb::pid_t pid,
                                    ProcessInstanceInfo &process_info) {
  bool success = false;
  if (IsHost()) {
    success = Platform::GetProcessInfo(pid, process_info);
  } else {
    if (m_remote_platform_sp)
      success = m_remote_platform_sp->GetProcessInfo(pid, process_info);
  }
  return success;
}

uint32_t
PlatformDarwin::FindProcesses(const ProcessInstanceInfoMatch &match_info,
                              ProcessInstanceInfoList &process_infos) {
  uint32_t match_count = 0;
  if (IsHost()) {
    // Let the base class figure out the host details
    match_count = Platform::FindProcesses(match_info, process_infos);
  } else {
    // If we are remote, we can only return results if we are connected
    if (m_remote_platform_sp)
      match_count =
          m_remote_platform_sp->FindProcesses(match_info, process_infos);
  }
  return match_count;
}

bool PlatformDarwin::ModuleIsExcludedForUnconstrainedSearches(
    lldb_private::Target &target, const lldb::ModuleSP &module_sp) {
  if (!module_sp)
    return false;

  ObjectFile *obj_file = module_sp->GetObjectFile();
  if (!obj_file)
    return false;

  ObjectFile::Type obj_type = obj_file->GetType();
  if (obj_type == ObjectFile::eTypeDynamicLinker)
    return true;
  else
    return false;
}

bool PlatformDarwin::x86GetSupportedArchitectureAtIndex(uint32_t idx,
                                                        ArchSpec &arch) {
  ArchSpec host_arch = HostInfo::GetArchitecture(HostInfo::eArchKindDefault);
  if (host_arch.GetCore() == ArchSpec::eCore_x86_64_x86_64h) {
    switch (idx) {
    case 0:
      arch = host_arch;
      return true;

    case 1:
      arch.SetTriple("x86_64-apple-macosx");
      return true;

    case 2:
      arch = HostInfo::GetArchitecture(HostInfo::eArchKind32);
      return true;

    default:
      return false;
    }
  } else {
    if (idx == 0) {
      arch = HostInfo::GetArchitecture(HostInfo::eArchKindDefault);
      return arch.IsValid();
    } else if (idx == 1) {
      ArchSpec platform_arch(
          HostInfo::GetArchitecture(HostInfo::eArchKindDefault));
      ArchSpec platform_arch64(
          HostInfo::GetArchitecture(HostInfo::eArchKind64));
      if (platform_arch.IsExactMatch(platform_arch64)) {
        // This macosx platform supports both 32 and 64 bit. Since we already
        // returned the 64 bit arch for idx == 0, return the 32 bit arch
        // for idx == 1
        arch = HostInfo::GetArchitecture(HostInfo::eArchKind32);
        return arch.IsValid();
      }
    }
  }
  return false;
}

// The architecture selection rules for arm processors
// These cpu subtypes have distinct names (e.g. armv7f) but armv7 binaries run
// fine on an armv7f processor.

bool PlatformDarwin::ARMGetSupportedArchitectureAtIndex(uint32_t idx,
                                                        ArchSpec &arch) {
  ArchSpec system_arch(GetSystemArchitecture());

// When lldb is running on a watch or tv, set the arch OS name appropriately.
#if defined(TARGET_OS_TV) && TARGET_OS_TV == 1
#define OSNAME "tvos"
#elif defined(TARGET_OS_WATCH) && TARGET_OS_WATCH == 1
#define OSNAME "watchos"
#else
#define OSNAME "ios"
#endif

  const ArchSpec::Core system_core = system_arch.GetCore();
  switch (system_core) {
  default:
    switch (idx) {
    case 0:
      arch.SetTriple("arm64-apple-" OSNAME);
      return true;
    case 1:
      arch.SetTriple("armv7-apple-" OSNAME);
      return true;
    case 2:
      arch.SetTriple("armv7f-apple-" OSNAME);
      return true;
    case 3:
      arch.SetTriple("armv7k-apple-" OSNAME);
      return true;
    case 4:
      arch.SetTriple("armv7s-apple-" OSNAME);
      return true;
    case 5:
      arch.SetTriple("armv7m-apple-" OSNAME);
      return true;
    case 6:
      arch.SetTriple("armv7em-apple-" OSNAME);
      return true;
    case 7:
      arch.SetTriple("armv6m-apple-" OSNAME);
      return true;
    case 8:
      arch.SetTriple("armv6-apple-" OSNAME);
      return true;
    case 9:
      arch.SetTriple("armv5-apple-" OSNAME);
      return true;
    case 10:
      arch.SetTriple("armv4-apple-" OSNAME);
      return true;
    case 11:
      arch.SetTriple("arm-apple-" OSNAME);
      return true;
    case 12:
      arch.SetTriple("thumbv7-apple-" OSNAME);
      return true;
    case 13:
      arch.SetTriple("thumbv7f-apple-" OSNAME);
      return true;
    case 14:
      arch.SetTriple("thumbv7k-apple-" OSNAME);
      return true;
    case 15:
      arch.SetTriple("thumbv7s-apple-" OSNAME);
      return true;
    case 16:
      arch.SetTriple("thumbv7m-apple-" OSNAME);
      return true;
    case 17:
      arch.SetTriple("thumbv7em-apple-" OSNAME);
      return true;
    case 18:
      arch.SetTriple("thumbv6m-apple-" OSNAME);
      return true;
    case 19:
      arch.SetTriple("thumbv6-apple-" OSNAME);
      return true;
    case 20:
      arch.SetTriple("thumbv5-apple-" OSNAME);
      return true;
    case 21:
      arch.SetTriple("thumbv4t-apple-" OSNAME);
      return true;
    case 22:
      arch.SetTriple("thumb-apple-" OSNAME);
      return true;
    default:
      break;
    }
    break;

  case ArchSpec::eCore_arm_arm64:
    switch (idx) {
    case 0:
      arch.SetTriple("arm64-apple-" OSNAME);
      return true;
    case 1:
      arch.SetTriple("armv7s-apple-" OSNAME);
      return true;
    case 2:
      arch.SetTriple("armv7f-apple-" OSNAME);
      return true;
    case 3:
      arch.SetTriple("armv7m-apple-" OSNAME);
      return true;
    case 4:
      arch.SetTriple("armv7em-apple-" OSNAME);
      return true;
    case 5:
      arch.SetTriple("armv7-apple-" OSNAME);
      return true;
    case 6:
      arch.SetTriple("armv6m-apple-" OSNAME);
      return true;
    case 7:
      arch.SetTriple("armv6-apple-" OSNAME);
      return true;
    case 8:
      arch.SetTriple("armv5-apple-" OSNAME);
      return true;
    case 9:
      arch.SetTriple("armv4-apple-" OSNAME);
      return true;
    case 10:
      arch.SetTriple("arm-apple-" OSNAME);
      return true;
    case 11:
      arch.SetTriple("thumbv7-apple-" OSNAME);
      return true;
    case 12:
      arch.SetTriple("thumbv7f-apple-" OSNAME);
      return true;
    case 13:
      arch.SetTriple("thumbv7k-apple-" OSNAME);
      return true;
    case 14:
      arch.SetTriple("thumbv7s-apple-" OSNAME);
      return true;
    case 15:
      arch.SetTriple("thumbv7m-apple-" OSNAME);
      return true;
    case 16:
      arch.SetTriple("thumbv7em-apple-" OSNAME);
      return true;
    case 17:
      arch.SetTriple("thumbv6m-apple-" OSNAME);
      return true;
    case 18:
      arch.SetTriple("thumbv6-apple-" OSNAME);
      return true;
    case 19:
      arch.SetTriple("thumbv5-apple-" OSNAME);
      return true;
    case 20:
      arch.SetTriple("thumbv4t-apple-" OSNAME);
      return true;
    case 21:
      arch.SetTriple("thumb-apple-" OSNAME);
      return true;
    default:
      break;
    }
    break;

  case ArchSpec::eCore_arm_armv7f:
    switch (idx) {
    case 0:
      arch.SetTriple("armv7f-apple-" OSNAME);
      return true;
    case 1:
      arch.SetTriple("armv7-apple-" OSNAME);
      return true;
    case 2:
      arch.SetTriple("armv6m-apple-" OSNAME);
      return true;
    case 3:
      arch.SetTriple("armv6-apple-" OSNAME);
      return true;
    case 4:
      arch.SetTriple("armv5-apple-" OSNAME);
      return true;
    case 5:
      arch.SetTriple("armv4-apple-" OSNAME);
      return true;
    case 6:
      arch.SetTriple("arm-apple-" OSNAME);
      return true;
    case 7:
      arch.SetTriple("thumbv7f-apple-" OSNAME);
      return true;
    case 8:
      arch.SetTriple("thumbv7-apple-" OSNAME);
      return true;
    case 9:
      arch.SetTriple("thumbv6m-apple-" OSNAME);
      return true;
    case 10:
      arch.SetTriple("thumbv6-apple-" OSNAME);
      return true;
    case 11:
      arch.SetTriple("thumbv5-apple-" OSNAME);
      return true;
    case 12:
      arch.SetTriple("thumbv4t-apple-" OSNAME);
      return true;
    case 13:
      arch.SetTriple("thumb-apple-" OSNAME);
      return true;
    default:
      break;
    }
    break;

  case ArchSpec::eCore_arm_armv7k:
    switch (idx) {
    case 0:
      arch.SetTriple("armv7k-apple-" OSNAME);
      return true;
    case 1:
      arch.SetTriple("armv7-apple-" OSNAME);
      return true;
    case 2:
      arch.SetTriple("armv6m-apple-" OSNAME);
      return true;
    case 3:
      arch.SetTriple("armv6-apple-" OSNAME);
      return true;
    case 4:
      arch.SetTriple("armv5-apple-" OSNAME);
      return true;
    case 5:
      arch.SetTriple("armv4-apple-" OSNAME);
      return true;
    case 6:
      arch.SetTriple("arm-apple-" OSNAME);
      return true;
    case 7:
      arch.SetTriple("thumbv7k-apple-" OSNAME);
      return true;
    case 8:
      arch.SetTriple("thumbv7-apple-" OSNAME);
      return true;
    case 9:
      arch.SetTriple("thumbv6m-apple-" OSNAME);
      return true;
    case 10:
      arch.SetTriple("thumbv6-apple-" OSNAME);
      return true;
    case 11:
      arch.SetTriple("thumbv5-apple-" OSNAME);
      return true;
    case 12:
      arch.SetTriple("thumbv4t-apple-" OSNAME);
      return true;
    case 13:
      arch.SetTriple("thumb-apple-" OSNAME);
      return true;
    default:
      break;
    }
    break;

  case ArchSpec::eCore_arm_armv7s:
    switch (idx) {
    case 0:
      arch.SetTriple("armv7s-apple-" OSNAME);
      return true;
    case 1:
      arch.SetTriple("armv7-apple-" OSNAME);
      return true;
    case 2:
      arch.SetTriple("armv6m-apple-" OSNAME);
      return true;
    case 3:
      arch.SetTriple("armv6-apple-" OSNAME);
      return true;
    case 4:
      arch.SetTriple("armv5-apple-" OSNAME);
      return true;
    case 5:
      arch.SetTriple("armv4-apple-" OSNAME);
      return true;
    case 6:
      arch.SetTriple("arm-apple-" OSNAME);
      return true;
    case 7:
      arch.SetTriple("thumbv7s-apple-" OSNAME);
      return true;
    case 8:
      arch.SetTriple("thumbv7-apple-" OSNAME);
      return true;
    case 9:
      arch.SetTriple("thumbv6m-apple-" OSNAME);
      return true;
    case 10:
      arch.SetTriple("thumbv6-apple-" OSNAME);
      return true;
    case 11:
      arch.SetTriple("thumbv5-apple-" OSNAME);
      return true;
    case 12:
      arch.SetTriple("thumbv4t-apple-" OSNAME);
      return true;
    case 13:
      arch.SetTriple("thumb-apple-" OSNAME);
      return true;
    default:
      break;
    }
    break;

  case ArchSpec::eCore_arm_armv7m:
    switch (idx) {
    case 0:
      arch.SetTriple("armv7m-apple-" OSNAME);
      return true;
    case 1:
      arch.SetTriple("armv7-apple-" OSNAME);
      return true;
    case 2:
      arch.SetTriple("armv6m-apple-" OSNAME);
      return true;
    case 3:
      arch.SetTriple("armv6-apple-" OSNAME);
      return true;
    case 4:
      arch.SetTriple("armv5-apple-" OSNAME);
      return true;
    case 5:
      arch.SetTriple("armv4-apple-" OSNAME);
      return true;
    case 6:
      arch.SetTriple("arm-apple-" OSNAME);
      return true;
    case 7:
      arch.SetTriple("thumbv7m-apple-" OSNAME);
      return true;
    case 8:
      arch.SetTriple("thumbv7-apple-" OSNAME);
      return true;
    case 9:
      arch.SetTriple("thumbv6m-apple-" OSNAME);
      return true;
    case 10:
      arch.SetTriple("thumbv6-apple-" OSNAME);
      return true;
    case 11:
      arch.SetTriple("thumbv5-apple-" OSNAME);
      return true;
    case 12:
      arch.SetTriple("thumbv4t-apple-" OSNAME);
      return true;
    case 13:
      arch.SetTriple("thumb-apple-" OSNAME);
      return true;
    default:
      break;
    }
    break;

  case ArchSpec::eCore_arm_armv7em:
    switch (idx) {
    case 0:
      arch.SetTriple("armv7em-apple-" OSNAME);
      return true;
    case 1:
      arch.SetTriple("armv7-apple-" OSNAME);
      return true;
    case 2:
      arch.SetTriple("armv6m-apple-" OSNAME);
      return true;
    case 3:
      arch.SetTriple("armv6-apple-" OSNAME);
      return true;
    case 4:
      arch.SetTriple("armv5-apple-" OSNAME);
      return true;
    case 5:
      arch.SetTriple("armv4-apple-" OSNAME);
      return true;
    case 6:
      arch.SetTriple("arm-apple-" OSNAME);
      return true;
    case 7:
      arch.SetTriple("thumbv7em-apple-" OSNAME);
      return true;
    case 8:
      arch.SetTriple("thumbv7-apple-" OSNAME);
      return true;
    case 9:
      arch.SetTriple("thumbv6m-apple-" OSNAME);
      return true;
    case 10:
      arch.SetTriple("thumbv6-apple-" OSNAME);
      return true;
    case 11:
      arch.SetTriple("thumbv5-apple-" OSNAME);
      return true;
    case 12:
      arch.SetTriple("thumbv4t-apple-" OSNAME);
      return true;
    case 13:
      arch.SetTriple("thumb-apple-" OSNAME);
      return true;
    default:
      break;
    }
    break;

  case ArchSpec::eCore_arm_armv7:
    switch (idx) {
    case 0:
      arch.SetTriple("armv7-apple-" OSNAME);
      return true;
    case 1:
      arch.SetTriple("armv6m-apple-" OSNAME);
      return true;
    case 2:
      arch.SetTriple("armv6-apple-" OSNAME);
      return true;
    case 3:
      arch.SetTriple("armv5-apple-" OSNAME);
      return true;
    case 4:
      arch.SetTriple("armv4-apple-" OSNAME);
      return true;
    case 5:
      arch.SetTriple("arm-apple-" OSNAME);
      return true;
    case 6:
      arch.SetTriple("thumbv7-apple-" OSNAME);
      return true;
    case 7:
      arch.SetTriple("thumbv6m-apple-" OSNAME);
      return true;
    case 8:
      arch.SetTriple("thumbv6-apple-" OSNAME);
      return true;
    case 9:
      arch.SetTriple("thumbv5-apple-" OSNAME);
      return true;
    case 10:
      arch.SetTriple("thumbv4t-apple-" OSNAME);
      return true;
    case 11:
      arch.SetTriple("thumb-apple-" OSNAME);
      return true;
    default:
      break;
    }
    break;

  case ArchSpec::eCore_arm_armv6m:
    switch (idx) {
    case 0:
      arch.SetTriple("armv6m-apple-" OSNAME);
      return true;
    case 1:
      arch.SetTriple("armv6-apple-" OSNAME);
      return true;
    case 2:
      arch.SetTriple("armv5-apple-" OSNAME);
      return true;
    case 3:
      arch.SetTriple("armv4-apple-" OSNAME);
      return true;
    case 4:
      arch.SetTriple("arm-apple-" OSNAME);
      return true;
    case 5:
      arch.SetTriple("thumbv6m-apple-" OSNAME);
      return true;
    case 6:
      arch.SetTriple("thumbv6-apple-" OSNAME);
      return true;
    case 7:
      arch.SetTriple("thumbv5-apple-" OSNAME);
      return true;
    case 8:
      arch.SetTriple("thumbv4t-apple-" OSNAME);
      return true;
    case 9:
      arch.SetTriple("thumb-apple-" OSNAME);
      return true;
    default:
      break;
    }
    break;

  case ArchSpec::eCore_arm_armv6:
    switch (idx) {
    case 0:
      arch.SetTriple("armv6-apple-" OSNAME);
      return true;
    case 1:
      arch.SetTriple("armv5-apple-" OSNAME);
      return true;
    case 2:
      arch.SetTriple("armv4-apple-" OSNAME);
      return true;
    case 3:
      arch.SetTriple("arm-apple-" OSNAME);
      return true;
    case 4:
      arch.SetTriple("thumbv6-apple-" OSNAME);
      return true;
    case 5:
      arch.SetTriple("thumbv5-apple-" OSNAME);
      return true;
    case 6:
      arch.SetTriple("thumbv4t-apple-" OSNAME);
      return true;
    case 7:
      arch.SetTriple("thumb-apple-" OSNAME);
      return true;
    default:
      break;
    }
    break;

  case ArchSpec::eCore_arm_armv5:
    switch (idx) {
    case 0:
      arch.SetTriple("armv5-apple-" OSNAME);
      return true;
    case 1:
      arch.SetTriple("armv4-apple-" OSNAME);
      return true;
    case 2:
      arch.SetTriple("arm-apple-" OSNAME);
      return true;
    case 3:
      arch.SetTriple("thumbv5-apple-" OSNAME);
      return true;
    case 4:
      arch.SetTriple("thumbv4t-apple-" OSNAME);
      return true;
    case 5:
      arch.SetTriple("thumb-apple-" OSNAME);
      return true;
    default:
      break;
    }
    break;

  case ArchSpec::eCore_arm_armv4:
    switch (idx) {
    case 0:
      arch.SetTriple("armv4-apple-" OSNAME);
      return true;
    case 1:
      arch.SetTriple("arm-apple-" OSNAME);
      return true;
    case 2:
      arch.SetTriple("thumbv4t-apple-" OSNAME);
      return true;
    case 3:
      arch.SetTriple("thumb-apple-" OSNAME);
      return true;
    default:
      break;
    }
    break;
  }
  arch.Clear();
  return false;
}

const char *PlatformDarwin::GetDeveloperDirectory() {
  std::lock_guard<std::mutex> guard(m_mutex);
  if (m_developer_directory.empty()) {
    bool developer_dir_path_valid = false;
    char developer_dir_path[PATH_MAX];
    FileSpec temp_file_spec;
    if (HostInfo::GetLLDBPath(ePathTypeLLDBShlibDir, temp_file_spec)) {
      if (temp_file_spec.GetPath(developer_dir_path,
                                 sizeof(developer_dir_path))) {
        char *shared_frameworks =
            strstr(developer_dir_path, "/SharedFrameworks/LLDB.framework");
        if (shared_frameworks) {
          ::snprintf(shared_frameworks,
                     sizeof(developer_dir_path) -
                         (shared_frameworks - developer_dir_path),
                     "/Developer");
          developer_dir_path_valid = true;
        } else {
          char *lib_priv_frameworks = strstr(
              developer_dir_path, "/Library/PrivateFrameworks/LLDB.framework");
          if (lib_priv_frameworks) {
            *lib_priv_frameworks = '\0';
            developer_dir_path_valid = true;
          }
        }
      }
    }

    if (!developer_dir_path_valid) {
      std::string xcode_dir_path;
      const char *xcode_select_prefix_dir = getenv("XCODE_SELECT_PREFIX_DIR");
      if (xcode_select_prefix_dir)
        xcode_dir_path.append(xcode_select_prefix_dir);
      xcode_dir_path.append("/usr/share/xcode-select/xcode_dir_path");
      temp_file_spec.SetFile(xcode_dir_path.c_str(), false);
      size_t bytes_read = temp_file_spec.ReadFileContents(
          0, developer_dir_path, sizeof(developer_dir_path), NULL);
      if (bytes_read > 0) {
        developer_dir_path[bytes_read] = '\0';
        while (developer_dir_path[bytes_read - 1] == '\r' ||
               developer_dir_path[bytes_read - 1] == '\n')
          developer_dir_path[--bytes_read] = '\0';
        developer_dir_path_valid = true;
      }
    }

    if (!developer_dir_path_valid) {
      FileSpec xcode_select_cmd("/usr/bin/xcode-select", false);
      if (xcode_select_cmd.Exists()) {
        int exit_status = -1;
        int signo = -1;
        std::string command_output;
        Error error =
            Host::RunShellCommand("/usr/bin/xcode-select --print-path",
                                  NULL, // current working directory
                                  &exit_status, &signo, &command_output,
                                  2,      // short timeout
                                  false); // don't run in a shell
        if (error.Success() && exit_status == 0 && !command_output.empty()) {
          const char *cmd_output_ptr = command_output.c_str();
          developer_dir_path[sizeof(developer_dir_path) - 1] = '\0';
          size_t i;
          for (i = 0; i < sizeof(developer_dir_path) - 1; i++) {
            if (cmd_output_ptr[i] == '\r' || cmd_output_ptr[i] == '\n' ||
                cmd_output_ptr[i] == '\0')
              break;
            developer_dir_path[i] = cmd_output_ptr[i];
          }
          developer_dir_path[i] = '\0';

          FileSpec devel_dir(developer_dir_path, false);
          if (devel_dir.Exists() && devel_dir.IsDirectory()) {
            developer_dir_path_valid = true;
          }
        }
      }
    }

    if (developer_dir_path_valid) {
      temp_file_spec.SetFile(developer_dir_path, false);
      if (temp_file_spec.Exists()) {
        m_developer_directory.assign(developer_dir_path);
        return m_developer_directory.c_str();
      }
    }
    // Assign a single NULL character so we know we tried to find the device
    // support directory and we don't keep trying to find it over and over.
    m_developer_directory.assign(1, '\0');
  }

  // We should have put a single NULL character into m_developer_directory
  // or it should have a valid path if the code gets here
  assert(m_developer_directory.empty() == false);
  if (m_developer_directory[0])
    return m_developer_directory.c_str();
  return NULL;
}

BreakpointSP PlatformDarwin::SetThreadCreationBreakpoint(Target &target) {
  BreakpointSP bp_sp;
  static const char *g_bp_names[] = {
      "start_wqthread", "_pthread_wqthread", "_pthread_start",
  };

  static const char *g_bp_modules[] = {"libsystem_c.dylib",
                                       "libSystem.B.dylib"};

  FileSpecList bp_modules;
  for (size_t i = 0; i < llvm::array_lengthof(g_bp_modules); i++) {
    const char *bp_module = g_bp_modules[i];
    bp_modules.Append(FileSpec(bp_module, false));
  }

  bool internal = true;
  bool hardware = false;
  LazyBool skip_prologue = eLazyBoolNo;
  bp_sp = target.CreateBreakpoint(&bp_modules, NULL, g_bp_names,
                                  llvm::array_lengthof(g_bp_names),
                                  eFunctionNameTypeFull, eLanguageTypeUnknown,
                                  0, skip_prologue, internal, hardware);
  bp_sp->SetBreakpointKind("thread-creation");

  return bp_sp;
}

int32_t
PlatformDarwin::GetResumeCountForLaunchInfo(ProcessLaunchInfo &launch_info) {
  const FileSpec &shell = launch_info.GetShell();
  if (!shell)
    return 1;

  std::string shell_string = shell.GetPath();
  const char *shell_name = strrchr(shell_string.c_str(), '/');
  if (shell_name == NULL)
    shell_name = shell_string.c_str();
  else
    shell_name++;

  if (strcmp(shell_name, "sh") == 0) {
    // /bin/sh re-exec's itself as /bin/bash requiring another resume.
    // But it only does this if the COMMAND_MODE environment variable
    // is set to "legacy".
    const char **envp =
        launch_info.GetEnvironmentEntries().GetConstArgumentVector();
    if (envp != NULL) {
      for (int i = 0; envp[i] != NULL; i++) {
        if (strcmp(envp[i], "COMMAND_MODE=legacy") == 0)
          return 2;
      }
    }
    return 1;
  } else if (strcmp(shell_name, "csh") == 0 ||
             strcmp(shell_name, "tcsh") == 0 ||
             strcmp(shell_name, "zsh") == 0) {
    // csh and tcsh always seem to re-exec themselves.
    return 2;
  } else
    return 1;
}

void PlatformDarwin::CalculateTrapHandlerSymbolNames() {
  m_trap_handlers.push_back(ConstString("_sigtramp"));
}

static const char *const sdk_strings[] = {
    "MacOSX", "iPhoneSimulator", "iPhoneOS",
};

static FileSpec CheckPathForXcode(const FileSpec &fspec) {
  if (fspec.Exists()) {
    const char substr[] = ".app/Contents/";

    std::string path_to_shlib = fspec.GetPath();
    size_t pos = path_to_shlib.rfind(substr);
    if (pos != std::string::npos) {
      path_to_shlib.erase(pos + strlen(substr));
      FileSpec ret(path_to_shlib.c_str(), false);

      FileSpec xcode_binary_path = ret;
      xcode_binary_path.AppendPathComponent("MacOS");
      xcode_binary_path.AppendPathComponent("Xcode");

      if (xcode_binary_path.Exists()) {
        return ret;
      }
    }
  }
  return FileSpec();
}

static FileSpec GetXcodeContentsPath() {
  static FileSpec g_xcode_filespec;
  static std::once_flag g_once_flag;
  std::call_once(g_once_flag, []() {

    FileSpec fspec;

    // First get the program file spec. If lldb.so or LLDB.framework is running
    // in a program and that program is Xcode, the path returned with be the
    // path
    // to Xcode.app/Contents/MacOS/Xcode, so this will be the correct Xcode to
    // use.
    fspec = HostInfo::GetProgramFileSpec();

    if (fspec) {
      // Ignore the current binary if it is python.
      std::string basename_lower = fspec.GetFilename().GetCString();
      std::transform(basename_lower.begin(), basename_lower.end(),
                     basename_lower.begin(), tolower);
      if (basename_lower != "python") {
        g_xcode_filespec = CheckPathForXcode(fspec);
      }
    }

    // Next check DEVELOPER_DIR environment variable
    if (!g_xcode_filespec) {
      const char *developer_dir_env_var = getenv("DEVELOPER_DIR");
      if (developer_dir_env_var && developer_dir_env_var[0]) {
        g_xcode_filespec =
            CheckPathForXcode(FileSpec(developer_dir_env_var, true));
      }

      // Fall back to using "xcrun" to find the selected Xcode
      if (!g_xcode_filespec) {
        int status = 0;
        int signo = 0;
        std::string output;
        const char *command = "/usr/bin/xcode-select -p";
        lldb_private::Error error = Host::RunShellCommand(
            command, // shell command to run
            NULL,    // current working directory
            &status, // Put the exit status of the process in here
            &signo,  // Put the signal that caused the process to exit in here
            &output, // Get the output from the command and place it in this
                     // string
            3);      // Timeout in seconds to wait for shell program to finish
        if (status == 0 && !output.empty()) {
          size_t first_non_newline = output.find_last_not_of("\r\n");
          if (first_non_newline != std::string::npos) {
            output.erase(first_non_newline + 1);
          }
          output.append("/..");

          g_xcode_filespec = CheckPathForXcode(FileSpec(output.c_str(), false));
        }
      }
    }
  });

  return g_xcode_filespec;
}

bool PlatformDarwin::SDKSupportsModules(SDKType sdk_type, uint32_t major,
                                        uint32_t minor, uint32_t micro) {
  switch (sdk_type) {
  case SDKType::MacOSX:
    if (major > 10 || (major == 10 && minor >= 10))
      return true;
    break;
  case SDKType::iPhoneOS:
  case SDKType::iPhoneSimulator:
    if (major >= 8)
      return true;
    break;
  }

  return false;
}

bool PlatformDarwin::SDKSupportsModules(SDKType desired_type,
                                        const FileSpec &sdk_path) {
  ConstString last_path_component = sdk_path.GetLastPathComponent();

  if (last_path_component) {
    const llvm::StringRef sdk_name = last_path_component.GetStringRef();

    llvm::StringRef version_part;

    if (sdk_name.startswith(sdk_strings[(int)desired_type])) {
      version_part =
          sdk_name.drop_front(strlen(sdk_strings[(int)desired_type]));
    } else {
      return false;
    }

    const size_t major_dot_offset = version_part.find('.');
    if (major_dot_offset == llvm::StringRef::npos)
      return false;

    const llvm::StringRef major_version =
        version_part.slice(0, major_dot_offset);
    const llvm::StringRef minor_part =
        version_part.drop_front(major_dot_offset + 1);

    const size_t minor_dot_offset = minor_part.find('.');
    if (minor_dot_offset == llvm::StringRef::npos)
      return false;

    const llvm::StringRef minor_version = minor_part.slice(0, minor_dot_offset);

    unsigned int major = 0;
    unsigned int minor = 0;
    unsigned int micro = 0;

    if (major_version.getAsInteger(10, major))
      return false;

    if (minor_version.getAsInteger(10, minor))
      return false;

    return SDKSupportsModules(desired_type, major, minor, micro);
  }

  return false;
}

FileSpec::EnumerateDirectoryResult
PlatformDarwin::DirectoryEnumerator(void *baton, FileSpec::FileType file_type,
                                    const FileSpec &spec) {
  SDKEnumeratorInfo *enumerator_info = static_cast<SDKEnumeratorInfo *>(baton);

  if (SDKSupportsModules(enumerator_info->sdk_type, spec)) {
    enumerator_info->found_path = spec;
    return FileSpec::EnumerateDirectoryResult::eEnumerateDirectoryResultNext;
  }

  return FileSpec::EnumerateDirectoryResult::eEnumerateDirectoryResultNext;
}

FileSpec PlatformDarwin::FindSDKInXcodeForModules(SDKType sdk_type,
                                                  const FileSpec &sdks_spec) {
  // Look inside Xcode for the required installed iOS SDK version

  if (!sdks_spec.IsDirectory())
    return FileSpec();

  const bool find_directories = true;
  const bool find_files = false;
  const bool find_other = true; // include symlinks

  SDKEnumeratorInfo enumerator_info;

  enumerator_info.sdk_type = sdk_type;

  FileSpec::EnumerateDirectory(sdks_spec.GetPath().c_str(), find_directories,
                               find_files, find_other, DirectoryEnumerator,
                               &enumerator_info);

  if (enumerator_info.found_path.IsDirectory())
    return enumerator_info.found_path;
  else
    return FileSpec();
}

FileSpec PlatformDarwin::GetSDKDirectoryForModules(SDKType sdk_type) {
  switch (sdk_type) {
  case SDKType::MacOSX:
  case SDKType::iPhoneSimulator:
  case SDKType::iPhoneOS:
    break;
  }

  FileSpec sdks_spec = GetXcodeContentsPath();
  sdks_spec.AppendPathComponent("Developer");
  sdks_spec.AppendPathComponent("Platforms");

  switch (sdk_type) {
  case SDKType::MacOSX:
    sdks_spec.AppendPathComponent("MacOSX.platform");
    break;
  case SDKType::iPhoneSimulator:
    sdks_spec.AppendPathComponent("iPhoneSimulator.platform");
    break;
  case SDKType::iPhoneOS:
    sdks_spec.AppendPathComponent("iPhoneOS.platform");
    break;
  }

  sdks_spec.AppendPathComponent("Developer");
  sdks_spec.AppendPathComponent("SDKs");

  if (sdk_type == SDKType::MacOSX) {
    uint32_t major = 0;
    uint32_t minor = 0;
    uint32_t micro = 0;

    if (HostInfo::GetOSVersion(major, minor, micro)) {
      if (SDKSupportsModules(SDKType::MacOSX, major, minor, micro)) {
        // We slightly prefer the exact SDK for this machine.  See if it is
        // there.

        FileSpec native_sdk_spec = sdks_spec;
        StreamString native_sdk_name;
        native_sdk_name.Printf("MacOSX%u.%u.sdk", major, minor);
        native_sdk_spec.AppendPathComponent(
            native_sdk_name.GetString().c_str());

        if (native_sdk_spec.Exists()) {
          return native_sdk_spec;
        }
      }
    }
  }

  return FindSDKInXcodeForModules(sdk_type, sdks_spec);
}

std::tuple<uint32_t, uint32_t, uint32_t, llvm::StringRef>
PlatformDarwin::ParseVersionBuildDir(llvm::StringRef dir) {
  uint32_t major, minor, update;
  llvm::StringRef build;
  llvm::StringRef version_str;
  llvm::StringRef build_str;
  std::tie(version_str, build_str) = dir.split(' ');
  if (Args::StringToVersion(version_str, major, minor, update) ||
      build_str.empty()) {
    if (build_str.consume_front("(")) {
      size_t pos = build_str.find(')');
      build = build_str.slice(0, pos);
    }
  }

  return std::make_tuple(major, minor, update, build);
}

void PlatformDarwin::AddClangModuleCompilationOptionsForSDKType(
    Target *target, std::vector<std::string> &options, SDKType sdk_type) {
  const std::vector<std::string> apple_arguments = {
      "-x",       "objective-c++", "-fobjc-arc",
      "-fblocks", "-D_ISO646_H",   "-D__ISO646_H"};

  options.insert(options.end(), apple_arguments.begin(), apple_arguments.end());

  StreamString minimum_version_option;
  uint32_t versions[3] = {0, 0, 0};
  bool use_current_os_version = false;
  switch (sdk_type) {
  case SDKType::iPhoneOS:
#if defined(__arm__) || defined(__arm64__) || defined(__aarch64__)
    use_current_os_version = true;
#else
    use_current_os_version = false;
#endif
    break;

  case SDKType::iPhoneSimulator:
    use_current_os_version = false;
    break;

  case SDKType::MacOSX:
#if defined(__i386__) || defined(__x86_64__)
    use_current_os_version = true;
#else
    use_current_os_version = false;
#endif
    break;
  }

  bool versions_valid = false;
  if (use_current_os_version)
    versions_valid = GetOSVersion(versions[0], versions[1], versions[2]);
  else if (target) {
    // Our OS doesn't match our executable so we need to get the min OS version
    // from the object file
    ModuleSP exe_module_sp = target->GetExecutableModule();
    if (exe_module_sp) {
      ObjectFile *object_file = exe_module_sp->GetObjectFile();
      if (object_file)
        versions_valid = object_file->GetMinimumOSVersion(versions, 3) > 0;
    }
  }
  // Only add the version-min options if we got a version from somewhere
  if (versions_valid && versions[0] != UINT32_MAX) {
    // Make any invalid versions be zero if needed
    if (versions[1] == UINT32_MAX)
      versions[1] = 0;
    if (versions[2] == UINT32_MAX)
      versions[2] = 0;

    switch (sdk_type) {
    case SDKType::iPhoneOS:
      minimum_version_option.PutCString("-mios-version-min=");
      minimum_version_option.PutCString(
          clang::VersionTuple(versions[0], versions[1], versions[2])
              .getAsString()
              .c_str());
      break;
    case SDKType::iPhoneSimulator:
      minimum_version_option.PutCString("-mios-simulator-version-min=");
      minimum_version_option.PutCString(
          clang::VersionTuple(versions[0], versions[1], versions[2])
              .getAsString()
              .c_str());
      break;
    case SDKType::MacOSX:
      minimum_version_option.PutCString("-mmacosx-version-min=");
      minimum_version_option.PutCString(
          clang::VersionTuple(versions[0], versions[1], versions[2])
              .getAsString()
              .c_str());
    }
    options.push_back(minimum_version_option.GetString());
  }

  FileSpec sysroot_spec;
  // Scope for mutex locker below
  {
    std::lock_guard<std::mutex> guard(m_mutex);
    sysroot_spec = GetSDKDirectoryForModules(sdk_type);
  }

  if (sysroot_spec.IsDirectory()) {
    options.push_back("-isysroot");
    options.push_back(sysroot_spec.GetPath());
  }
}

ConstString PlatformDarwin::GetFullNameForDylib(ConstString basename) {
  if (basename.IsEmpty())
    return basename;

  StreamString stream;
  stream.Printf("lib%s.dylib", basename.GetCString());
  return ConstString(stream.GetData());
}

bool PlatformDarwin::GetOSVersion(uint32_t &major, uint32_t &minor,
                                  uint32_t &update, Process *process) {
  if (process && strstr(GetPluginName().GetCString(), "-simulator")) {
    lldb_private::ProcessInstanceInfo proc_info;
    if (Host::GetProcessInfo(process->GetID(), proc_info)) {
      Args &env = proc_info.GetEnvironmentEntries();
      const size_t n = env.GetArgumentCount();
      const llvm::StringRef k_runtime_version("SIMULATOR_RUNTIME_VERSION=");
      const llvm::StringRef k_dyld_root_path("DYLD_ROOT_PATH=");
      std::string dyld_root_path;

      for (size_t i = 0; i < n; ++i) {
        const char *env_cstr = env.GetArgumentAtIndex(i);
        if (env_cstr) {
          llvm::StringRef env_str(env_cstr);
          if (env_str.consume_front(k_runtime_version)) {
            if (Args::StringToVersion(env_str, major, minor, update))
              return true;
          } else if (env_str.consume_front(k_dyld_root_path)) {
            dyld_root_path = env_str;
          }
        }
      }

      if (!dyld_root_path.empty()) {
        dyld_root_path += "/System/Library/CoreServices/SystemVersion.plist";
        ApplePropertyList system_version_plist(dyld_root_path.c_str());
        std::string product_version;
        if (system_version_plist.GetValueAsString("ProductVersion",
                                                  product_version)) {
          return Args::StringToVersion(product_version, major, minor, update);
        }
      }
    }
    // For simulator platforms, do NOT call back through
    // Platform::GetOSVersion()
    // as it might call Process::GetHostOSVersion() which we don't want as it
    // will be
    // incorrect
    return false;
  }

  return Platform::GetOSVersion(major, minor, update, process);
}

lldb_private::FileSpec PlatformDarwin::LocateExecutable(const char *basename) {
  // A collection of SBFileSpec whose SBFileSpec.m_directory members are filled
  // in with
  // any executable directories that should be searched.
  static std::vector<FileSpec> g_executable_dirs;

  // Find the global list of directories that we will search for
  // executables once so we don't keep doing the work over and over.
  static std::once_flag g_once_flag;
  std::call_once(g_once_flag, []() {

    // When locating executables, trust the DEVELOPER_DIR first if it is set
    FileSpec xcode_contents_dir = GetXcodeContentsPath();
    if (xcode_contents_dir) {
      FileSpec xcode_lldb_resources = xcode_contents_dir;
      xcode_lldb_resources.AppendPathComponent("SharedFrameworks");
      xcode_lldb_resources.AppendPathComponent("LLDB.framework");
      xcode_lldb_resources.AppendPathComponent("Resources");
      if (xcode_lldb_resources.Exists()) {
        FileSpec dir;
        dir.GetDirectory().SetCString(xcode_lldb_resources.GetPath().c_str());
        g_executable_dirs.push_back(dir);
      }
    }
  });

  // Now search the global list of executable directories for the executable we
  // are looking for
  for (const auto &executable_dir : g_executable_dirs) {
    FileSpec executable_file;
    executable_file.GetDirectory() = executable_dir.GetDirectory();
    executable_file.GetFilename().SetCString(basename);
    if (executable_file.Exists())
      return executable_file;
  }

  return FileSpec();
}

lldb_private::Error
PlatformDarwin::LaunchProcess(lldb_private::ProcessLaunchInfo &launch_info) {
  // Starting in Fall 2016 OSes, NSLog messages only get mirrored to stderr
  // if the OS_ACTIVITY_DT_MODE environment variable is set.  (It doesn't
  // require any specific value; rather, it just needs to exist).
  // We will set it here as long as the IDE_DISABLED_OS_ACTIVITY_DT_MODE flag
  // is not set.  Xcode makes use of IDE_DISABLED_OS_ACTIVITY_DT_MODE to tell
  // LLDB *not* to muck with the OS_ACTIVITY_DT_MODE flag when they
  // specifically want it unset.
  const char *disable_env_var = "IDE_DISABLED_OS_ACTIVITY_DT_MODE";
  auto &env_vars = launch_info.GetEnvironmentEntries();
  if (!env_vars.ContainsEnvironmentVariable(disable_env_var)) {
    // We want to make sure that OS_ACTIVITY_DT_MODE is set so that
    // we get os_log and NSLog messages mirrored to the target process
    // stderr.
    if (!env_vars.ContainsEnvironmentVariable("OS_ACTIVITY_DT_MODE"))
      env_vars.AppendArgument("OS_ACTIVITY_DT_MODE=enable");
  }

  // Let our parent class do the real launching.
  return PlatformPOSIX::LaunchProcess(launch_info);
}
