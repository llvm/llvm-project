//===-- Symbols.cpp ---------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Host/Symbols.h"
#include "lldb/Core/ArchSpec.h"
#include "lldb/Core/DataBuffer.h"
#include "lldb/Core/DataExtractor.h"
#include "lldb/Core/Log.h"
#include "lldb/Core/Module.h"
#include "lldb/Core/ModuleSpec.h"
#include "lldb/Core/Timer.h"
#include "lldb/Core/UUID.h"
#include "lldb/Symbol/ObjectFile.h"
#include "lldb/Target/Target.h"
#include "lldb/Utility/SafeMachO.h"
#include "lldb/Utility/StreamString.h"

#include "llvm/Support/FileSystem.h"

// From MacOSX system header "mach/machine.h"
typedef int cpu_type_t;
typedef int cpu_subtype_t;

using namespace lldb;
using namespace lldb_private;
using namespace llvm::MachO;

#if defined(__APPLE__)

// Forward declaration of method defined in source/Host/macosx/Symbols.cpp
int LocateMacOSXFilesUsingDebugSymbols(const ModuleSpec &module_spec,
                                       ModuleSpec &return_module_spec);

#else

int LocateMacOSXFilesUsingDebugSymbols(const ModuleSpec &module_spec,
                                       ModuleSpec &return_module_spec) {
  // Cannot find MacOSX files using debug symbols on non MacOSX.
  return 0;
}

#endif

static bool FileAtPathContainsArchAndUUID(const FileSpec &file_fspec,
                                          const ArchSpec *arch,
                                          const lldb_private::UUID *uuid) {
  ModuleSpecList module_specs;
  if (ObjectFile::GetModuleSpecifications(file_fspec, 0, 0, module_specs)) {
    ModuleSpec spec;
    for (size_t i = 0; i < module_specs.GetSize(); ++i) {
      assert(module_specs.GetModuleSpecAtIndex(i, spec));
      if ((uuid == NULL || (spec.GetUUIDPtr() && spec.GetUUID() == *uuid)) &&
          (arch == NULL || (spec.GetArchitecturePtr() &&
                            spec.GetArchitecture().IsCompatibleMatch(*arch)))) {
        return true;
      }
    }
  }
  return false;
}

static bool LocateDSYMInVincinityOfExecutable(const ModuleSpec &module_spec,
                                              FileSpec &dsym_fspec) {
  Log *log = lldb_private::GetLogIfAllCategoriesSet(LIBLLDB_LOG_HOST);
  const FileSpec *exec_fspec = module_spec.GetFileSpecPtr();
  if (exec_fspec) {
    char path[PATH_MAX];
    if (exec_fspec->GetPath(path, sizeof(path))) {
      // Make sure the module isn't already just a dSYM file...
      if (strcasestr(path, ".dSYM/Contents/Resources/DWARF") == NULL) {
        if (log) {
          if (module_spec.GetUUIDPtr() && module_spec.GetUUIDPtr()->IsValid()) {
            log->Printf(
                "Searching for dSYM bundle next to executable %s, UUID %s",
                path, module_spec.GetUUIDPtr()->GetAsString().c_str());
          } else {
            log->Printf("Searching for dSYM bundle next to executable %s",
                        path);
          }
        }
        size_t obj_file_path_length = strlen(path);
        ::strncat(path, ".dSYM/Contents/Resources/DWARF/",
                  sizeof(path) - strlen(path) - 1);
        ::strncat(path, exec_fspec->GetFilename().AsCString(),
                  sizeof(path) - strlen(path) - 1);

        dsym_fspec.SetFile(path, false);

        ModuleSpecList module_specs;
        ModuleSpec matched_module_spec;
        if (dsym_fspec.Exists() &&
            FileAtPathContainsArchAndUUID(dsym_fspec,
                                          module_spec.GetArchitecturePtr(),
                                          module_spec.GetUUIDPtr())) {
          if (log) {
            log->Printf("dSYM with matching UUID & arch found at %s", path);
          }
          return true;
        } else {
          path[obj_file_path_length] = '\0';

          char *last_dot = strrchr(path, '.');
          while (last_dot != NULL && last_dot[0]) {
            char *next_slash = strchr(last_dot, '/');
            if (next_slash != NULL) {
              *next_slash = '\0';
              ::strncat(path, ".dSYM/Contents/Resources/DWARF/",
                        sizeof(path) - strlen(path) - 1);
              ::strncat(path, exec_fspec->GetFilename().AsCString(),
                        sizeof(path) - strlen(path) - 1);
              dsym_fspec.SetFile(path, false);
              if (dsym_fspec.Exists() &&
                  FileAtPathContainsArchAndUUID(
                      dsym_fspec, module_spec.GetArchitecturePtr(),
                      module_spec.GetUUIDPtr())) {
                if (log) {
                  log->Printf("dSYM with matching UUID & arch found at %s",
                              path);
                }
                return true;
              } else {
                *last_dot = '\0';
                char *prev_slash = strrchr(path, '/');
                if (prev_slash != NULL)
                  *prev_slash = '\0';
                else
                  break;
              }
            } else {
              break;
            }
          }
        }
      }
    }
  }
  dsym_fspec.Clear();
  return false;
}

FileSpec LocateExecutableSymbolFileDsym(const ModuleSpec &module_spec) {
  const FileSpec *exec_fspec = module_spec.GetFileSpecPtr();
  const ArchSpec *arch = module_spec.GetArchitecturePtr();
  const UUID *uuid = module_spec.GetUUIDPtr();

  Timer scoped_timer(
      LLVM_PRETTY_FUNCTION,
      "LocateExecutableSymbolFileDsym (file = %s, arch = %s, uuid = %p)",
      exec_fspec ? exec_fspec->GetFilename().AsCString("<NULL>") : "<NULL>",
      arch ? arch->GetArchitectureName() : "<NULL>", (const void *)uuid);

  FileSpec symbol_fspec;
  ModuleSpec dsym_module_spec;
  // First try and find the dSYM in the same directory as the executable or in
  // an appropriate parent directory
  if (LocateDSYMInVincinityOfExecutable(module_spec, symbol_fspec) == false) {
    // We failed to easily find the dSYM above, so use DebugSymbols
    LocateMacOSXFilesUsingDebugSymbols(module_spec, dsym_module_spec);
  } else {
    dsym_module_spec.GetSymbolFileSpec() = symbol_fspec;
  }
  return dsym_module_spec.GetSymbolFileSpec();
}

ModuleSpec Symbols::LocateExecutableObjectFile(const ModuleSpec &module_spec) {
  ModuleSpec result;
  const FileSpec *exec_fspec = module_spec.GetFileSpecPtr();
  const ArchSpec *arch = module_spec.GetArchitecturePtr();
  const UUID *uuid = module_spec.GetUUIDPtr();
  Timer scoped_timer(
      LLVM_PRETTY_FUNCTION,
      "LocateExecutableObjectFile (file = %s, arch = %s, uuid = %p)",
      exec_fspec ? exec_fspec->GetFilename().AsCString("<NULL>") : "<NULL>",
      arch ? arch->GetArchitectureName() : "<NULL>", (const void *)uuid);

  ModuleSpecList module_specs;
  ModuleSpec matched_module_spec;
  if (exec_fspec &&
      ObjectFile::GetModuleSpecifications(*exec_fspec, 0, 0, module_specs) &&
      module_specs.FindMatchingModuleSpec(module_spec, matched_module_spec)) {
    result.GetFileSpec() = exec_fspec;
  } else {
    LocateMacOSXFilesUsingDebugSymbols(module_spec, result);
  }
  return result;
}

FileSpec Symbols::LocateExecutableSymbolFile(const ModuleSpec &module_spec) {
  FileSpec symbol_file_spec = module_spec.GetSymbolFileSpec();
  if (symbol_file_spec.IsAbsolute() && symbol_file_spec.Exists())
    return symbol_file_spec;

  const char *symbol_filename = symbol_file_spec.GetFilename().AsCString();
  if (symbol_filename && symbol_filename[0]) {
    FileSpecList debug_file_search_paths(
        Target::GetDefaultDebugFileSearchPaths());

    // Add module directory.
    const ConstString &file_dir = module_spec.GetFileSpec().GetDirectory();
    debug_file_search_paths.AppendIfUnique(
        FileSpec(file_dir.AsCString("."), true));

    // Add current working directory.
    debug_file_search_paths.AppendIfUnique(FileSpec(".", true));

#ifndef LLVM_ON_WIN32
    // Add /usr/lib/debug directory.
    debug_file_search_paths.AppendIfUnique(FileSpec("/usr/lib/debug", true));
#endif // LLVM_ON_WIN32

    std::string uuid_str;
    const UUID &module_uuid = module_spec.GetUUID();
    if (module_uuid.IsValid()) {
      // Some debug files are stored in the .build-id directory like this:
      //   /usr/lib/debug/.build-id/ff/e7fe727889ad82bb153de2ad065b2189693315.debug
      uuid_str = module_uuid.GetAsString("");
      uuid_str.insert(2, 1, '/');
      uuid_str = uuid_str + ".debug";
    }

    size_t num_directories = debug_file_search_paths.GetSize();
    for (size_t idx = 0; idx < num_directories; ++idx) {
      FileSpec dirspec = debug_file_search_paths.GetFileSpecAtIndex(idx);
      dirspec.ResolvePath();
      if (!dirspec.Exists() || !dirspec.IsDirectory())
        continue;

      std::vector<std::string> files;
      std::string dirname = dirspec.GetPath();

      files.push_back(dirname + "/" + symbol_filename);
      files.push_back(dirname + "/.debug/" + symbol_filename);
      files.push_back(dirname + "/.build-id/" + uuid_str);

      // Some debug files may stored in the module directory like this:
      //   /usr/lib/debug/usr/lib/library.so.debug
      if (!file_dir.IsEmpty())
        files.push_back(dirname + file_dir.AsCString() + "/" + symbol_filename);

      const uint32_t num_files = files.size();
      for (size_t idx_file = 0; idx_file < num_files; ++idx_file) {
        const std::string &filename = files[idx_file];
        FileSpec file_spec(filename, true);

        if (llvm::sys::fs::equivalent(file_spec.GetPath(),
                                      module_spec.GetFileSpec().GetPath()))
          continue;

        if (file_spec.Exists()) {
          lldb_private::ModuleSpecList specs;
          const size_t num_specs =
              ObjectFile::GetModuleSpecifications(file_spec, 0, 0, specs);
          assert(num_specs <= 1 &&
                 "Symbol Vendor supports only a single architecture");
          if (num_specs == 1) {
            ModuleSpec mspec;
            if (specs.GetModuleSpecAtIndex(0, mspec)) {
              if (mspec.GetUUID() == module_uuid)
                return file_spec;
            }
          }
        }
      }
    }
  }

  return LocateExecutableSymbolFileDsym(module_spec);
}

#if !defined(__APPLE__)

FileSpec Symbols::FindSymbolFileInBundle(const FileSpec &symfile_bundle,
                                         const lldb_private::UUID *uuid,
                                         const ArchSpec *arch) {
  // FIXME
  return FileSpec();
}

bool Symbols::DownloadObjectAndSymbolFile(ModuleSpec &module_spec,
                                          bool force_lookup) {
  // Fill in the module_spec.GetFileSpec() for the object file and/or the
  // module_spec.GetSymbolFileSpec() for the debug symbols file.
  return false;
}

#endif
