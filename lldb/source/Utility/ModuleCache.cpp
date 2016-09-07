//===--------------------- ModuleCache.cpp ----------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "ModuleCache.h"

#include "lldb/Core/Log.h"
#include "lldb/Core/Module.h"
#include "lldb/Core/ModuleList.h"
#include "lldb/Core/ModuleSpec.h"
#include "lldb/Host/File.h"
#include "lldb/Host/FileSystem.h"
#include "lldb/Host/LockFile.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FileUtilities.h"

#include <assert.h>

#include <cstdio>

using namespace lldb;
using namespace lldb_private;

namespace {

const char *kModulesSubdir = ".cache";
const char *kLockDirName = ".lock";
const char *kTempFileName = ".temp";
const char *kTempSymFileName = ".symtemp";
const char *kSymFileExtension = ".sym";
const char *kFSIllegalChars = "\\/:*?\"<>|";

std::string GetEscapedHostname(const char *hostname) {
  std::string result(hostname);
  size_t size = result.size();
  for (size_t i = 0; i < size; ++i) {
    if ((result[i] >= 1 && result[i] <= 31) ||
        strchr(kFSIllegalChars, result[i]) != nullptr)
      result[i] = '_';
  }
  return result;
}

class ModuleLock {
private:
  File m_file;
  std::unique_ptr<lldb_private::LockFile> m_lock;
  FileSpec m_file_spec;

public:
  ModuleLock(const FileSpec &root_dir_spec, const UUID &uuid, Error &error);
  void Delete();
};

FileSpec JoinPath(const FileSpec &path1, const char *path2) {
  FileSpec result_spec(path1);
  result_spec.AppendPathComponent(path2);
  return result_spec;
}

Error MakeDirectory(const FileSpec &dir_path) {
  if (dir_path.Exists()) {
    if (!dir_path.IsDirectory())
      return Error("Invalid existing path");

    return Error();
  }

  return FileSystem::MakeDirectory(dir_path, eFilePermissionsDirectoryDefault);
}

FileSpec GetModuleDirectory(const FileSpec &root_dir_spec, const UUID &uuid) {
  const auto modules_dir_spec = JoinPath(root_dir_spec, kModulesSubdir);
  return JoinPath(modules_dir_spec, uuid.GetAsString().c_str());
}

FileSpec GetSymbolFileSpec(const FileSpec &module_file_spec) {
  return FileSpec((module_file_spec.GetPath() + kSymFileExtension).c_str(),
                  false);
}

void DeleteExistingModule(const FileSpec &root_dir_spec,
                          const FileSpec &sysroot_module_path_spec) {
  Log *log(GetLogIfAllCategoriesSet(LIBLLDB_LOG_MODULES));
  UUID module_uuid;
  {
    auto module_sp =
        std::make_shared<Module>(ModuleSpec(sysroot_module_path_spec));
    module_uuid = module_sp->GetUUID();
  }

  if (!module_uuid.IsValid())
    return;

  Error error;
  ModuleLock lock(root_dir_spec, module_uuid, error);
  if (error.Fail()) {
    if (log)
      log->Printf("Failed to lock module %s: %s",
                  module_uuid.GetAsString().c_str(), error.AsCString());
  }

  auto link_count = FileSystem::GetHardlinkCount(sysroot_module_path_spec);
  if (link_count == -1)
    return;

  if (link_count > 2) // module is referred by other hosts.
    return;

  const auto module_spec_dir = GetModuleDirectory(root_dir_spec, module_uuid);
  FileSystem::DeleteDirectory(module_spec_dir, true);
  lock.Delete();
}

void DecrementRefExistingModule(const FileSpec &root_dir_spec,
                                const FileSpec &sysroot_module_path_spec) {
  // Remove $platform/.cache/$uuid folder if nobody else references it.
  DeleteExistingModule(root_dir_spec, sysroot_module_path_spec);

  // Remove sysroot link.
  FileSystem::Unlink(sysroot_module_path_spec);

  FileSpec symfile_spec = GetSymbolFileSpec(sysroot_module_path_spec);
  if (symfile_spec.Exists()) // delete module's symbol file if exists.
    FileSystem::Unlink(symfile_spec);
}

Error CreateHostSysRootModuleLink(const FileSpec &root_dir_spec,
                                  const char *hostname,
                                  const FileSpec &platform_module_spec,
                                  const FileSpec &local_module_spec,
                                  bool delete_existing) {
  const auto sysroot_module_path_spec =
      JoinPath(JoinPath(root_dir_spec, hostname),
               platform_module_spec.GetPath().c_str());
  if (sysroot_module_path_spec.Exists()) {
    if (!delete_existing)
      return Error();

    DecrementRefExistingModule(root_dir_spec, sysroot_module_path_spec);
  }

  const auto error = MakeDirectory(
      FileSpec(sysroot_module_path_spec.GetDirectory().AsCString(), false));
  if (error.Fail())
    return error;

  return FileSystem::Hardlink(sysroot_module_path_spec, local_module_spec);
}

} // namespace

ModuleLock::ModuleLock(const FileSpec &root_dir_spec, const UUID &uuid,
                       Error &error) {
  const auto lock_dir_spec = JoinPath(root_dir_spec, kLockDirName);
  error = MakeDirectory(lock_dir_spec);
  if (error.Fail())
    return;

  m_file_spec = JoinPath(lock_dir_spec, uuid.GetAsString().c_str());
  m_file.Open(m_file_spec.GetCString(), File::eOpenOptionWrite |
                                            File::eOpenOptionCanCreate |
                                            File::eOpenOptionCloseOnExec);
  if (!m_file) {
    error.SetErrorToErrno();
    return;
  }

  m_lock.reset(new lldb_private::LockFile(m_file.GetDescriptor()));
  error = m_lock->WriteLock(0, 1);
  if (error.Fail())
    error.SetErrorStringWithFormat("Failed to lock file: %s",
                                   error.AsCString());
}

void ModuleLock::Delete() {
  if (!m_file)
    return;

  m_file.Close();
  FileSystem::Unlink(m_file_spec);
}

/////////////////////////////////////////////////////////////////////////

Error ModuleCache::Put(const FileSpec &root_dir_spec, const char *hostname,
                       const ModuleSpec &module_spec, const FileSpec &tmp_file,
                       const FileSpec &target_file) {
  const auto module_spec_dir =
      GetModuleDirectory(root_dir_spec, module_spec.GetUUID());
  const auto module_file_path =
      JoinPath(module_spec_dir, target_file.GetFilename().AsCString());

  const auto tmp_file_path = tmp_file.GetPath();
  const auto err_code = llvm::sys::fs::rename(
      tmp_file_path.c_str(), module_file_path.GetPath().c_str());
  if (err_code)
    return Error("Failed to rename file %s to %s: %s", tmp_file_path.c_str(),
                 module_file_path.GetPath().c_str(),
                 err_code.message().c_str());

  const auto error = CreateHostSysRootModuleLink(
      root_dir_spec, hostname, target_file, module_file_path, true);
  if (error.Fail())
    return Error("Failed to create link to %s: %s",
                 module_file_path.GetPath().c_str(), error.AsCString());
  return Error();
}

Error ModuleCache::Get(const FileSpec &root_dir_spec, const char *hostname,
                       const ModuleSpec &module_spec,
                       ModuleSP &cached_module_sp, bool *did_create_ptr) {
  const auto find_it =
      m_loaded_modules.find(module_spec.GetUUID().GetAsString());
  if (find_it != m_loaded_modules.end()) {
    cached_module_sp = (*find_it).second.lock();
    if (cached_module_sp)
      return Error();
    m_loaded_modules.erase(find_it);
  }

  const auto module_spec_dir =
      GetModuleDirectory(root_dir_spec, module_spec.GetUUID());
  const auto module_file_path = JoinPath(
      module_spec_dir, module_spec.GetFileSpec().GetFilename().AsCString());

  if (!module_file_path.Exists())
    return Error("Module %s not found", module_file_path.GetPath().c_str());
  if (module_file_path.GetByteSize() != module_spec.GetObjectSize())
    return Error("Module %s has invalid file size",
                 module_file_path.GetPath().c_str());

  // We may have already cached module but downloaded from an another host - in
  // this case let's create a link to it.
  auto error = CreateHostSysRootModuleLink(root_dir_spec, hostname,
                                           module_spec.GetFileSpec(),
                                           module_file_path, false);
  if (error.Fail())
    return Error("Failed to create link to %s: %s",
                 module_file_path.GetPath().c_str(), error.AsCString());

  auto cached_module_spec(module_spec);
  cached_module_spec.GetUUID().Clear(); // Clear UUID since it may contain md5
                                        // content hash instead of real UUID.
  cached_module_spec.GetFileSpec() = module_file_path;
  cached_module_spec.GetPlatformFileSpec() = module_spec.GetFileSpec();

  error = ModuleList::GetSharedModule(cached_module_spec, cached_module_sp,
                                      nullptr, nullptr, did_create_ptr, false);
  if (error.Fail())
    return error;

  FileSpec symfile_spec = GetSymbolFileSpec(cached_module_sp->GetFileSpec());
  if (symfile_spec.Exists())
    cached_module_sp->SetSymbolFileFileSpec(symfile_spec);

  m_loaded_modules.insert(
      std::make_pair(module_spec.GetUUID().GetAsString(), cached_module_sp));

  return Error();
}

Error ModuleCache::GetAndPut(const FileSpec &root_dir_spec,
                             const char *hostname,
                             const ModuleSpec &module_spec,
                             const ModuleDownloader &module_downloader,
                             const SymfileDownloader &symfile_downloader,
                             lldb::ModuleSP &cached_module_sp,
                             bool *did_create_ptr) {
  const auto module_spec_dir =
      GetModuleDirectory(root_dir_spec, module_spec.GetUUID());
  auto error = MakeDirectory(module_spec_dir);
  if (error.Fail())
    return error;

  ModuleLock lock(root_dir_spec, module_spec.GetUUID(), error);
  if (error.Fail())
    return Error("Failed to lock module %s: %s",
                 module_spec.GetUUID().GetAsString().c_str(),
                 error.AsCString());

  const auto escaped_hostname(GetEscapedHostname(hostname));
  // Check local cache for a module.
  error = Get(root_dir_spec, escaped_hostname.c_str(), module_spec,
              cached_module_sp, did_create_ptr);
  if (error.Success())
    return error;

  const auto tmp_download_file_spec = JoinPath(module_spec_dir, kTempFileName);
  error = module_downloader(module_spec, tmp_download_file_spec);
  llvm::FileRemover tmp_file_remover(tmp_download_file_spec.GetPath().c_str());
  if (error.Fail())
    return Error("Failed to download module: %s", error.AsCString());

  // Put downloaded file into local module cache.
  error = Put(root_dir_spec, escaped_hostname.c_str(), module_spec,
              tmp_download_file_spec, module_spec.GetFileSpec());
  if (error.Fail())
    return Error("Failed to put module into cache: %s", error.AsCString());

  tmp_file_remover.releaseFile();
  error = Get(root_dir_spec, escaped_hostname.c_str(), module_spec,
              cached_module_sp, did_create_ptr);
  if (error.Fail())
    return error;

  // Fetching a symbol file for the module
  const auto tmp_download_sym_file_spec =
      JoinPath(module_spec_dir, kTempSymFileName);
  error = symfile_downloader(cached_module_sp, tmp_download_sym_file_spec);
  llvm::FileRemover tmp_symfile_remover(
      tmp_download_sym_file_spec.GetPath().c_str());
  if (error.Fail())
    // Failed to download a symfile but fetching the module was successful. The
    // module might
    // contain the necessary symbols and the debugging is also possible without
    // a symfile.
    return Error();

  error = Put(root_dir_spec, escaped_hostname.c_str(), module_spec,
              tmp_download_sym_file_spec,
              GetSymbolFileSpec(module_spec.GetFileSpec()));
  if (error.Fail())
    return Error("Failed to put symbol file into cache: %s", error.AsCString());

  tmp_symfile_remover.releaseFile();

  FileSpec symfile_spec = GetSymbolFileSpec(cached_module_sp->GetFileSpec());
  cached_module_sp->SetSymbolFileFileSpec(symfile_spec);
  return Error();
}
