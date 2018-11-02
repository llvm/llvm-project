//===-- FileSystem.cpp ------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Host/FileSystem.h"

#include "lldb/Utility/LLDBAssert.h"
#include "lldb/Utility/TildeExpressionResolver.h"

#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/Threading.h"

#include <algorithm>
#include <fstream>
#include <vector>

using namespace lldb;
using namespace lldb_private;
using namespace llvm;

FileSystem &FileSystem::Instance() { return *InstanceImpl(); }

void FileSystem::Initialize() {
  lldbassert(!InstanceImpl() && "Already initialized.");
  InstanceImpl().emplace();
}

void FileSystem::Initialize(IntrusiveRefCntPtr<vfs::FileSystem> fs) {
  lldbassert(!InstanceImpl() && "Already initialized.");
  InstanceImpl().emplace(fs);
}

void FileSystem::Terminate() {
  lldbassert(InstanceImpl() && "Already terminated.");
  InstanceImpl().reset();
}

Optional<FileSystem> &FileSystem::InstanceImpl() {
  static Optional<FileSystem> g_fs;
  return g_fs;
}

void FileSystem::SetFileSystem(IntrusiveRefCntPtr<vfs::FileSystem> fs) {
  m_fs = fs;
}

sys::TimePoint<>
FileSystem::GetModificationTime(const FileSpec &file_spec) const {
  return GetModificationTime(file_spec.GetPath());
}

sys::TimePoint<> FileSystem::GetModificationTime(const Twine &path) const {
  ErrorOr<vfs::Status> status = m_fs->status(path);
  if (!status)
    return sys::TimePoint<>();
  return status->getLastModificationTime();
}

uint64_t FileSystem::GetByteSize(const FileSpec &file_spec) const {
  return GetByteSize(file_spec.GetPath());
}

uint64_t FileSystem::GetByteSize(const Twine &path) const {
  ErrorOr<vfs::Status> status = m_fs->status(path);
  if (!status)
    return 0;
  return status->getSize();
}

uint32_t FileSystem::GetPermissions(const FileSpec &file_spec) const {
  return GetPermissions(file_spec.GetPath());
}

uint32_t FileSystem::GetPermissions(const Twine &path) const {
  ErrorOr<vfs::Status> status = m_fs->status(path);
  if (!status)
    return sys::fs::perms::perms_not_known;
  return status->getPermissions();
}

bool FileSystem::Exists(const Twine &path) const { return m_fs->exists(path); }

bool FileSystem::Exists(const FileSpec &file_spec) const {
  return Exists(file_spec.GetPath());
}

bool FileSystem::Readable(const Twine &path) const {
  return GetPermissions(path) & sys::fs::perms::all_read;
}

bool FileSystem::Readable(const FileSpec &file_spec) const {
  return Readable(file_spec.GetPath());
}

void FileSystem::EnumerateDirectory(Twine path, bool find_directories,
                                    bool find_files, bool find_other,
                                    EnumerateDirectoryCallbackType callback,
                                    void *callback_baton) {
  std::error_code EC;
  vfs::recursive_directory_iterator Iter(*m_fs, path, EC);
  vfs::recursive_directory_iterator End;
  for (; Iter != End && !EC; Iter.increment(EC)) {
    const auto &Item = *Iter;
    ErrorOr<vfs::Status> Status = m_fs->status(Item.path());
    if (!Status)
      break;
    if (!find_files && Status->isRegularFile())
      continue;
    if (!find_directories && Status->isDirectory())
      continue;
    if (!find_other && Status->isOther())
      continue;

    auto Result = callback(callback_baton, Status->getType(), Item.path());
    if (Result == eEnumerateDirectoryResultQuit)
      return;
    if (Result == eEnumerateDirectoryResultNext) {
      // Default behavior is to recurse. Opt out if the callback doesn't want
      // this behavior.
      Iter.no_push();
    }
  }
}

std::error_code FileSystem::MakeAbsolute(SmallVectorImpl<char> &path) const {
  return m_fs->makeAbsolute(path);
}

std::error_code FileSystem::MakeAbsolute(FileSpec &file_spec) const {
  SmallString<128> path;
  file_spec.GetPath(path, false);

  auto EC = MakeAbsolute(path);
  if (EC)
    return EC;

  FileSpec new_file_spec(path, file_spec.GetPathStyle());
  file_spec = new_file_spec;
  return {};
}

std::error_code FileSystem::GetRealPath(const Twine &path,
                                        SmallVectorImpl<char> &output) const {
  return m_fs->getRealPath(path, output);
}

void FileSystem::Resolve(SmallVectorImpl<char> &path) {
  if (path.empty())
    return;

  // Resolve tilde.
  SmallString<128> original_path(path.begin(), path.end());
  StandardTildeExpressionResolver Resolver;
  Resolver.ResolveFullPath(original_path, path);

  // Try making the path absolute if it exists.
  SmallString<128> absolute_path(path.begin(), path.end());
  MakeAbsolute(path);
  if (!Exists(path)) {
    path.clear();
    path.append(original_path.begin(), original_path.end());
  }
}

void FileSystem::Resolve(FileSpec &file_spec) {
  // Extract path from the FileSpec.
  SmallString<128> path;
  file_spec.GetPath(path);

  // Resolve the path.
  Resolve(path);

  // Update the FileSpec with the resolved path.
  file_spec.SetPath(path);
  file_spec.SetIsResolved(true);
}

bool FileSystem::ResolveExecutableLocation(FileSpec &file_spec) {
  // If the directory is set there's nothing to do.
  const ConstString &directory = file_spec.GetDirectory();
  if (directory)
    return false;

  // We cannot look for a file if there's no file name.
  const ConstString &filename = file_spec.GetFilename();
  if (!filename)
    return false;

  // Search for the file on the host.
  const std::string filename_str(filename.GetCString());
  llvm::ErrorOr<std::string> error_or_path =
      llvm::sys::findProgramByName(filename_str);
  if (!error_or_path)
    return false;

  // findProgramByName returns "." if it can't find the file.
  llvm::StringRef path = *error_or_path;
  llvm::StringRef parent = llvm::sys::path::parent_path(path);
  if (parent.empty() || parent == ".")
    return false;

  // Make sure that the result exists.
  FileSpec result(*error_or_path);
  if (!Exists(result))
    return false;

  file_spec = result;
  return true;
}
