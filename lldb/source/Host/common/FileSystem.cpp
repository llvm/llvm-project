//===-- FileSystem.cpp ------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Host/FileSystem.h"

#include "lldb/Utility/TildeExpressionResolver.h"

#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Threading.h"

#include <algorithm>
#include <fstream>
#include <vector>

using namespace lldb;
using namespace lldb_private;
using namespace llvm;

FileSystem &FileSystem::Instance() { return *InstanceImpl(); }

void FileSystem::Initialize() {
  assert(!InstanceImpl());
  InstanceImpl().emplace();
}

void FileSystem::Initialize(IntrusiveRefCntPtr<vfs::FileSystem> fs) {
  assert(!InstanceImpl());
  InstanceImpl().emplace(fs);
}

void FileSystem::Terminate() {
  assert(InstanceImpl());
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

std::error_code FileSystem::MakeAbsolute(SmallVectorImpl<char> &path) const {
  return m_fs->makeAbsolute(path);
}

std::error_code FileSystem::MakeAbsolute(FileSpec &file_spec) const {
  SmallString<128> path;
  file_spec.GetPath(path, false);

  auto EC = MakeAbsolute(path);
  if (EC)
    return EC;

  FileSpec new_file_spec(path, false, file_spec.GetPathStyle());
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
}
