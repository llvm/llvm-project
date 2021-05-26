//===-- FileSystem.cpp ----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Host/FileSystem.h"

#include "lldb/Utility/LLDBAssert.h"
#include "lldb/Utility/TildeExpressionResolver.h"

#include "llvm/Support/Errc.h"
#include "llvm/Support/Errno.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/Threading.h"

#include <cerrno>
#include <climits>
#include <cstdarg>
#include <cstdio>
#include <fcntl.h>

#ifdef _WIN32
#include "lldb/Host/windows/windows.h"
#else
#include <sys/ioctl.h>
#include <sys/stat.h>
#include <termios.h>
#include <unistd.h>
#endif

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

void FileSystem::Initialize(std::shared_ptr<FileCollectorBase> collector) {
  lldbassert(!InstanceImpl() && "Already initialized.");
  InstanceImpl().emplace(collector);
}

llvm::Error FileSystem::Initialize(const FileSpec &mapping) {
  lldbassert(!InstanceImpl() && "Already initialized.");

  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> buffer =
      llvm::vfs::getRealFileSystem()->getBufferForFile(mapping.GetPath());

  if (!buffer)
    return llvm::errorCodeToError(buffer.getError());

  InstanceImpl().emplace(llvm::vfs::getVFSFromYAML(std::move(buffer.get()),
                                                   nullptr, mapping.GetPath()),
                         true);

  return llvm::Error::success();
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

vfs::directory_iterator FileSystem::DirBegin(const FileSpec &file_spec,
                                             std::error_code &ec) {
  if (!file_spec) {
    ec = std::error_code(static_cast<int>(errc::no_such_file_or_directory),
                         std::system_category());
    return {};
  }
  return DirBegin(file_spec.GetPath(), ec);
}

vfs::directory_iterator FileSystem::DirBegin(const Twine &dir,
                                             std::error_code &ec) {
  return m_fs->dir_begin(dir, ec);
}

llvm::ErrorOr<vfs::Status>
FileSystem::GetStatus(const FileSpec &file_spec) const {
  if (!file_spec)
    return std::error_code(static_cast<int>(errc::no_such_file_or_directory),
                           std::system_category());
  return GetStatus(file_spec.GetPath());
}

llvm::ErrorOr<vfs::Status> FileSystem::GetStatus(const Twine &path) const {
  return m_fs->status(path);
}

sys::TimePoint<>
FileSystem::GetModificationTime(const FileSpec &file_spec) const {
  if (!file_spec)
    return sys::TimePoint<>();
  return GetModificationTime(file_spec.GetPath());
}

sys::TimePoint<> FileSystem::GetModificationTime(const Twine &path) const {
  ErrorOr<vfs::Status> status = m_fs->status(path);
  if (!status)
    return sys::TimePoint<>();
  return status->getLastModificationTime();
}

uint64_t FileSystem::GetByteSize(const FileSpec &file_spec) const {
  if (!file_spec)
    return 0;
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

uint32_t FileSystem::GetPermissions(const FileSpec &file_spec,
                                    std::error_code &ec) const {
  if (!file_spec)
    return sys::fs::perms::perms_not_known;
  return GetPermissions(file_spec.GetPath(), ec);
}

uint32_t FileSystem::GetPermissions(const Twine &path) const {
  std::error_code ec;
  return GetPermissions(path, ec);
}

uint32_t FileSystem::GetPermissions(const Twine &path,
                                    std::error_code &ec) const {
  ErrorOr<vfs::Status> status = m_fs->status(path);
  if (!status) {
    ec = status.getError();
    return sys::fs::perms::perms_not_known;
  }
  return status->getPermissions();
}

bool FileSystem::Exists(const Twine &path) const { return m_fs->exists(path); }

bool FileSystem::Exists(const FileSpec &file_spec) const {
  return file_spec && Exists(file_spec.GetPath());
}

bool FileSystem::Readable(const Twine &path) const {
  return GetPermissions(path) & sys::fs::perms::all_read;
}

bool FileSystem::Readable(const FileSpec &file_spec) const {
  return file_spec && Readable(file_spec.GetPath());
}

bool FileSystem::IsDirectory(const Twine &path) const {
  ErrorOr<vfs::Status> status = m_fs->status(path);
  if (!status)
    return false;
  return status->isDirectory();
}

bool FileSystem::IsDirectory(const FileSpec &file_spec) const {
  return file_spec && IsDirectory(file_spec.GetPath());
}

bool FileSystem::IsLocal(const Twine &path) const {
  bool b = false;
  m_fs->isLocal(path, b);
  return b;
}

bool FileSystem::IsLocal(const FileSpec &file_spec) const {
  return file_spec && IsLocal(file_spec.GetPath());
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

  // Resolve tilde in path.
  SmallString<128> resolved(path.begin(), path.end());
  StandardTildeExpressionResolver Resolver;
  Resolver.ResolveFullPath(llvm::StringRef(path.begin(), path.size()),
                           resolved);

  // Try making the path absolute if it exists.
  SmallString<128> absolute(resolved.begin(), resolved.end());
  MakeAbsolute(absolute);

  path.clear();
  if (Exists(absolute)) {
    path.append(absolute.begin(), absolute.end());
  } else {
    path.append(resolved.begin(), resolved.end());
  }
}

void FileSystem::Resolve(FileSpec &file_spec) {
  if (!file_spec)
    return;

  // Extract path from the FileSpec.
  SmallString<128> path;
  file_spec.GetPath(path);

  // Resolve the path.
  Resolve(path);

  // Update the FileSpec with the resolved path.
  if (file_spec.GetFilename().IsEmpty())
    file_spec.GetDirectory().SetString(path);
  else
    file_spec.SetPath(path);
  file_spec.SetIsResolved(true);
}

std::shared_ptr<DataBufferLLVM>
FileSystem::CreateDataBuffer(const llvm::Twine &path, uint64_t size,
                             uint64_t offset) {
  Collect(path);

  const bool is_volatile = !IsLocal(path);
  const ErrorOr<std::string> external_path = GetExternalPath(path);

  if (!external_path)
    return nullptr;

  std::unique_ptr<llvm::WritableMemoryBuffer> buffer;
  if (size == 0) {
    auto buffer_or_error =
        llvm::WritableMemoryBuffer::getFile(*external_path, is_volatile);
    if (!buffer_or_error)
      return nullptr;
    buffer = std::move(*buffer_or_error);
  } else {
    auto buffer_or_error = llvm::WritableMemoryBuffer::getFileSlice(
        *external_path, size, offset, is_volatile);
    if (!buffer_or_error)
      return nullptr;
    buffer = std::move(*buffer_or_error);
  }
  return std::shared_ptr<DataBufferLLVM>(new DataBufferLLVM(std::move(buffer)));
}

std::shared_ptr<DataBufferLLVM>
FileSystem::CreateDataBuffer(const FileSpec &file_spec, uint64_t size,
                             uint64_t offset) {
  return CreateDataBuffer(file_spec.GetPath(), size, offset);
}

bool FileSystem::ResolveExecutableLocation(FileSpec &file_spec) {
  // If the directory is set there's nothing to do.
  ConstString directory = file_spec.GetDirectory();
  if (directory)
    return false;

  // We cannot look for a file if there's no file name.
  ConstString filename = file_spec.GetFilename();
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

bool FileSystem::GetHomeDirectory(SmallVectorImpl<char> &path) const {
  if (!m_home_directory.empty()) {
    path.assign(m_home_directory.begin(), m_home_directory.end());
    return true;
  }
  return llvm::sys::path::home_directory(path);
}

bool FileSystem::GetHomeDirectory(FileSpec &file_spec) const {
  SmallString<128> home_dir;
  if (!GetHomeDirectory(home_dir))
    return false;
  file_spec.SetPath(home_dir);
  return true;
}

static int OpenWithFS(const FileSystem &fs, const char *path, int flags,
                      int mode) {
  return const_cast<FileSystem &>(fs).Open(path, flags, mode);
}

static int GetOpenFlags(uint32_t options) {
  const bool read = options & File::eOpenOptionRead;
  const bool write = options & File::eOpenOptionWrite;

  int open_flags = 0;
  if (write) {
    if (read)
      open_flags |= O_RDWR;
    else
      open_flags |= O_WRONLY;

    if (options & File::eOpenOptionAppend)
      open_flags |= O_APPEND;

    if (options & File::eOpenOptionTruncate)
      open_flags |= O_TRUNC;

    if (options & File::eOpenOptionCanCreate)
      open_flags |= O_CREAT;

    if (options & File::eOpenOptionCanCreateNewOnly)
      open_flags |= O_CREAT | O_EXCL;
  } else if (read) {
    open_flags |= O_RDONLY;

#ifndef _WIN32
    if (options & File::eOpenOptionDontFollowSymlinks)
      open_flags |= O_NOFOLLOW;
#endif
  }

#ifndef _WIN32
  if (options & File::eOpenOptionNonBlocking)
    open_flags |= O_NONBLOCK;
  if (options & File::eOpenOptionCloseOnExec)
    open_flags |= O_CLOEXEC;
#else
  open_flags |= O_BINARY;
#endif

  return open_flags;
}

static mode_t GetOpenMode(uint32_t permissions) {
  mode_t mode = 0;
  if (permissions & lldb::eFilePermissionsUserRead)
    mode |= S_IRUSR;
  if (permissions & lldb::eFilePermissionsUserWrite)
    mode |= S_IWUSR;
  if (permissions & lldb::eFilePermissionsUserExecute)
    mode |= S_IXUSR;
  if (permissions & lldb::eFilePermissionsGroupRead)
    mode |= S_IRGRP;
  if (permissions & lldb::eFilePermissionsGroupWrite)
    mode |= S_IWGRP;
  if (permissions & lldb::eFilePermissionsGroupExecute)
    mode |= S_IXGRP;
  if (permissions & lldb::eFilePermissionsWorldRead)
    mode |= S_IROTH;
  if (permissions & lldb::eFilePermissionsWorldWrite)
    mode |= S_IWOTH;
  if (permissions & lldb::eFilePermissionsWorldExecute)
    mode |= S_IXOTH;
  return mode;
}

Expected<FileUP> FileSystem::Open(const FileSpec &file_spec,
                                  File::OpenOptions options,
                                  uint32_t permissions, bool should_close_fd) {
  Collect(file_spec.GetPath());

  const int open_flags = GetOpenFlags(options);
  const mode_t open_mode =
      (open_flags & O_CREAT) ? GetOpenMode(permissions) : 0;

  auto path = GetExternalPath(file_spec);
  if (!path)
    return errorCodeToError(path.getError());

  int descriptor = llvm::sys::RetryAfterSignal(
      -1, OpenWithFS, *this, path->c_str(), open_flags, open_mode);

  if (!File::DescriptorIsValid(descriptor))
    return llvm::errorCodeToError(
        std::error_code(errno, std::system_category()));

  auto file = std::unique_ptr<File>(
      new NativeFile(descriptor, options, should_close_fd));
  assert(file->IsValid());
  return std::move(file);
}

ErrorOr<std::string> FileSystem::GetExternalPath(const llvm::Twine &path) {
  if (!m_mapped)
    return path.str();

  // If VFS mapped we know the underlying FS is a RedirectingFileSystem.
  ErrorOr<vfs::RedirectingFileSystem::LookupResult> Result =
      static_cast<vfs::RedirectingFileSystem &>(*m_fs).lookupPath(path.str());
  if (!Result) {
    if (Result.getError() == llvm::errc::no_such_file_or_directory) {
      return path.str();
    }
    return Result.getError();
  }

  if (Optional<StringRef> ExtRedirect = Result->getExternalRedirect())
    return std::string(*ExtRedirect);
  return make_error_code(llvm::errc::not_supported);
}

ErrorOr<std::string> FileSystem::GetExternalPath(const FileSpec &file_spec) {
  return GetExternalPath(file_spec.GetPath());
}

void FileSystem::Collect(const FileSpec &file_spec) {
  Collect(file_spec.GetPath());
}

void FileSystem::Collect(const llvm::Twine &file) {
  if (!m_collector)
    return;

  if (llvm::sys::fs::is_directory(file))
    m_collector->addDirectory(file);
  else
    m_collector->addFile(file);
}

void FileSystem::SetHomeDirectory(std::string home_directory) {
  m_home_directory = std::move(home_directory);
}
