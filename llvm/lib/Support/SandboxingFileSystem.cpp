//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// VFS implementation that restricts access to only certain directories.
///
//===----------------------------------------------------------------------===//

#include "llvm/Support/SandboxingFileSystem.h"
#include "llvm/Support/VirtualFileSystem.h"

using namespace llvm;

// FIXME: Move to llvm/Support/Path.h?
/// \returns true if \p Path is a nested directory in \p ParentPath or equal to
/// it. \p Path should be absolute. Returns false if \p ParentPath is relative.
static bool isPathNestedIn(StringRef Path, StringRef ParentPath) {
  assert(sys::path::is_absolute(Path));
  if (Path.size() < ParentPath.size())
    return false;
  if (!Path.starts_with(ParentPath))
    return false;
  if (Path.size() == ParentPath.size())
    return true;
  return sys::path::is_separator(Path.drop_front(ParentPath.size()).front());
}

namespace {
class SandboxingFileSystem final
    : public llvm::RTTIExtends<SandboxingFileSystem, vfs::FileSystem> {
  IntrusiveRefCntPtr<vfs::FileSystem> FS;
  SmallVector<std::string, 3> AllowedPaths;

  ErrorOr<bool> isInSandbox(const Twine &Path) const {
    SmallString<256> PathToCheck;
    Path.toVector(PathToCheck);
    if (std::error_code EC = FS->makeAbsolute(PathToCheck))
      return EC;

    for (const auto &AllowedPath : AllowedPaths) {
      if (isPathNestedIn(PathToCheck, AllowedPath))
        return true;
    }
    return false;
  }

public:
  SandboxingFileSystem(IntrusiveRefCntPtr<vfs::FileSystem> FS)
      : FS(std::move(FS)) {}

  static const char ID;

  Error addAllowedPaths(ArrayRef<StringRef> AllowedPaths) {
    SmallString<256> PathBuf;
    for (StringRef Path : AllowedPaths) {
      PathBuf.clear();
      PathBuf += Path;
      if (std::error_code EC = FS->makeAbsolute(PathBuf))
        return createFileError(Path, EC);
      this->AllowedPaths.push_back(PathBuf.str().str());
    }
    return Error::success();
  }

  ErrorOr<vfs::Status> status(const Twine &Path) override {
    ErrorOr<bool> inSandbox = isInSandbox(Path);
    if (!inSandbox)
      return inSandbox.getError();
    if (!*inSandbox)
      return std::make_error_code(std::errc::no_such_file_or_directory);

    return FS->status(Path);
  }

  llvm::ErrorOr<std::unique_ptr<vfs::File>>
  openFileForRead(const Twine &Path) override {
    ErrorOr<bool> inSandbox = isInSandbox(Path);
    if (!inSandbox)
      return inSandbox.getError();
    if (!*inSandbox)
      return std::make_error_code(std::errc::no_such_file_or_directory);

    return FS->openFileForRead(Path);
  }

  vfs::directory_iterator dir_begin(const Twine &Dir,
                                    std::error_code &EC) override {
    ErrorOr<bool> inSandbox = isInSandbox(Dir);
    if (!inSandbox) {
      EC = inSandbox.getError();
      return vfs::directory_iterator();
    }
    if (!*inSandbox) {
      EC = std::make_error_code(std::errc::no_such_file_or_directory);
      return vfs::directory_iterator();
    }

    return FS->dir_begin(Dir, EC);
  }

  std::error_code setCurrentWorkingDirectory(const Twine &Path) override {
    return FS->setCurrentWorkingDirectory(Path);
  }

  llvm::ErrorOr<std::string> getCurrentWorkingDirectory() const override {
    return FS->getCurrentWorkingDirectory();
  }
};
} // namespace

const char SandboxingFileSystem::ID = 0;

Expected<std::unique_ptr<vfs::FileSystem>> vfs::createSandboxingFileSystem(
    IntrusiveRefCntPtr<vfs::FileSystem> UnderlyingFS,
    ArrayRef<StringRef> AllowedPaths) {
  auto FS = std::make_unique<SandboxingFileSystem>(std::move(UnderlyingFS));
  if (Error E = FS->addAllowedPaths(AllowedPaths))
    return E;
  return FS;
}
