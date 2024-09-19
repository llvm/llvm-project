//===-- MockSymlinkFileSystem.h
//--------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Utility/FileSpec.h"
#include "llvm/Support/VirtualFileSystem.h"

namespace lldb_private {

// A mock file system that realpath's a given symlink to a given realpath.
class MockSymlinkFileSystem : public llvm::vfs::FileSystem {
public:
  // Treat all files as non-symlinks.
  MockSymlinkFileSystem() = default;

  /// Treat \a symlink as a symlink to \a realpath. Treat all other files as
  /// non-symlinks.
  MockSymlinkFileSystem(FileSpec &&symlink, FileSpec &&realpath,
                        FileSpec::Style style = FileSpec::Style::native)
      : m_symlink(std::move(symlink)), m_realpath(std::move(realpath)),
        m_style(style) {}

  /// If \a Path matches the symlink given in the ctor, put the realpath given
  /// in the ctor into \a Output.
  std::error_code getRealPath(const llvm::Twine &Path,
                              llvm::SmallVectorImpl<char> &Output) override {
    if (FileSpec(Path.str(), m_style) == m_symlink) {
      std::string path = m_realpath.GetPath();
      Output.assign(path.begin(), path.end());
    } else {
      Path.toVector(Output);
    }
    return {};
  }

  // Implement the rest of the interface
  llvm::ErrorOr<llvm::vfs::Status> status(const llvm::Twine &Path) override {
    return llvm::errc::operation_not_permitted;
  }
  llvm::ErrorOr<std::unique_ptr<llvm::vfs::File>>
  openFileForRead(const llvm::Twine &Path, bool IsText = true) override {
    return llvm::errc::operation_not_permitted;
  }
  llvm::vfs::directory_iterator dir_begin(const llvm::Twine &Dir,
                                          std::error_code &EC) override {
    return llvm::vfs::directory_iterator();
  }
  std::error_code setCurrentWorkingDirectory(const llvm::Twine &Path) override {
    return llvm::errc::operation_not_permitted;
  }
  llvm::ErrorOr<std::string> getCurrentWorkingDirectory() const override {
    return llvm::errc::operation_not_permitted;
  }

private:
  FileSpec m_symlink;
  FileSpec m_realpath;
  FileSpec::Style m_style;
};

} // namespace lldb_private
