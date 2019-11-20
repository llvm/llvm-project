//===-- FileCollector.h -----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_FILE_COLLECTOR_H
#define LLVM_SUPPORT_FILE_COLLECTOR_H

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/VirtualFileSystem.h"

#include <mutex>

namespace llvm {

/// Collects files into a directory and generates a mapping that can be used by
/// the VFS.
class FileCollector {
public:
  FileCollector(std::string Root, std::string OverlayRoot);

  void addFile(const Twine &file);

  /// Write the yaml mapping (for the VFS) to the given file.
  std::error_code writeMapping(StringRef mapping_file);

  /// Copy the files into the root directory.
  ///
  /// When StopOnError is true (the default) we abort as soon as one file
  /// cannot be copied. This is relatively common, for example when a file was
  /// removed after it was added to the mapping.
  std::error_code copyFiles(bool StopOnError = true);

  /// Create a VFS that collects all the paths that might be looked at by the
  /// file system accesses.
  static IntrusiveRefCntPtr<vfs::FileSystem>
  createCollectorVFS(IntrusiveRefCntPtr<vfs::FileSystem> BaseFS,
                     std::shared_ptr<FileCollector> Collector);

private:
  void addFileImpl(StringRef SrcPath);

  bool markAsSeen(StringRef Path) {
    if (Path.empty())
      return false;
    return Seen.insert(Path).second;
  }

  bool getRealPath(StringRef SrcPath, SmallVectorImpl<char> &Result);

  void addFileToMapping(StringRef VirtualPath, StringRef RealPath) {
    VFSWriter.addFileMapping(VirtualPath, RealPath);
  }

protected:
  /// Synchronizes adding files.
  std::mutex Mutex;

  /// The root directory where files are copied.
  std::string Root;

  /// The root directory where the VFS overlay lives.
  std::string OverlayRoot;

  /// Tracks already seen files so they can be skipped.
  StringSet<> Seen;

  /// The yaml mapping writer.
  vfs::YAMLVFSWriter VFSWriter;

  /// Caches RealPath calls when resolving symlinks.
  StringMap<std::string> SymlinkMap;
};

} // end namespace llvm

#endif // LLVM_SUPPORT_FILE_COLLECTOR_H
