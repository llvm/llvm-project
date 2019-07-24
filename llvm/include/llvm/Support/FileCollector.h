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
  FileCollector(std::string root, std::string overlay);

  void AddFile(const Twine &file);

  /// Write the yaml mapping (for the VFS) to the given file.
  std::error_code WriteMapping(StringRef mapping_file);

  /// Copy the files into the root directory.
  ///
  /// When stop_on_error is true (the default) we abort as soon as one file
  /// cannot be copied. This is relatively common, for example when a file was
  /// removed after it was added to the mapping.
  std::error_code CopyFiles(bool stop_on_error = true);

private:
  void AddFileImpl(StringRef src_path);

  bool MarkAsSeen(StringRef path) { return m_seen.insert(path).second; }

  bool GetRealPath(StringRef src_path,
                   SmallVectorImpl<char> &result);

  void AddFileToMapping(StringRef virtual_path,
                        StringRef real_path) {
    m_vfs_writer.addFileMapping(virtual_path, real_path);
  }

protected:
  /// Synchronizes adding files.
  std::mutex m_mutex;

  /// The root directory where files are copied.
  std::string m_root;

  /// The root directory where the VFS overlay lives.
  std::string m_overlay_root;

  /// Tracks already seen files so they can be skipped.
  StringSet<> m_seen;

  /// The yaml mapping writer.
  vfs::YAMLVFSWriter m_vfs_writer;

  /// Caches real_path calls when resolving symlinks.
  StringMap<std::string> m_symlink_map;
};

} // end namespace llvm

#endif // LLVM_SUPPORT_FILE_COLLECTOR_H
