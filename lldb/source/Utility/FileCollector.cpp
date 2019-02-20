//===-- FileCollector.cpp ---------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Utility/FileCollector.h"

#include "llvm/ADT/SmallString.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"

using namespace lldb_private;
using namespace llvm;

static bool IsCaseSensitivePath(StringRef path) {
  SmallString<256> tmp_dest = path, upper_dest, real_dest;

  // Remove component traversals, links, etc.
  if (!sys::fs::real_path(path, tmp_dest))
    return true; // Current default value in vfs.yaml
  path = tmp_dest;

  // Change path to all upper case and ask for its real path, if the latter
  // exists and is equal to path, it's not case sensitive. Default to case
  // sensitive in the absence of real_path, since this is the YAMLVFSWriter
  // default.
  upper_dest = path.upper();
  if (sys::fs::real_path(upper_dest, real_dest) && path.equals(real_dest))
    return false;
  return true;
}

FileCollector::FileCollector(const FileSpec &root) : m_root(root) {
  sys::fs::create_directories(m_root.GetPath(), true);
}

bool FileCollector::GetRealPath(StringRef src_path,
                                SmallVectorImpl<char> &result) {
  SmallString<256> real_path;
  StringRef FileName = sys::path::filename(src_path);
  std::string directory = sys::path::parent_path(src_path).str();
  auto dir_with_symlink = m_symlink_map.find(directory);

  // Use real_path to fix any symbolic link component present in a path.
  // Computing the real path is expensive, cache the search through the
  // parent path directory.
  if (dir_with_symlink == m_symlink_map.end()) {
    auto ec = sys::fs::real_path(directory, real_path);
    if (ec)
      return false;
    m_symlink_map[directory] = real_path.str();
  } else {
    real_path = dir_with_symlink->second;
  }

  sys::path::append(real_path, FileName);
  result.swap(real_path);
  return true;
}

void FileCollector::AddFile(const Twine &file) {
  std::lock_guard<std::mutex> lock(m_mutex);
  std::string file_str = file.str();
  if (MarkAsSeen(file_str))
    AddFileImpl(file_str);
}

void FileCollector::AddFileImpl(StringRef src_path) {
  std::string root = m_root.GetPath();

  // We need an absolute src path to append to the root.
  SmallString<256> absolute_src = src_path;
  sys::fs::make_absolute(absolute_src);

  // Canonicalize src to a native path to avoid mixed separator styles.
  sys::path::native(absolute_src);

  // Remove redundant leading "./" pieces and consecutive separators.
  absolute_src = sys::path::remove_leading_dotslash(absolute_src);

  // Canonicalize the source path by removing "..", "." components.
  SmallString<256> virtual_path = absolute_src;
  sys::path::remove_dots(virtual_path, /*remove_dot_dot=*/true);

  // If a ".." component is present after a symlink component, remove_dots may
  // lead to the wrong real destination path. Let the source be canonicalized
  // like that but make sure we always use the real path for the destination.
  SmallString<256> copy_from;
  if (!GetRealPath(absolute_src, copy_from))
    copy_from = virtual_path;

  SmallString<256> dst_path = StringRef(root);
  sys::path::append(dst_path, sys::path::relative_path(copy_from));

  // Always map a canonical src path to its real path into the YAML, by doing
  // this we map different virtual src paths to the same entry in the VFS
  // overlay, which is a way to emulate symlink inside the VFS; this is also
  // needed for correctness, not doing that can lead to module redefinition
  // errors.
  AddFileToMapping(virtual_path, dst_path);
}

std::error_code FileCollector::CopyFiles(bool stop_on_error) {
  for (auto &entry : m_vfs_writer.getMappings()) {
    // Create directory tree.
    if (std::error_code ec =
            sys::fs::create_directories(sys::path::parent_path(entry.RPath),
                                        /*IgnoreExisting=*/true)) {
      if (stop_on_error)
        return ec;
    }

    // Copy file over.
    if (std::error_code ec = sys::fs::copy_file(entry.VPath, entry.RPath)) {
      if (stop_on_error)
        return ec;
    }

    // Copy over permissions.
    if (auto perms = sys::fs::getPermissions(entry.VPath)) {
      if (std::error_code ec = sys::fs::setPermissions(entry.RPath, *perms)) {
        if (stop_on_error)
          return ec;
      }
    }
  }
  return {};
}

std::error_code FileCollector::WriteMapping(const FileSpec &mapping_file) {
  std::lock_guard<std::mutex> lock(m_mutex);

  const std::string root = m_root.GetPath();
  m_vfs_writer.setCaseSensitivity(IsCaseSensitivePath(root));
  m_vfs_writer.setUseExternalNames(false);

  std::error_code ec;
  raw_fd_ostream os(mapping_file.GetPath(), ec, sys::fs::F_Text);
  if (ec)
    return ec;

  m_vfs_writer.write(os);

  return {};
}
