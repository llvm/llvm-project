//===- VirtualCachedDirectoryEntry.h ----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_VIRTUALCACHEDDIRECTORYENTRY_H
#define LLVM_SUPPORT_VIRTUALCACHEDDIRECTORYENTRY_H

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Path.h"

namespace llvm {
namespace vfs {

/// Base class for a cached directory entry, exposing details of the filesystem
/// tree.
///
/// TODO: Move more of \a llvm::cas::FileSystemCache::DirectoryEntry to here.
class CachedDirectoryEntry {
public:
  /// The name of this filesystem node.
  StringRef getName() const { return Name; }

  /// The path to this filesystem node (including the name). The parent path of
  /// the tree path is guaranteed not to contain symbolic links.
  ///
  /// For example, given:
  ///
  ///     /a/sym -> b
  ///     /a/b/c
  ///     /d
  ///
  /// Then directory entries exist with the following tree paths:
  ///
  ///     - "/a"
  ///         - "/a/b"
  ///             - "/a/b/c"
  ///         - "/a/sym"
  ///     - "/d"
  ///
  /// The directory entry referred to by "/a/sym/c" will have the tree path
  /// "/a/b/c".
  StringRef getTreePath() const { return TreePath; }

  explicit CachedDirectoryEntry(
      StringRef TreePath, sys::path::Style Style = sys::path::Style::native)
      : TreePath(TreePath), Name(sys::path::filename(TreePath, Style)) {}

protected:
  StringRef TreePath;
  StringRef Name;
};

} // namespace vfs
} // namespace llvm

#endif // LLVM_SUPPORT_VIRTUALCACHEDDIRECTORYENTRY_H
