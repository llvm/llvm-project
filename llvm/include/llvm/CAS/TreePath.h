//===- llvm/CAS/TreePath.h - Tree path utility ------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CAS_TREEPATH_H
#define LLVM_CAS_TREEPATH_H

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Path.h"
#include <string>

namespace llvm::cas {

// On Windows, prepend a dummy root backslash to tree paths internally
// (\C:\foo) and distinguish them from normal file paths (C:\foo) that
// come from and return to clients and are passed to sys::fs
// functions. This is so that we can have a single root and CAS ID
// representing the root of the filesystem (that may otherwise spread
// across multiple trees / drives) and the path handling to be more
// uniform with the Posix systems. These convert paths between them.
// TODO: Reassess this way of handling multiple roots for Windows later.
inline std::string getTreePath(StringRef FilePath, sys::path::Style PathStyle) {
  if (sys::path::is_style_windows(PathStyle)) {
    assert(FilePath[0] != '\\');
    std::string TreePath = "\\" + std::string(FilePath);
    return TreePath;
  }
  return FilePath.str();
}

inline StringRef getFilePath(StringRef TreePath, sys::path::Style PathStyle) {
  if (sys::path::is_style_windows(PathStyle)) {
    assert(TreePath[0] == '\\');
    return TreePath.drop_front(1);
  }
  return TreePath;
}

} // namespace llvm::cas

#endif // LLVM_CAS_TREEPATH_H
