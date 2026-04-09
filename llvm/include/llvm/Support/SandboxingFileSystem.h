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

#ifndef LLVM_SUPPORT_SANDBOXINGFILESYSTEM_H
#define LLVM_SUPPORT_SANDBOXINGFILESYSTEM_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/IntrusiveRefCntPtr.h"
#include "llvm/Support/Error.h"

namespace llvm::vfs {
class FileSystem;

/// Creates a VFS that restricts access to only certain directories.
///
/// \param UnderlyingFS The base file-system that queries will be delegated for
/// the unrestricted paths.
/// \param Path A list of paths (and their sub-paths) to restrict access to.
/// Queries for paths not included in the list will fail with "no such file or
/// directory" error. If a path is relative it will be made absolute using the
/// current working directory of \p UnderlyingFS at creation time.
/// The check for whether a path is contained within one of the \p AllowedPaths
/// is case-sensitive, and there's no path canonicalization happening.
Expected<std::unique_ptr<FileSystem>>
createSandboxingFileSystem(IntrusiveRefCntPtr<FileSystem> UnderlyingFS,
                           ArrayRef<StringRef> AllowedPaths);

} // end namespace llvm::vfs

#endif // LLVM_SUPPORT_SANDBOXINGFILESYSTEM_H
