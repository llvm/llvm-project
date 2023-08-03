//===- llvm/Support/PrefixMappingFileSystem.h - -----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_PREFIXMAPPINGFILESYSTEM_H
#define LLVM_SUPPORT_PREFIXMAPPINGFILESYSTEM_H

#include "llvm/ADT/IntrusiveRefCntPtr.h"

namespace llvm {
class PrefixMapper;

namespace vfs {
class FileSystem;

/// Creates a VFS that remaps paths using the given \p Mapper, before looking
/// them up.
std::unique_ptr<FileSystem>
createPrefixMappingFileSystem(PrefixMapper Mapper,
                              IntrusiveRefCntPtr<FileSystem> UnderlyingFS);

} // end namespace vfs
} // end namespace llvm

#endif // LLVM_SUPPORT_PREFIXMAPPINGFILESYSTEM_H
