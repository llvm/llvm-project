//===- VirtualFileSystemFwd.h - Virtual File System Forward -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// Contains the forward declaration for vfs::FileSystem, as well as the
/// IntrusiveRefCntPtrInfo specialization for it. This allows the
/// vfs::FileSystem class to be used with IntrusiveRefCntPtr without including
/// its full definition.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_VIRTUALFILESYSTEMFWD_H
#define LLVM_SUPPORT_VIRTUALFILESYSTEMFWD_H

#include "llvm/ADT/IntrusiveRefCntPtr.h"
#include "llvm/Support/Compiler.h"

namespace llvm {
namespace vfs {
class FileSystem;
} // namespace vfs

template <> struct LLVM_ABI IntrusiveRefCntPtrInfo<::llvm::vfs::FileSystem> {
  static unsigned useCount(const ::llvm::vfs::FileSystem *FS);
  static void retain(::llvm::vfs::FileSystem *FS);
  static void release(::llvm::vfs::FileSystem *FS);
};

} // namespace llvm

#endif // LLVM_SUPPORT_VIRTUALFILESYSTEMFWD_H
