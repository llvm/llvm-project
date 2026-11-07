//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CAS_CASPROVIDINGFILESYSTEM_H
#define LLVM_CAS_CASPROVIDINGFILESYSTEM_H

#include "llvm/ADT/IntrusiveRefCntPtr.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Error.h"

namespace llvm {
namespace vfs {
class FileSystem;
}

namespace cas {
class ObjectStore;

/// Create a \c CASBackedFileSystem on top of \p UnderlyingFS. Each file opened
/// for reading is ingested into \p DB, so the result is a \c CASBackedFile
/// whose \c getObjectRefForContent() refers to the stored contents.
LLVM_ABI std::unique_ptr<llvm::vfs::FileSystem> createCASProvidingFileSystem(
    std::shared_ptr<ObjectStore> DB,
    IntrusiveRefCntPtr<llvm::vfs::FileSystem> UnderlyingFS);

} // namespace cas
} // namespace llvm

#endif // LLVM_CAS_CASPROVIDINGFILESYSTEM_H
