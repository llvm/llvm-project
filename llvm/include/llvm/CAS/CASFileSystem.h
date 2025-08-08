//===- llvm/CAS/CASFileSystem.h -------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CAS_CASFILESYSTEM_H
#define LLVM_CAS_CASFILESYSTEM_H

#include <llvm/CAS/ThreadSafeFileSystem.h>
#include <llvm/Support/Error.h>

namespace llvm {
namespace cas {
class ObjectStore;
class CASID;

Expected<std::unique_ptr<vfs::FileSystem>>
createCASFileSystem(std::shared_ptr<ObjectStore> DB, const CASID &RootID,
                    sys::path::Style PathStyle = sys::path::Style::native);

Expected<std::unique_ptr<vfs::FileSystem>>
createCASFileSystem(ObjectStore &DB, const CASID &RootID,
                    sys::path::Style PathStyle = sys::path::Style::native);

} // namespace cas
} // namespace llvm

#endif // LLVM_CAS_CASFILESYSTEM_H
