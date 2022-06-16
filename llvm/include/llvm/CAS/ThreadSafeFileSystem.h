//===- llvm/CAS/ThreadSafeFileSystem.h ------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CAS_THREADSAFEFILESYSTEM_H
#define LLVM_CAS_THREADSAFEFILESYSTEM_H

#include "llvm/Support/Error.h"
#include "llvm/Support/VirtualFileSystem.h"

namespace llvm {
namespace cas {

class CASDB;
class CASID;

/// For thread-safe filesystem implementations.
class ThreadSafeFileSystem : public vfs::FileSystem {
  virtual void anchor();

public:
  /// Get a proxy FS that has an independent working directory.
  virtual IntrusiveRefCntPtr<ThreadSafeFileSystem>
  createThreadSafeProxyFS() = 0;
};

/// For filesystems that use a CAS.
class CASFileSystemBase : public ThreadSafeFileSystem {
  virtual void anchor() override;

public:
  /// Get a proxy FS that has an independent working directory.
  virtual CASDB &getCAS() const = 0;

  /// An extra API to pull out the \a CASID if \p Path refers to a file.
  virtual Optional<CASID> getFileCASID(const Twine &Path) = 0;

  bool isCASFS() const final { return true; }
};

} // namespace cas
} // namespace llvm

#endif // LLVM_CAS_THREADSAFEFILESYSTEM_H
