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

class ObjectStore;
class CASID;

/// For thread-safe filesystem implementations.
class ThreadSafeFileSystem
    : public llvm::RTTIExtends<ThreadSafeFileSystem, vfs::FileSystem> {
  virtual void anchor() override;

public:
  static const char ID;

  /// Get a proxy FS that has an independent working directory.
  virtual IntrusiveRefCntPtr<ThreadSafeFileSystem>
  createThreadSafeProxyFS() = 0;
};

} // namespace cas
} // namespace llvm

#endif // LLVM_CAS_THREADSAFEFILESYSTEM_H
