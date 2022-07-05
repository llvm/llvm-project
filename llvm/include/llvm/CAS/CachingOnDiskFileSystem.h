//===- llvm/CAS/CachingOnDiskFileSystem.h ---------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CAS_CACHINGONDISKFILESYSTEM_H
#define LLVM_CAS_CACHINGONDISKFILESYSTEM_H

#include "llvm/CAS/FileSystemCache.h"
#include "llvm/CAS/ThreadSafeFileSystem.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/VirtualFileSystem.h"
#include <map>

namespace llvm {
namespace cas {

class ObjectProxy;

/// A mostly-thread-safe caching on-disk filesystem. The only unsafe operation
/// is \a setCurrentWorkingDirectory(), which would be dangerous to call in a
/// multi-threaded context anyway.
///
/// \a createProxyFS() will return a proxy filesystem with an independent
/// working directory. This allows a single caching on-disk filesystem to be
/// used across the filesystem, with each thread using a different proxy to set
/// the working directory.
class CachingOnDiskFileSystem : public CASFileSystemBase {
  void anchor() override;

public:
  /// An extra API to pull out the \a CASID if \p Path refers to a file.
  virtual ErrorOr<vfs::Status> statusAndFileID(const Twine &Path,
                                               Optional<CASID> &FileID) = 0;

  /// Start tracking all stats (and other accesses). Only affects this
  /// filesystem instance, not current (or future) proxies.
  virtual void trackNewAccesses() = 0;

  /// Create a tree that represents all stats tracked since the call to \a
  /// trackNewAccesses(). Stops tracking new accesses.
  ///
  /// If provided, \p RemapPath is used to adjust paths in the created CAS
  /// tree.
  ///
  /// FIXME: Targets of symbolic links are not currently remapped. For example,
  /// given:
  ///
  ///     /sym1 -> old
  ///     /old/filename
  ///     /old/sym2 -> filename
  ///     /old/sym3 -> /old/filename
  ///
  /// If \p RemapPath uses a mapping prefix "/old=/new", then the resulting
  /// tree will be:
  ///
  ///     /sym1 -> old               [broken]
  ///     /new/filename
  ///     /new/sym2 -> filename
  ///     /new/sym3 -> /old/filename [broken]
  ///
  /// ... but the correct result would keep /sym1 and /old/sym3 working:
  ///
  ///     /sym1 -> new               [broken]
  ///     /new/filename
  ///     /new/sym2 -> filename
  ///     /new/sym3 -> /new/filename [broken]
  virtual Expected<ObjectProxy> createTreeFromNewAccesses(
      llvm::function_ref<StringRef(const vfs::CachedDirectoryEntry &)>
          RemapPath = nullptr) = 0;

  /// Create a tree that represents all known directories, files, and symlinks.
  virtual Expected<ObjectProxy> createTreeFromAllAccesses() = 0;

  /// Helper class to build a tree with a subset of what has been read.
  class TreeBuilder {
  public:
    /// Add \p Path to hierarchical tree-in-progress.
    ///
    /// If \p Path resolves to a symlink, its target is implicitly pushed as
    /// well.
    ///
    /// If \p Path resolves to a directory, the recursive directory contents
    /// will be pushed, implicitly pushing the targets of any contained
    /// symlinks.
    ///
    /// If \p Path does not exist, an error will be returned. If \p Path's
    /// parent path exists but the filename refers to a broken symlink, that is
    /// not an error; the symlink will be added without the target.
    virtual Error push(const Twine &Path) = 0;
    virtual Expected<ObjectProxy> create() = 0;
    virtual ~TreeBuilder() = default;
  };

  /// Get a builder instance for creating a tree containing a subset of the
  /// cached filesystem contents.
  virtual std::unique_ptr<TreeBuilder> createTreeBuilder() = 0;

  CASDB &getCAS() const final { return DB; }

  /// Get a proxy FS that has an independent working directory but uses the
  /// same thread-safe cache.
  virtual IntrusiveRefCntPtr<CachingOnDiskFileSystem> createProxyFS() = 0;

  IntrusiveRefCntPtr<ThreadSafeFileSystem> createThreadSafeProxyFS() final {
    return createProxyFS();
  }

protected:
  CachingOnDiskFileSystem(std::shared_ptr<CASDB> DB);
  CachingOnDiskFileSystem(CASDB &DB);
  CachingOnDiskFileSystem(const CachingOnDiskFileSystem &) = default;

  CASDB &DB;
  std::shared_ptr<CASDB> OwnedDB;
};

Expected<IntrusiveRefCntPtr<CachingOnDiskFileSystem>>
createCachingOnDiskFileSystem(std::shared_ptr<CASDB> DB);

Expected<IntrusiveRefCntPtr<CachingOnDiskFileSystem>>
createCachingOnDiskFileSystem(CASDB &DB);

} // namespace cas
} // namespace llvm

#endif // LLVM_CAS_CACHINGONDISKFILESYSTEM_H
