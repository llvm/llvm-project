//===- LazyMappedFileRegion.h -----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CAS_LAZYMAPPEDFILEREGION_H
#define LLVM_CAS_LAZYMAPPEDFILEREGION_H

#include "llvm/Config/llvm-config.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include <atomic>
#include <mutex>
#include <optional>

#if LLVM_ENABLE_ONDISK_CAS

namespace llvm {

class MemoryBuffer;

namespace cas {

/// Mapped file region with a lazy size. The mapped-in memory is larger than
/// the actual size on disk, and the on-disk file is resized when necessary.
///
/// Process-safe. Uses file locks to increase the capacity. Adds more capacity
/// than requested to avoid unnecessary contention on file locks.
///
/// Thread-safe, assuming all threads use the same instance to talk to a given
/// file/mapping. Unsafe to have multiple instances talking to the same file
/// in the same process since file locks will misbehave. Clients should
/// coordinate (somehow).
///
/// TODO: Implement for Windows. See comment next to implementation of \a
/// create() for a sketch of how to do this.
///
/// FIXME: Probably should move to Support.
class LazyMappedFileRegion {
public:
  /// Create a region.
  ///
  /// If there could be multiple instances pointing at the same underlying
  /// file this is not safe. Use \a createShared() instead.
  ///
  /// \p NewFileConstructor is for constructing new files. It has exclusive
  /// access to the file, and must extend the size before returning.
  static Expected<LazyMappedFileRegion>
  create(const Twine &Path, uint64_t Capacity,
         function_ref<Error(LazyMappedFileRegion &)> NewFileConstructor,
         uint64_t MaxSizeIncrement = 4ULL * 1024ULL * 1024ULL);

  /// Create a region, shared across the process via a singleton map.
  ///
  /// FIXME: Singleton map should be based on sys::fs::UniqueID, but currently
  /// it is just based on \p Path.
  ///
  /// \p NewFileConstructor is for constructing new files. It has exclusive
  /// access to the file, and must extend the size before returning.
  static Expected<std::shared_ptr<LazyMappedFileRegion>>
  createShared(const Twine &Path, uint64_t Capacity,
               function_ref<Error(LazyMappedFileRegion &)> NewFileConstructor,
               uint64_t MaxSizeIncrement = 4ULL * 1024ULL * 1024ULL);

  /// Get the path this was opened with.
  ///
  /// FIXME: While this is useful for debugging, might be better to remove
  /// and/or update callers not to rely on this.
  StringRef getPath() const { return Path; }

  char *data() const { return Map.data(); }

  /// Resize to at least \p MinSize.
  ///
  /// Errors if \p MinSize is bigger than \a capacity() or if the operation
  /// fails.
  Error extendSize(uint64_t MinSize) {
    // Common case.
    if (MinSize <= size())
      return Error::success();
    return extendSizeImpl(MinSize);
  }

  /// Size allocated on disk.
  size_t size() const { return CachedSize; }

  /// Size of the underlying \a mapped_file_region. This cannot be extended.
  size_t capacity() const { return Map.size(); }

  explicit operator bool() const { return bool(Map); }

  ~LazyMappedFileRegion() { destroyImpl(); }

  LazyMappedFileRegion() : CachedSize(0) {}
  LazyMappedFileRegion(LazyMappedFileRegion &&RHS) { moveImpl(RHS); }
  LazyMappedFileRegion &operator=(LazyMappedFileRegion &&RHS) {
    destroyImpl();
    moveImpl(RHS);
    return *this;
  }

  LazyMappedFileRegion(const LazyMappedFileRegion &) = delete;
  LazyMappedFileRegion &operator=(const LazyMappedFileRegion &) = delete;

private:
  Error extendSizeImpl(uint64_t MinSize);
  void destroyImpl() {
    if (FD) {
      sys::fs::closeFile(*FD);
      FD = std::nullopt;
    }
  }
  void moveImpl(LazyMappedFileRegion &RHS) {
    Path = std::move(RHS.Path);
    FD = std::move(RHS.FD);
    RHS.FD = std::nullopt;
    Map = std::move(RHS.Map);
    CachedSize = RHS.CachedSize.load();
    RHS.CachedSize = 0;
    MaxSizeIncrement = RHS.MaxSizeIncrement;

    assert(!RHS.Map && "Expected Optional(Optional&&) to clear RHS.Map");
  }

  std::string Path;
  std::optional<sys::fs::file_t> FD;
  sys::fs::mapped_file_region Map;
  std::atomic<uint64_t> CachedSize;
  std::mutex Mutex;
  uint64_t MaxSizeIncrement = 0;

  /// Set to \c true only for the duration of calls to \c NewFileConstructor in
  /// \a create() and \a createShared(). This allows \a extendSize() to be used
  /// without locking \a Mutex and without (erroneous) new file locks.
  ///
  /// FIXME: Steal a bit from MaxSizeIncrement for this.
  bool IsConstructingNewFile = false;
};

} // namespace cas
} // namespace llvm

#endif // LLVM_ENABLE_ONDISK_CAS
#endif // LLVM_CAS_LAZYMAPPEDFILEREGION_H
