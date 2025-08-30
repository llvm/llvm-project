//===- MappedFileRegionBumpPtr.h --------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// This file declares interface for MappedFileRegionBumpPtr, a bump pointer
/// allocator, backed by a memory-mapped file.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_CAS_MAPPEDFILEREGIONBUMPPTR_H
#define LLVM_CAS_MAPPEDFILEREGIONBUMPPTR_H

#include "llvm/Support/Alignment.h"
#include "llvm/Support/FileSystem.h"
#include <atomic>

namespace llvm::cas {

/// Allocator for an owned mapped file region that supports thread-safe and
/// process-safe bump pointer allocation.
///
/// This allocator is designed to create a sparse file when supported by the
/// filesystem's \c ftruncate so that it can be used with a large maximum size.
/// It will also attempt to shrink the underlying file down to its current
/// allocation size when the last concurrent mapping is closed.
///
/// Process-safe. Uses file locks when resizing the file during initialization
/// and destruction.
///
/// Thread-safe, assuming all threads use the same instance to talk to a given
/// file/mapping. Unsafe to have multiple instances talking to the same file
/// in the same process since file locks will misbehave. Clients should
/// coordinate (somehow).
///
/// Provides 8-byte alignment for all allocations.
class MappedFileRegionBumpPtr {
public:
  using RegionT = sys::fs::mapped_file_region;

  /// Header for MappedFileRegionBumpPtr. It can be configured to be located
  /// at any location within the file and the allocation will be appended after
  /// the header.
  struct Header {
    std::atomic<uint64_t> BumpPtr;
    std::atomic<uint64_t> AllocatedSize;
  };

  /// Create a \c MappedFileRegionBumpPtr.
  ///
  /// \param Path the path to open the mapped region.
  /// \param Capacity the maximum size for the mapped file region.
  /// \param HeaderOffset the offset at which to store the header. This is so
  /// that information can be stored before the header, like a file magic.
  /// \param NewFileConstructor is for constructing new files. It has exclusive
  /// access to the file. Must call \c initializeBumpPtr.
  static Expected<MappedFileRegionBumpPtr>
  create(const Twine &Path, uint64_t Capacity, uint64_t HeaderOffset,
         function_ref<Error(MappedFileRegionBumpPtr &)> NewFileConstructor);

  /// Finish initializing the header. Must be called by \c NewFileConstructor.
  void initializeHeader(uint64_t HeaderOffset);

  /// Minimum alignment for allocations, currently hardcoded to 8B.
  static constexpr Align getAlign() {
    // Trick Align into giving us '8' as a constexpr.
    struct alignas(8) T {};
    static_assert(alignof(T) == 8, "Tautology failed?");
    return Align::Of<T>();
  }

  /// Allocate at least \p AllocSize. Rounds up to \a getAlign().
  Expected<char *> allocate(uint64_t AllocSize) {
    auto Offset = allocateOffset(AllocSize);
    if (LLVM_UNLIKELY(!Offset))
      return Offset.takeError();
    return data() + *Offset;
  }
  /// Allocate, returning the offset from \a data() instead of a pointer.
  Expected<int64_t> allocateOffset(uint64_t AllocSize);

  char *data() const { return Region.data(); }
  uint64_t size() const { return H->BumpPtr; }
  uint64_t capacity() const { return Region.size(); }

  RegionT &getRegion() { return Region; }

  ~MappedFileRegionBumpPtr() { destroyImpl(); }

  MappedFileRegionBumpPtr() = default;
  MappedFileRegionBumpPtr(MappedFileRegionBumpPtr &&RHS) { moveImpl(RHS); }
  MappedFileRegionBumpPtr &operator=(MappedFileRegionBumpPtr &&RHS) {
    destroyImpl();
    moveImpl(RHS);
    return *this;
  }

  MappedFileRegionBumpPtr(const MappedFileRegionBumpPtr &) = delete;
  MappedFileRegionBumpPtr &operator=(const MappedFileRegionBumpPtr &) = delete;

private:
  void destroyImpl();
  void moveImpl(MappedFileRegionBumpPtr &RHS) {
    std::swap(Region, RHS.Region);
    std::swap(H, RHS.H);
    std::swap(Path, RHS.Path);
    std::swap(FD, RHS.FD);
    std::swap(SharedLockFD, RHS.SharedLockFD);
  }

private:
  RegionT Region;
  Header *H = nullptr;
  std::string Path;
  // File descriptor for the main storage file.
  std::optional<int> FD;
  // File descriptor for the file used as reader/writer lock.
  std::optional<int> SharedLockFD;
};

} // namespace llvm::cas

#endif // LLVM_CAS_MAPPEDFILEREGIONBUMPPTR_H
