//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file Implements MappedFileRegionBumpPtr.
///
/// A bump pointer allocator, backed by a memory-mapped file.
///
/// The effect we want is:
///
/// 1. If it doesn't exist, create the file with an initial size.
/// 2. Reserve virtual memory large enough for the max file size.
/// 3. Map the file into memory in the reserved region.
/// 4. Increase the file size and update the mapping when necessary.
///
/// However, updating the mapping is challenging when it needs to work portably,
/// and across multiple processes without locking for every read. Our current
/// implementation strategy is:
///
/// 1. Use \c sys::fs::resize_file_sparse to grow the file to its max size
///    (typically several GB). If the file system doesn't support sparse file,
///    this may return a fully allocated file.
/// 2. Call \c sys::fs::mapped_file_region to map the entire file.
/// 3. [Automatic as part of 2.]
/// 4. If supported, use \c fallocate or similiar APIs to ensure the file system
///    storage for the sparse file so we won't end up with partial file if the
///    disk is out of space.
///
/// Additionally, we attempt to resize the file to its actual data size when
/// closing the mapping, if this is the only concurrent instance. This is done
/// using file locks. Shrinking the file mitigates problems with having large
/// files: on filesystems without sparse files it avoids unnecessary space use;
/// it also avoids allocating the full size if another process copies the file,
/// which typically loses sparseness. These mitigations only work while the file
/// is not in use.
///
/// If different values of the capacity is used for concurrent users of the same
/// mapping, the actual capacity will be the largest value requested at the time
/// of the creation. As a result, the mapped region in one process can be
/// smaller than the size of the file on disk and can run out of reserved space
/// when the file has still space. It is highly recommanded to use the same
/// capacity for all the concurrent users of the same instance of
/// MappedFileRegionBumpPtr.
///
/// To support resizing, we use two separate file locks:
/// 1. We use a shared reader lock on a ".shared" file until destruction.
/// 2. We use a lock on the main file during initialization - shared to check
///    the status, upgraded to exclusive to resize/initialize the file.
///
/// Then during destruction we attempt to get exclusive access on (1), which
/// requires no concurrent readers. If so, we shrink the file. Using two
/// separate locks simplifies the implementation and enables it to work on
/// platforms (e.g. Windows) where a shared/reader lock prevents writing.
//===----------------------------------------------------------------------===//

#include "llvm/CAS/MappedFileRegionBumpPtr.h"
#include "OnDiskCommon.h"

#if LLVM_ON_UNIX
#include <sys/stat.h>
#if __has_include(<sys/param.h>)
#include <sys/param.h>
#endif
#ifdef DEV_BSIZE
#define MAPPED_FILE_BSIZE DEV_BSIZE
#elif __linux__
#define MAPPED_FILE_BSIZE 512
#endif
#endif

using namespace llvm;
using namespace llvm::cas;
using namespace llvm::cas::ondisk;

namespace {
struct FileLockRAII {
  std::string Path;
  int FD;
  std::optional<sys::fs::LockKind> Locked;

  FileLockRAII(std::string Path, int FD) : Path(std::move(Path)), FD(FD) {}
  ~FileLockRAII() { consumeError(unlock()); }

  Error lock(sys::fs::LockKind LK) {
    if (std::error_code EC = lockFileThreadSafe(FD, LK))
      return createFileError(Path, EC);
    Locked = LK;
    return Error::success();
  }

  Error unlock() {
    if (Locked) {
      Locked = std::nullopt;
      if (std::error_code EC = unlockFileThreadSafe(FD))
        return createFileError(Path, EC);
    }
    return Error::success();
  }
};

struct FileSizeInfo {
  uint64_t Size;
  uint64_t AllocatedSize;

  static ErrorOr<FileSizeInfo> get(sys::fs::file_t File);
};
} // end anonymous namespace

Expected<MappedFileRegionBumpPtr> MappedFileRegionBumpPtr::create(
    const Twine &Path, uint64_t Capacity, uint64_t HeaderOffset,
    function_ref<Error(MappedFileRegionBumpPtr &)> NewFileConstructor) {
  MappedFileRegionBumpPtr Result;
  Result.Path = Path.str();
  // Open the main file.
  int FD;
  if (std::error_code EC = sys::fs::openFileForReadWrite(
          Result.Path, FD, sys::fs::CD_OpenAlways, sys::fs::OF_None))
    return createFileError(Path, EC);
  Result.FD = FD;

  // Open the shared lock file. See file comment for details of locking scheme.
  SmallString<128> SharedLockPath(Result.Path);
  SharedLockPath.append(".shared");
  int SharedLockFD;
  if (std::error_code EC = sys::fs::openFileForReadWrite(
          SharedLockPath, SharedLockFD, sys::fs::CD_OpenAlways,
          sys::fs::OF_None))
    return createFileError(SharedLockPath, EC);
  Result.SharedLockFD = SharedLockFD;

  // Take shared/reader lock that will be held until we close the file; unlocked
  // by destroyImpl.
  if (std::error_code EC =
          lockFileThreadSafe(SharedLockFD, sys::fs::LockKind::Shared))
    return createFileError(Path, EC);

  // Take shared/reader lock for initialization.
  FileLockRAII InitLock(Result.Path, FD);
  if (Error E = InitLock.lock(sys::fs::LockKind::Shared))
    return std::move(E);

  sys::fs::file_t File = sys::fs::convertFDToNativeFile(FD);
  auto FileSize = FileSizeInfo::get(File);
  if (!FileSize)
    return createFileError(Result.Path, FileSize.getError());

  if (FileSize->Size < Capacity) {
    // Lock the file exclusively so only one process will do the initialization.
    if (Error E = InitLock.unlock())
      return std::move(E);
    if (Error E = InitLock.lock(sys::fs::LockKind::Exclusive))
      return std::move(E);
    // Retrieve the current size now that we have exclusive access.
    FileSize = FileSizeInfo::get(File);
    if (!FileSize)
      return createFileError(Result.Path, FileSize.getError());
  }

  // At this point either the file is still under-sized, or we have the size for
  // the completely initialized file.

  if (FileSize->Size < Capacity) {
    // We are initializing the file; it may be empty, or may have been shrunk
    // during a previous close.
    // TODO: Detect a case where someone opened it with a smaller capacity.
    assert(InitLock.Locked == sys::fs::LockKind::Exclusive);
    if (std::error_code EC = sys::fs::resize_file_sparse(FD, Capacity))
      return createFileError(Result.Path, EC);
  } else {
    // Someone else initialized it.
    Capacity = FileSize->Size;
  }

  // Create the mapped region.
  {
    std::error_code EC;
    sys::fs::mapped_file_region Map(
        File, sys::fs::mapped_file_region::readwrite, Capacity, 0, EC);
    if (EC)
      return createFileError(Result.Path, EC);
    Result.Region = std::move(Map);
  }

  if (FileSize->Size == 0) {
    assert(InitLock.Locked == sys::fs::LockKind::Exclusive);
    // We are creating a new file; run the constructor.
    if (Error E = NewFileConstructor(Result))
      return std::move(E);
  } else {
    Result.initializeHeader(HeaderOffset);
  }

  if (FileSize->Size < Capacity && FileSize->AllocatedSize < Capacity) {
    // We are initializing the file; sync the allocated size in case it
    // changed when truncating or during construction.
    FileSize = FileSizeInfo::get(File);
    if (!FileSize)
      return createFileError(Result.Path, FileSize.getError());
    assert(InitLock.Locked == sys::fs::LockKind::Exclusive);
    Result.H->AllocatedSize.exchange(FileSize->AllocatedSize);
  }

  return Result;
}

void MappedFileRegionBumpPtr::destroyImpl() {
  if (!FD)
    return;

  // Drop the shared lock indicating we are no longer accessing the file.
  if (SharedLockFD)
    (void)unlockFileThreadSafe(*SharedLockFD);

  // Attempt to truncate the file if we can get exclusive access. Ignore any
  // errors.
  if (H) {
    assert(SharedLockFD && "Must have shared lock file open");
    if (tryLockFileThreadSafe(*SharedLockFD) == std::error_code()) {
      size_t Size = size();
      size_t Capacity = capacity();
      assert(Size < Capacity);
      // sync to file system to make sure all contents are up-to-date.
      (void)Region.sync();
      // unmap the file before resizing since that is the requirement for
      // some platforms.
      Region.unmap();
      (void)sys::fs::resize_file(*FD, Size);
      (void)unlockFileThreadSafe(*SharedLockFD);
    }
  }

  auto Close = [](std::optional<int> &FD) {
    if (FD) {
      sys::fs::file_t File = sys::fs::convertFDToNativeFile(*FD);
      sys::fs::closeFile(File);
      FD = std::nullopt;
    }
  };

  // Close the file and shared lock.
  Close(FD);
  Close(SharedLockFD);
}

void MappedFileRegionBumpPtr::initializeHeader(uint64_t HeaderOffset) {
  assert(capacity() < (uint64_t)INT64_MAX && "capacity must fit in int64_t");
  uint64_t HeaderEndOffset = HeaderOffset + sizeof(decltype(*H));
  assert(HeaderEndOffset <= capacity() &&
         "Expected end offset to be pre-allocated");
  assert(isAligned(Align::Of<decltype(*H)>(), HeaderOffset) &&
         "Expected end offset to be aligned");
  H = reinterpret_cast<decltype(H)>(data() + HeaderOffset);

  uint64_t ExistingValue = 0;
  if (!H->BumpPtr.compare_exchange_strong(ExistingValue, HeaderEndOffset))
    assert(ExistingValue >= HeaderEndOffset &&
           "Expected 0, or past the end of the BumpPtr itself");
}

static Error createAllocatorOutOfSpaceError() {
  return createStringError(std::make_error_code(std::errc::not_enough_memory),
                           "memory mapped file allocator is out of space");
}

Expected<int64_t> MappedFileRegionBumpPtr::allocateOffset(uint64_t AllocSize) {
  AllocSize = alignTo(AllocSize, getAlign());
  uint64_t OldEnd = H->BumpPtr.fetch_add(AllocSize);
  uint64_t NewEnd = OldEnd + AllocSize;
  if (LLVM_UNLIKELY(NewEnd > capacity())) {
    // Return the allocation. If the start already passed the end, that means
    // some other concurrent allocations already consumed all the capacity.
    // There is no need to return the original value. If the start was not
    // passed the end, current allocation certainly bumped it passed the end.
    // All other allocation afterwards must have failed and current allocation
    // is in charge of return the allocation back to a valid value.
    if (OldEnd <= capacity())
      (void)H->BumpPtr.exchange(OldEnd);

    return createAllocatorOutOfSpaceError();
  }

  uint64_t DiskSize = H->AllocatedSize;
  if (LLVM_UNLIKELY(NewEnd > DiskSize)) {
    uint64_t NewSize;
    // The minimum increment is a page, but allocate more to amortize the cost.
    constexpr uint64_t Increment = 1 * 1024 * 1024; // 1 MB
    if (Error E = preallocateFileTail(*FD, DiskSize, DiskSize + Increment)
                      .moveInto(NewSize))
      return std::move(E);
    assert(NewSize >= DiskSize + Increment);
    // FIXME: on Darwin this can under-count the size if there is a race to
    // preallocate disk, because the semantics of F_PREALLOCATE are to add bytes
    // to the end of the file, not to allocate up to a fixed size.
    // Any discrepancy will be resolved the next time the file is truncated and
    // then reopend.
    while (DiskSize < NewSize)
      H->AllocatedSize.compare_exchange_strong(DiskSize, NewSize);
  }
  return OldEnd;
}

ErrorOr<FileSizeInfo> FileSizeInfo::get(sys::fs::file_t File) {
#if LLVM_ON_UNIX && defined(MAPPED_FILE_BSIZE)
  struct stat Status;
  int StatRet = ::fstat(File, &Status);
  if (StatRet)
    return errnoAsErrorCode();
  uint64_t AllocatedSize = uint64_t(Status.st_blksize) * MAPPED_FILE_BSIZE;
  return FileSizeInfo{uint64_t(Status.st_size), AllocatedSize};
#else
  // Fallback: assume the file is fully allocated. Note: this may result in
  // data loss on out-of-space.
  sys::fs::file_status Status;
  if (std::error_code EC = sys::fs::status(File, Status))
    return EC;
  return FileSizeInfo{Status.getSize(), Status.getSize()};
#endif
}
