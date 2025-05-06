//===- MappedFileRegionBumpPtr.cpp ------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
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
/// 1. Use \c ftruncate (\c sys::fs::resize_file) to grow the file to its max
///    size (typically several GB). Many modern filesystems will create a sparse
///    file, so that the trailing unused pages do not take space on disk.
/// 2. Call \c mmap (\c sys::fs::mapped_file_region)
/// 3. [Automatic as part of 2.]
/// 4. [Automatic as part of 2.]
///
/// Additionally, we attempt to resize the file to its actual data size when
/// closing the mapping, if this is the only concurrent instance. This is done
/// using file locks. Shrinking the file mitigates problems with having large
/// files: on filesystems without sparse files it avoids unnecessary space use;
/// it also avoids allocating the full size if another process copies the file,
/// which typically loses sparseness. These mitigations only work while the file
/// is not in use.
///
/// FIXME: we assume that all concurrent users of the file will use the same
/// value for Capacity. Otherwise a process with a larger capacity can write
/// data that is "out of bounds" for processes with smaller capacity. Currently
/// this is true in the CAS.
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
#include "llvm/CAS/OnDiskCASLogger.h"

using namespace llvm;
using namespace llvm::cas;
using namespace llvm::cas::ondisk;

namespace {
struct FileLockRAII {
  std::string Path;
  int FD;
  enum LockKind { Shared, Exclusive };
  std::optional<LockKind> Locked;

  FileLockRAII(std::string Path, int FD) : Path(std::move(Path)), FD(FD) {}
  ~FileLockRAII() { consumeError(unlock()); }

  Error lock(LockKind LK) {
    if (std::error_code EC = lockFileThreadSafe(FD, LK == Exclusive))
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
} // end anonymous namespace

Expected<MappedFileRegionBumpPtr> MappedFileRegionBumpPtr::create(
    const Twine &Path, uint64_t Capacity, int64_t BumpPtrOffset,
    std::shared_ptr<ondisk::OnDiskCASLogger> Logger,
    function_ref<Error(MappedFileRegionBumpPtr &)> NewFileConstructor) {
  MappedFileRegionBumpPtr Result;
  Result.Path = Path.str();
  Result.Logger = std::move(Logger);
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
          lockFileThreadSafe(SharedLockFD, /*Exclusive=*/false))
    return createFileError(Path, EC);

  // Take shared/reader lock for initialization.
  FileLockRAII InitLock(Result.Path, FD);
  if (Error E = InitLock.lock(FileLockRAII::Shared))
    return std::move(E);

  sys::fs::file_t File = sys::fs::convertFDToNativeFile(FD);
  sys::fs::file_status Status;
  if (std::error_code EC = sys::fs::status(File, Status))
    return createFileError(Result.Path, EC);

  if (Status.getSize() < Capacity) {
    // Lock the file exclusively so only one process will do the initialization.
    if (Error E = InitLock.unlock())
      return std::move(E);
    if (Error E = InitLock.lock(FileLockRAII::Exclusive))
      return std::move(E);
    // Retrieve the current size now that we have exclusive access.
    if (std::error_code EC = sys::fs::status(File, Status))
      return createFileError(Result.Path, EC);
  }

  // At this point either the file is still under-sized, or we have the size for
  // the completely initialized file.

  if (Status.getSize() < Capacity) {
    // We are initializing the file; it may be empty, or may have been shrunk
    // during a previous close.
    // FIXME: Detect a case where someone opened it with a smaller capacity.
    // FIXME: On Windows we should use FSCTL_SET_SPARSE and FSCTL_SET_ZERO_DATA
    // to make this a sparse region, if supported.
    if (std::error_code EC = sys::fs::resize_file(FD, Capacity))
      return createFileError(Result.Path, EC);

    if (Result.Logger)
      Result.Logger->log_MappedFileRegionBumpPtr_resizeFile(
          Result.Path, Status.getSize(), Capacity);
  } else {
    // Someone else initialized it.
    Capacity = Status.getSize();
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

  if (Status.getSize() == 0) {
    // We are creating a new file; run the constructor.
    if (Error E = NewFileConstructor(Result))
      return std::move(E);
  } else {
    Result.initializeBumpPtr(BumpPtrOffset);
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
  if (BumpPtr) {
    assert(SharedLockFD && "Must have shared lock file open");
    if (tryLockFileThreadSafe(*SharedLockFD) == std::error_code()) {
      size_t Size = size();
      size_t Capacity = capacity();
      assert(Size < Capacity);
      // sync to file system to make sure all contents are up-to-date.
      (void)Region.sync();
      (void)sys::fs::resize_file(*FD, size());
      (void)unlockFileThreadSafe(*SharedLockFD);

      if (Logger)
        Logger->log_MappedFileRegionBumpPtr_resizeFile(Path, Capacity, Size);
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

  if (Logger)
    Logger->log_MappedFileRegionBumpPtr_close(Path);
}

void MappedFileRegionBumpPtr::initializeBumpPtr(int64_t BumpPtrOffset) {
  assert(capacity() < (uint64_t)INT64_MAX && "capacity must fit in int64_t");
  int64_t BumpPtrEndOffset = BumpPtrOffset + sizeof(decltype(*BumpPtr));
  assert(BumpPtrEndOffset <= (int64_t)capacity() &&
         "Expected end offset to be pre-allocated");
  assert(isAligned(Align::Of<decltype(*BumpPtr)>(), BumpPtrOffset) &&
         "Expected end offset to be aligned");
  BumpPtr = reinterpret_cast<decltype(BumpPtr)>(data() + BumpPtrOffset);

  int64_t ExistingValue = 0;
  if (!BumpPtr->compare_exchange_strong(ExistingValue, BumpPtrEndOffset))
    assert(ExistingValue >= BumpPtrEndOffset &&
           "Expected 0, or past the end of the BumpPtr itself");

  if (Logger)
    Logger->log_MappedFileRegionBumpPtr_create(Path, *FD, data(), capacity(),
                                               size());
}

static Error createAllocatorOutOfSpaceError() {
  return createStringError(std::make_error_code(std::errc::not_enough_memory),
                           "memory mapped file allocator is out of space");
}

Expected<int64_t> MappedFileRegionBumpPtr::allocateOffset(uint64_t AllocSize) {
  AllocSize = alignTo(AllocSize, getAlign());
  int64_t OldEnd = BumpPtr->fetch_add(AllocSize);
  int64_t NewEnd = OldEnd + AllocSize;
  if (LLVM_UNLIKELY(NewEnd > (int64_t)capacity())) {
    // Return the allocation. If the start already passed the end, that means
    // some other concurrent allocations already consumed all the capacity.
    // There is no need to return the original value. If the start was not
    // passed the end, current allocation certainly bumped it passed the end.
    // All other allocation afterwards must have failed and current allocation
    // is in charge of return the allocation back to a valid value.
    if (OldEnd <= (int64_t)capacity())
      (void)BumpPtr->exchange(OldEnd);

    if (Logger)
      Logger->log_MappedFileRegionBumpPtr_oom(Path, capacity(), OldEnd,
                                              AllocSize);

    return createAllocatorOutOfSpaceError();
  }

  if (Logger)
    Logger->log_MappedFileRegionBumpPtr_allocate(data(), OldEnd, AllocSize);

  return OldEnd;
}
