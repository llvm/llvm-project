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
/// 1. Use \ref sys::fs::resize_file_sparse to grow the file to its max size
///    (typically several GB). If the file system doesn't support sparse file,
///    this may return a fully allocated file.
/// 2. Call \ref sys::fs::mapped_file_region to map the entire file.
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
/// The capacity and the header offset is determined by the first user of the
/// MappedFileRegionBumpPtr instance and any future mismatched value from the
/// original will result in error on creation.
///
/// To support resizing, we use two separate file locks:
/// 1. We use a shared reader lock on a ".shared" file until destruction. The
///    shared file also contains the configuration for the main file, including
///    the offset to the header and the capacity.
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
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/EndianStream.h"

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
    assert(!Locked && "already locked");
    if (std::error_code EC = lockFileThreadSafe(FD, LK))
      return createFileError(Path, EC);
    Locked = LK;
    return Error::success();
  }

  Error switchLock(sys::fs::LockKind LK) {
    assert(Locked && "not locked");
    if (auto E = unlock())
      return E;

    return lock(LK);
  }

  Error unlock() {
    if (Locked) {
      Locked = std::nullopt;
      if (std::error_code EC = unlockFileThreadSafe(FD))
        return createFileError(Path, EC);
    }
    return Error::success();
  }

  // Return true if succeed to lock the file exclusively.
  bool tryLockExclusive() {
    assert(!Locked && "can only try to lock if not locked");
    if (tryLockFileThreadSafe(FD) == std::error_code()) {
      Locked = sys::fs::LockKind::Exclusive;
      return true;
    }

    return false;
  }

  // Release the lock so it will not be unlocked on destruction.
  void release() {
    Locked = std::nullopt;
    FD = -1;
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
  uint64_t MinCapacity = HeaderOffset + sizeof(Header);
  if (Capacity < MinCapacity)
    return createStringError(
        std::make_error_code(std::errc::invalid_argument),
        "capacity is too small to hold MappedFileRegionBumpPtr");

  MappedFileRegionBumpPtr Result;
  Result.Path = Path.str();
  // Open the main file.
  int FD;
  if (std::error_code EC = sys::fs::openFileForReadWrite(
          Result.Path, FD, sys::fs::CD_OpenAlways, sys::fs::OF_None))
    return createFileError(Path, EC);
  Result.FD = FD;

  // Open the support file. See file comment for details of locking scheme.
  SmallString<128> SupportFilePath(Result.Path);
  SupportFilePath.append(".shared");
  int SupportFD;
  if (std::error_code EC = sys::fs::openFileForReadWrite(
          SupportFilePath, SupportFD, sys::fs::CD_OpenAlways, sys::fs::OF_None))
    return createFileError(SupportFilePath, EC);
  Result.SupportFD = SupportFD;

  sys::fs::file_status SupportFileStatus;
  bool FreshConfig = false;
  while (true) {
    if (auto EC = sys::fs::status(SupportFD, SupportFileStatus))
      return createFileError(SupportFilePath, EC);
    if (SupportFileStatus.getSize() != 0)
      break;

    // Try exclusive lock and initialize the file.
    FileLockRAII SupportFileLock(std::string(SupportFilePath), SupportFD);
    if (SupportFileLock.tryLockExclusive()) {
      // Check size again and exit if it is initialized.
      if (auto EC = sys::fs::status(SupportFD, SupportFileStatus))
        return createFileError(SupportFilePath, EC);
      if (SupportFileStatus.getSize() != 0)
        break;

      // Write capacity and header offset to file.
      llvm::raw_fd_ostream OS(SupportFD, /*shouldClose=*/false,
                              /*unbuffered=*/true);
      support::endian::write<uint64_t>(OS, Capacity, endianness::little);
      support::endian::write<uint64_t>(OS, HeaderOffset, endianness::little);

      FreshConfig = true;
      break;
    }
  }

  // Take shared/reader lock that will be held until we close the file;
  // unlocked by destroyImpl if construction is successful.
  FileLockRAII SupportFileLock(std::string(SupportFilePath), SupportFD);
  if (auto E = SupportFileLock.lock(sys::fs::LockKind::Shared))
    return std::move(E);

  if (!FreshConfig) {
    // Read the configuration and error out if the configuration is different.
    SmallVector<char, 16> ConfigBuffer;
    sys::fs::file_t SupportFile = sys::fs::convertFDToNativeFile(SupportFD);
    if (auto E = sys::fs::readNativeFileToEOF(SupportFile, ConfigBuffer))
      return std::move(E);

    if (ConfigBuffer.size() != 2 * sizeof(uint64_t))
      return createStringError(
          std::make_error_code(std::errc::invalid_argument),
          SupportFilePath + " does not have correct size");

    uint64_t ExpectedCapacity =
        support::endian::read<uint64_t, endianness::little>(
            ConfigBuffer.data());
    uint64_t ExpectedOffset =
        support::endian::read<uint64_t, endianness::little>(
            ConfigBuffer.data() + sizeof(uint64_t));

    // If the capacity doesn't match, use the existing capacity instead.
    if (ExpectedCapacity != Capacity)
      Capacity = ExpectedCapacity;

    if (ExpectedOffset != HeaderOffset)
      return createStringError(
          std::make_error_code(std::errc::invalid_argument),
          "specified header offset (" + utostr(HeaderOffset) +
              ") does not match existing config (" + utostr(ExpectedOffset) +
              ")");
  }

  // Take shared/reader lock for initialization.
  FileLockRAII InitLock(Result.Path, FD);
  if (Error E = InitLock.lock(sys::fs::LockKind::Shared))
    return std::move(E);

  sys::fs::file_t File = sys::fs::convertFDToNativeFile(FD);
  auto FileSize = FileSizeInfo::get(File);
  if (!FileSize)
    return createFileError(Result.Path, FileSize.getError());

  // If the size is smaller than the capacity, we need to initialize the file.
  // It maybe empty, or may have been shrunk during a previous close.
  if (FileSize->Size < Capacity) {
    // Lock the file exclusively so only one process will do the initialization.
    if (Error E = InitLock.switchLock(sys::fs::LockKind::Exclusive))
      return std::move(E);
    // Retrieve the current size now that we have exclusive access.
    FileSize = FileSizeInfo::get(File);
    if (!FileSize)
      return createFileError(Result.Path, FileSize.getError());
  }

  // If the size is still smaller than capacity, we need to resize the file.
  if (FileSize->Size < Capacity) {
    assert(InitLock.Locked == sys::fs::LockKind::Exclusive);
    if (std::error_code EC = sys::fs::resize_file_sparse(FD, Capacity))
      return createFileError(Result.Path, EC);
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

  if (FileSize->Size < MinCapacity) {
    assert(InitLock.Locked == sys::fs::LockKind::Exclusive);
    // If we need to fully initialize the file, call NewFileConstructor.
    if (Error E = NewFileConstructor(Result))
      return std::move(E);
  } else
    Result.initializeHeader(HeaderOffset);

  if (InitLock.Locked == sys::fs::LockKind::Exclusive) {
    // If holding an exclusive lock, we might have resized the file and
    // performed some read/write to the file. Query the file size again to make
    // sure everything is up-to-date. Otherwise, FileSize info is already
    // up-to-date.
    FileSize = FileSizeInfo::get(File);
    if (!FileSize)
      return createFileError(Result.Path, FileSize.getError());
  }

  Result.H->AllocatedSize.exchange(FileSize->AllocatedSize);

  // Release the shared lock from RAII so it can be closed in destoryImpl().
  SupportFileLock.release();

  return Result;
}

void MappedFileRegionBumpPtr::destroyImpl() {
  if (!FD)
    return;

  // Drop the shared lock indicating we are no longer accessing the file.
  if (SupportFD)
    (void)unlockFileThreadSafe(*SupportFD);

  // Attempt to truncate the file if we can get exclusive access. Ignore any
  // errors.
  if (H) {
    assert(SupportFD && "Must have shared lock file open");
    if (tryLockFileThreadSafe(*SupportFD) == std::error_code()) {
      size_t Size = size();
      // sync to file system to make sure all contents are up-to-date.
      (void)Region.sync();
      // unmap the file before resizing since that is the requirement for
      // some platforms.
      Region.unmap();
      (void)sys::fs::resize_file(*FD, Size);
      (void)unlockFileThreadSafe(*SupportFD);
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
  Close(SupportFD);
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
           "Expected 0, or past the end of the header itself");
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
