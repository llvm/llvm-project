//===- LazyMappedFileRegion.cpp -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/CAS/LazyMappedFileRegion.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"
#include <optional>

#if LLVM_ENABLE_ONDISK_CAS

using namespace llvm;
using namespace llvm::cas;

/// Open / resize / map in a file on-disk.
///
/// FIXME: This isn't portable. Windows will resize the file to match the map
/// size, which means immediately creating a very large file. Instead, maybe
/// we can increase the mapped region size on windows after creation, or
/// default to a more reasonable size.
///
/// The effect we want is:
///
/// 1. Reserve virtual memory large enough for the max file size (1GB).
/// 2. If it doesn't exist, give the file an initial smaller size (1MB).
/// 3. Map the file into memory.
/// 4. Assign the file to the reserved virtual memory.
/// 5. Increase the file size and update the mapping when necessary.
///
/// Here's the current implementation for Unix:
///
/// 1. [Automatic as part of 3.]
/// 2. Call ::ftruncate to 1MB (sys::fs::resize_file).
/// 3. Call ::mmap with 1GB (sys::fs::mapped_file_region).
/// 4. [Automatic as part of 3.]
/// 5. Call ::ftruncate with the new size.
///
/// On Windows, I *think* this can be implemented with:
///
/// 1. Call VirtualAlloc2 to reserve 1GB of virtual memory.
/// 2. [Automatic as part of 3.]
/// 3. Call CreateFileMapping to with 1MB, or existing size.
/// 4. Call MapViewOfFileN to place it in the reserved memory.
/// 5. Repeat step (3) with the new size and step (4).
Expected<LazyMappedFileRegion> LazyMappedFileRegion::create(
    const Twine &Path, uint64_t Capacity,
    function_ref<Error(LazyMappedFileRegion &)> NewFileConstructor,
    uint64_t MaxSizeIncrement) {
  LazyMappedFileRegion LMFR;
  LMFR.Path = Path.str();
  LMFR.MaxSizeIncrement = MaxSizeIncrement;

  if (Error E = sys::fs::openNativeFileForReadWrite(
          LMFR.Path, sys::fs::CD_OpenAlways, sys::fs::OF_None).moveInto(LMFR.FD))
    return std::move(E);
  assert(LMFR.FD && "Expected valid file descriptor");

  {
    std::error_code EC;
    sys::fs::mapped_file_region
        Map(*LMFR.FD, sys::fs::mapped_file_region::readwrite, Capacity, 0, EC);
    if (EC)
      return createFileError(LMFR.Path, EC);
    LMFR.Map = std::move(Map);
  }

  // Lock the file so we can initialize it.
  if (std::error_code EC = sys::fs::lockFile(*LMFR.FD))
    return createFileError(Path, EC);
  auto Unlock = make_scope_exit([FD = *LMFR.FD]() { sys::fs::unlockFile(FD); });

  sys::fs::file_status Status;
  if (std::error_code EC = sys::fs::status(*LMFR.FD, Status))
    return errorCodeToError(EC);
  if (Status.getSize() > 0) {
    // The file was already constructed.
    LMFR.CachedSize = Status.getSize();
    return std::move(LMFR);
  }

  // This is a new file. Resize to NewFileSize and run the constructor.
  LMFR.IsConstructingNewFile = true;
  if (Error E = NewFileConstructor(LMFR))
    return std::move(E);
  assert(LMFR.size() > 0 && "Constructor must set a non-zero size");
  LMFR.IsConstructingNewFile = false;
  return std::move(LMFR);
}

Error LazyMappedFileRegion::extendSizeImpl(uint64_t MinSize) {
  assert(Map && "Expected a valid map");
  assert(FD && "Expected a valid file descriptor");

  // Synchronize with other threads. Skip if constructing a new file since
  // exclusive access is already guaranteed.
  std::optional<std::lock_guard<std::mutex>> Lock;
  if (!IsConstructingNewFile)
    Lock.emplace(Mutex);

  uint64_t OldSize = CachedSize;
  if (MinSize <= OldSize)
    return Error::success();

  // Increase sizes by doubling up to 8MB, and then limit the over-allocation
  // to 4MB.
  uint64_t NewSize;
  if (MinSize < MaxSizeIncrement)
    NewSize = NextPowerOf2(MinSize);
  else
    NewSize = alignTo(MinSize, MaxSizeIncrement);

  if (NewSize > Map.size())
    NewSize = Map.size();
  if (NewSize < MinSize)
    return errorCodeToError(std::make_error_code(std::errc::not_enough_memory));

  // Synchronize with other processes. Skip if constructing a new file since
  // file locks are already in place.
  if (!IsConstructingNewFile)
    if (std::error_code EC = sys::fs::lockFile(*FD))
      return errorCodeToError(EC);
  auto Unlock = make_scope_exit([&]() {
    if (!IsConstructingNewFile)
      sys::fs::unlockFile(*FD);
  });

  sys::fs::file_status Status;
  if (std::error_code EC = sys::fs::status(*FD, Status))
    return errorCodeToError(EC);
  if (Status.getSize() >= MinSize) {
    // Another process already resized the file. Be careful not to let size()
    // increase beyond capacity(), in case that process used a bigger map size.
    CachedSize = std::min(Status.getSize(), (uint64_t)Map.size());
    return Error::success();
  }

  // Resize.
  if (std::error_code EC = sys::fs::resize_file(*FD, NewSize))
    return errorCodeToError(EC);
  CachedSize = NewSize;
  return Error::success();
}

Expected<std::shared_ptr<LazyMappedFileRegion>>
LazyMappedFileRegion::createShared(
    const Twine &PathTwine, uint64_t Capacity,
    function_ref<Error(LazyMappedFileRegion &)> NewFileConstructor,
    uint64_t MaxSizeIncrement) {
  struct MapNode {
    std::mutex Mutex;
    std::weak_ptr<LazyMappedFileRegion> LMFR;
  };
  static std::mutex Mutex;

  // FIXME: Map should be by sys::fs::UniqueID instead of by path. Here's how
  // it should work:
  //
  // 1. Open the file.
  // 2. Stat the file descriptor to get the UniqueID.
  // 3. Check the map.
  // 4. If new, pass the open file descriptor to a helper extracted from
  //    LazyMappedFileRegion::create().
  static StringMap<MapNode> Regions;

  SmallString<128> PathStorage;
  const StringRef Path = PathTwine.toStringRef(PathStorage);

  MapNode *Node;
  {
    std::lock_guard<std::mutex> Lock(Mutex);
    Node = &Regions[Path];
  }

  if (std::shared_ptr<LazyMappedFileRegion> LMFR = Node->LMFR.lock())
    return LMFR;

  // Construct a new region. Use a fine-grained lock to allow other regions to
  // be opened concurrently.
  std::lock_guard<std::mutex> Lock(Node->Mutex);

  // Open / create / initialize files on disk.
  Expected<LazyMappedFileRegion> ExpectedLMFR = LazyMappedFileRegion::create(
      Path, Capacity, NewFileConstructor, MaxSizeIncrement);
  if (!ExpectedLMFR)
    return ExpectedLMFR.takeError();

  auto SharedLMFR =
      std::make_shared<LazyMappedFileRegion>(std::move(*ExpectedLMFR));

  // Success.
  Node->LMFR = SharedLMFR;
  return std::move(SharedLMFR);
}
#endif // LLVM_ENABLE_ONDISK_CAS
