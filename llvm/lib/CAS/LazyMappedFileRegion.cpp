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

#ifdef _WIN32
#include "llvm/Support/Windows/WindowsSupport.h"
#include "llvm/Support/WindowsError.h"
#endif

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
#ifdef _WIN32
// FIXME: VirtualAlloc2 declaration for older SDKs and building for older
// windows verions. Currently only support windows 10+.
PVOID WINAPI VirtualAlloc2(HANDLE Process, PVOID BaseAddress, SIZE_T Size,
                           ULONG AllocationType, ULONG PageProtection,
                           MEM_EXTENDED_PARAMETER *ExtendedParameters,
                           ULONG ParameterCount);

Expected<LazyMappedFileRegion> LazyMappedFileRegion::create(
    const Twine &Path, uint64_t Capacity,
    function_ref<Error(LazyMappedFileRegion &)> NewFileConstructor,
    uint64_t MaxSizeIncrement) {
  LazyMappedFileRegion LMFR;
  LMFR.Path = Path.str();
  LMFR.MaxSizeIncrement = std::min(Capacity, MaxSizeIncrement);

  int FD;
  if (std::error_code EC = sys::fs::openFileForReadWrite(
          LMFR.Path, FD, sys::fs::CD_OpenAlways, sys::fs::OF_None))
    return errorCodeToError(EC);
  LMFR.FD = FD;

  // Lock the file so we can initialize it.
  if (std::error_code EC = sys::fs::lockFile(*LMFR.FD))
    return createFileError(Path, EC);
  auto Unlock = make_scope_exit([FD = *LMFR.FD]() { sys::fs::unlockFile(FD); });

  sys::fs::file_t File = sys::fs::convertFDToNativeFile(FD);
  // Status the file so we can decide if we need to run init.
  sys::fs::file_status Status;
  if (std::error_code EC = sys::fs::status(File, Status))
    return errorCodeToError(EC);

  // Reserve VM using VirtualAlloc2. This will error out on Windows 10 or
  // before.
  auto pVirtualAlloc2 = (decltype(&::VirtualAlloc2))GetProcAddress(
      GetModuleHandle(L"kernelbase"), "VirtualAlloc2");
  // FIXME: return error if the new alloc function is not found.
  if (!pVirtualAlloc2)
    return errorCodeToError(std::make_error_code(std::errc::not_supported));

  LMFR.VM = (char *)pVirtualAlloc2(0, 0, Capacity,
                                   MEM_RESERVE | MEM_RESERVE_PLACEHOLDER,
                                   PAGE_NOACCESS, 0, 0);
  if (!LMFR.VM)
    return errorCodeToError(mapWindowsError(::GetLastError()));
  LMFR.MaxSize = Capacity;

  if (Status.getSize() > 0)
    // The file was already constructed.
    LMFR.CachedSize = Status.getSize();
  else
    LMFR.IsConstructingNewFile = true;

  // Create a memory mapped region. The larger of the current size or the max
  // file increment.
  uint64_t AllocSize = std::max(LMFR.MaxSizeIncrement, Status.getSize());
  sys::fs::file_t FileMap = ::CreateFileMappingA(
      File, 0, PAGE_READWRITE, Hi_32(AllocSize), Lo_32(AllocSize), 0);
  if (!FileMap)
    return errorCodeToError(mapWindowsError(::GetLastError()));

  // If there is still space in reserved area after allocation, split the
  // placeholder.
  if (AllocSize < LMFR.MaxSize) {
    if (!VirtualFree(LMFR.VM, AllocSize,
                     MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER))
      return errorCodeToError(mapWindowsError(::GetLastError()));
  }

  // Free up AllocSize from VM then mapped the file in.
  if (!VirtualFree(LMFR.VM, 0, MEM_RELEASE))
    return errorCodeToError(mapWindowsError(::GetLastError()));

  void *Mapped =
      ::MapViewOfFileEx(FileMap, FILE_MAP_ALL_ACCESS, 0, 0, AllocSize, LMFR.VM);
  if (!Mapped)
    return errorCodeToError(mapWindowsError(::GetLastError()));

  LMFR.MappedRegions.push_back(Mapped);

  CloseHandle(FileMap);
  if (!LMFR.IsConstructingNewFile)
    return std::move(LMFR);

  // This is a new file. Resize to NewFileSize and run the constructor.
  if (Error E = NewFileConstructor(LMFR))
    return std::move(E);

  assert(LMFR.size() > 0 && "Constructor must set a non-zero size");
  LMFR.IsConstructingNewFile = false;
  return std::move(LMFR);
}

Error LazyMappedFileRegion::extendSizeImpl(uint64_t MinSize) {
  assert(VM && "Expected a valid map");
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

  if (NewSize > MaxSize)
    NewSize = MaxSize;
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
    CachedSize = std::min(Status.getSize(), MaxSize);
    return Error::success();
  }

  // Resize.
  uint64_t AllocSize = NewSize - OldSize;
  sys::fs::file_t File = sys::fs::convertFDToNativeFile(*FD);
  sys::fs::file_t FileMap = ::CreateFileMappingA(
      File, 0, PAGE_READWRITE, Hi_32(NewSize), Lo_32(NewSize), 0);
  if (!FileMap)
    return errorCodeToError(mapWindowsError(::GetLastError()));

  if (NewSize < MaxSize) {
    if (!VirtualFree(VM + OldSize, AllocSize,
                     MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER))
      return errorCodeToError(mapWindowsError(::GetLastError()));
  }

  if (!VirtualFree(VM + OldSize, 0, MEM_RELEASE))
    return errorCodeToError(mapWindowsError(::GetLastError()));

  void *Mapped =
      ::MapViewOfFileEx(FileMap, FILE_MAP_ALL_ACCESS, Hi_32(OldSize),
                        Lo_32(OldSize), AllocSize, VM + OldSize);
  if (!Mapped)
    return errorCodeToError(mapWindowsError(::GetLastError()));

  MappedRegions.push_back(Mapped);
  CloseHandle(FileMap);
  CachedSize = NewSize;
  return Error::success();
}

#else
Expected<LazyMappedFileRegion> LazyMappedFileRegion::create(
    const Twine &Path, uint64_t Capacity,
    function_ref<Error(LazyMappedFileRegion &)> NewFileConstructor,
    uint64_t MaxSizeIncrement) {
  LazyMappedFileRegion LMFR;
  LMFR.Path = Path.str();
  LMFR.MaxSizeIncrement = std::min(Capacity, MaxSizeIncrement);

  int FD;
  if (std::error_code EC = sys::fs::openFileForReadWrite(
          LMFR.Path, FD, sys::fs::CD_OpenAlways, sys::fs::OF_None))
    return errorCodeToError(EC);
  LMFR.FD = FD;

  sys::fs::file_t File = sys::fs::convertFDToNativeFile(FD);

  struct FileLockRAII {
    std::string Path;
    int FD;
    bool IsLocked = false;

    enum LockKind { Shared, Exclusive };

    FileLockRAII(LazyMappedFileRegion &LMFR) : Path(LMFR.Path), FD(*LMFR.FD) {}
    ~FileLockRAII() { consumeError(unlock()); }

    Error lock(LockKind LK) {
      if (IsLocked)
        return createStringError(inconvertibleErrorCode(),
                                 Path + " already locked");
      if (std::error_code EC = sys::fs::lockFile(FD, LK == Exclusive))
        return createFileError(Path, EC);
      IsLocked = true;
      return Error::success();
    }

    Error unlock() {
      if (IsLocked) {
        IsLocked = false;
        if (std::error_code EC = sys::fs::unlockFile(FD))
          return createFileError(Path, EC);
      }
      return Error::success();
    }

  } FileLock(LMFR);

  // Use shared/reader locking in case another process is in the process of
  // initializing the file.
  if (Error E = FileLock.lock(FileLockRAII::Shared))
    return std::move(E);

  sys::fs::file_status Status;
  if (std::error_code EC = sys::fs::status(File, Status))
    return errorCodeToError(EC);

  if (Status.getSize() == 0) {
    // Lock the file exclusively so only one process will do the initialization.
    if (Error E = FileLock.unlock())
      return std::move(E);
    if (Error E = FileLock.lock(FileLockRAII::Exclusive))
      return std::move(E);
    if (std::error_code EC = sys::fs::status(File, Status))
      return errorCodeToError(EC);
  }

  // At this point either the file is still empty (this process won the race to
  // do the initialization) or we have the size for the completely initialized
  // file.

  {
    std::error_code EC;
    sys::fs::mapped_file_region Map(
        File, sys::fs::mapped_file_region::readwrite, Capacity, 0, EC);
    if (EC)
      return createFileError(LMFR.Path, EC);
    LMFR.Map = std::move(Map);
  }

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
#endif

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

void LazyMappedFileRegion::destroyImpl() {
  if (FD) {
    sys::fs::file_t File = sys::fs::convertFDToNativeFile(*FD);
    sys::fs::closeFile(File);
    FD = std::nullopt;
  }
#ifdef _WIN32
  for (auto *Region : MappedRegions)
    CloseHandle(Region);
#endif
}
