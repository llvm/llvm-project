//===- RecordReplay.cpp - Target independent kernel record replay ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#include "PluginInterface.h"

#include "Shared/APITypes.h"

#include "ErrorReporting.h"
#include "Shared/RecordReplayBlocks.h"
#include "Shared/RecordReplayDiff.h"
#include "Shared/RecordReplayPointers.h"
#include "Shared/Utils.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/Compression.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/xxhash.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <functional>

using namespace llvm;
using namespace omp;
using namespace target;
using namespace plugin;
using namespace error;

RecordReplayTy::InstanceTy::InstanceTy(const KernelInfoTy &Kernel,
                                       uint32_t NumTeams, uint32_t NumThreads,
                                       uint32_t SharedMemorySize,
                                       KernelReplayOutcomeTy *ReplayOutcome)
    : KernelName(Kernel.getName()), NumTeams(NumTeams), NumThreads(NumThreads),
      SharedMemorySize(SharedMemorySize), ReplayOutcome(ReplayOutcome) {
  KernelHash = stable_hash_name(Kernel.getName());
  LaunchConfigHash =
      stable_hash_combine((stable_hash)NumTeams, (stable_hash)NumThreads,
                          (stable_hash)SharedMemorySize);
}

Error RecordReplayTy::init(uint64_t MemSize, void *VAddr) {
  // When replaying without an address, the memory can be placed anywhere. This
  // requires relocating the pointers into the recorded memory.
  if (!VAddr && isRecording())
    VAddr = Device.getSuggestedVirtualAddress();

  auto StartAddrOrErr = Device.allocateWithVirtualAddress(MemSize, VAddr);
  if (!StartAddrOrErr)
    return StartAddrOrErr.takeError();

  if (!*StartAddrOrErr)
    return Plugin::error(ErrorCode::OUT_OF_RESOURCES, "allocating memory");
  if (isReplaying() && VAddr && *StartAddrOrErr != VAddr)
    return Plugin::error(ErrorCode::INVALID_ARGUMENT,
                         "could not reserve recorded virtual address");

  StartAddr = *StartAddrOrErr;
  TotalSize = MemSize;

  // Create the output directory if necessary.
  std::error_code EC;
  std::filesystem::create_directories(OutputDirectory, EC);
  if (EC)
    return Plugin::error(ErrorCode::HOST_IO, "creating output directory");

  INFO(OMP_INFOTYPE_PLUGIN_KERNEL, Device.getDeviceId(),
       "%s initialized with starting address %p, "
       "memory size %" PRIu64 " bytes, and output directory in %s\n",
       Status == StatusTy::Recording ? "Record" : "Replay", StartAddr,
       TotalSize, OutputDirectory.string().c_str());

  return Plugin::success();
}

Error RecordReplayTy::deinit() {
  if (isRecording() && EmitReport)
    if (auto Err = emitInstanceReport())
      return Err;

  if (BlockStore) {
    BlockStore->close();
    if (BlockStore->has_error()) {
      BlockStore->clear_error();
      return Plugin::error(ErrorCode::HOST_IO, "saving block store file");
    }
    BlockStore.reset();
  }

  if (StartAddr)
    return Device.deallocateWithVirtualAddress(StartAddr, TotalSize);

  return Plugin::success();
}

Error RecordReplayTy::emitInstanceReport() {
  llvm::raw_ostream *OutStream = &llvm::outs();
  std::unique_ptr<llvm::raw_fd_ostream> FileOut;

  if (!ReportFilename.empty()) {
    // The report file is emitted in the output directory.
    std::string ReportFilepath =
        (std::filesystem::absolute(OutputDirectory) / ReportFilename).string();
    std::error_code EC;
    FileOut = std::make_unique<llvm::raw_fd_ostream>(ReportFilepath, EC);
    if (EC)
      return Plugin::error(ErrorCode::HOST_IO, "saving report file");
    OutStream = FileOut.get();
  }

  std::lock_guard<std::mutex> LG(InstancesLock);
  *OutStream << "=== Kernel Record Report ===\n";
  *OutStream << "Directory: "
             << std::filesystem::absolute(OutputDirectory).string() << "\n";
  *OutStream << "Total Instances: " << OrderedInstances.size() << "\n";
  *OutStream << "JSON Filename, Kernel Name, Time (ns), Occurrences:\n";

  SmallString<128> Filename;
  for (const auto *Inst : OrderedInstances)
    *OutStream
        << getFilename(*Inst, FileTy::Descriptor, /*IncludeDir=*/false).c_str()
        << ", " << Inst->KernelName << ", " << Inst->getRecordedTimeNs() << ", "
        << Inst->Occurrences << "\n";
  *OutStream << "=== End Kernel Record Report ===\n";

  return Plugin::success();
}

std::pair<const RecordReplayTy::InstanceTy &, bool>
RecordReplayTy::registerInstance(const KernelInfoTy &Kernel, uint32_t NumTeams,
                                 uint32_t NumThreads, uint32_t SharedMemorySize,
                                 KernelReplayOutcomeTy *ReplayOutcome) {
  std::lock_guard<std::mutex> LG(InstancesLock);
  auto [It, Inserted] = Instances.emplace(Kernel, NumTeams, NumThreads,
                                          SharedMemorySize, ReplayOutcome);
  // Keep insertion order.
  if (Inserted)
    OrderedInstances.push_back(&(*It));

  // Increase the number of occurrences.
  It->Occurrences += 1;

  // Return reference and whether it was registered for the first time. Notice
  // that registering an unregistered instance counts as a new registration.
  return {*It, (It->Occurrences == 1)};
}

Error RecordReplayTy::unregisterInstance(const InstanceTy &Instance) {
  assert(isReplaying() && "Cannot unregister instance when recording.");

  // Do not remove it, it may be reused in the future.
  std::lock_guard<std::mutex> LG(InstancesLock);
  Instance.Occurrences = 0;
  return Plugin::success();
}

Expected<void *> RecordReplayTy::allocate(uint64_t Size) {
  assert(StartAddr && "Expected memory has been pre-allocated");
  constexpr int Alignment = 16;
  // Assume alignment is a power of 2.
  int64_t AlignedSize = (Size + (Alignment - 1)) & (~(Alignment - 1));

  std::lock_guard<std::mutex> LG(AllocationLock);
  if (CurrentSize + AlignedSize > TotalSize)
    return Plugin::error(ErrorCode::OUT_OF_RESOURCES,
                         "run out of record replay memory");
  void *Alloc = (char *)StartAddr + CurrentSize;
  CurrentSize += AlignedSize;
  return Alloc;
}

Error RecordReplayTy::deallocate(void *Ptr) { return Plugin::success(); }

void RecordReplayTy::addPointerLocation(void *TgtPtrAddr) {
  if (!isRecording())
    return;

  uint64_t Addr = reinterpret_cast<uintptr_t>(TgtPtrAddr);
  uint64_t Start = reinterpret_cast<uintptr_t>(StartAddr);
  if (Addr < Start || Addr + sizeof(void *) > Start + TotalSize)
    return;

  std::lock_guard<std::mutex> LG(AllocationLock);
  PointerLocations.push_back(Addr - Start);
}

uint64_t RecordReplayTy::countUnknownPointers(const char *Data, uint64_t Size,
                                              uint64_t Offset,
                                              ArrayRef<uint64_t> Known,
                                              uint64_t MemorySize) const {
  uint64_t Count = 0;
  for (uint64_t I = alignTo(Offset, sizeof(uint64_t)) - Offset;
       I + sizeof(uint64_t) <= Size; I += sizeof(uint64_t)) {
    uint64_t Value;
    std::memcpy(&Value, Data + I, sizeof(Value));
    if (pointsToMemory(Value, MemorySize) &&
        !llvm::binary_search(Known, Offset + I))
      ++Count;
  }
  return Count;
}

Expected<RecordReplayTy::HandleTy>
RecordReplayTy::recordPrologue(const KernelInfoTy &Kernel,
                               const KernelRecordArgsTy &RecordArgs) {
  if (!isRecordingOrReplaying())
    return HandleTy{nullptr, false};

  // Register the instance and avoid recording if it is inactive or replaying.
  auto [Instance, First] = registerInstance(
      Kernel, RecordArgs.NumBlocks[0], RecordArgs.NumThreads[0],
      RecordArgs.DynCGroupMem, RecordArgs.ReplayOutcome);

  HandleTy Handle{&Instance, First};
  if (!First)
    return Handle;

  if (isRecording())
    if (auto Err = recordDescImpl(Kernel, Instance, RecordArgs))
      return Err;

  if (shouldRecordPrologue())
    if (auto Err = recordPrologueImpl(Kernel, Instance))
      return Err;

  // Start the timer for the kernel execution.
  Instance.recordBeginTime();

  return Handle;
}

Error RecordReplayTy::recordEpilogue(const KernelInfoTy &Kernel,
                                     HandleTy Handle) {
  if (!Handle.Active)
    return Plugin::success();

  // Stop the timer for the kernel execution.
  const InstanceTy &Instance = *Handle.Instance;
  Instance.recordEndTime();

  if (shouldRecordEpilogue())
    if (auto Err = recordEpilogueImpl(Kernel, Instance))
      return Err;

  if (isRecording())
    if (auto Err = recordPointersImpl(Instance))
      return Err;

  if (isReplaying() && Instance.ReplayOutcome)
    populateReplayOutcome(Instance, *Instance.ReplayOutcome);

  // After a replay, unregister the instance so it can be replayed again. Do
  // not access the instance object beyond this point.
  if (isReplaying())
    return unregisterInstance(Instance);

  return Plugin::success();
}

void RecordReplayTy::populateReplayOutcome(const InstanceTy &Instance,
                                           KernelReplayOutcomeTy &Outcome) {
  // Only save the epilogue output filename if it was recorded.
  if (shouldRecordEpilogue()) {
    SmallString<128> Filename = getFilename(Instance, FileTy::EpilogueSnapshot);
    Outcome.OutputFilepath = Filename;
  }
  // Save the kernel replay time.
  Outcome.KernelReplayTimeNs = Instance.getRecordedTimeNs();
}

Error NativeRecordReplayTy::recordPrologueImpl(const KernelInfoTy &Kernel,
                                               const InstanceTy &Instance) {
  // When replaying, the device memory was initialized from an input snapshot
  // on the host. If it covers the whole replay memory, use it instead of
  // retrieving the device memory.
  if (const KernelReplayOutcomeTy *Outcome = Instance.ReplayOutcome) {
    AllocationLock.lock();
    uint64_t ReplaySize = CurrentSize;
    AllocationLock.unlock();

    if (Outcome->InputSnapshotSize == ReplaySize &&
        (!ReplaySize ||
         (Outcome->InputSnapshot && Outcome->ReplayDeviceAlloc == StartAddr))) {
      StringRef Snapshot(static_cast<const char *>(Outcome->InputSnapshot),
                         ReplaySize);
      Instance.InputSnapshot =
          MemoryBuffer::getMemBuffer(Snapshot, /*BufferName=*/"",
                                     /*RequiresNullTerminator=*/false);
      return Plugin::success();
    }
  }

  auto SnapshotOrErr = retrieveSnapshot();
  if (!SnapshotOrErr)
    return SnapshotOrErr.takeError();
  std::unique_ptr<WritableMemoryBuffer> Snapshot = std::move(*SnapshotOrErr);

  if (isRecording()) {
    SmallString<128> SnapshotFilename =
        getFilename(Instance, FileTy::PrologueSnapshot);
    if (auto Err = UseBlockStore
                       ? recordSnapshotBlocks(SnapshotFilename, Snapshot.get())
                       : recordSnapshot(SnapshotFilename, Snapshot.get()))
      return Err;

    findInputPointers(Instance, Snapshot.get());

    SmallString<128> GlobalsFilename = getFilename(Instance, FileTy::Globals);
    uint64_t NumUnknownGlobalPointers = 0;
    if (auto Err =
            recordGlobals(GlobalsFilename.c_str(), NumUnknownGlobalPointers))
      return Err;
    Instance.NumUnknownInputPointers += NumUnknownGlobalPointers;

    SmallString<128> ImageFilename = getFilename(Instance, FileTy::Program);

    SmallString<128> IRImageFilename = getFilename(Instance, FileTy::IRImage);
    if (auto Err =
            recordImage(Kernel, ImageFilename.c_str(), IRImageFilename.c_str()))
      return Err;
  }

  // Keep the input snapshot to record the output snapshot as a diff against it.
  // When replaying, this is the only purpose of the prologue.
  if (shouldRecordEpilogue())
    Instance.InputSnapshot = std::move(Snapshot);
  return Plugin::success();
}

Error NativeRecordReplayTy::recordEpilogueImpl(const KernelInfoTy &Kernel,
                                               const InstanceTy &Instance) {
  SmallString<128> SnapshotFilename =
      getFilename(Instance, FileTy::EpilogueSnapshot);
  return recordSnapshotDiff(SnapshotFilename, Instance);
}

Error NativeRecordReplayTy::recordDescImpl(
    const KernelInfoTy &Kernel, const InstanceTy &Instance,
    const KernelRecordArgsTy &RecordArgs) {
  json::Object JsonKernelInfo;
  JsonKernelInfo["Name"] = Instance.KernelName;
  JsonKernelInfo["NumArgs"] = RecordArgs.NumArgs;
  JsonKernelInfo["NumTeams"] = Instance.NumTeams;
  JsonKernelInfo["NumThreads"] = Instance.NumThreads;
  JsonKernelInfo["SharedMemorySize"] = Instance.SharedMemorySize;
  JsonKernelInfo["LoopTripCount"] = RecordArgs.Tripcount;
  JsonKernelInfo["DeviceId"] = Device.getDeviceId();
  JsonKernelInfo["VAllocAddr"] = (intptr_t)StartAddr;
  JsonKernelInfo["VAllocSize"] = TotalSize;

  // Export minimum and maximum for allowed number of teams. If zero, it means
  // there was no restriction provided by the program.
  uint32_t MinMaxBlocks = RecordArgs.RequestedNumBlocks;
  json::Array JsonTeamsLimits;
  JsonTeamsLimits.push_back(MinMaxBlocks);
  JsonTeamsLimits.push_back(MinMaxBlocks);
  JsonKernelInfo["TeamsLimits"] = json::Value(std::move(JsonTeamsLimits));

  // Export minimum and maximum for allowed number of threads. If zero, it means
  // there was no restriction provided by the program.
  uint32_t MaxThreads = RecordArgs.MaxNumThreads;
  json::Array JsonThreadsLimits;
  JsonThreadsLimits.push_back(1);
  JsonThreadsLimits.push_back(MaxThreads);
  JsonKernelInfo["ThreadsLimits"] = json::Value(std::move(JsonThreadsLimits));

  json::Array JsonArgPtrs;
  for (uint32_t I = 0; I < RecordArgs.NumArgs; ++I)
    JsonArgPtrs.push_back((intptr_t)(*(void **)RecordArgs.Args[I]));
  JsonKernelInfo["ArgPtrs"] = json::Value(std::move(JsonArgPtrs));

  json::Array JsonArgOffsets;
  for (uint32_t I = 0; I < RecordArgs.NumArgs; ++I)
    JsonArgOffsets.push_back(0);
  JsonKernelInfo["ArgOffsets"] = json::Value(std::move(JsonArgOffsets));

  SmallString<128> JsonFilename = getFilename(Instance, FileTy::Descriptor);

  std::error_code EC;
  raw_fd_ostream JsonOS(JsonFilename.c_str(), EC);
  if (EC)
    return Plugin::error(ErrorCode::HOST_IO, "saving kernel json file");
  JsonOS << json::Value(std::move(JsonKernelInfo));
  JsonOS.close();
  return Plugin::success();
}

StringRef NativeRecordReplayTy::getExtension(FileTy FileType) {
  switch (FileType) {
  case FileTy::PrologueSnapshot:
    return "record_input";
  case FileTy::EpilogueSnapshot:
    return isRecording() ? "record_output" : "replay_output";
  case FileTy::Descriptor:
    return "json";
  case FileTy::Globals:
    return "globals";
  case FileTy::Program:
    return "image";
  case FileTy::IRImage:
    return "bc";
  case FileTy::Pointers:
    return "pointers";
  }
  return "";
}

SmallString<128>
NativeRecordReplayTy::getFilenameImpl(const InstanceTy &Instance,
                                      FileTy FileType, bool IncludeDirectory) {
  std::filesystem::path Filepath = IncludeDirectory ? OutputDirectory : "";
  Filepath /= std::to_string(Instance.KernelHash) + "_" +
              std::to_string(Instance.LaunchConfigHash);
  Filepath.replace_extension(getExtension(FileType).data());
  SmallString<128> Filename(Filepath.string());
  return Filename;
}

Expected<std::unique_ptr<WritableMemoryBuffer>>
NativeRecordReplayTy::retrieveSnapshot() {
  // Another thread may be allocating memory. The size can only increase.
  AllocationLock.lock();
  uint64_t RecordSize = CurrentSize;
  AllocationLock.unlock();

  if (!RecordSize)
    return nullptr;

  auto DeviceMB = WritableMemoryBuffer::getNewUninitMemBuffer(RecordSize);
  if (!DeviceMB)
    return Plugin::error(ErrorCode::OUT_OF_RESOURCES,
                         "creating MemoryBuffer for device memory");

  if (auto Err = Device.dataRetrieve(DeviceMB->getBufferStart(), StartAddr,
                                     RecordSize, nullptr))
    return std::move(Err);
  return std::move(DeviceMB);
}

void NativeRecordReplayTy::findInputPointers(const InstanceTy &Instance,
                                             const MemoryBuffer *Snapshot) {
  AllocationLock.lock();
  SmallVector<uint64_t> Locations = PointerLocations;
  AllocationLock.unlock();

  llvm::sort(Locations);
  Locations.erase(llvm::unique(Locations), Locations.end());

  Instance.PointerOffsets.clear();
  Instance.NumUnknownInputPointers = 0;
  if (!Snapshot)
    return;

  // Only the logged locations that still hold a pointer into the memory need
  // to be relocated.
  const char *Data = Snapshot->getBufferStart();
  uint64_t Size = Snapshot->getBufferSize();
  for (uint64_t Location : Locations) {
    uint64_t Value;
    if (Location + sizeof(Value) > Size)
      continue;
    std::memcpy(&Value, Data + Location, sizeof(Value));
    if (pointsToMemory(Value, Size))
      Instance.PointerOffsets.push_back(Location);
  }
  Instance.NumUnknownInputPointers =
      countUnknownPointers(Data, Size, /*Offset=*/0, Instance.PointerOffsets,
                           /*MemorySize=*/Size);
}

Error NativeRecordReplayTy::recordPointersImpl(const InstanceTy &Instance) {
  SmallString<128> Filename = getFilename(Instance, FileTy::Pointers);
  std::error_code EC;
  raw_fd_ostream OS(Filename, EC);
  if (EC)
    return Plugin::error(ErrorCode::HOST_IO, "saving pointers file");

  RecordReplayPointersHeaderTy Header;
  std::memcpy(Header.Magic, RecordReplayPointersMagic, sizeof(Header.Magic));
  Header.NumPointers = Instance.PointerOffsets.size();
  Header.NumUnknownInputPointers = Instance.NumUnknownInputPointers;
  Header.NumUnknownOutputPointers = Instance.NumUnknownOutputPointers;
  OS.write(reinterpret_cast<const char *>(&Header), sizeof(Header));
  OS.write(reinterpret_cast<const char *>(Instance.PointerOffsets.data()),
           Instance.PointerOffsets.size() * sizeof(uint64_t));
  OS.close();

  // The instance is never recorded again.
  Instance.PointerOffsets = {};
  return Plugin::success();
}

Error NativeRecordReplayTy::recordSnapshot(StringRef Filename,
                                           const MemoryBuffer *Snapshot) {
  std::error_code EC;
  raw_fd_ostream OS(Filename, EC);
  if (EC)
    return Plugin::error(ErrorCode::HOST_IO, "saving memory snapshot file");
  if (Snapshot)
    OS.write(Snapshot->getBufferStart(), Snapshot->getBufferSize());
  OS.close();
  return Plugin::success();
}

Error NativeRecordReplayTy::recordSnapshotBlocks(StringRef Filename,
                                                 const MemoryBuffer *Snapshot) {
  constexpr uint64_t BlockSize = RecordReplayBlockSize;
  static const char ZeroBlock[BlockSize] = {};
  ArrayRef<uint8_t> Data;
  if (Snapshot)
    Data = arrayRefFromStringRef(Snapshot->getBuffer());
  uint64_t NumBlocks = divideCeil(Data.size(), BlockSize);
  SmallVector<RecordReplayBlockTy> Blocks(NumBlocks);

  {
    std::lock_guard<std::mutex> LG(BlockStoreLock);
    if (!BlockStore) {
      // Append to the block store of the directory, so that the snapshots
      // recorded into it before remain valid.
      SmallString<128> StorePath(OutputDirectory.string());
      sys::path::append(StorePath, RecordReplayBlockStoreName);
      uint64_t ExistingSize = 0;
      if (sys::fs::exists(StorePath) &&
          sys::fs::file_size(StorePath, ExistingSize))
        return Plugin::error(ErrorCode::HOST_IO, "reading block store file");
      std::error_code EC;
      BlockStore =
          std::make_unique<raw_fd_ostream>(StorePath, EC, sys::fs::OF_Append);
      if (EC)
        return Plugin::error(ErrorCode::HOST_IO, "opening block store file");
      BlockStoreSize = ExistingSize;
    }

    SmallVector<uint8_t> Compressed;
    for (uint64_t I = 0; I < NumBlocks; ++I) {
      ArrayRef<uint8_t> Block = Data.slice(
          I * BlockSize, std::min(BlockSize, Data.size() - I * BlockSize));
      if (!std::memcmp(Block.data(), ZeroBlock, Block.size())) {
        Blocks[I] = {/*Offset=*/0, /*StoredSize=*/0, RecordReplayBlockZero};
        continue;
      }

      // Store each distinct block content once.
      XXH128_hash_t Hash = xxh3_128bits(Block.data(), Block.size());
      auto [It, Inserted] = StoredBlocks.try_emplace({Hash.high64, Hash.low64});
      if (Inserted) {
        ArrayRef<uint8_t> Stored = Block;
        uint32_t Codec = RecordReplayBlockRaw;
        Compressed.clear();
        if (compression::zstd::isAvailable()) {
          compression::zstd::compress(Block, Compressed, /*Level=*/3);
          Codec = RecordReplayBlockZstd;
        } else if (compression::zlib::isAvailable()) {
          compression::zlib::compress(Block, Compressed);
          Codec = RecordReplayBlockZlib;
        }
        if (Codec != RecordReplayBlockRaw && Compressed.size() < Block.size())
          Stored = Compressed;
        else
          Codec = RecordReplayBlockRaw;

        It->second = {BlockStoreSize, static_cast<uint32_t>(Stored.size()),
                      Codec};
        BlockStore->write(reinterpret_cast<const char *>(Stored.data()),
                          Stored.size());
        BlockStoreSize += Stored.size();
      }
      Blocks[I] = It->second;
    }

    // Make the stored blocks visible before the snapshot that uses them.
    BlockStore->flush();
    if (BlockStore->has_error())
      return Plugin::error(ErrorCode::HOST_IO, "saving block store file");
  }

  std::error_code EC;
  raw_fd_ostream OS(Filename, EC);
  if (EC)
    return Plugin::error(ErrorCode::HOST_IO, "saving memory snapshot file");
  RecordReplayBlocksHeaderTy Header;
  std::memcpy(Header.Magic, RecordReplayBlocksMagic, sizeof(Header.Magic));
  Header.Size = Data.size();
  Header.BlockSize = BlockSize;
  Header.NumBlocks = NumBlocks;
  OS.write(reinterpret_cast<const char *>(&Header), sizeof(Header));
  OS.write(reinterpret_cast<const char *>(Blocks.data()),
           Blocks.size() * sizeof(RecordReplayBlockTy));
  OS.close();
  if (OS.has_error()) {
    OS.clear_error();
    return Plugin::error(ErrorCode::HOST_IO, "saving memory snapshot file");
  }
  return Plugin::success();
}

/// Return the number of leading bytes that are equal in \p A and \p B.
static uint64_t getNumEqualBytes(const char *A, const char *B, uint64_t Size) {
  // Skip the equal blocks before comparing byte by byte.
  constexpr uint64_t BlockSize = 4096;
  uint64_t I = 0;
  while (I + BlockSize <= Size && std::memcmp(A + I, B + I, BlockSize) == 0)
    I += BlockSize;
  while (I < Size && A[I] == B[I])
    ++I;
  return I;
}

Error NativeRecordReplayTy::recordSnapshotDiff(StringRef Filename,
                                               const InstanceTy &Instance) {
  // Another thread may be allocating memory. The size can only increase.
  AllocationLock.lock();
  uint64_t OutputSize = CurrentSize;
  AllocationLock.unlock();

  // Take the input snapshot so it is released once the diff is recorded.
  std::unique_ptr<MemoryBuffer> Input = std::move(Instance.InputSnapshot);
  const char *InputData = Input ? Input->getBufferStart() : nullptr;
  uint64_t InputSize = Input ? Input->getBufferSize() : 0;

  std::error_code EC;
  raw_fd_ostream OS(Filename, EC);
  if (EC)
    return Plugin::error(ErrorCode::HOST_IO, "saving memory snapshot file");

  RecordReplayDiffHeaderTy Header;
  std::memcpy(Header.Magic, RecordReplayDiffMagic, sizeof(Header.Magic));
  Header.InputSize = InputSize;
  Header.OutputSize = OutputSize;
  OS.write(reinterpret_cast<const char *>(&Header), sizeof(Header));

  // The region being written, if any, and the end of its bytes written so far.
  // Its descriptor is written once its size is known, at the file position
  // where it was reserved.
  bool RegionOpen = false;
  RecordReplayDiffRegionTy Region;
  uint64_t RegionFilePos = 0;
  uint64_t RegionEnd = 0;

  // When recording, count the unknown pointers in the output, which prevent
  // verifying a relocated replay.
  if (isRecording())
    Instance.NumUnknownOutputPointers = 0;
  auto CloseRegion = [&]() {
    Region.Size = RegionEnd - Region.Offset;
    OS.pwrite(reinterpret_cast<const char *>(&Region), sizeof(Region),
              RegionFilePos);
  };

  // Retrieve the output memory in chunks to bound the host memory usage. The
  // regions do not depend on the chunk size, since a region can span chunks.
  constexpr uint64_t MaxChunkSize = 64 * 1024 * 1024;
  std::unique_ptr<WritableMemoryBuffer> ChunkMB;
  if (OutputSize) {
    ChunkMB = WritableMemoryBuffer::getNewUninitMemBuffer(
        std::min(MaxChunkSize, OutputSize));
    if (!ChunkMB)
      return Plugin::error(ErrorCode::OUT_OF_RESOURCES,
                           "creating MemoryBuffer for device memory");
  }

  for (uint64_t ChunkOffset = 0; ChunkOffset < OutputSize;
       ChunkOffset += MaxChunkSize) {
    uint64_t ChunkSize = std::min(MaxChunkSize, OutputSize - ChunkOffset);
    char *Chunk = ChunkMB->getBufferStart();
    if (auto Err = Device.dataRetrieve(
            Chunk, utils::advancePtr(StartAddr, ChunkOffset), ChunkSize,
            nullptr))
      return Err;

    if (isRecording())
      Instance.NumUnknownOutputPointers += countUnknownPointers(
          Chunk, ChunkSize, ChunkOffset, Instance.PointerOffsets, OutputSize);

    // A byte is changed if it differs from the input snapshot or lies beyond
    // it.
    auto IsChanged = [&](uint64_t I) {
      uint64_t Offset = ChunkOffset + I;
      return Offset >= InputSize || Chunk[I] != InputData[Offset];
    };

    uint64_t I = 0;
    while (I < ChunkSize) {
      // Skip the unchanged bytes.
      uint64_t Offset = ChunkOffset + I;
      if (Offset < InputSize) {
        uint64_t Size = std::min(ChunkSize - I, InputSize - Offset);
        I += getNumEqualBytes(Chunk + I, InputData + Offset, Size);
        if (I == ChunkSize)
          break;
        Offset = ChunkOffset + I;
      }

      // Find the end of the changed bytes.
      uint64_t End = I + 1;
      while (End < ChunkSize && IsChanged(End))
        ++End;

      // Extend the current region over a short gap of unchanged bytes, which
      // are taken from the input snapshot, or start a new region.
      if (RegionOpen && Offset - RegionEnd < RecordReplayDiffMaxGap) {
        if (Offset > RegionEnd)
          OS.write(InputData + RegionEnd, Offset - RegionEnd);
      } else {
        if (RegionOpen)
          CloseRegion();
        RegionOpen = true;
        Region.Offset = Offset;
        RegionFilePos = OS.tell();
        OS.write_zeros(sizeof(Region));
      }
      OS.write(Chunk + I, End - I);
      RegionEnd = ChunkOffset + End;
      I = End;
    }
  }
  if (RegionOpen)
    CloseRegion();

  OS.close();
  if (OS.has_error()) {
    OS.clear_error();
    return Plugin::error(ErrorCode::HOST_IO, "saving memory snapshot file");
  }
  return Plugin::success();
}

Error NativeRecordReplayTy::recordImage(const KernelInfoTy &Kernel,
                                        StringRef Filename,
                                        StringRef IRImageFilename) {
  std::error_code EC;
  raw_fd_ostream OS(Filename, EC);
  if (EC)
    return Plugin::error(ErrorCode::HOST_IO, "saving image file");
  OS << Kernel.getImage().getBuffer();
  OS.close();

  if (!IRImageFilename.empty() && Kernel.hasIRImage()) {
    raw_fd_ostream IROS(IRImageFilename, EC);
    if (EC)
      return Plugin::error(ErrorCode::HOST_IO, "saving IR image file");
    IROS << Kernel.getIRImage().getBuffer();
    IROS.close();
  }

  return Plugin::success();
}

Error NativeRecordReplayTy::recordGlobals(StringRef Filename,
                                          uint64_t &NumUnknownPointers) {
  NumUnknownPointers = 0;

  AllocationLock.lock();
  // Copy the globals into a local vector so we can read it safely from this
  // thread. This vector should have a few entries in general. No need to lock
  // the entire function.
  SmallVector<GlobalEntryTy> Globals = GlobalEntries;
  uint64_t MemorySize = CurrentSize;
  AllocationLock.unlock();

  uint64_t TotalSize = sizeof(uint32_t);
  uint32_t NumGlobals = 0;
  for (auto &Global : Globals) {
    if (!Global.Size)
      continue;
    // Get the total size of the string and entry including the null byte.
    TotalSize += Global.Size + sizeof(uint32_t) + sizeof(uint64_t) +
                 Global.Name.length() + 1;
    NumGlobals++;
  }

  auto GlobalsMB = WritableMemoryBuffer::getNewUninitMemBuffer(TotalSize);
  if (!GlobalsMB)
    return Plugin::error(ErrorCode::OUT_OF_RESOURCES,
                         "creating MemoryBuffer for globals memory");

  void *BufferPtr = GlobalsMB->getBufferStart();
  *((uint32_t *)(BufferPtr)) = NumGlobals;
  BufferPtr = utils::advancePtr(BufferPtr, sizeof(uint32_t));

  for (auto &Global : Globals) {
    if (!Global.Size)
      continue;

    uint32_t NameLength = Global.Name.length() + 1;
    *((uint32_t *)(BufferPtr)) = NameLength;
    BufferPtr = utils::advancePtr(BufferPtr, sizeof(uint32_t));

    *((uint64_t *)(BufferPtr)) = Global.Size;
    BufferPtr = utils::advancePtr(BufferPtr, sizeof(uint64_t));

    memcpy(BufferPtr, Global.Name.data(), NameLength);
    BufferPtr = utils::advancePtr(BufferPtr, NameLength);

    if (auto Err =
            Device.dataRetrieve(BufferPtr, Global.Addr, Global.Size, nullptr))
      return Err;

    // The globals are restored by name, so a pointer into the record replay
    // memory stored in a global cannot be relocated.
    NumUnknownPointers +=
        countUnknownPointers(static_cast<const char *>(BufferPtr), Global.Size,
                             /*Offset=*/0, /*Known=*/{}, MemorySize);
    BufferPtr = utils::advancePtr(BufferPtr, Global.Size);
  }
  assert(BufferPtr == GlobalsMB->getBufferEnd() &&
         "Buffer over or under-filled.");
  assert(TotalSize == (uint64_t)utils::getPtrDiff(
                          BufferPtr, GlobalsMB->getBufferStart()) &&
         "Buffer size mismatch.");

  std::error_code EC;
  raw_fd_ostream OS(Filename, EC);
  OS.write(GlobalsMB->getBufferStart(), TotalSize);
  OS.close();
  return Plugin::success();
}
