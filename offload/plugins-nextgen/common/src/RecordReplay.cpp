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
#include "Shared/Utils.h"

#include "llvm/Support/Error.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdint>
#include <functional>

using namespace llvm;
using namespace omp;
using namespace target;
using namespace plugin;
using namespace error;

RecordReplayTy::InstanceTy::InstanceTy(const GenericKernelTy &Kernel,
                                       uint32_t NumTeams, uint32_t NumThreads,
                                       uint32_t SharedMemorySize)
    : Kernel(Kernel), NumTeams(NumTeams), NumThreads(NumThreads),
      SharedMemorySize(SharedMemorySize) {
  KernelHash = stable_hash_name(Kernel.getName());
  LaunchConfigHash =
      stable_hash_combine((stable_hash)NumTeams, (stable_hash)NumThreads,
                          (stable_hash)SharedMemorySize);
}

Error RecordReplayTy::init(uint64_t MemSize, void *VAddr) {
  if (!VAddr && isReplaying())
    return Plugin::error(ErrorCode::INVALID_ARGUMENT,
                         "VAddr cannot be null when replaying");
  if (!VAddr)
    VAddr = Device.getSuggestedVirtualAddress();

  auto StartAddrOrErr = Device.allocateWithVirtualAddress(MemSize, VAddr);
  if (!StartAddrOrErr)
    return StartAddrOrErr.takeError();

  if (!*StartAddrOrErr)
    return Plugin::error(ErrorCode::OUT_OF_RESOURCES, "allocating memory");
  if (isReplaying() && *StartAddrOrErr != VAddr)
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
       "memory size %lu bytes, and output directory in %s\n",
       Status == StatusTy::Recording ? "Record" : "Replay", StartAddr,
       TotalSize, OutputDirectory.c_str());

  return Plugin::success();
}

Error RecordReplayTy::deinit() {
  if (isRecording() && EmitReport)
    if (auto Err = emitInstanceReport())
      return Err;

  if (StartAddr)
    return Device.deallocateWithVirtualAddress(StartAddr, TotalSize);

  return Plugin::success();
}

Error RecordReplayTy::emitInstanceReport() {
  std::lock_guard<std::mutex> LG(InstancesLock);
  llvm::outs() << "=== record report begin ===\n";
  llvm::outs() << "directory: "
               << std::filesystem::absolute(OutputDirectory).string() << "\n";
  llvm::outs() << "kernels: " << Instances.size() << "\n";
  for (const auto &Inst : Instances) {
    llvm::outs() << getFilename(Inst, "json", /*IncludeDir=*/false) << ": "
                 << Inst.Kernel.getName() << "\n";
  }
  llvm::outs() << "=== record report end ===\n";
  return Plugin::success();
}

std::pair<const RecordReplayTy::InstanceTy &, bool>
RecordReplayTy::registerInstance(const GenericKernelTy &Kernel,
                                 uint32_t NumTeams, uint32_t NumThreads,
                                 uint32_t SharedMemorySize) {
  std::lock_guard<std::mutex> LG(InstancesLock);
  auto [It, Inserted] =
      Instances.emplace(Kernel, NumTeams, NumThreads, SharedMemorySize);
  // Increase the number of occurrences.
  It->Occurrences += 1;
  return {*It, Inserted};
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

Expected<RecordReplayTy::HandleTy> RecordReplayTy::recordPrologue(
    const GenericKernelTy &Kernel, const KernelArgsTy &KernelArgs,
    const KernelLaunchParamsTy &LaunchParams, uint32_t NumTeams[3],
    uint32_t NumThreads[3], uint32_t SharedMemorySize) {
  if (!isRecordingOrReplaying())
    return HandleTy{nullptr, false};

  // Register the instance and avoid recording if it is inactive or replaying.
  auto [Instance, First] =
      registerInstance(Kernel, NumTeams[0], NumThreads[0], SharedMemorySize);

  HandleTy Handle{&Instance, First};
  if (isReplaying() || !First)
    return Handle;

  if (auto Err = recordDescImpl(Kernel, Instance, KernelArgs, LaunchParams))
    return Err;

  if (auto Err = recordPrologueImpl(Kernel, Instance, KernelArgs, LaunchParams))
    return Err;

  return Handle;
}

Error RecordReplayTy::recordEpilogue(const GenericKernelTy &Kernel,
                                     HandleTy Handle) {
  if (!shouldRecordEpilogue() || !Handle.Active)
    return Plugin::success();

  return recordEpilogueImpl(Kernel, *Handle.Instance);
}

Error NativeRecordReplayTy::recordPrologueImpl(
    const GenericKernelTy &Kernel, const InstanceTy &Instance,
    const KernelArgsTy &KernelArgs, const KernelLaunchParamsTy &LaunchParams) {
  std::string SnapshotFilename = getFilename(Instance, "record_input");
  if (auto Err = recordSnapshot(SnapshotFilename))
    return Err;

  std::string GlobalsFilename = getFilename(Instance, "globals");
  if (auto Err = recordGlobals(GlobalsFilename))
    return Err;

  std::string ImageFilename = getFilename(Instance, "image");
  return recordImage(Kernel, ImageFilename);
}

Error NativeRecordReplayTy::recordEpilogueImpl(const GenericKernelTy &Kernel,
                                               const InstanceTy &Instance) {
  std::string SnapshotFilename =
      getFilename(Instance, isRecording() ? "record_output" : "replay_output");
  return recordSnapshot(SnapshotFilename);
}

Error NativeRecordReplayTy::recordDescImpl(
    const GenericKernelTy &Kernel, const InstanceTy &Instance,
    const KernelArgsTy &KernelArgs, const KernelLaunchParamsTy &LaunchParams) {
  json::Object JsonKernelInfo;
  JsonKernelInfo["Name"] = Kernel.getName();
  JsonKernelInfo["NumArgs"] = KernelArgs.NumArgs;
  JsonKernelInfo["NumTeamsClause"] = Instance.NumTeams;
  JsonKernelInfo["ThreadLimitClause"] = Instance.NumThreads;
  JsonKernelInfo["SharedMemorySize"] = Instance.SharedMemorySize;
  JsonKernelInfo["LoopTripCount"] = KernelArgs.Tripcount;
  JsonKernelInfo["DeviceId"] = Device.getDeviceId();
  JsonKernelInfo["VAllocAddr"] = (intptr_t)StartAddr;
  JsonKernelInfo["VAllocSize"] = TotalSize;

  json::Array JsonArgPtrs;
  for (uint32_t I = 0; I < KernelArgs.NumArgs; ++I)
    JsonArgPtrs.push_back((intptr_t)(*(void **)LaunchParams.Ptrs[I]));
  JsonKernelInfo["ArgPtrs"] = json::Value(std::move(JsonArgPtrs));

  json::Array JsonArgOffsets;
  for (uint32_t I = 0; I < KernelArgs.NumArgs; ++I)
    JsonArgOffsets.push_back(0);
  JsonKernelInfo["ArgOffsets"] = json::Value(std::move(JsonArgOffsets));

  std::string JsonFilename = getFilename(Instance, "json");
  std::error_code EC;
  raw_fd_ostream JsonOS(JsonFilename, EC);
  if (EC)
    return Plugin::error(ErrorCode::HOST_IO, "saving kernel json file");
  JsonOS << json::Value(std::move(JsonKernelInfo));
  JsonOS.close();
  return Plugin::success();
}

std::string NativeRecordReplayTy::getFilenameImpl(const InstanceTy &Instance,
                                                  StringRef Suffix,
                                                  bool IncludeDirectory) {
  std::filesystem::path Filepath = IncludeDirectory ? OutputDirectory : "";
  Filepath /= std::to_string(Instance.KernelHash) + "_" +
              std::to_string(Instance.LaunchConfigHash);
  Filepath.replace_extension(Suffix.data());
  return Filepath.string();
}

Error NativeRecordReplayTy::recordSnapshot(const std::string &Filename) {
  // Another thread may be allocating memory. The size can only increase.
  AllocationLock.lock();
  uint64_t RecordSize = CurrentSize;
  AllocationLock.unlock();

  ErrorOr<std::unique_ptr<WritableMemoryBuffer>> DeviceMemoryMB =
      WritableMemoryBuffer::getNewUninitMemBuffer(RecordSize);
  if (!DeviceMemoryMB)
    return Plugin::error(ErrorCode::OUT_OF_RESOURCES,
                         "creating MemoryBuffer for device memory");

  if (auto Err = Device.dataRetrieve(DeviceMemoryMB.get()->getBufferStart(),
                                     StartAddr, RecordSize, nullptr))
    return Err;

  StringRef DeviceMemory(DeviceMemoryMB.get()->getBufferStart(), RecordSize);

  std::error_code EC;
  raw_fd_ostream OS(Filename, EC);
  if (EC)
    return Plugin::error(ErrorCode::HOST_IO, "saving memory snapshot file");
  OS << DeviceMemory;
  OS.close();
  return Plugin::success();
}

Error NativeRecordReplayTy::recordImage(const GenericKernelTy &Kernel,
                                        const std::string &Filename) {
  std::error_code EC;
  raw_fd_ostream OS(Filename, EC);
  if (EC)
    return Plugin::error(ErrorCode::HOST_IO, "saving image file");
  OS << Kernel.getImage().getMemoryBuffer().getBuffer();
  OS.close();
  return Plugin::success();
}

Error NativeRecordReplayTy::recordGlobals(const std::string &Filename) {
  AllocationLock.lock();
  // Copy the globals into a local vector so we can read it safely from this
  // thread. This vector should have a few entries in general. No need to lock
  // the entire function.
  SmallVector<GlobalEntryTy> Globals = GlobalEntries;
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

  ErrorOr<std::unique_ptr<WritableMemoryBuffer>> GlobalsMB =
      WritableMemoryBuffer::getNewUninitMemBuffer(TotalSize);
  if (!GlobalsMB)
    return Plugin::error(ErrorCode::OUT_OF_RESOURCES,
                         "creating MemoryBuffer for globals memory");

  void *BufferPtr = GlobalsMB.get()->getBufferStart();
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
    BufferPtr = utils::advancePtr(BufferPtr, Global.Size);
  }
  assert(BufferPtr == GlobalsMB->get()->getBufferEnd() &&
         "Buffer over or under-filled.");
  assert(TotalSize == (uint64_t)utils::getPtrDiff(
                          BufferPtr, GlobalsMB->get()->getBufferStart()) &&
         "Buffer size mismatch.");

  StringRef GlobalsMemory(GlobalsMB.get()->getBufferStart(), TotalSize);
  std::error_code EC;
  raw_fd_ostream OS(Filename, EC);
  OS << GlobalsMemory;
  OS.close();
  return Plugin::success();
}
