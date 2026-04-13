#include "PluginInterface.h"

#include "Shared/APITypes.h"

#include "ErrorReporting.h"
#include "Shared/Utils.h"

#include "llvm/Support/Error.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdint>
#include <filesystem>
#include <functional>

using namespace llvm;
using namespace omp;
using namespace target;
using namespace plugin;
using namespace error;

Error RecordReplayTy::init(uint64_t MemSize, void *VAddr) {
  if (!VAddr)
    VAddr = Device.getSuggestedVirtualAddress();

  auto StartAddrOrErr = Device.allocateWithVirtualAddress(MemSize, VAddr);
  if (!StartAddrOrErr)
    return StartAddrOrErr.takeError();
  if (!*StartAddrOrErr)
    return Plugin::error(ErrorCode::OUT_OF_RESOURCES, "allocating memory");

  StartAddr = *StartAddrOrErr;
  TotalSize = MemSize;

  INFO(OMP_INFOTYPE_PLUGIN_KERNEL, Device.getDeviceId(),
       "Record initialized with starting address %p, "
       "memory size %lu bytes and status %s\n",
       StartAddr, TotalSize,
       Status == StatusTy::Recording ? "recording" : "replaying");

  return Plugin::success();
}

Error RecordReplayTy::deinit() {
  if (StartAddr)
    return Device.deallocateWithVirtualAddress(StartAddr, TotalSize);
  return Plugin::success();
}

std::pair<const RecordReplayTy::InstanceTy &, bool>
RecordReplayTy::registerInstance(StringRef KernelName, uint32_t NumTeams,
                                 uint32_t NumThreads,
                                 uint32_t SharedMemorySize) {
  std::lock_guard<std::mutex> LG(InstancesLock);
  auto [It, Inserted] =
      Instances.emplace(KernelName, NumTeams, NumThreads, SharedMemorySize);
  // Increase the number of occurrences.
  It->Occurrences += 1;
  return {*It, Inserted};
}

void *RecordReplayTy::allocate(uint64_t Size) {
  assert(StartAddr && "Expected memory has been pre-allocated");
  constexpr int Alignment = 16;
  // Assume alignment is a power of 2.
  int64_t AlignedSize = (Size + (Alignment - 1)) & (~(Alignment - 1));

  std::lock_guard<std::mutex> LG(AllocationLock);
  void *Alloc = (char *)StartAddr + CurrentSize;
  CurrentSize += AlignedSize;
  return Alloc;
}

Expected<RecordReplayTy::HandleTy> RecordReplayTy::recordPrologue(
    const GenericKernelTy &Kernel, const KernelArgsTy &KernelArgs,
    const KernelLaunchParamsTy &LaunchParams, uint32_t NumTeams[3],
    uint32_t NumThreads[3], uint32_t SharedMemorySize) {
  if (!isRecordingOrReplaying())
    return HandleTy{nullptr, false};

  // Register the instance and avoid recording if it is inactive or replaying.
  auto [Instance, First] = registerInstance(Kernel.getName(), NumTeams[0],
                                            NumThreads[0], SharedMemorySize);

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
  SmallString<128> SnapshotFilename = {Kernel.getName(), ".memory"};
  if (auto Err = recordSnapshot(SnapshotFilename))
    return Err;

  SmallString<128> GlobalsFilename = {Kernel.getName(), ".globals"};
  if (auto Err = recordGlobals(GlobalsFilename))
    return Err;

  SmallString<128> ImageFilename = {Kernel.getName(), ".image"};
  return recordImage(Kernel, ImageFilename);
}

Error NativeRecordReplayTy::recordEpilogueImpl(const GenericKernelTy &Kernel,
                                               const InstanceTy &Instance) {
  SmallString<128> SnapshotFilename = {
      Kernel.getName(),
      (isRecording() ? ".original.output" : ".replay.output")};
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

  SmallString<128> JsonFilename = {Kernel.getName(), ".json"};
  std::error_code EC;
  raw_fd_ostream JsonOS(JsonFilename.str(), EC);
  if (EC)
    return Plugin::error(ErrorCode::UNKNOWN, "saving kernel json file");
  JsonOS << json::Value(std::move(JsonKernelInfo));
  JsonOS.close();
  return Plugin::success();
}

Error NativeRecordReplayTy::recordSnapshot(StringRef Filename) {
  // Another thread may be allocating memory.
  AllocationLock.lock();
  uint64_t RecordSize = CurrentSize;
  AllocationLock.unlock();

  ErrorOr<std::unique_ptr<WritableMemoryBuffer>> DeviceMemoryMB =
      WritableMemoryBuffer::getNewUninitMemBuffer(RecordSize);
  if (!DeviceMemoryMB)
    return Plugin::error(ErrorCode::UNKNOWN,
                         "creating MemoryBuffer for device memory");

  if (auto Err = Device.dataRetrieve(DeviceMemoryMB.get()->getBufferStart(),
                                     StartAddr, RecordSize, nullptr))
    return Err;

  StringRef DeviceMemory(DeviceMemoryMB.get()->getBufferStart(), RecordSize);
  std::error_code EC;
  raw_fd_ostream OS(Filename, EC);
  if (EC)
    return Plugin::error(ErrorCode::UNKNOWN, "dumping memory to file");
  OS << DeviceMemory;
  OS.close();
  return Plugin::success();
}

Error NativeRecordReplayTy::recordImage(const GenericKernelTy &Kernel,
                                        StringRef Filename) {
  std::error_code EC;
  raw_fd_ostream OS(Filename, EC);
  if (EC)
    return Plugin::error(ErrorCode::UNKNOWN, "saving image");
  OS << Kernel.getImage().getMemoryBuffer().getBuffer();
  OS.close();
  return Plugin::success();
}

Error NativeRecordReplayTy::recordGlobals(StringRef Filename) {
  uint64_t TotalSize = 0;
  uint32_t NumGlobals = 0;

  for (auto &OffloadEntry : GlobalEntries) {
    if (!OffloadEntry.Size)
      continue;
    // Get the total size of the string and entry including the null byte.
    TotalSize += OffloadEntry.Size + sizeof(uint32_t) + sizeof(uint64_t) +
                 OffloadEntry.Name.length() + 1;
    NumGlobals++;
  }

  ErrorOr<std::unique_ptr<WritableMemoryBuffer>> GlobalsMB =
      WritableMemoryBuffer::getNewUninitMemBuffer(TotalSize);
  if (!GlobalsMB)
    return Plugin::error(ErrorCode::UNKNOWN,
                         "creating MemoryBuffer for globals memory");

  void *BufferPtr = GlobalsMB.get()->getBufferStart();
  *((uint32_t *)(BufferPtr)) = NumGlobals;
  BufferPtr = utils::advancePtr(BufferPtr, sizeof(uint32_t));

  for (auto &OffloadEntry : GlobalEntries) {
    if (!OffloadEntry.Size)
      continue;

    uint32_t NameLength = OffloadEntry.Name.length() + 1;
    *((uint32_t *)(BufferPtr)) = NameLength;
    BufferPtr = utils::advancePtr(BufferPtr, sizeof(uint32_t));

    *((uint64_t *)(BufferPtr)) = OffloadEntry.Size;
    BufferPtr = utils::advancePtr(BufferPtr, sizeof(uint64_t));

    memcpy(BufferPtr, OffloadEntry.Name.data(), NameLength);
    BufferPtr = utils::advancePtr(BufferPtr, NameLength);

    if (auto Err = Device.dataRetrieve(BufferPtr, OffloadEntry.Addr,
                                       OffloadEntry.Size, nullptr))
      return Err;
    BufferPtr = utils::advancePtr(BufferPtr, OffloadEntry.Size);
  }
  assert(BufferPtr == GlobalsMB->get()->getBufferEnd() &&
         "Buffer over/under-filled.");
  assert(TotalSize ==
             utils::getPtrDiff(BufferPtr, GlobalsMB->get()->getBufferStart()) &&
         "Buffer size mismatch");

  StringRef GlobalsMemory(GlobalsMB.get()->getBufferStart(), TotalSize);
  std::error_code EC;
  raw_fd_ostream OS(Filename, EC);
  OS << GlobalsMemory;
  OS.close();
  return Plugin::success();
}
