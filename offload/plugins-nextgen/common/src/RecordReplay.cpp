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
                                       uint32_t SharedMemorySize,
                                       KernelReplayOutcomeTy *ReplayOutcome)
    : Kernel(Kernel), NumTeams(NumTeams), NumThreads(NumThreads),
      SharedMemorySize(SharedMemorySize), ReplayOutcome(ReplayOutcome) {
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
       "memory size %" PRIu64 " bytes, and output directory in %s\n",
       Status == StatusTy::Recording ? "Record" : "Replay", StartAddr,
       TotalSize, OutputDirectory.string().c_str());

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
        << ", " << Inst->Kernel.getName() << ", " << Inst->getRecordedTimeNs()
        << ", " << Inst->Occurrences << "\n";
  *OutStream << "=== End Kernel Record Report ===\n";

  return Plugin::success();
}

std::pair<const RecordReplayTy::InstanceTy &, bool>
RecordReplayTy::registerInstance(const GenericKernelTy &Kernel,
                                 uint32_t NumTeams, uint32_t NumThreads,
                                 uint32_t SharedMemorySize,
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

Expected<RecordReplayTy::HandleTy> RecordReplayTy::recordPrologue(
    const GenericKernelTy &Kernel, const KernelArgsTy &KernelArgs,
    const KernelExtraArgsTy *KernelExtraArgs,
    const KernelLaunchParamsTy &LaunchParams, uint32_t NumTeams[3],
    uint32_t NumThreads[3], uint32_t SharedMemorySize) {
  if (!isRecordingOrReplaying())
    return HandleTy{nullptr, false};

  // Register the instance and avoid recording if it is inactive or replaying.
  auto [Instance, First] = registerInstance(
      Kernel, NumTeams[0], NumThreads[0], SharedMemorySize,
      (KernelExtraArgs) ? KernelExtraArgs->ReplayOutcome : nullptr);

  HandleTy Handle{&Instance, First};
  if (!First)
    return Handle;

  if (isRecording()) {
    if (auto Err = recordDescImpl(Kernel, Instance, KernelArgs, LaunchParams))
      return Err;

    if (auto Err =
            recordPrologueImpl(Kernel, Instance, KernelArgs, LaunchParams))
      return Err;
  }

  // Start the timer for the kernel execution.
  Instance.recordBeginTime();

  return Handle;
}

Error RecordReplayTy::recordEpilogue(const GenericKernelTy &Kernel,
                                     HandleTy Handle) {
  if (!Handle.Active)
    return Plugin::success();

  // Stop the timer for the kernel execution.
  const InstanceTy &Instance = *Handle.Instance;
  Instance.recordEndTime();

  if (shouldRecordEpilogue())
    if (auto Err = recordEpilogueImpl(Kernel, Instance))
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

Error NativeRecordReplayTy::recordPrologueImpl(
    const GenericKernelTy &Kernel, const InstanceTy &Instance,
    const KernelArgsTy &KernelArgs, const KernelLaunchParamsTy &LaunchParams) {
  SmallString<128> SnapshotFilename =
      getFilename(Instance, FileTy::PrologueSnapshot);
  if (auto Err = recordSnapshot(SnapshotFilename.c_str()))
    return Err;

  SmallString<128> GlobalsFilename = getFilename(Instance, FileTy::Globals);
  if (auto Err = recordGlobals(GlobalsFilename.c_str()))
    return Err;

  SmallString<128> ImageFilename = getFilename(Instance, FileTy::Program);

  SmallString<128> IRImageFilename = getFilename(Instance, FileTy::IRImage);
  return recordImage(Kernel, ImageFilename.c_str(), IRImageFilename.c_str());
}

Error NativeRecordReplayTy::recordEpilogueImpl(const GenericKernelTy &Kernel,
                                               const InstanceTy &Instance) {
  SmallString<128> SnapshotFilename =
      getFilename(Instance, FileTy::EpilogueSnapshot);
  return recordSnapshot(SnapshotFilename);
}

Error NativeRecordReplayTy::recordDescImpl(
    const GenericKernelTy &Kernel, const InstanceTy &Instance,
    const KernelArgsTy &KernelArgs, const KernelLaunchParamsTy &LaunchParams) {
  json::Object JsonKernelInfo;
  JsonKernelInfo["Name"] = Kernel.getName();
  JsonKernelInfo["NumArgs"] = KernelArgs.NumArgs;
  JsonKernelInfo["NumTeams"] = Instance.NumTeams;
  JsonKernelInfo["NumThreads"] = Instance.NumThreads;
  JsonKernelInfo["SharedMemorySize"] = Instance.SharedMemorySize;
  JsonKernelInfo["LoopTripCount"] = KernelArgs.Tripcount;
  JsonKernelInfo["DeviceId"] = Device.getDeviceId();
  JsonKernelInfo["VAllocAddr"] = (intptr_t)StartAddr;
  JsonKernelInfo["VAllocSize"] = TotalSize;

  // Export minimum and maximum for allowed number of teams. If zero, it means
  // there was no restriction provided by the program.
  uint32_t MinMaxBlocks = std::max(KernelArgs.UserNumBlocks[0], uint32_t(0));
  json::Array JsonTeamsLimits;
  JsonTeamsLimits.push_back(MinMaxBlocks);
  JsonTeamsLimits.push_back(MinMaxBlocks);
  JsonKernelInfo["TeamsLimits"] = json::Value(std::move(JsonTeamsLimits));

  // Export minimum and maximum for allowed number of threads. If zero, it means
  // there was no restriction provided by the program.
  uint32_t UserThreads = std::max(KernelArgs.UserThreadLimit[0], uint32_t(0));
  uint32_t MaxThreads = UserThreads
                            ? std::min(UserThreads, Kernel.getMaxThreads())
                            : Kernel.getMaxThreads();
  json::Array JsonThreadsLimits;
  JsonThreadsLimits.push_back(1);
  JsonThreadsLimits.push_back(MaxThreads);
  JsonKernelInfo["ThreadsLimits"] = json::Value(std::move(JsonThreadsLimits));

  json::Array JsonArgPtrs;
  for (uint32_t I = 0; I < KernelArgs.NumArgs; ++I)
    JsonArgPtrs.push_back((intptr_t)(*(void **)LaunchParams.Args[I]));
  JsonKernelInfo["ArgPtrs"] = json::Value(std::move(JsonArgPtrs));

  json::Array JsonArgOffsets;
  for (uint32_t I = 0; I < KernelArgs.NumArgs; ++I)
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

Error NativeRecordReplayTy::recordSnapshot(StringRef Filename) {
  // Another thread may be allocating memory. The size can only increase.
  AllocationLock.lock();
  uint64_t RecordSize = CurrentSize;
  AllocationLock.unlock();

  std::unique_ptr<WritableMemoryBuffer> DeviceMB;
  if (RecordSize) {
    DeviceMB = WritableMemoryBuffer::getNewUninitMemBuffer(RecordSize);
    if (!DeviceMB)
      return Plugin::error(ErrorCode::OUT_OF_RESOURCES,
                           "creating MemoryBuffer for device memory");

    if (auto Err = Device.dataRetrieve(DeviceMB->getBufferStart(), StartAddr,
                                       RecordSize, nullptr))
      return Err;
  }

  std::error_code EC;
  raw_fd_ostream OS(Filename, EC);
  if (EC)
    return Plugin::error(ErrorCode::HOST_IO, "saving memory snapshot file");
  if (DeviceMB)
    OS.write(DeviceMB->getBufferStart(), RecordSize);
  OS.close();
  return Plugin::success();
}

Error NativeRecordReplayTy::recordImage(const GenericKernelTy &Kernel,
                                        StringRef Filename,
                                        StringRef IRImageFilename) {
  std::error_code EC;
  raw_fd_ostream OS(Filename, EC);
  if (EC)
    return Plugin::error(ErrorCode::HOST_IO, "saving image file");
  OS << Kernel.getImage().getMemoryBuffer().getBuffer();
  OS.close();

  if (!IRImageFilename.empty() && Kernel.getImage().hasIRImage()) {
    raw_fd_ostream IROS(IRImageFilename, EC);
    if (EC)
      return Plugin::error(ErrorCode::HOST_IO, "saving IR image file");
    IROS << Kernel.getImage().getIRImageMemoryBuffer().getBuffer();
    IROS.close();
  }

  return Plugin::success();
}

Error NativeRecordReplayTy::recordGlobals(StringRef Filename) {
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
