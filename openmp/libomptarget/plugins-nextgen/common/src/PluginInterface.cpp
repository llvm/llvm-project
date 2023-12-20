//===- PluginInterface.cpp - Target independent plugin device interface ---===//
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
#include "Shared/Debug.h"
#include "Shared/Environment.h"
#include "Shared/PluginAPI.h"

#include "GlobalHandler.h"
#include "JIT.h"
#include "Utils/ELF.h"
#include "omptarget.h"

#ifdef OMPT_SUPPORT
#include "OpenMP/OMPT/Callback.h"
#include "omp-tools.h"
#endif

#include "llvm/Frontend/OpenMP/OMPConstants.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/MemoryBuffer.h"

#include <cstdint>
#include <limits>

using namespace llvm;
using namespace omp;
using namespace target;
using namespace plugin;

GenericPluginTy *Plugin::SpecificPlugin = nullptr;

// TODO: Fix any thread safety issues for multi-threaded kernel recording.
struct RecordReplayTy {

  // Describes the state of the record replay mechanism.
  enum RRStatusTy { RRDeactivated = 0, RRRecording, RRReplaying };

private:
  // Memory pointers for recording, replaying memory.
  void *MemoryStart = nullptr;
  void *MemoryPtr = nullptr;
  size_t MemorySize = 0;
  size_t TotalSize = 0;
  GenericDeviceTy *Device = nullptr;
  std::mutex AllocationLock;

  RRStatusTy Status = RRDeactivated;
  bool ReplaySaveOutput = false;
  bool UsedVAMap = false;
  uintptr_t MemoryOffset = 0;

  void *suggestAddress(uint64_t MaxMemoryAllocation) {
    // Get a valid pointer address for this system
    void *Addr =
        Device->allocate(1024, /* HstPtr */ nullptr, TARGET_ALLOC_DEFAULT);
    Device->free(Addr);
    // Align Address to MaxMemoryAllocation
    Addr = (void *)alignPtr((Addr), MaxMemoryAllocation);
    return Addr;
  }

  Error preAllocateVAMemory(uint64_t MaxMemoryAllocation, void *VAddr) {
    size_t ASize = MaxMemoryAllocation;

    if (!VAddr && isRecording())
      VAddr = suggestAddress(MaxMemoryAllocation);

    DP("Request %ld bytes allocated at %p\n", MaxMemoryAllocation, VAddr);

    if (auto Err = Device->memoryVAMap(&MemoryStart, VAddr, &ASize))
      return Err;

    if (isReplaying() && VAddr != MemoryStart) {
      return Plugin::error("Record-Replay cannot assign the"
                           "requested recorded address (%p, %p)",
                           VAddr, MemoryStart);
    }

    INFO(OMP_INFOTYPE_PLUGIN_KERNEL, Device->getDeviceId(),
         "Allocated %" PRIu64 " bytes at %p for replay.\n", ASize, MemoryStart);

    MemoryPtr = MemoryStart;
    MemorySize = 0;
    TotalSize = ASize;
    UsedVAMap = true;
    return Plugin::success();
  }

  Error preAllocateHeuristic(uint64_t MaxMemoryAllocation,
                             uint64_t RequiredMemoryAllocation, void *VAddr) {
    const size_t MAX_MEMORY_ALLOCATION = MaxMemoryAllocation;
    constexpr size_t STEP = 1024 * 1024 * 1024ULL;
    MemoryStart = nullptr;
    for (TotalSize = MAX_MEMORY_ALLOCATION; TotalSize > 0; TotalSize -= STEP) {
      MemoryStart = Device->allocate(TotalSize, /* HstPtr */ nullptr,
                                     TARGET_ALLOC_DEFAULT);
      if (MemoryStart)
        break;
    }
    if (!MemoryStart)
      return Plugin::error("Allocating record/replay memory");

    if (VAddr && VAddr != MemoryStart)
      MemoryOffset = uintptr_t(VAddr) - uintptr_t(MemoryStart);

    MemoryPtr = MemoryStart;
    MemorySize = 0;

    // Check if we need adjustment.
    if (MemoryOffset > 0 &&
        TotalSize >= RequiredMemoryAllocation + MemoryOffset) {
      // If we are off but "before" the required address and with enough space,
      // we just "allocate" the offset to match the required address.
      MemoryPtr = (char *)MemoryPtr + MemoryOffset;
      MemorySize += MemoryOffset;
      MemoryOffset = 0;
      assert(MemoryPtr == VAddr && "Expected offset adjustment to work");
    } else if (MemoryOffset) {
      // If we are off and in a situation we cannot just "waste" memory to force
      // a match, we hope adjusting the arguments is sufficient.
      REPORT(
          "WARNING Failed to allocate replay memory at required location %p, "
          "got %p, trying to offset argument pointers by %" PRIi64 "\n",
          VAddr, MemoryStart, MemoryOffset);
    }

    INFO(OMP_INFOTYPE_PLUGIN_KERNEL, Device->getDeviceId(),
         "Allocated %" PRIu64 " bytes at %p for replay.\n", TotalSize,
         MemoryStart);

    return Plugin::success();
  }

  Error preallocateDeviceMemory(uint64_t DeviceMemorySize, void *ReqVAddr) {
    if (Device->supportVAManagement()) {
      auto Err = preAllocateVAMemory(DeviceMemorySize, ReqVAddr);
      if (Err) {
        REPORT("WARNING VA mapping failed, fallback to heuristic: "
               "(Error: %s)\n",
               toString(std::move(Err)).data());
      }
    }

    uint64_t DevMemSize;
    if (Device->getDeviceMemorySize(DevMemSize))
      return Plugin::error("Cannot determine Device Memory Size");

    return preAllocateHeuristic(DevMemSize, DeviceMemorySize, ReqVAddr);
  }

  void dumpDeviceMemory(StringRef Filename) {
    ErrorOr<std::unique_ptr<WritableMemoryBuffer>> DeviceMemoryMB =
        WritableMemoryBuffer::getNewUninitMemBuffer(MemorySize);
    if (!DeviceMemoryMB)
      report_fatal_error("Error creating MemoryBuffer for device memory");

    auto Err = Device->dataRetrieve(DeviceMemoryMB.get()->getBufferStart(),
                                    MemoryStart, MemorySize, nullptr);
    if (Err)
      report_fatal_error("Error retrieving data for target pointer");

    StringRef DeviceMemory(DeviceMemoryMB.get()->getBufferStart(), MemorySize);
    std::error_code EC;
    raw_fd_ostream OS(Filename, EC);
    if (EC)
      report_fatal_error("Error dumping memory to file " + Filename + " :" +
                         EC.message());
    OS << DeviceMemory;
    OS.close();
  }

public:
  bool isRecording() const { return Status == RRStatusTy::RRRecording; }
  bool isReplaying() const { return Status == RRStatusTy::RRReplaying; }
  bool isRecordingOrReplaying() const {
    return (Status != RRStatusTy::RRDeactivated);
  }
  void setStatus(RRStatusTy Status) { this->Status = Status; }
  bool isSaveOutputEnabled() const { return ReplaySaveOutput; }

  void saveImage(const char *Name, const DeviceImageTy &Image) {
    SmallString<128> ImageName = {Name, ".image"};
    std::error_code EC;
    raw_fd_ostream OS(ImageName, EC);
    if (EC)
      report_fatal_error("Error saving image : " + StringRef(EC.message()));
    if (const auto *TgtImageBitcode = Image.getTgtImageBitcode()) {
      size_t Size =
          getPtrDiff(TgtImageBitcode->ImageEnd, TgtImageBitcode->ImageStart);
      MemoryBufferRef MBR = MemoryBufferRef(
          StringRef((const char *)TgtImageBitcode->ImageStart, Size), "");
      OS << MBR.getBuffer();
    } else {
      OS << Image.getMemoryBuffer().getBuffer();
    }
    OS.close();
  }

  void dumpGlobals(StringRef Filename, DeviceImageTy &Image) {
    int32_t Size = 0;

    for (auto &OffloadEntry : Image.getOffloadEntryTable()) {
      if (!OffloadEntry.size)
        continue;
      Size += std::strlen(OffloadEntry.name) + /* '\0' */ 1 +
              /* OffloadEntry.size value */ sizeof(uint32_t) +
              OffloadEntry.size;
    }

    ErrorOr<std::unique_ptr<WritableMemoryBuffer>> GlobalsMB =
        WritableMemoryBuffer::getNewUninitMemBuffer(Size);
    if (!GlobalsMB)
      report_fatal_error("Error creating MemoryBuffer for globals memory");

    void *BufferPtr = GlobalsMB.get()->getBufferStart();
    for (auto &OffloadEntry : Image.getOffloadEntryTable()) {
      if (!OffloadEntry.size)
        continue;

      int32_t NameLength = std::strlen(OffloadEntry.name) + 1;
      memcpy(BufferPtr, OffloadEntry.name, NameLength);
      BufferPtr = advanceVoidPtr(BufferPtr, NameLength);

      *((uint32_t *)(BufferPtr)) = OffloadEntry.size;
      BufferPtr = advanceVoidPtr(BufferPtr, sizeof(uint32_t));

      auto Err = Plugin::success();
      {
        if (auto Err = Device->dataRetrieve(BufferPtr, OffloadEntry.addr,
                                            OffloadEntry.size, nullptr))
          report_fatal_error("Error retrieving data for global");
      }
      if (Err)
        report_fatal_error("Error retrieving data for global");
      BufferPtr = advanceVoidPtr(BufferPtr, OffloadEntry.size);
    }
    assert(BufferPtr == GlobalsMB->get()->getBufferEnd() &&
           "Buffer over/under-filled.");
    assert(Size == getPtrDiff(BufferPtr, GlobalsMB->get()->getBufferStart()) &&
           "Buffer size mismatch");

    StringRef GlobalsMemory(GlobalsMB.get()->getBufferStart(), Size);
    std::error_code EC;
    raw_fd_ostream OS(Filename, EC);
    OS << GlobalsMemory;
    OS.close();
  }

  void saveKernelDescr(const char *Name, void **ArgPtrs, int32_t NumArgs,
                       uint64_t NumTeamsClause, uint32_t ThreadLimitClause,
                       uint64_t LoopTripCount) {
    json::Object JsonKernelInfo;
    JsonKernelInfo["Name"] = Name;
    JsonKernelInfo["NumArgs"] = NumArgs;
    JsonKernelInfo["NumTeamsClause"] = NumTeamsClause;
    JsonKernelInfo["ThreadLimitClause"] = ThreadLimitClause;
    JsonKernelInfo["LoopTripCount"] = LoopTripCount;
    JsonKernelInfo["DeviceMemorySize"] = MemorySize;
    JsonKernelInfo["DeviceId"] = Device->getDeviceId();
    JsonKernelInfo["BumpAllocVAStart"] = (intptr_t)MemoryStart;

    json::Array JsonArgPtrs;
    for (int I = 0; I < NumArgs; ++I)
      JsonArgPtrs.push_back((intptr_t)ArgPtrs[I]);
    JsonKernelInfo["ArgPtrs"] = json::Value(std::move(JsonArgPtrs));

    json::Array JsonArgOffsets;
    for (int I = 0; I < NumArgs; ++I)
      JsonArgOffsets.push_back(0);
    JsonKernelInfo["ArgOffsets"] = json::Value(std::move(JsonArgOffsets));

    SmallString<128> JsonFilename = {Name, ".json"};
    std::error_code EC;
    raw_fd_ostream JsonOS(JsonFilename.str(), EC);
    if (EC)
      report_fatal_error("Error saving kernel json file : " +
                         StringRef(EC.message()));
    JsonOS << json::Value(std::move(JsonKernelInfo));
    JsonOS.close();
  }

  void saveKernelInput(const char *Name, DeviceImageTy &Image) {
    SmallString<128> GlobalsFilename = {Name, ".globals"};
    dumpGlobals(GlobalsFilename, Image);

    SmallString<128> MemoryFilename = {Name, ".memory"};
    dumpDeviceMemory(MemoryFilename);
  }

  void saveKernelOutputInfo(const char *Name) {
    SmallString<128> OutputFilename = {
        Name, (isRecording() ? ".original.output" : ".replay.output")};
    dumpDeviceMemory(OutputFilename);
  }

  void *alloc(uint64_t Size) {
    assert(MemoryStart && "Expected memory has been pre-allocated");
    void *Alloc = nullptr;
    constexpr int Alignment = 16;
    // Assumes alignment is a power of 2.
    int64_t AlignedSize = (Size + (Alignment - 1)) & (~(Alignment - 1));
    std::lock_guard<std::mutex> LG(AllocationLock);
    Alloc = MemoryPtr;
    MemoryPtr = (char *)MemoryPtr + AlignedSize;
    MemorySize += AlignedSize;
    DP("Memory Allocator return " DPxMOD "\n", DPxPTR(Alloc));
    return Alloc;
  }

  Error init(GenericDeviceTy *Device, uint64_t MemSize, void *VAddr,
             RRStatusTy Status, bool SaveOutput, uint64_t &ReqPtrArgOffset) {
    this->Device = Device;
    this->Status = Status;
    this->ReplaySaveOutput = SaveOutput;

    if (auto Err = preallocateDeviceMemory(MemSize, VAddr))
      return Err;

    INFO(OMP_INFOTYPE_PLUGIN_KERNEL, Device->getDeviceId(),
         "Record Replay Initialized (%p)"
         " as starting address, %lu Memory Size"
         " and set on status %s\n",
         MemoryStart, TotalSize,
         Status == RRStatusTy::RRRecording ? "Recording" : "Replaying");

    // Tell the user to offset pointer arguments as the memory allocation does
    // not match.
    ReqPtrArgOffset = MemoryOffset;
    return Plugin::success();
  }

  void deinit() {
    if (UsedVAMap) {
      if (auto Err = Device->memoryVAUnMap(MemoryStart, TotalSize))
        report_fatal_error("Error on releasing virtual memory space");
    } else {
      Device->free(MemoryStart);
    }
  }
};

static RecordReplayTy RecordReplay;

// Extract the mapping of host function pointers to device function pointers
// from the entry table. Functions marked as 'indirect' in OpenMP will have
// offloading entries generated for them which map the host's function pointer
// to a global containing the corresponding function pointer on the device.
static Expected<std::pair<void *, uint64_t>>
setupIndirectCallTable(GenericPluginTy &Plugin, GenericDeviceTy &Device,
                       DeviceImageTy &Image) {
  GenericGlobalHandlerTy &Handler = Plugin.getGlobalHandler();

  llvm::ArrayRef<__tgt_offload_entry> Entries(Image.getTgtImage()->EntriesBegin,
                                              Image.getTgtImage()->EntriesEnd);
  llvm::SmallVector<std::pair<void *, void *>> IndirectCallTable;
  for (const auto &Entry : Entries) {
    if (Entry.size == 0 || !(Entry.flags & OMP_DECLARE_TARGET_INDIRECT))
      continue;

    assert(Entry.size == sizeof(void *) && "Global not a function pointer?");
    auto &[HstPtr, DevPtr] = IndirectCallTable.emplace_back();

    GlobalTy DeviceGlobal(Entry.name, Entry.size);
    if (auto Err =
            Handler.getGlobalMetadataFromDevice(Device, Image, DeviceGlobal))
      return std::move(Err);

    HstPtr = Entry.addr;
    if (auto Err = Device.dataRetrieve(&DevPtr, DeviceGlobal.getPtr(),
                                       Entry.size, nullptr))
      return std::move(Err);
  }

  // If we do not have any indirect globals we exit early.
  if (IndirectCallTable.empty())
    return std::pair{nullptr, 0};

  // Sort the array to allow for more efficient lookup of device pointers.
  llvm::sort(IndirectCallTable,
             [](const auto &x, const auto &y) { return x.first < y.first; });

  uint64_t TableSize =
      IndirectCallTable.size() * sizeof(std::pair<void *, void *>);
  void *DevicePtr = Device.allocate(TableSize, nullptr, TARGET_ALLOC_DEVICE);
  if (auto Err = Device.dataSubmit(DevicePtr, IndirectCallTable.data(),
                                   TableSize, nullptr))
    return std::move(Err);
  return std::pair<void *, uint64_t>(DevicePtr, IndirectCallTable.size());
}

AsyncInfoWrapperTy::AsyncInfoWrapperTy(GenericDeviceTy &Device,
                                       __tgt_async_info *AsyncInfoPtr)
    : Device(Device),
      AsyncInfoPtr(AsyncInfoPtr ? AsyncInfoPtr : &LocalAsyncInfo) {}

void AsyncInfoWrapperTy::finalize(Error &Err) {
  assert(AsyncInfoPtr && "AsyncInfoWrapperTy already finalized");

  // If we used a local async info object we want synchronous behavior. In that
  // case, and assuming the current status code is correct, we will synchronize
  // explicitly when the object is deleted. Update the error with the result of
  // the synchronize operation.
  if (AsyncInfoPtr == &LocalAsyncInfo && LocalAsyncInfo.Queue && !Err)
    Err = Device.synchronize(&LocalAsyncInfo);

  // Invalidate the wrapper object.
  AsyncInfoPtr = nullptr;
}

Error GenericKernelTy::init(GenericDeviceTy &GenericDevice,
                            DeviceImageTy &Image) {

  ImagePtr = &Image;

  // Retrieve kernel environment object for the kernel.
  GlobalTy KernelEnv(std::string(Name) + "_kernel_environment",
                     sizeof(KernelEnvironment), &KernelEnvironment);
  GenericGlobalHandlerTy &GHandler = Plugin::get().getGlobalHandler();
  if (auto Err =
          GHandler.readGlobalFromImage(GenericDevice, *ImagePtr, KernelEnv)) {
    [[maybe_unused]] std::string ErrStr = toString(std::move(Err));
    DP("Failed to read kernel environment for '%s': %s\n"
       "Using default SPMD (2) execution mode\n",
       Name, ErrStr.data());
    assert(KernelEnvironment.Configuration.ReductionDataSize == 0 &&
           "Default initialization failed.");
    IsBareKernel = true;
  }

  // Max = Config.Max > 0 ? min(Config.Max, Device.Max) : Device.Max;
  MaxNumThreads = KernelEnvironment.Configuration.MaxThreads > 0
                      ? std::min(KernelEnvironment.Configuration.MaxThreads,
                                 int32_t(GenericDevice.getThreadLimit()))
                      : GenericDevice.getThreadLimit();

  // Pref = Config.Pref > 0 ? max(Config.Pref, Device.Pref) : Device.Pref;
  PreferredNumThreads =
      KernelEnvironment.Configuration.MinThreads > 0
          ? std::max(KernelEnvironment.Configuration.MinThreads,
                     int32_t(GenericDevice.getDefaultNumThreads()))
          : GenericDevice.getDefaultNumThreads();

  return initImpl(GenericDevice, Image);
}

Expected<KernelLaunchEnvironmentTy *>
GenericKernelTy::getKernelLaunchEnvironment(
    GenericDeviceTy &GenericDevice,
    AsyncInfoWrapperTy &AsyncInfoWrapper) const {
  // Ctor/Dtor have no arguments, replaying uses the original kernel launch
  // environment.
  if (isCtorOrDtor() || RecordReplay.isReplaying())
    return nullptr;

  if (!KernelEnvironment.Configuration.ReductionDataSize ||
      !KernelEnvironment.Configuration.ReductionBufferLength)
    return reinterpret_cast<KernelLaunchEnvironmentTy *>(~0);

  // TODO: Check if the kernel needs a launch environment.
  auto AllocOrErr = GenericDevice.dataAlloc(sizeof(KernelLaunchEnvironmentTy),
                                            /*HostPtr=*/nullptr,
                                            TargetAllocTy::TARGET_ALLOC_DEVICE);
  if (!AllocOrErr)
    return AllocOrErr.takeError();

  // Remember to free the memory later.
  AsyncInfoWrapper.freeAllocationAfterSynchronization(*AllocOrErr);

  /// Use the KLE in the __tgt_async_info to ensure a stable address for the
  /// async data transfer.
  auto &LocalKLE = (*AsyncInfoWrapper).KernelLaunchEnvironment;
  LocalKLE = KernelLaunchEnvironment;
  {
    auto AllocOrErr = GenericDevice.dataAlloc(
        KernelEnvironment.Configuration.ReductionDataSize *
            KernelEnvironment.Configuration.ReductionBufferLength,
        /*HostPtr=*/nullptr, TargetAllocTy::TARGET_ALLOC_DEVICE);
    if (!AllocOrErr)
      return AllocOrErr.takeError();
    LocalKLE.ReductionBuffer = *AllocOrErr;
    // Remember to free the memory later.
    AsyncInfoWrapper.freeAllocationAfterSynchronization(*AllocOrErr);
  }

  INFO(OMP_INFOTYPE_DATA_TRANSFER, GenericDevice.getDeviceId(),
       "Copying data from host to device, HstPtr=" DPxMOD ", TgtPtr=" DPxMOD
       ", Size=%" PRId64 ", Name=KernelLaunchEnv\n",
       DPxPTR(&LocalKLE), DPxPTR(*AllocOrErr),
       sizeof(KernelLaunchEnvironmentTy));

  auto Err = GenericDevice.dataSubmit(*AllocOrErr, &LocalKLE,
                                      sizeof(KernelLaunchEnvironmentTy),
                                      AsyncInfoWrapper);
  if (Err)
    return Err;
  return static_cast<KernelLaunchEnvironmentTy *>(*AllocOrErr);
}

Error GenericKernelTy::printLaunchInfo(GenericDeviceTy &GenericDevice,
                                       KernelArgsTy &KernelArgs,
                                       uint32_t NumThreads,
                                       uint64_t NumBlocks) const {
  INFO(OMP_INFOTYPE_PLUGIN_KERNEL, GenericDevice.getDeviceId(),
       "Launching kernel %s with %" PRIu64
       " blocks and %d threads in %s mode\n",
       getName(), NumBlocks, NumThreads, getExecutionModeName());
  return printLaunchInfoDetails(GenericDevice, KernelArgs, NumThreads,
                                NumBlocks);
}

Error GenericKernelTy::printLaunchInfoDetails(GenericDeviceTy &GenericDevice,
                                              KernelArgsTy &KernelArgs,
                                              uint32_t NumThreads,
                                              uint64_t NumBlocks) const {
  return Plugin::success();
}

Error GenericKernelTy::launch(GenericDeviceTy &GenericDevice, void **ArgPtrs,
                              ptrdiff_t *ArgOffsets, KernelArgsTy &KernelArgs,
                              AsyncInfoWrapperTy &AsyncInfoWrapper) const {
  llvm::SmallVector<void *, 16> Args;
  llvm::SmallVector<void *, 16> Ptrs;

  auto KernelLaunchEnvOrErr =
      getKernelLaunchEnvironment(GenericDevice, AsyncInfoWrapper);
  if (!KernelLaunchEnvOrErr)
    return KernelLaunchEnvOrErr.takeError();

  void *KernelArgsPtr =
      prepareArgs(GenericDevice, ArgPtrs, ArgOffsets, KernelArgs.NumArgs, Args,
                  Ptrs, *KernelLaunchEnvOrErr);

  uint32_t NumThreads = getNumThreads(GenericDevice, KernelArgs.ThreadLimit);
  uint64_t NumBlocks =
      getNumBlocks(GenericDevice, KernelArgs.NumTeams, KernelArgs.Tripcount,
                   NumThreads, KernelArgs.ThreadLimit[0] > 0);

  // Record the kernel description after we modified the argument count and num
  // blocks/threads.
  if (RecordReplay.isRecording()) {
    RecordReplay.saveImage(getName(), getImage());
    RecordReplay.saveKernelInput(getName(), getImage());
    RecordReplay.saveKernelDescr(getName(), Ptrs.data(), KernelArgs.NumArgs,
                                 NumBlocks, NumThreads, KernelArgs.Tripcount);
  }

  if (auto Err =
          printLaunchInfo(GenericDevice, KernelArgs, NumThreads, NumBlocks))
    return Err;

  return launchImpl(GenericDevice, NumThreads, NumBlocks, KernelArgs,
                    KernelArgsPtr, AsyncInfoWrapper);
}

void *GenericKernelTy::prepareArgs(
    GenericDeviceTy &GenericDevice, void **ArgPtrs, ptrdiff_t *ArgOffsets,
    uint32_t &NumArgs, llvm::SmallVectorImpl<void *> &Args,
    llvm::SmallVectorImpl<void *> &Ptrs,
    KernelLaunchEnvironmentTy *KernelLaunchEnvironment) const {
  if (isCtorOrDtor())
    return nullptr;

  uint32_t KLEOffset = !!KernelLaunchEnvironment;
  NumArgs += KLEOffset;

  Args.resize(NumArgs);
  Ptrs.resize(NumArgs);

  if (KernelLaunchEnvironment) {
    Ptrs[0] = KernelLaunchEnvironment;
    Args[0] = &Ptrs[0];
  }

  for (int I = KLEOffset; I < NumArgs; ++I) {
    Ptrs[I] =
        (void *)((intptr_t)ArgPtrs[I - KLEOffset] + ArgOffsets[I - KLEOffset]);
    Args[I] = &Ptrs[I];
  }
  return &Args[0];
}

uint32_t GenericKernelTy::getNumThreads(GenericDeviceTy &GenericDevice,
                                        uint32_t ThreadLimitClause[3]) const {
  assert(ThreadLimitClause[1] == 0 && ThreadLimitClause[2] == 0 &&
         "Multi dimensional launch not supported yet.");

  if (IsBareKernel && ThreadLimitClause[0] > 0)
    return ThreadLimitClause[0];

  if (ThreadLimitClause[0] > 0 && isGenericMode())
    ThreadLimitClause[0] += GenericDevice.getWarpSize();

  return std::min(MaxNumThreads, (ThreadLimitClause[0] > 0)
                                     ? ThreadLimitClause[0]
                                     : PreferredNumThreads);
}

uint64_t GenericKernelTy::getNumBlocks(GenericDeviceTy &GenericDevice,
                                       uint32_t NumTeamsClause[3],
                                       uint64_t LoopTripCount,
                                       uint32_t &NumThreads,
                                       bool IsNumThreadsFromUser) const {
  assert(NumTeamsClause[1] == 0 && NumTeamsClause[2] == 0 &&
         "Multi dimensional launch not supported yet.");

  if (IsBareKernel && NumTeamsClause[0] > 0)
    return NumTeamsClause[0];

  if (NumTeamsClause[0] > 0) {
    // TODO: We need to honor any value and consequently allow more than the
    // block limit. For this we might need to start multiple kernels or let the
    // blocks start again until the requested number has been started.
    return std::min(NumTeamsClause[0], GenericDevice.getBlockLimit());
  }

  uint64_t DefaultNumBlocks = GenericDevice.getDefaultNumBlocks();
  uint64_t TripCountNumBlocks = std::numeric_limits<uint64_t>::max();
  if (LoopTripCount > 0) {
    if (isSPMDMode()) {
      // We have a combined construct, i.e. `target teams distribute
      // parallel for [simd]`. We launch so many teams so that each thread
      // will execute one iteration of the loop; rounded up to the nearest
      // integer. However, if that results in too few teams, we artificially
      // reduce the thread count per team to increase the outer parallelism.
      auto MinThreads = GenericDevice.getMinThreadsForLowTripCountLoop();
      MinThreads = std::min(MinThreads, NumThreads);

      // Honor the thread_limit clause; only lower the number of threads.
      [[maybe_unused]] auto OldNumThreads = NumThreads;
      if (LoopTripCount >= DefaultNumBlocks * NumThreads ||
          IsNumThreadsFromUser) {
        // Enough parallelism for teams and threads.
        TripCountNumBlocks = ((LoopTripCount - 1) / NumThreads) + 1;
        assert(IsNumThreadsFromUser ||
               TripCountNumBlocks >= DefaultNumBlocks &&
                   "Expected sufficient outer parallelism.");
      } else if (LoopTripCount >= DefaultNumBlocks * MinThreads) {
        // Enough parallelism for teams, limit threads.

        // This case is hard; for now, we force "full warps":
        // First, compute a thread count assuming DefaultNumBlocks.
        auto NumThreadsDefaultBlocks =
            (LoopTripCount + DefaultNumBlocks - 1) / DefaultNumBlocks;
        // Now get a power of two that is larger or equal.
        auto NumThreadsDefaultBlocksP2 =
            llvm::PowerOf2Ceil(NumThreadsDefaultBlocks);
        // Do not increase a thread limit given be the user.
        NumThreads = std::min(NumThreads, uint32_t(NumThreadsDefaultBlocksP2));
        assert(NumThreads >= MinThreads &&
               "Expected sufficient inner parallelism.");
        TripCountNumBlocks = ((LoopTripCount - 1) / NumThreads) + 1;
      } else {
        // Not enough parallelism for teams and threads, limit both.
        NumThreads = std::min(NumThreads, MinThreads);
        TripCountNumBlocks = ((LoopTripCount - 1) / NumThreads) + 1;
      }

      assert(NumThreads * TripCountNumBlocks >= LoopTripCount &&
             "Expected sufficient parallelism");
      assert(OldNumThreads >= NumThreads &&
             "Number of threads cannot be increased!");
    } else {
      assert((isGenericMode() || isGenericSPMDMode()) &&
             "Unexpected execution mode!");
      // If we reach this point, then we have a non-combined construct, i.e.
      // `teams distribute` with a nested `parallel for` and each team is
      // assigned one iteration of the `distribute` loop. E.g.:
      //
      // #pragma omp target teams distribute
      // for(...loop_tripcount...) {
      //   #pragma omp parallel for
      //   for(...) {}
      // }
      //
      // Threads within a team will execute the iterations of the `parallel`
      // loop.
      TripCountNumBlocks = LoopTripCount;
    }
  }
  // If the loops are long running we rather reuse blocks than spawn too many.
  uint32_t PreferredNumBlocks = std::min(TripCountNumBlocks, DefaultNumBlocks);
  return std::min(PreferredNumBlocks, GenericDevice.getBlockLimit());
}

GenericDeviceTy::GenericDeviceTy(int32_t DeviceId, int32_t NumDevices,
                                 const llvm::omp::GV &OMPGridValues)
    : MemoryManager(nullptr), OMP_TeamLimit("OMP_TEAM_LIMIT"),
      OMP_NumTeams("OMP_NUM_TEAMS"),
      OMP_TeamsThreadLimit("OMP_TEAMS_THREAD_LIMIT"),
      OMPX_DebugKind("LIBOMPTARGET_DEVICE_RTL_DEBUG"),
      OMPX_SharedMemorySize("LIBOMPTARGET_SHARED_MEMORY_SIZE"),
      // Do not initialize the following two envars since they depend on the
      // device initialization. These cannot be consulted until the device is
      // initialized correctly. We intialize them in GenericDeviceTy::init().
      OMPX_TargetStackSize(), OMPX_TargetHeapSize(),
      // By default, the initial number of streams and events is 1.
      OMPX_InitialNumStreams("LIBOMPTARGET_NUM_INITIAL_STREAMS", 1),
      OMPX_InitialNumEvents("LIBOMPTARGET_NUM_INITIAL_EVENTS", 1),
      DeviceId(DeviceId), GridValues(OMPGridValues),
      PeerAccesses(NumDevices, PeerAccessState::PENDING), PeerAccessesLock(),
      PinnedAllocs(*this), RPCServer(nullptr) {
#ifdef OMPT_SUPPORT
  OmptInitialized.store(false);
  // Bind the callbacks to this device's member functions
#define bindOmptCallback(Name, Type, Code)                                     \
  if (ompt::Initialized && ompt::lookupCallbackByCode) {                       \
    ompt::lookupCallbackByCode((ompt_callbacks_t)(Code),                       \
                               ((ompt_callback_t *)&(Name##_fn)));             \
    DP("OMPT: class bound %s=%p\n", #Name, ((void *)(uint64_t)Name##_fn));     \
  }

  FOREACH_OMPT_DEVICE_EVENT(bindOmptCallback);
#undef bindOmptCallback

#endif
}

Error GenericDeviceTy::init(GenericPluginTy &Plugin) {
  if (auto Err = initImpl(Plugin))
    return Err;

#ifdef OMPT_SUPPORT
  if (ompt::Initialized) {
    bool ExpectedStatus = false;
    if (OmptInitialized.compare_exchange_strong(ExpectedStatus, true))
      performOmptCallback(device_initialize,
                          /* device_num */ DeviceId +
                              Plugin.getDeviceIdStartIndex(),
                          /* type */ getComputeUnitKind().c_str(),
                          /* device */ reinterpret_cast<ompt_device_t *>(this),
                          /* lookup */ ompt::lookupCallbackByName,
                          /* documentation */ nullptr);
  }
#endif

  // Read and reinitialize the envars that depend on the device initialization.
  // Notice these two envars may change the stack size and heap size of the
  // device, so they need the device properly initialized.
  auto StackSizeEnvarOrErr = UInt64Envar::create(
      "LIBOMPTARGET_STACK_SIZE",
      [this](uint64_t &V) -> Error { return getDeviceStackSize(V); },
      [this](uint64_t V) -> Error { return setDeviceStackSize(V); });
  if (!StackSizeEnvarOrErr)
    return StackSizeEnvarOrErr.takeError();
  OMPX_TargetStackSize = std::move(*StackSizeEnvarOrErr);

  auto HeapSizeEnvarOrErr = UInt64Envar::create(
      "LIBOMPTARGET_HEAP_SIZE",
      [this](uint64_t &V) -> Error { return getDeviceHeapSize(V); },
      [this](uint64_t V) -> Error { return setDeviceHeapSize(V); });
  if (!HeapSizeEnvarOrErr)
    return HeapSizeEnvarOrErr.takeError();
  OMPX_TargetHeapSize = std::move(*HeapSizeEnvarOrErr);

  // Update the maximum number of teams and threads after the device
  // initialization sets the corresponding hardware limit.
  if (OMP_NumTeams > 0)
    GridValues.GV_Max_Teams =
        std::min(GridValues.GV_Max_Teams, uint32_t(OMP_NumTeams));

  if (OMP_TeamsThreadLimit > 0)
    GridValues.GV_Max_WG_Size =
        std::min(GridValues.GV_Max_WG_Size, uint32_t(OMP_TeamsThreadLimit));

  // Enable the memory manager if required.
  auto [ThresholdMM, EnableMM] = MemoryManagerTy::getSizeThresholdFromEnv();
  if (EnableMM)
    MemoryManager = new MemoryManagerTy(*this, ThresholdMM);

  return Plugin::success();
}

Error GenericDeviceTy::deinit(GenericPluginTy &Plugin) {
  for (DeviceImageTy *Image : LoadedImages)
    if (auto Err = callGlobalDestructors(Plugin, *Image))
      return Err;

  if (OMPX_DebugKind.get() & uint32_t(DeviceDebugKind::AllocationTracker)) {
    GenericGlobalHandlerTy &GHandler = Plugin.getGlobalHandler();
    for (auto *Image : LoadedImages) {
      DeviceMemoryPoolTrackingTy ImageDeviceMemoryPoolTracking = {0, 0, ~0U, 0};
      GlobalTy TrackerGlobal("__omp_rtl_device_memory_pool_tracker",
                             sizeof(DeviceMemoryPoolTrackingTy),
                             &ImageDeviceMemoryPoolTracking);
      if (auto Err =
              GHandler.readGlobalFromDevice(*this, *Image, TrackerGlobal)) {
        consumeError(std::move(Err));
        continue;
      }
      DeviceMemoryPoolTracking.combine(ImageDeviceMemoryPoolTracking);
    }

    // TODO: Write this by default into a file.
    printf("\n\n|-----------------------\n"
           "| Device memory tracker:\n"
           "|-----------------------\n"
           "| #Allocations: %lu\n"
           "| Byes allocated: %lu\n"
           "| Minimal allocation: %lu\n"
           "| Maximal allocation: %lu\n"
           "|-----------------------\n\n\n",
           DeviceMemoryPoolTracking.NumAllocations,
           DeviceMemoryPoolTracking.AllocationTotal,
           DeviceMemoryPoolTracking.AllocationMin,
           DeviceMemoryPoolTracking.AllocationMax);
  }

  // Delete the memory manager before deinitializing the device. Otherwise,
  // we may delete device allocations after the device is deinitialized.
  if (MemoryManager)
    delete MemoryManager;
  MemoryManager = nullptr;

  if (RecordReplay.isRecordingOrReplaying())
    RecordReplay.deinit();

  if (RPCServer)
    if (auto Err = RPCServer->deinitDevice(*this))
      return Err;

#ifdef OMPT_SUPPORT
  if (ompt::Initialized) {
    bool ExpectedStatus = true;
    if (OmptInitialized.compare_exchange_strong(ExpectedStatus, false))
      performOmptCallback(device_finalize,
                          /* device_num */ DeviceId +
                              Plugin.getDeviceIdStartIndex());
  }
#endif

  return deinitImpl();
}
Expected<__tgt_target_table *>
GenericDeviceTy::loadBinary(GenericPluginTy &Plugin,
                            const __tgt_device_image *InputTgtImage) {
  assert(InputTgtImage && "Expected non-null target image");
  DP("Load data from image " DPxMOD "\n", DPxPTR(InputTgtImage->ImageStart));

  auto PostJITImageOrErr = Plugin.getJIT().process(*InputTgtImage, *this);
  if (!PostJITImageOrErr) {
    auto Err = PostJITImageOrErr.takeError();
    REPORT("Failure to jit IR image %p on device %d: %s\n", InputTgtImage,
           DeviceId, toString(std::move(Err)).data());
    return nullptr;
  }

  // Load the binary and allocate the image object. Use the next available id
  // for the image id, which is the number of previously loaded images.
  auto ImageOrErr =
      loadBinaryImpl(PostJITImageOrErr.get(), LoadedImages.size());
  if (!ImageOrErr)
    return ImageOrErr.takeError();

  DeviceImageTy *Image = *ImageOrErr;
  assert(Image != nullptr && "Invalid image");
  if (InputTgtImage != PostJITImageOrErr.get())
    Image->setTgtImageBitcode(InputTgtImage);

  // Add the image to list.
  LoadedImages.push_back(Image);

  // Setup the device environment if needed.
  if (auto Err = setupDeviceEnvironment(Plugin, *Image))
    return std::move(Err);

  // Setup the global device memory pool if needed.
  if (!RecordReplay.isReplaying() && shouldSetupDeviceMemoryPool()) {
    uint64_t HeapSize;
    auto SizeOrErr = getDeviceHeapSize(HeapSize);
    if (SizeOrErr) {
      REPORT("No global device memory pool due to error: %s\n",
             toString(std::move(SizeOrErr)).data());
    } else if (auto Err = setupDeviceMemoryPool(Plugin, *Image, HeapSize))
      return std::move(Err);
  }

  // Register all offload entries of the image.
  if (auto Err = registerOffloadEntries(*Image))
    return std::move(Err);

  if (auto Err = setupRPCServer(Plugin, *Image))
    return std::move(Err);

#ifdef OMPT_SUPPORT
  if (ompt::Initialized) {
    size_t Bytes =
        getPtrDiff(InputTgtImage->ImageEnd, InputTgtImage->ImageStart);
    performOmptCallback(device_load,
                        /* device_num */ DeviceId +
                            Plugin.getDeviceIdStartIndex(),
                        /* FileName */ nullptr,
                        /* File Offset */ 0,
                        /* VmaInFile */ nullptr,
                        /* ImgSize */ Bytes,
                        /* HostAddr */ InputTgtImage->ImageStart,
                        /* DeviceAddr */ nullptr,
                        /* FIXME: ModuleId */ 0);
  }
#endif

  // Call any global constructors present on the device.
  if (auto Err = callGlobalConstructors(Plugin, *Image))
    return std::move(Err);

  // Return the pointer to the table of entries.
  return Image->getOffloadEntryTable();
}

Error GenericDeviceTy::setupDeviceEnvironment(GenericPluginTy &Plugin,
                                              DeviceImageTy &Image) {
  // There are some plugins that do not need this step.
  if (!shouldSetupDeviceEnvironment())
    return Plugin::success();

  // Obtain a table mapping host function pointers to device function pointers.
  auto CallTablePairOrErr = setupIndirectCallTable(Plugin, *this, Image);
  if (!CallTablePairOrErr)
    return CallTablePairOrErr.takeError();

  DeviceEnvironmentTy DeviceEnvironment;
  DeviceEnvironment.DeviceDebugKind = OMPX_DebugKind;
  DeviceEnvironment.NumDevices = Plugin.getNumDevices();
  // TODO: The device ID used here is not the real device ID used by OpenMP.
  DeviceEnvironment.DeviceNum = DeviceId;
  DeviceEnvironment.DynamicMemSize = OMPX_SharedMemorySize;
  DeviceEnvironment.ClockFrequency = getClockFrequency();
  DeviceEnvironment.IndirectCallTable =
      reinterpret_cast<uintptr_t>(CallTablePairOrErr->first);
  DeviceEnvironment.IndirectCallTableSize = CallTablePairOrErr->second;
  DeviceEnvironment.HardwareParallelism = getHardwareParallelism();

  // Create the metainfo of the device environment global.
  GlobalTy DevEnvGlobal("__omp_rtl_device_environment",
                        sizeof(DeviceEnvironmentTy), &DeviceEnvironment);

  // Write device environment values to the device.
  GenericGlobalHandlerTy &GHandler = Plugin.getGlobalHandler();
  if (auto Err = GHandler.writeGlobalToDevice(*this, Image, DevEnvGlobal)) {
    DP("Missing symbol %s, continue execution anyway.\n",
       DevEnvGlobal.getName().data());
    consumeError(std::move(Err));
  }
  return Plugin::success();
}

Error GenericDeviceTy::setupDeviceMemoryPool(GenericPluginTy &Plugin,
                                             DeviceImageTy &Image,
                                             uint64_t PoolSize) {
  // Free the old pool, if any.
  if (DeviceMemoryPool.Ptr) {
    if (auto Err = dataDelete(DeviceMemoryPool.Ptr,
                              TargetAllocTy::TARGET_ALLOC_DEVICE))
      return Err;
  }

  DeviceMemoryPool.Size = PoolSize;
  auto AllocOrErr = dataAlloc(PoolSize, /*HostPtr=*/nullptr,
                              TargetAllocTy::TARGET_ALLOC_DEVICE);
  if (AllocOrErr) {
    DeviceMemoryPool.Ptr = *AllocOrErr;
  } else {
    auto Err = AllocOrErr.takeError();
    REPORT("Failure to allocate device memory for global memory pool: %s\n",
           toString(std::move(Err)).data());
    DeviceMemoryPool.Ptr = nullptr;
    DeviceMemoryPool.Size = 0;
  }

  // Create the metainfo of the device environment global.
  GenericGlobalHandlerTy &GHandler = Plugin.getGlobalHandler();
  if (!GHandler.isSymbolInImage(*this, Image,
                                "__omp_rtl_device_memory_pool_tracker")) {
    DP("Skip the memory pool as there is no tracker symbol in the image.");
    return Error::success();
  }

  GlobalTy TrackerGlobal("__omp_rtl_device_memory_pool_tracker",
                         sizeof(DeviceMemoryPoolTrackingTy),
                         &DeviceMemoryPoolTracking);
  if (auto Err = GHandler.writeGlobalToDevice(*this, Image, TrackerGlobal))
    return Err;

  // Create the metainfo of the device environment global.
  GlobalTy DevEnvGlobal("__omp_rtl_device_memory_pool",
                        sizeof(DeviceMemoryPoolTy), &DeviceMemoryPool);

  // Write device environment values to the device.
  return GHandler.writeGlobalToDevice(*this, Image, DevEnvGlobal);
}

Error GenericDeviceTy::setupRPCServer(GenericPluginTy &Plugin,
                                      DeviceImageTy &Image) {
  // The plugin either does not need an RPC server or it is unavailible.
  if (!shouldSetupRPCServer())
    return Plugin::success();

  // Check if this device needs to run an RPC server.
  RPCServerTy &Server = Plugin.getRPCServer();
  auto UsingOrErr =
      Server.isDeviceUsingRPC(*this, Plugin.getGlobalHandler(), Image);
  if (!UsingOrErr)
    return UsingOrErr.takeError();

  if (!UsingOrErr.get())
    return Plugin::success();

  if (auto Err = Server.initDevice(*this, Plugin.getGlobalHandler(), Image))
    return Err;

  RPCServer = &Server;
  DP("Running an RPC server on device %d\n", getDeviceId());
  return Plugin::success();
}

Error GenericDeviceTy::registerOffloadEntries(DeviceImageTy &Image) {
  const __tgt_offload_entry *Begin = Image.getTgtImage()->EntriesBegin;
  const __tgt_offload_entry *End = Image.getTgtImage()->EntriesEnd;
  for (const __tgt_offload_entry *Entry = Begin; Entry != End; ++Entry) {
    // The host should have always something in the address to uniquely
    // identify the entry.
    if (!Entry->addr)
      return Plugin::error("Failure to register entry without address");

    __tgt_offload_entry DeviceEntry = {0};

    if (Entry->size) {
      if (auto Err = registerGlobalOffloadEntry(Image, *Entry, DeviceEntry))
        return Err;
    } else {
      if (auto Err = registerKernelOffloadEntry(Image, *Entry, DeviceEntry))
        return Err;
    }

    assert(DeviceEntry.addr && "Device addr of offload entry cannot be null");

    DP("Entry point " DPxMOD " maps to%s %s (" DPxMOD ")\n",
       DPxPTR(Entry - Begin), (Entry->size) ? " global" : "", Entry->name,
       DPxPTR(DeviceEntry.addr));
  }
  return Plugin::success();
}

Error GenericDeviceTy::registerGlobalOffloadEntry(
    DeviceImageTy &Image, const __tgt_offload_entry &GlobalEntry,
    __tgt_offload_entry &DeviceEntry) {

  GenericPluginTy &Plugin = Plugin::get();

  DeviceEntry = GlobalEntry;

  // Create a metadata object for the device global.
  GlobalTy DeviceGlobal(GlobalEntry.name, GlobalEntry.size);

  // Get the address of the device of the global.
  GenericGlobalHandlerTy &GHandler = Plugin.getGlobalHandler();
  if (auto Err =
          GHandler.getGlobalMetadataFromDevice(*this, Image, DeviceGlobal))
    return Err;

  // Store the device address on the device entry.
  DeviceEntry.addr = DeviceGlobal.getPtr();
  assert(DeviceEntry.addr && "Invalid device global's address");

  // Note: In the current implementation declare target variables
  // can either be link or to. This means that once unified
  // memory is activated via the requires directive, the variable
  // can be used directly from the host in both cases.
  if (Plugin.getRequiresFlags() & OMP_REQ_UNIFIED_SHARED_MEMORY) {
    // If unified memory is present any target link or to variables
    // can access host addresses directly. There is no longer a
    // need for device copies.
    GlobalTy HostGlobal(GlobalEntry);
    if (auto Err =
            GHandler.writeGlobalToDevice(*this, HostGlobal, DeviceGlobal))
      return Err;
  }

  // Add the device entry on the entry table.
  Image.getOffloadEntryTable().addEntry(DeviceEntry);

  return Plugin::success();
}

Error GenericDeviceTy::registerKernelOffloadEntry(
    DeviceImageTy &Image, const __tgt_offload_entry &KernelEntry,
    __tgt_offload_entry &DeviceEntry) {
  DeviceEntry = KernelEntry;

  // Create a kernel object.
  auto KernelOrErr = constructKernel(KernelEntry);
  if (!KernelOrErr)
    return KernelOrErr.takeError();

  GenericKernelTy &Kernel = *KernelOrErr;

  // Initialize the kernel.
  if (auto Err = Kernel.init(*this, Image))
    return Err;

  // Set the device entry address to the kernel address and store the entry on
  // the entry table.
  DeviceEntry.addr = (void *)&Kernel;
  Image.getOffloadEntryTable().addEntry(DeviceEntry);

  return Plugin::success();
}

Error PinnedAllocationMapTy::insertEntry(void *HstPtr, void *DevAccessiblePtr,
                                         size_t Size, bool ExternallyLocked) {
  // Insert the new entry into the map.
  auto Res = Allocs.insert({HstPtr, DevAccessiblePtr, Size, ExternallyLocked});
  if (!Res.second)
    return Plugin::error("Cannot insert locked buffer entry");

  // Check whether the next entry overlaps with the inserted entry.
  auto It = std::next(Res.first);
  if (It == Allocs.end())
    return Plugin::success();

  const EntryTy *NextEntry = &(*It);
  if (intersects(NextEntry->HstPtr, NextEntry->Size, HstPtr, Size))
    return Plugin::error("Partial overlapping not allowed in locked buffers");

  return Plugin::success();
}

Error PinnedAllocationMapTy::eraseEntry(const EntryTy &Entry) {
  // Erase the existing entry. Notice this requires an additional map lookup,
  // but this should not be a performance issue. Using iterators would make
  // the code more difficult to read.
  size_t Erased = Allocs.erase({Entry.HstPtr});
  if (!Erased)
    return Plugin::error("Cannot erase locked buffer entry");
  return Plugin::success();
}

Error PinnedAllocationMapTy::registerEntryUse(const EntryTy &Entry,
                                              void *HstPtr, size_t Size) {
  if (!contains(Entry.HstPtr, Entry.Size, HstPtr, Size))
    return Plugin::error("Partial overlapping not allowed in locked buffers");

  ++Entry.References;
  return Plugin::success();
}

Expected<bool> PinnedAllocationMapTy::unregisterEntryUse(const EntryTy &Entry) {
  if (Entry.References == 0)
    return Plugin::error("Invalid number of references");

  // Return whether this was the last user.
  return (--Entry.References == 0);
}

Error PinnedAllocationMapTy::registerHostBuffer(void *HstPtr,
                                                void *DevAccessiblePtr,
                                                size_t Size) {
  assert(HstPtr && "Invalid pointer");
  assert(DevAccessiblePtr && "Invalid pointer");
  assert(Size && "Invalid size");

  std::lock_guard<std::shared_mutex> Lock(Mutex);

  // No pinned allocation should intersect.
  const EntryTy *Entry = findIntersecting(HstPtr);
  if (Entry)
    return Plugin::error("Cannot insert entry due to an existing one");

  // Now insert the new entry.
  return insertEntry(HstPtr, DevAccessiblePtr, Size);
}

Error PinnedAllocationMapTy::unregisterHostBuffer(void *HstPtr) {
  assert(HstPtr && "Invalid pointer");

  std::lock_guard<std::shared_mutex> Lock(Mutex);

  const EntryTy *Entry = findIntersecting(HstPtr);
  if (!Entry)
    return Plugin::error("Cannot find locked buffer");

  // The address in the entry should be the same we are unregistering.
  if (Entry->HstPtr != HstPtr)
    return Plugin::error("Unexpected host pointer in locked buffer entry");

  // Unregister from the entry.
  auto LastUseOrErr = unregisterEntryUse(*Entry);
  if (!LastUseOrErr)
    return LastUseOrErr.takeError();

  // There should be no other references to the pinned allocation.
  if (!(*LastUseOrErr))
    return Plugin::error("The locked buffer is still being used");

  // Erase the entry from the map.
  return eraseEntry(*Entry);
}

Expected<void *> PinnedAllocationMapTy::lockHostBuffer(void *HstPtr,
                                                       size_t Size) {
  assert(HstPtr && "Invalid pointer");
  assert(Size && "Invalid size");

  std::lock_guard<std::shared_mutex> Lock(Mutex);

  const EntryTy *Entry = findIntersecting(HstPtr);

  if (Entry) {
    // An already registered intersecting buffer was found. Register a new use.
    if (auto Err = registerEntryUse(*Entry, HstPtr, Size))
      return std::move(Err);

    // Return the device accessible pointer with the correct offset.
    return advanceVoidPtr(Entry->DevAccessiblePtr,
                          getPtrDiff(HstPtr, Entry->HstPtr));
  }

  // No intersecting registered allocation found in the map. First, lock the
  // host buffer and retrieve the device accessible pointer.
  auto DevAccessiblePtrOrErr = Device.dataLockImpl(HstPtr, Size);
  if (!DevAccessiblePtrOrErr)
    return DevAccessiblePtrOrErr.takeError();

  // Now insert the new entry into the map.
  if (auto Err = insertEntry(HstPtr, *DevAccessiblePtrOrErr, Size))
    return std::move(Err);

  // Return the device accessible pointer.
  return *DevAccessiblePtrOrErr;
}

Error PinnedAllocationMapTy::unlockHostBuffer(void *HstPtr) {
  assert(HstPtr && "Invalid pointer");

  std::lock_guard<std::shared_mutex> Lock(Mutex);

  const EntryTy *Entry = findIntersecting(HstPtr);
  if (!Entry)
    return Plugin::error("Cannot find locked buffer");

  // Unregister from the locked buffer. No need to do anything if there are
  // others using the allocation.
  auto LastUseOrErr = unregisterEntryUse(*Entry);
  if (!LastUseOrErr)
    return LastUseOrErr.takeError();

  // No need to do anything if there are others using the allocation.
  if (!(*LastUseOrErr))
    return Plugin::success();

  // This was the last user of the allocation. Unlock the original locked buffer
  // if it was locked by the plugin. Do not unlock it if it was locked by an
  // external entity. Unlock the buffer using the host pointer of the entry.
  if (!Entry->ExternallyLocked)
    if (auto Err = Device.dataUnlockImpl(Entry->HstPtr))
      return Err;

  // Erase the entry from the map.
  return eraseEntry(*Entry);
}

Error PinnedAllocationMapTy::lockMappedHostBuffer(void *HstPtr, size_t Size) {
  assert(HstPtr && "Invalid pointer");
  assert(Size && "Invalid size");

  std::lock_guard<std::shared_mutex> Lock(Mutex);

  // If previously registered, just register a new user on the entry.
  const EntryTy *Entry = findIntersecting(HstPtr);
  if (Entry)
    return registerEntryUse(*Entry, HstPtr, Size);

  size_t BaseSize;
  void *BaseHstPtr, *BaseDevAccessiblePtr;

  // Check if it was externally pinned by a vendor-specific API.
  auto IsPinnedOrErr = Device.isPinnedPtrImpl(HstPtr, BaseHstPtr,
                                              BaseDevAccessiblePtr, BaseSize);
  if (!IsPinnedOrErr)
    return IsPinnedOrErr.takeError();

  // If pinned, just insert the entry representing the whole pinned buffer.
  if (*IsPinnedOrErr)
    return insertEntry(BaseHstPtr, BaseDevAccessiblePtr, BaseSize,
                       /* Externally locked */ true);

  // Not externally pinned. Do nothing if locking of mapped buffers is disabled.
  if (!LockMappedBuffers)
    return Plugin::success();

  // Otherwise, lock the buffer and insert the new entry.
  auto DevAccessiblePtrOrErr = Device.dataLockImpl(HstPtr, Size);
  if (!DevAccessiblePtrOrErr) {
    // Errors may be tolerated.
    if (!IgnoreLockMappedFailures)
      return DevAccessiblePtrOrErr.takeError();

    consumeError(DevAccessiblePtrOrErr.takeError());
    return Plugin::success();
  }

  return insertEntry(HstPtr, *DevAccessiblePtrOrErr, Size);
}

Error PinnedAllocationMapTy::unlockUnmappedHostBuffer(void *HstPtr) {
  assert(HstPtr && "Invalid pointer");

  std::lock_guard<std::shared_mutex> Lock(Mutex);

  // Check whether there is any intersecting entry.
  const EntryTy *Entry = findIntersecting(HstPtr);

  // No entry but automatic locking of mapped buffers is disabled, so
  // nothing to do.
  if (!Entry && !LockMappedBuffers)
    return Plugin::success();

  // No entry, automatic locking is enabled, but the locking may have failed, so
  // do nothing.
  if (!Entry && IgnoreLockMappedFailures)
    return Plugin::success();

  // No entry, but the automatic locking is enabled, so this is an error.
  if (!Entry)
    return Plugin::error("Locked buffer not found");

  // There is entry, so unregister a user and check whether it was the last one.
  auto LastUseOrErr = unregisterEntryUse(*Entry);
  if (!LastUseOrErr)
    return LastUseOrErr.takeError();

  // If it is not the last one, there is nothing to do.
  if (!(*LastUseOrErr))
    return Plugin::success();

  // Otherwise, if it was the last and the buffer was locked by the plugin,
  // unlock it.
  if (!Entry->ExternallyLocked)
    if (auto Err = Device.dataUnlockImpl(Entry->HstPtr))
      return Err;

  // Finally erase the entry from the map.
  return eraseEntry(*Entry);
}

Error GenericDeviceTy::synchronize(__tgt_async_info *AsyncInfo) {
  if (!AsyncInfo || !AsyncInfo->Queue)
    return Plugin::error("Invalid async info queue");

  if (auto Err = synchronizeImpl(*AsyncInfo))
    return Err;

  for (auto *Ptr : AsyncInfo->AssociatedAllocations)
    if (auto Err = dataDelete(Ptr, TargetAllocTy::TARGET_ALLOC_DEVICE))
      return Err;
  AsyncInfo->AssociatedAllocations.clear();

  return Plugin::success();
}

Error GenericDeviceTy::queryAsync(__tgt_async_info *AsyncInfo) {
  if (!AsyncInfo || !AsyncInfo->Queue)
    return Plugin::error("Invalid async info queue");

  return queryAsyncImpl(*AsyncInfo);
}

Error GenericDeviceTy::memoryVAMap(void **Addr, void *VAddr, size_t *RSize) {
  return Plugin::error("Device does not suppport VA Management");
}

Error GenericDeviceTy::memoryVAUnMap(void *VAddr, size_t Size) {
  return Plugin::error("Device does not suppport VA Management");
}

Error GenericDeviceTy::getDeviceMemorySize(uint64_t &DSize) {
  return Plugin::error(
      "Mising getDeviceMemorySize impelmentation (required by RR-heuristic");
}

Expected<void *> GenericDeviceTy::dataAlloc(int64_t Size, void *HostPtr,
                                            TargetAllocTy Kind) {
  void *Alloc = nullptr;

  if (RecordReplay.isRecordingOrReplaying())
    return RecordReplay.alloc(Size);

  switch (Kind) {
  case TARGET_ALLOC_DEFAULT:
  case TARGET_ALLOC_DEVICE:
    if (MemoryManager) {
      Alloc = MemoryManager->allocate(Size, HostPtr);
      if (!Alloc)
        return Plugin::error("Failed to allocate from memory manager");
      break;
    }
    [[fallthrough]];
  case TARGET_ALLOC_HOST:
  case TARGET_ALLOC_SHARED:
    Alloc = allocate(Size, HostPtr, Kind);
    if (!Alloc)
      return Plugin::error("Failed to allocate from device allocator");
  }

  // Report error if the memory manager or the device allocator did not return
  // any memory buffer.
  if (!Alloc)
    return Plugin::error("Invalid target data allocation kind or requested "
                         "allocator not implemented yet");

  // Register allocated buffer as pinned memory if the type is host memory.
  if (Kind == TARGET_ALLOC_HOST)
    if (auto Err = PinnedAllocs.registerHostBuffer(Alloc, Alloc, Size))
      return std::move(Err);

  return Alloc;
}

Error GenericDeviceTy::dataDelete(void *TgtPtr, TargetAllocTy Kind) {
  // Free is a noop when recording or replaying.
  if (RecordReplay.isRecordingOrReplaying())
    return Plugin::success();

  int Res;
  if (MemoryManager)
    Res = MemoryManager->free(TgtPtr);
  else
    Res = free(TgtPtr, Kind);

  if (Res)
    return Plugin::error("Failure to deallocate device pointer %p", TgtPtr);

  // Unregister deallocated pinned memory buffer if the type is host memory.
  if (Kind == TARGET_ALLOC_HOST)
    if (auto Err = PinnedAllocs.unregisterHostBuffer(TgtPtr))
      return Err;

  return Plugin::success();
}

Error GenericDeviceTy::dataSubmit(void *TgtPtr, const void *HstPtr,
                                  int64_t Size, __tgt_async_info *AsyncInfo) {
  AsyncInfoWrapperTy AsyncInfoWrapper(*this, AsyncInfo);

  auto Err = dataSubmitImpl(TgtPtr, HstPtr, Size, AsyncInfoWrapper);
  AsyncInfoWrapper.finalize(Err);
  return Err;
}

Error GenericDeviceTy::dataRetrieve(void *HstPtr, const void *TgtPtr,
                                    int64_t Size, __tgt_async_info *AsyncInfo) {
  AsyncInfoWrapperTy AsyncInfoWrapper(*this, AsyncInfo);

  auto Err = dataRetrieveImpl(HstPtr, TgtPtr, Size, AsyncInfoWrapper);
  AsyncInfoWrapper.finalize(Err);
  return Err;
}

Error GenericDeviceTy::dataExchange(const void *SrcPtr, GenericDeviceTy &DstDev,
                                    void *DstPtr, int64_t Size,
                                    __tgt_async_info *AsyncInfo) {
  AsyncInfoWrapperTy AsyncInfoWrapper(*this, AsyncInfo);

  auto Err = dataExchangeImpl(SrcPtr, DstDev, DstPtr, Size, AsyncInfoWrapper);
  AsyncInfoWrapper.finalize(Err);
  return Err;
}

Error GenericDeviceTy::launchKernel(void *EntryPtr, void **ArgPtrs,
                                    ptrdiff_t *ArgOffsets,
                                    KernelArgsTy &KernelArgs,
                                    __tgt_async_info *AsyncInfo) {
  AsyncInfoWrapperTy AsyncInfoWrapper(
      *this, RecordReplay.isRecordingOrReplaying() ? nullptr : AsyncInfo);

  GenericKernelTy &GenericKernel =
      *reinterpret_cast<GenericKernelTy *>(EntryPtr);

  auto Err = GenericKernel.launch(*this, ArgPtrs, ArgOffsets, KernelArgs,
                                  AsyncInfoWrapper);

  // 'finalize' here to guarantee next record-replay actions are in-sync
  AsyncInfoWrapper.finalize(Err);

  if (RecordReplay.isRecordingOrReplaying() &&
      RecordReplay.isSaveOutputEnabled())
    RecordReplay.saveKernelOutputInfo(GenericKernel.getName());

  return Err;
}

Error GenericDeviceTy::initAsyncInfo(__tgt_async_info **AsyncInfoPtr) {
  assert(AsyncInfoPtr && "Invalid async info");

  *AsyncInfoPtr = new __tgt_async_info();

  AsyncInfoWrapperTy AsyncInfoWrapper(*this, *AsyncInfoPtr);

  auto Err = initAsyncInfoImpl(AsyncInfoWrapper);
  AsyncInfoWrapper.finalize(Err);
  return Err;
}

Error GenericDeviceTy::initDeviceInfo(__tgt_device_info *DeviceInfo) {
  assert(DeviceInfo && "Invalid device info");

  return initDeviceInfoImpl(DeviceInfo);
}

Error GenericDeviceTy::printInfo() {
  InfoQueueTy InfoQueue;

  // Get the vendor-specific info entries describing the device properties.
  if (auto Err = obtainInfoImpl(InfoQueue))
    return Err;

  // Print all info entries.
  InfoQueue.print();

  return Plugin::success();
}

Error GenericDeviceTy::createEvent(void **EventPtrStorage) {
  return createEventImpl(EventPtrStorage);
}

Error GenericDeviceTy::destroyEvent(void *EventPtr) {
  return destroyEventImpl(EventPtr);
}

Error GenericDeviceTy::recordEvent(void *EventPtr,
                                   __tgt_async_info *AsyncInfo) {
  AsyncInfoWrapperTy AsyncInfoWrapper(*this, AsyncInfo);

  auto Err = recordEventImpl(EventPtr, AsyncInfoWrapper);
  AsyncInfoWrapper.finalize(Err);
  return Err;
}

Error GenericDeviceTy::waitEvent(void *EventPtr, __tgt_async_info *AsyncInfo) {
  AsyncInfoWrapperTy AsyncInfoWrapper(*this, AsyncInfo);

  auto Err = waitEventImpl(EventPtr, AsyncInfoWrapper);
  AsyncInfoWrapper.finalize(Err);
  return Err;
}

Error GenericDeviceTy::syncEvent(void *EventPtr) {
  return syncEventImpl(EventPtr);
}

Error GenericPluginTy::init() {
  auto NumDevicesOrErr = initImpl();
  if (!NumDevicesOrErr)
    return NumDevicesOrErr.takeError();

  NumDevices = *NumDevicesOrErr;
  if (NumDevices == 0)
    return Plugin::success();

  assert(Devices.size() == 0 && "Plugin already initialized");
  Devices.resize(NumDevices, nullptr);

  GlobalHandler = Plugin::createGlobalHandler();
  assert(GlobalHandler && "Invalid global handler");

  RPCServer = new RPCServerTy(NumDevices);
  assert(RPCServer && "Invalid RPC server");

  return Plugin::success();
}

Error GenericPluginTy::deinit() {
  // Deinitialize all active devices.
  for (int32_t DeviceId = 0; DeviceId < NumDevices; ++DeviceId) {
    if (Devices[DeviceId]) {
      if (auto Err = deinitDevice(DeviceId))
        return Err;
    }
    assert(!Devices[DeviceId] && "Device was not deinitialized");
  }

  // There is no global handler if no device is available.
  if (GlobalHandler)
    delete GlobalHandler;

  if (RPCServer)
    delete RPCServer;

  // Perform last deinitializations on the plugin.
  return deinitImpl();
}

Error GenericPluginTy::initDevice(int32_t DeviceId) {
  assert(!Devices[DeviceId] && "Device already initialized");

  // Create the device and save the reference.
  GenericDeviceTy *Device = Plugin::createDevice(DeviceId, NumDevices);
  assert(Device && "Invalid device");

  // Save the device reference into the list.
  Devices[DeviceId] = Device;

  // Initialize the device and its resources.
  return Device->init(*this);
}

Error GenericPluginTy::deinitDevice(int32_t DeviceId) {
  // The device may be already deinitialized.
  if (Devices[DeviceId] == nullptr)
    return Plugin::success();

  // Deinitialize the device and release its resources.
  if (auto Err = Devices[DeviceId]->deinit(*this))
    return Err;

  // Delete the device and invalidate its reference.
  delete Devices[DeviceId];
  Devices[DeviceId] = nullptr;

  return Plugin::success();
}

Expected<bool> GenericPluginTy::checkELFImage(__tgt_device_image &Image) const {
  StringRef Buffer(reinterpret_cast<const char *>(Image.ImageStart),
                   target::getPtrDiff(Image.ImageEnd, Image.ImageStart));

  // First check if this image is a regular ELF file.
  if (!utils::elf::isELF(Buffer))
    return false;

  // Check if this image is an ELF with a matching machine value.
  auto MachineOrErr = utils::elf::checkMachine(Buffer, getMagicElfBits());
  if (!MachineOrErr)
    return MachineOrErr.takeError();

  if (!*MachineOrErr)
    return false;

  // Perform plugin-dependent checks for the specific architecture if needed.
  return isELFCompatible(Buffer);
}

const bool llvm::omp::target::plugin::libomptargetSupportsRPC() {
#ifdef LIBOMPTARGET_RPC_SUPPORT
  return true;
#else
  return false;
#endif
}

/// Exposed library API function, basically wrappers around the GenericDeviceTy
/// functionality with the same name. All non-async functions are redirected
/// to the async versions right away with a NULL AsyncInfoPtr.
#ifdef __cplusplus
extern "C" {
#endif

int32_t __tgt_rtl_init_plugin() {
  auto Err = Plugin::initIfNeeded();
  if (Err) {
    REPORT("Failure to initialize plugin " GETNAME(TARGET_NAME) ": %s\n",
           toString(std::move(Err)).data());
    return OFFLOAD_FAIL;
  }

  return OFFLOAD_SUCCESS;
}

int32_t __tgt_rtl_is_valid_binary(__tgt_device_image *TgtImage) {
  // TODO: We should be able to perform a trivial ELF machine check without
  // initializing the plugin first to save time if the plugin is not needed.
  if (!Plugin::isActive())
    return false;

  // Check if this is a valid ELF with a matching machine and processor.
  auto MatchOrErr = Plugin::get().checkELFImage(*TgtImage);
  if (Error Err = MatchOrErr.takeError()) {
    [[maybe_unused]] std::string ErrStr = toString(std::move(Err));
    DP("Failure to check validity of image %p: %s", TgtImage, ErrStr.c_str());
    return false;
  } else if (*MatchOrErr) {
    return true;
  }

  // Check if this is a valid LLVM-IR file with matching triple.
  if (Plugin::get().getJIT().checkBitcodeImage(*TgtImage))
    return true;

  return false;
}

int32_t __tgt_rtl_supports_empty_images() {
  return Plugin::get().supportsEmptyImages();
}

int32_t __tgt_rtl_init_device(int32_t DeviceId) {
  auto Err = Plugin::get().initDevice(DeviceId);
  if (Err) {
    REPORT("Failure to initialize device %d: %s\n", DeviceId,
           toString(std::move(Err)).data());
    return OFFLOAD_FAIL;
  }

  return OFFLOAD_SUCCESS;
}

int32_t __tgt_rtl_number_of_devices() { return Plugin::get().getNumDevices(); }

int64_t __tgt_rtl_init_requires(int64_t RequiresFlags) {
  Plugin::get().setRequiresFlag(RequiresFlags);
  return OFFLOAD_SUCCESS;
}

int32_t __tgt_rtl_is_data_exchangable(int32_t SrcDeviceId,
                                      int32_t DstDeviceId) {
  return Plugin::get().isDataExchangable(SrcDeviceId, DstDeviceId);
}

int32_t __tgt_rtl_initialize_record_replay(int32_t DeviceId, int64_t MemorySize,
                                           void *VAddr, bool isRecord,
                                           bool SaveOutput,
                                           uint64_t &ReqPtrArgOffset) {
  GenericPluginTy &Plugin = Plugin::get();
  GenericDeviceTy &Device = Plugin.getDevice(DeviceId);
  RecordReplayTy::RRStatusTy Status =
      isRecord ? RecordReplayTy::RRStatusTy::RRRecording
               : RecordReplayTy::RRStatusTy::RRReplaying;

  if (auto Err = RecordReplay.init(&Device, MemorySize, VAddr, Status,
                                   SaveOutput, ReqPtrArgOffset)) {
    REPORT("WARNING RR did not intialize RR-properly with %lu bytes"
           "(Error: %s)\n",
           MemorySize, toString(std::move(Err)).data());
    RecordReplay.setStatus(RecordReplayTy::RRStatusTy::RRDeactivated);

    if (!isRecord) {
      return OFFLOAD_FAIL;
    }
  }
  return OFFLOAD_SUCCESS;
}

__tgt_target_table *__tgt_rtl_load_binary(int32_t DeviceId,
                                          __tgt_device_image *TgtImage) {
  GenericPluginTy &Plugin = Plugin::get();
  GenericDeviceTy &Device = Plugin.getDevice(DeviceId);

  auto TableOrErr = Device.loadBinary(Plugin, TgtImage);
  if (!TableOrErr) {
    auto Err = TableOrErr.takeError();
    REPORT("Failure to load binary image %p on device %d: %s\n", TgtImage,
           DeviceId, toString(std::move(Err)).data());
    return nullptr;
  }

  __tgt_target_table *Table = *TableOrErr;
  assert(Table != nullptr && "Invalid table");

  return Table;
}

void *__tgt_rtl_data_alloc(int32_t DeviceId, int64_t Size, void *HostPtr,
                           int32_t Kind) {
  auto AllocOrErr = Plugin::get().getDevice(DeviceId).dataAlloc(
      Size, HostPtr, (TargetAllocTy)Kind);
  if (!AllocOrErr) {
    auto Err = AllocOrErr.takeError();
    REPORT("Failure to allocate device memory: %s\n",
           toString(std::move(Err)).data());
    return nullptr;
  }
  assert(*AllocOrErr && "Null pointer upon successful allocation");

  return *AllocOrErr;
}

int32_t __tgt_rtl_data_delete(int32_t DeviceId, void *TgtPtr, int32_t Kind) {
  auto Err =
      Plugin::get().getDevice(DeviceId).dataDelete(TgtPtr, (TargetAllocTy)Kind);
  if (Err) {
    REPORT("Failure to deallocate device pointer %p: %s\n", TgtPtr,
           toString(std::move(Err)).data());
    return OFFLOAD_FAIL;
  }

  return OFFLOAD_SUCCESS;
}

int32_t __tgt_rtl_data_lock(int32_t DeviceId, void *Ptr, int64_t Size,
                            void **LockedPtr) {
  auto LockedPtrOrErr = Plugin::get().getDevice(DeviceId).dataLock(Ptr, Size);
  if (!LockedPtrOrErr) {
    auto Err = LockedPtrOrErr.takeError();
    REPORT("Failure to lock memory %p: %s\n", Ptr,
           toString(std::move(Err)).data());
    return OFFLOAD_FAIL;
  }

  if (!(*LockedPtrOrErr)) {
    REPORT("Failure to lock memory %p: obtained a null locked pointer\n", Ptr);
    return OFFLOAD_FAIL;
  }
  *LockedPtr = *LockedPtrOrErr;

  return OFFLOAD_SUCCESS;
}

int32_t __tgt_rtl_data_unlock(int32_t DeviceId, void *Ptr) {
  auto Err = Plugin::get().getDevice(DeviceId).dataUnlock(Ptr);
  if (Err) {
    REPORT("Failure to unlock memory %p: %s\n", Ptr,
           toString(std::move(Err)).data());
    return OFFLOAD_FAIL;
  }

  return OFFLOAD_SUCCESS;
}

int32_t __tgt_rtl_data_notify_mapped(int32_t DeviceId, void *HstPtr,
                                     int64_t Size) {
  auto Err = Plugin::get().getDevice(DeviceId).notifyDataMapped(HstPtr, Size);
  if (Err) {
    REPORT("Failure to notify data mapped %p: %s\n", HstPtr,
           toString(std::move(Err)).data());
    return OFFLOAD_FAIL;
  }

  return OFFLOAD_SUCCESS;
}

int32_t __tgt_rtl_data_notify_unmapped(int32_t DeviceId, void *HstPtr) {
  auto Err = Plugin::get().getDevice(DeviceId).notifyDataUnmapped(HstPtr);
  if (Err) {
    REPORT("Failure to notify data unmapped %p: %s\n", HstPtr,
           toString(std::move(Err)).data());
    return OFFLOAD_FAIL;
  }

  return OFFLOAD_SUCCESS;
}

int32_t __tgt_rtl_data_submit(int32_t DeviceId, void *TgtPtr, void *HstPtr,
                              int64_t Size) {
  return __tgt_rtl_data_submit_async(DeviceId, TgtPtr, HstPtr, Size,
                                     /* AsyncInfoPtr */ nullptr);
}

int32_t __tgt_rtl_data_submit_async(int32_t DeviceId, void *TgtPtr,
                                    void *HstPtr, int64_t Size,
                                    __tgt_async_info *AsyncInfoPtr) {
  auto Err = Plugin::get().getDevice(DeviceId).dataSubmit(TgtPtr, HstPtr, Size,
                                                          AsyncInfoPtr);
  if (Err) {
    REPORT("Failure to copy data from host to device. Pointers: host "
           "= " DPxMOD ", device = " DPxMOD ", size = %" PRId64 ": %s\n",
           DPxPTR(HstPtr), DPxPTR(TgtPtr), Size,
           toString(std::move(Err)).data());
    return OFFLOAD_FAIL;
  }

  return OFFLOAD_SUCCESS;
}

int32_t __tgt_rtl_data_retrieve(int32_t DeviceId, void *HstPtr, void *TgtPtr,
                                int64_t Size) {
  return __tgt_rtl_data_retrieve_async(DeviceId, HstPtr, TgtPtr, Size,
                                       /* AsyncInfoPtr */ nullptr);
}

int32_t __tgt_rtl_data_retrieve_async(int32_t DeviceId, void *HstPtr,
                                      void *TgtPtr, int64_t Size,
                                      __tgt_async_info *AsyncInfoPtr) {
  auto Err = Plugin::get().getDevice(DeviceId).dataRetrieve(HstPtr, TgtPtr,
                                                            Size, AsyncInfoPtr);
  if (Err) {
    REPORT("Faliure to copy data from device to host. Pointers: host "
           "= " DPxMOD ", device = " DPxMOD ", size = %" PRId64 ": %s\n",
           DPxPTR(HstPtr), DPxPTR(TgtPtr), Size,
           toString(std::move(Err)).data());
    return OFFLOAD_FAIL;
  }

  return OFFLOAD_SUCCESS;
}

int32_t __tgt_rtl_data_exchange(int32_t SrcDeviceId, void *SrcPtr,
                                int32_t DstDeviceId, void *DstPtr,
                                int64_t Size) {
  return __tgt_rtl_data_exchange_async(SrcDeviceId, SrcPtr, DstDeviceId, DstPtr,
                                       Size,
                                       /* AsyncInfoPtr */ nullptr);
}

int32_t __tgt_rtl_data_exchange_async(int32_t SrcDeviceId, void *SrcPtr,
                                      int DstDeviceId, void *DstPtr,
                                      int64_t Size,
                                      __tgt_async_info *AsyncInfo) {
  GenericDeviceTy &SrcDevice = Plugin::get().getDevice(SrcDeviceId);
  GenericDeviceTy &DstDevice = Plugin::get().getDevice(DstDeviceId);
  auto Err = SrcDevice.dataExchange(SrcPtr, DstDevice, DstPtr, Size, AsyncInfo);
  if (Err) {
    REPORT("Failure to copy data from device (%d) to device (%d). Pointers: "
           "host = " DPxMOD ", device = " DPxMOD ", size = %" PRId64 ": %s\n",
           SrcDeviceId, DstDeviceId, DPxPTR(SrcPtr), DPxPTR(DstPtr), Size,
           toString(std::move(Err)).data());
    return OFFLOAD_FAIL;
  }

  return OFFLOAD_SUCCESS;
}

int32_t __tgt_rtl_launch_kernel(int32_t DeviceId, void *TgtEntryPtr,
                                void **TgtArgs, ptrdiff_t *TgtOffsets,
                                KernelArgsTy *KernelArgs,
                                __tgt_async_info *AsyncInfoPtr) {
  auto Err = Plugin::get().getDevice(DeviceId).launchKernel(
      TgtEntryPtr, TgtArgs, TgtOffsets, *KernelArgs, AsyncInfoPtr);
  if (Err) {
    REPORT("Failure to run target region " DPxMOD " in device %d: %s\n",
           DPxPTR(TgtEntryPtr), DeviceId, toString(std::move(Err)).data());
    return OFFLOAD_FAIL;
  }

  return OFFLOAD_SUCCESS;
}

int32_t __tgt_rtl_synchronize(int32_t DeviceId,
                              __tgt_async_info *AsyncInfoPtr) {
  auto Err = Plugin::get().getDevice(DeviceId).synchronize(AsyncInfoPtr);
  if (Err) {
    REPORT("Failure to synchronize stream %p: %s\n", AsyncInfoPtr->Queue,
           toString(std::move(Err)).data());
    return OFFLOAD_FAIL;
  }

  return OFFLOAD_SUCCESS;
}

int32_t __tgt_rtl_query_async(int32_t DeviceId,
                              __tgt_async_info *AsyncInfoPtr) {
  auto Err = Plugin::get().getDevice(DeviceId).queryAsync(AsyncInfoPtr);
  if (Err) {
    REPORT("Failure to query stream %p: %s\n", AsyncInfoPtr->Queue,
           toString(std::move(Err)).data());
    return OFFLOAD_FAIL;
  }

  return OFFLOAD_SUCCESS;
}

void __tgt_rtl_print_device_info(int32_t DeviceId) {
  if (auto Err = Plugin::get().getDevice(DeviceId).printInfo())
    REPORT("Failure to print device %d info: %s\n", DeviceId,
           toString(std::move(Err)).data());
}

int32_t __tgt_rtl_create_event(int32_t DeviceId, void **EventPtr) {
  auto Err = Plugin::get().getDevice(DeviceId).createEvent(EventPtr);
  if (Err) {
    REPORT("Failure to create event: %s\n", toString(std::move(Err)).data());
    return OFFLOAD_FAIL;
  }

  return OFFLOAD_SUCCESS;
}

int32_t __tgt_rtl_record_event(int32_t DeviceId, void *EventPtr,
                               __tgt_async_info *AsyncInfoPtr) {
  auto Err =
      Plugin::get().getDevice(DeviceId).recordEvent(EventPtr, AsyncInfoPtr);
  if (Err) {
    REPORT("Failure to record event %p: %s\n", EventPtr,
           toString(std::move(Err)).data());
    return OFFLOAD_FAIL;
  }

  return OFFLOAD_SUCCESS;
}

int32_t __tgt_rtl_wait_event(int32_t DeviceId, void *EventPtr,
                             __tgt_async_info *AsyncInfoPtr) {
  auto Err =
      Plugin::get().getDevice(DeviceId).waitEvent(EventPtr, AsyncInfoPtr);
  if (Err) {
    REPORT("Failure to wait event %p: %s\n", EventPtr,
           toString(std::move(Err)).data());
    return OFFLOAD_FAIL;
  }

  return OFFLOAD_SUCCESS;
}

int32_t __tgt_rtl_sync_event(int32_t DeviceId, void *EventPtr) {
  auto Err = Plugin::get().getDevice(DeviceId).syncEvent(EventPtr);
  if (Err) {
    REPORT("Failure to synchronize event %p: %s\n", EventPtr,
           toString(std::move(Err)).data());
    return OFFLOAD_FAIL;
  }

  return OFFLOAD_SUCCESS;
}

int32_t __tgt_rtl_destroy_event(int32_t DeviceId, void *EventPtr) {
  auto Err = Plugin::get().getDevice(DeviceId).destroyEvent(EventPtr);
  if (Err) {
    REPORT("Failure to destroy event %p: %s\n", EventPtr,
           toString(std::move(Err)).data());
    return OFFLOAD_FAIL;
  }

  return OFFLOAD_SUCCESS;
}

void __tgt_rtl_set_info_flag(uint32_t NewInfoLevel) {
  std::atomic<uint32_t> &InfoLevel = getInfoLevelInternal();
  InfoLevel.store(NewInfoLevel);
}

int32_t __tgt_rtl_init_async_info(int32_t DeviceId,
                                  __tgt_async_info **AsyncInfoPtr) {
  assert(AsyncInfoPtr && "Invalid async info");

  auto Err = Plugin::get().getDevice(DeviceId).initAsyncInfo(AsyncInfoPtr);
  if (Err) {
    REPORT("Failure to initialize async info at " DPxMOD " on device %d: %s\n",
           DPxPTR(*AsyncInfoPtr), DeviceId, toString(std::move(Err)).data());
    return OFFLOAD_FAIL;
  }

  return OFFLOAD_SUCCESS;
}

int32_t __tgt_rtl_init_device_info(int32_t DeviceId,
                                   __tgt_device_info *DeviceInfo,
                                   const char **ErrStr) {
  *ErrStr = "";

  auto Err = Plugin::get().getDevice(DeviceId).initDeviceInfo(DeviceInfo);
  if (Err) {
    REPORT("Failure to initialize device info at " DPxMOD " on device %d: %s\n",
           DPxPTR(DeviceInfo), DeviceId, toString(std::move(Err)).data());
    return OFFLOAD_FAIL;
  }

  return OFFLOAD_SUCCESS;
}

int32_t __tgt_rtl_set_device_offset(int32_t DeviceIdOffset) {
  Plugin::get().setDeviceIdStartIndex(DeviceIdOffset);

  return OFFLOAD_SUCCESS;
}

#ifdef __cplusplus
}
#endif
