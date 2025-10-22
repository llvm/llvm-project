//===--- Level Zero Target RTL Implementation -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// GenericDevice instatiation for SPIR-V/Xe machine
//
//===----------------------------------------------------------------------===//

#include "L0Device.h"
#include "L0Defs.h"
#include "L0Interop.h"
#include "L0Plugin.h"
#include "L0Program.h"
#include "L0Trace.h"

namespace llvm::omp::target::plugin {

L0DeviceTLSTy &L0DeviceTy::getTLS() {
  return getPlugin().getDeviceTLS(getDeviceId());
}

// clang-format off
/// Mapping from device arch to GPU runtime's device identifiers
static struct {
  DeviceArchTy arch;
  PCIIdTy ids[10];
} DeviceArchMap[] = {{DeviceArchTy::DeviceArch_Gen,
                      {PCIIdTy::SKL,
                       PCIIdTy::KBL,
                       PCIIdTy::CFL, PCIIdTy::CFL_2,
                       PCIIdTy::ICX,
                       PCIIdTy::None}},
                     {DeviceArchTy::DeviceArch_Gen,
                      {PCIIdTy::TGL, PCIIdTy::TGL_2,
                       PCIIdTy::DG1,
                       PCIIdTy::RKL,
                       PCIIdTy::ADLS,
                       PCIIdTy::RTL,
                       PCIIdTy::None}},
                     {DeviceArchTy::DeviceArch_XeLPG,
                      {PCIIdTy::MTL,
                       PCIIdTy::None}},
                     {DeviceArchTy::DeviceArch_XeHPC,
                      {PCIIdTy::PVC,
                       PCIIdTy::None}},
                     {DeviceArchTy::DeviceArch_XeHPG,
                      {PCIIdTy::DG2_ATS_M,
                       PCIIdTy::DG2_ATS_M_2,
                       PCIIdTy::None}},
                     {DeviceArchTy::DeviceArch_Xe2LP,
                      {PCIIdTy::LNL,
                       PCIIdTy::None}},
                     {DeviceArchTy::DeviceArch_Xe2HP,
                      {PCIIdTy::BMG,
                       PCIIdTy::None}},
};
constexpr int DeviceArchMapSize = sizeof(DeviceArchMap) / sizeof(DeviceArchMap[0]);
// clang-format on

DeviceArchTy L0DeviceTy::computeArch() const {
  const auto PCIDeviceId = getPCIId();
  if (PCIDeviceId != 0) {
    for (int ArchIndex = 0; ArchIndex < DeviceArchMapSize; ArchIndex++) {
      for (int i = 0;; i++) {
        const auto Id = DeviceArchMap[ArchIndex].ids[i];
        if (Id == PCIIdTy::None)
          break;

        auto maskedId = static_cast<PCIIdTy>(PCIDeviceId & 0xFF00);
        if (maskedId == Id)
          return DeviceArchMap[ArchIndex].arch; // Exact match or prefix match
      }
    }
  }

  DP("Warning: Cannot decide device arch for %s.\n", getNameCStr());
  return DeviceArchTy::DeviceArch_None;
}

bool L0DeviceTy::isDeviceIPorNewer(uint32_t Version) const {
  ze_device_ip_version_ext_t IPVersion{};
  IPVersion.stype = ZE_STRUCTURE_TYPE_DEVICE_IP_VERSION_EXT;
  IPVersion.pNext = nullptr;
  ze_device_properties_t DevicePR{};
  DevicePR.stype = ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES;
  DevicePR.pNext = &IPVersion;
  CALL_ZE_RET(false, zeDeviceGetProperties, zeDevice, &DevicePR);
  return IPVersion.ipVersion >= Version;
}

/// Get default compute group ordinal. Returns Ordinal-NumQueues pair
std::pair<uint32_t, uint32_t> L0DeviceTy::findComputeOrdinal() {
  std::pair<uint32_t, uint32_t> Ordinal{UINT32_MAX, 0};
  uint32_t Count = 0;
  const auto zeDevice = getZeDevice();
  CALL_ZE_RET(Ordinal, zeDeviceGetCommandQueueGroupProperties, zeDevice, &Count,
              nullptr);
  ze_command_queue_group_properties_t Init{
      ZE_STRUCTURE_TYPE_COMMAND_QUEUE_GROUP_PROPERTIES, nullptr, 0, 0, 0};
  std::vector<ze_command_queue_group_properties_t> Properties(Count, Init);
  CALL_ZE_RET(Ordinal, zeDeviceGetCommandQueueGroupProperties, zeDevice, &Count,
              Properties.data());
  for (uint32_t I = 0; I < Count; I++) {
    // TODO: add a separate set of ordinals for compute queue groups which
    // support cooperative kernels
    if (Properties[I].flags & ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COMPUTE) {
      Ordinal.first = I;
      Ordinal.second = Properties[I].numQueues;
      break;
    }
  }
  if (Ordinal.first == UINT32_MAX)
    DP("Error: no command queues are found\n");

  return Ordinal;
}

/// Get copy command queue group ordinal. Returns Ordinal-NumQueues pair
std::pair<uint32_t, uint32_t> L0DeviceTy::findCopyOrdinal(bool LinkCopy) {
  std::pair<uint32_t, uint32_t> Ordinal{UINT32_MAX, 0};
  uint32_t Count = 0;
  const auto zeDevice = getZeDevice();
  CALL_ZE_RET(Ordinal, zeDeviceGetCommandQueueGroupProperties, zeDevice, &Count,
              nullptr);
  ze_command_queue_group_properties_t Init{
      ZE_STRUCTURE_TYPE_COMMAND_QUEUE_GROUP_PROPERTIES, nullptr, 0, 0, 0};
  std::vector<ze_command_queue_group_properties_t> Properties(Count, Init);
  CALL_ZE_RET(Ordinal, zeDeviceGetCommandQueueGroupProperties, zeDevice, &Count,
              Properties.data());

  for (uint32_t I = 0; I < Count; I++) {
    const auto &Flags = Properties[I].flags;
    if ((Flags & ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COPY) &&
        (Flags & ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COMPUTE) == 0) {
      auto NumQueues = Properties[I].numQueues;
      if (LinkCopy && NumQueues > 1) {
        Ordinal = {I, NumQueues};
        DP("Found link copy command queue for device " DPxMOD
           ", ordinal = %" PRIu32 ", number of queues = %" PRIu32 "\n",
           DPxPTR(zeDevice), Ordinal.first, Ordinal.second);
        break;
      } else if (!LinkCopy && NumQueues == 1) {
        Ordinal = {I, NumQueues};
        DP("Found copy command queue for device " DPxMOD ", ordinal = %" PRIu32
           "\n",
           DPxPTR(zeDevice), Ordinal.first);
        break;
      }
    }
  }
  return Ordinal;
}

void L0DeviceTy::reportDeviceInfo() const {
  DP("Device %" PRIu32 "\n", DeviceId);
  DP("-- Name                         : %s\n", getNameCStr());
  DP("-- PCI ID                       : 0x%" PRIx32 "\n", getPCIId());
  DP("-- UUID                         : %s\n", getUuid().data());
  DP("-- Number of total EUs          : %" PRIu32 "\n", getNumEUs());
  DP("-- Number of threads per EU     : %" PRIu32 "\n", getNumThreadsPerEU());
  DP("-- EU SIMD width                : %" PRIu32 "\n", getSIMDWidth());
  DP("-- Number of EUs per subslice   : %" PRIu32 "\n", getNumEUsPerSubslice());
  DP("-- Number of subslices per slice: %" PRIu32 "\n",
     getNumSubslicesPerSlice());
  DP("-- Number of slices             : %" PRIu32 "\n", getNumSlices());
  DP("-- Local memory size (bytes)    : %" PRIu32 "\n",
     getMaxSharedLocalMemory());
  DP("-- Global memory size (bytes)   : %" PRIu64 "\n", getGlobalMemorySize());
  DP("-- Cache size (bytes)           : %" PRIu64 "\n", getCacheSize());
  DP("-- Max clock frequency (MHz)    : %" PRIu32 "\n", getClockRate());
}

Error L0DeviceTy::initImpl(GenericPluginTy &Plugin) {
  const auto &Options = getPlugin().getOptions();

  uint32_t Count = 1;
  const auto zeDevice = getZeDevice();
  CALL_ZE_RET_ERROR(zeDeviceGetProperties, zeDevice, &DeviceProperties);
  CALL_ZE_RET_ERROR(zeDeviceGetComputeProperties, zeDevice, &ComputeProperties);
  CALL_ZE_RET_ERROR(zeDeviceGetMemoryProperties, zeDevice, &Count,
                    &MemoryProperties);
  CALL_ZE_RET_ERROR(zeDeviceGetCacheProperties, zeDevice, &Count,
                    &CacheProperties);

  DeviceName =
      std::string(DeviceProperties.name, sizeof(DeviceProperties.name));

  DP("Found a GPU device, Name = %s\n", DeviceProperties.name);

  DeviceArch = computeArch();
  // Default allocation kind for this device
  AllocKind = isDiscreteDevice() ? TARGET_ALLOC_DEVICE : TARGET_ALLOC_SHARED;

  ze_kernel_indirect_access_flags_t Flags =
      (AllocKind == TARGET_ALLOC_DEVICE)
          ? ZE_KERNEL_INDIRECT_ACCESS_FLAG_DEVICE
          : ZE_KERNEL_INDIRECT_ACCESS_FLAG_SHARED;
  IndirectAccessFlags = Flags;

  // Get the UUID
  std::string uid = "";
  for (int n = 0; n < ZE_MAX_DEVICE_UUID_SIZE; n++)
    uid += std::to_string(DeviceProperties.uuid.id[n]);
  DeviceUuid = std::move(uid);

  ComputeOrdinal = findComputeOrdinal();

  CopyOrdinal = findCopyOrdinal();

  IsAsyncEnabled =
      isDiscreteDevice() && Options.CommandMode != CommandModeTy::Sync;
  if (auto Err = MemAllocator.initDevicePools(*this, getPlugin().getOptions()))
    return Err;
  l0Context.getHostMemAllocator().updateMaxAllocSize(*this);
  return Plugin::success();
}

Error L0DeviceTy::deinitImpl() {
  for (auto &PGM : Programs)
    if (auto Err = PGM.deinit())
      return Err;
  return MemAllocator.deinit();
}

Expected<DeviceImageTy *>
L0DeviceTy::loadBinaryImpl(std::unique_ptr<MemoryBuffer> &&TgtImage,
                           int32_t ImageId) {
  auto *PGM = getProgramFromImage(TgtImage->getMemBufferRef());
  if (PGM) {
    // Program already exists
    return PGM;
  }

  INFO(OMP_INFOTYPE_PLUGIN_KERNEL, getDeviceId(),
       "Device %" PRId32 ": Loading binary from " DPxMOD "\n", getDeviceId(),
       DPxPTR(TgtImage->getBufferStart()));

  const auto &Options = getPlugin().getOptions();
  std::string CompilationOptions(Options.CompilationOptions);
  CompilationOptions += " " + Options.UserCompilationOptions;

  INFO(OMP_INFOTYPE_PLUGIN_KERNEL, getDeviceId(),
       "Base L0 module compilation options: %s\n", CompilationOptions.c_str());

  CompilationOptions += " ";
  CompilationOptions += Options.InternalCompilationOptions;
  auto &Program = addProgram(ImageId, std::move(TgtImage));

  if (auto Err = Program.buildModules(CompilationOptions))
    return std::move(Err);

  if (auto Err = Program.linkModules())
    return std::move(Err);

  if (auto Err = Program.loadModuleKernels())
    return std::move(Err);

  return &Program;
}

Error L0DeviceTy::unloadBinaryImpl(DeviceImageTy *Image) {
  // Ignoring for now
  // TODO: call properly L0Program unload
  return Plugin::success();
}

Error L0DeviceTy::synchronizeImpl(__tgt_async_info &AsyncInfo,
                                  bool ReleaseQueue) {
  bool IsAsync = asyncEnabled();
  if (!IsAsync)
    return Plugin::success();

  auto &Plugin = getPlugin();

  AsyncQueueTy *AsyncQueue = (AsyncQueueTy *)AsyncInfo.Queue;

  if (!AsyncQueue->WaitEvents.empty()) {
    const auto &WaitEvents = AsyncQueue->WaitEvents;
    if (Plugin.getOptions().CommandMode == CommandModeTy::AsyncOrdered) {
      // Only need to wait for the last event
      CALL_ZE_RET_ERROR(zeEventHostSynchronize, WaitEvents.back(), UINT64_MAX);
      // Synchronize on kernel event to support printf()
      auto KE = AsyncQueue->KernelEvent;
      if (KE && KE != WaitEvents.back()) {
        CALL_ZE_RET_ERROR(zeEventHostSynchronize, KE, UINT64_MAX);
      }
      for (auto &Event : WaitEvents) {
        if (auto Err = releaseEvent(Event))
          return Err;
      }
    } else { // Async
      // Wait for all events. We should wait and reset events in reverse order
      // to avoid premature event reset. If we have a kernel event in the
      // queue, it is the last event to wait for since all wait events of the
      // kernel are signaled before the kernel is invoked. We always invoke
      // synchronization on kernel event to support printf().
      bool WaitDone = false;
      for (auto Itr = WaitEvents.rbegin(); Itr != WaitEvents.rend(); Itr++) {
        if (!WaitDone) {
          CALL_ZE_RET_ERROR(zeEventHostSynchronize, *Itr, UINT64_MAX);
          if (*Itr == AsyncQueue->KernelEvent)
            WaitDone = true;
        }
        if (auto Err = releaseEvent(*Itr))
          return Err;
      }
    }
  }

  // Commit delayed USM2M copies
  for (auto &USM2M : AsyncQueue->USM2MList) {
    std::copy_n(static_cast<const char *>(std::get<0>(USM2M)),
                std::get<2>(USM2M), static_cast<char *>(std::get<1>(USM2M)));
  }
  // Commit delayed H2M copies
  for (auto &H2M : AsyncQueue->H2MList) {
    std::copy_n(static_cast<char *>(std::get<0>(H2M)), std::get<2>(H2M),
                static_cast<char *>(std::get<1>(H2M)));
  }
  if (ReleaseQueue) {
    Plugin.releaseAsyncQueue(AsyncQueue);
    getStagingBuffer().reset();
    AsyncInfo.Queue = nullptr;
  }

  return Plugin::success();
}

Expected<bool>
L0DeviceTy::hasPendingWorkImpl(AsyncInfoWrapperTy &AsyncInfoWrapper) {
  auto &AsyncInfo = *static_cast<__tgt_async_info *>(AsyncInfoWrapper);
  const bool IsAsync = AsyncInfo.Queue && asyncEnabled();
  if (!IsAsync)
    return false;

  auto *AsyncQueue = static_cast<AsyncQueueTy *>(AsyncInfo.Queue);

  if (AsyncQueue->WaitEvents.empty())
    return false;

  return true;
}

Error L0DeviceTy::queryAsyncImpl(__tgt_async_info &AsyncInfo) {
  const bool IsAsync = AsyncInfo.Queue && asyncEnabled();
  if (!IsAsync)
    return Plugin::success();

  auto &Plugin = getPlugin();
  auto *AsyncQueue = static_cast<AsyncQueueTy *>(AsyncInfo.Queue);

  if (!AsyncQueue->WaitEvents.empty())
    return Plugin::success();

  // Commit delayed USM2M copies
  for (auto &USM2M : AsyncQueue->USM2MList) {
    std::copy_n(static_cast<const char *>(std::get<0>(USM2M)),
                std::get<2>(USM2M), static_cast<char *>(std::get<1>(USM2M)));
  }
  // Commit delayed H2M copies
  for (auto &H2M : AsyncQueue->H2MList) {
    std::copy_n(static_cast<char *>(std::get<0>(H2M)), std::get<2>(H2M),
                static_cast<char *>(std::get<1>(H2M)));
  }
  Plugin.releaseAsyncQueue(AsyncQueue);
  getStagingBuffer().reset();
  AsyncInfo.Queue = nullptr;

  return Plugin::success();
}

Expected<void *> L0DeviceTy::allocate(size_t Size, void *HstPtr,
                                      TargetAllocTy Kind) {
  return dataAlloc(Size, /*Align=*/0, Kind,
                   /*Offset=*/0, /*UserAlloc=*/HstPtr == nullptr,
                   /*DevMalloc=*/false);
}

Error L0DeviceTy::free(void *TgtPtr, TargetAllocTy Kind) {
  return dataDelete(TgtPtr);
}

Error L0DeviceTy::dataSubmitImpl(void *TgtPtr, const void *HstPtr, int64_t Size,
                                 AsyncInfoWrapperTy &AsyncInfoWrapper) {
  if (Size == 0)
    return Plugin::success();

  auto &Plugin = getPlugin();
  __tgt_async_info *AsyncInfo = AsyncInfoWrapper;

  const auto DeviceId = getDeviceId();
  bool IsAsync = AsyncInfo && asyncEnabled();
  if (IsAsync && !AsyncInfo->Queue) {
    AsyncInfo->Queue = reinterpret_cast<void *>(Plugin.getAsyncQueue());
    if (!AsyncInfo->Queue)
      IsAsync = false; // Couldn't get a queue, revert to sync
  }
  const auto TgtPtrType = getMemAllocType(TgtPtr);
  if (TgtPtrType == ZE_MEMORY_TYPE_SHARED ||
      TgtPtrType == ZE_MEMORY_TYPE_HOST) {
    std::copy_n(static_cast<const char *>(HstPtr), Size,
                static_cast<char *>(TgtPtr));
  } else {
    const void *SrcPtr = HstPtr;
    if (isDiscreteDevice() &&
        static_cast<size_t>(Size) <= Plugin.getOptions().StagingBufferSize &&
        getMemAllocType(HstPtr) != ZE_MEMORY_TYPE_HOST) {
      auto PtrOrErr = getStagingBuffer().get(IsAsync);
      if (!PtrOrErr)
        return PtrOrErr.takeError();
      SrcPtr = *PtrOrErr;
      std::copy_n(static_cast<const char *>(HstPtr), Size,
                  static_cast<char *>(const_cast<void *>(SrcPtr)));
    }
    if (IsAsync) {
      if (auto Err = enqueueMemCopyAsync(TgtPtr, SrcPtr, Size, AsyncInfo))
        return Err;
    } else {
      if (auto Err = enqueueMemCopy(TgtPtr, SrcPtr, Size, AsyncInfo))
        return Err;
    }
  }
  INFO(OMP_INFOTYPE_PLUGIN_KERNEL, DeviceId,
       "%s %" PRId64 " bytes (hst:" DPxMOD ") -> (tgt:" DPxMOD ")\n",
       IsAsync ? "Submitted copy" : "Copied", Size, DPxPTR(HstPtr),
       DPxPTR(TgtPtr));
  return Plugin::success();
}

Error L0DeviceTy::dataRetrieveImpl(void *HstPtr, const void *TgtPtr,
                                   int64_t Size,
                                   AsyncInfoWrapperTy &AsyncInfoWrapper) {
  if (Size == 0)
    return Plugin::success();

  auto &Plugin = getPlugin();
  __tgt_async_info *AsyncInfo = AsyncInfoWrapper;

  const auto DeviceId = getDeviceId();
  bool IsAsync = AsyncInfo && asyncEnabled();
  if (IsAsync && !AsyncInfo->Queue) {
    AsyncInfo->Queue = Plugin.getAsyncQueue();
    if (!AsyncInfo->Queue)
      IsAsync = false; // Couldn't get a queue, revert to sync
  }
  auto AsyncQueue =
      IsAsync ? static_cast<AsyncQueueTy *>(AsyncInfo->Queue) : nullptr;
  auto TgtPtrType = getMemAllocType(TgtPtr);
  if (TgtPtrType == ZE_MEMORY_TYPE_HOST ||
      TgtPtrType == ZE_MEMORY_TYPE_SHARED) {
    bool CopyNow = true;
    if (IsAsync) {
      if (AsyncQueue->KernelEvent) {
        // Delay Host/Shared USM to host memory copy since it must wait for
        // kernel completion.
        AsyncQueue->USM2MList.emplace_back(TgtPtr, HstPtr, Size);
        CopyNow = false;
      }
    }
    if (CopyNow) {
      std::copy_n(static_cast<const char *>(TgtPtr), Size,
                  static_cast<char *>(HstPtr));
    }
  } else {
    void *DstPtr = HstPtr;
    if (isDiscreteDevice() &&
        static_cast<size_t>(Size) <=
            getPlugin().getOptions().StagingBufferSize &&
        getMemAllocType(HstPtr) != ZE_MEMORY_TYPE_HOST) {
      auto PtrOrErr = getStagingBuffer().get(IsAsync);
      if (!PtrOrErr)
        return PtrOrErr.takeError();
      DstPtr = *PtrOrErr;
    }
    if (IsAsync) {
      if (auto Err = enqueueMemCopyAsync(DstPtr, TgtPtr, Size, AsyncInfo,
                                         /* CopyTo */ false))
        return Err;
    } else {
      if (auto Err = enqueueMemCopy(DstPtr, TgtPtr, Size, AsyncInfo))
        return Err;
    }
    if (DstPtr != HstPtr) {
      if (IsAsync) {
        // Store delayed H2M data copies
        auto &H2MList = AsyncQueue->H2MList;
        H2MList.emplace_back(DstPtr, HstPtr, static_cast<size_t>(Size));
      } else {
        std::copy_n(static_cast<char *>(DstPtr), Size,
                    static_cast<char *>(HstPtr));
      }
    }
  }
  INFO(OMP_INFOTYPE_PLUGIN_KERNEL, DeviceId,
       "%s %" PRId64 " bytes (tgt:" DPxMOD ") -> (hst:" DPxMOD ")\n",
       IsAsync ? "Submitted copy" : "Copied", Size, DPxPTR(TgtPtr),
       DPxPTR(HstPtr));
  return Plugin::success();
}

Error L0DeviceTy::dataExchangeImpl(const void *SrcPtr, GenericDeviceTy &DstDev,
                                   void *DstPtr, int64_t Size,
                                   AsyncInfoWrapperTy &AsyncInfoWrapper) {

  L0DeviceTy &L0DstDev = L0DeviceTy::makeL0Device(DstDev);
  // Use copy engine only for across-tile/device copies.
  const bool UseCopyEngine = getZeDevice() != L0DstDev.getZeDevice();

  if (asyncEnabled() && AsyncInfoWrapper.hasQueue()) {
    if (auto Err = enqueueMemCopyAsync(DstPtr, SrcPtr, Size,
                                       (__tgt_async_info *)AsyncInfoWrapper))
      return Err;
  } else {
    if (auto Err = enqueueMemCopy(DstPtr, SrcPtr, Size,
                                  /* AsyncInfo */ nullptr, UseCopyEngine))
      return Err;
  }
  return Plugin::success();
}

Error L0DeviceTy::initAsyncInfoImpl(AsyncInfoWrapperTy &AsyncInfoWrapper) {
  AsyncQueueTy *Queue = AsyncInfoWrapper.getQueueAs<AsyncQueueTy *>();
  if (!Queue) {
    Queue = getPlugin().getAsyncQueue();
    AsyncInfoWrapper.setQueueAs<AsyncQueueTy *>(Queue);
  }
  return Plugin::success();
}

static const char *DriverVersionToStrTable[] = {
    "1.0", "1.1", "1.2", "1.3",  "1.4",  "1.5", "1.6",
    "1.7", "1.8", "1.9", "1.10", "1.11", "1.12"};
constexpr size_t DriverVersionToStrTableSize =
    sizeof(DriverVersionToStrTable) / sizeof(DriverVersionToStrTable[0]);

Expected<InfoTreeNode> L0DeviceTy::obtainInfoImpl() {
  InfoTreeNode Info;
  Info.add("Device Number", getDeviceId());
  Info.add("Device Name", getNameCStr(), "", DeviceInfo::NAME);
  Info.add("Device Type", "GPU", "", DeviceInfo::TYPE);
  Info.add("Vendor", "Intel", "", DeviceInfo::VENDOR);
  Info.add("Vendor ID", getVendorId(), "", DeviceInfo::VENDOR_ID);
  auto DriverVersion = getDriverAPIVersion();
  if (DriverVersion < DriverVersionToStrTableSize)
    Info.add("Driver Version", DriverVersionToStrTable[DriverVersion], "",
             DeviceInfo::DRIVER_VERSION);
  else
    Info.add("Driver Version", "Unknown", "", DeviceInfo::DRIVER_VERSION);
  Info.add("Device PCI ID", getPCIId());
  Info.add("Device UUID", getUuid().data());
  Info.add("Number of total EUs", getNumEUs(), "",
           DeviceInfo::NUM_COMPUTE_UNITS);
  Info.add("Number of threads per EU", getNumThreadsPerEU());
  Info.add("EU SIMD width", getSIMDWidth());
  Info.add("Number of EUs per subslice", getNumEUsPerSubslice());
  Info.add("Number of subslices per slice", getNumSubslicesPerSlice());
  Info.add("Number of slices", getNumSlices());
  Info.add("Max Group size", getMaxGroupSize(), "",
           DeviceInfo::MAX_WORK_GROUP_SIZE);
  Info.add("Local memory size (bytes)", getMaxSharedLocalMemory());
  Info.add("Global memory size (bytes)", getGlobalMemorySize(), "",
           DeviceInfo::GLOBAL_MEM_SIZE);
  Info.add("Cache size (bytes)", getCacheSize());
  Info.add("Max Memory Allocation Size (bytes)", getMaxMemAllocSize(), "",
           DeviceInfo::MAX_MEM_ALLOC_SIZE);
  Info.add("Max clock frequency (MHz)", getClockRate(), "",
           DeviceInfo::MAX_CLOCK_FREQUENCY);
  return Info;
}

Expected<GenericKernelTy &> L0DeviceTy::constructKernel(const char *Name) {
  // Allocate and construct the L0 kernel.
  L0KernelTy *L0Kernel = getPlugin().allocate<L0KernelTy>();
  if (!L0Kernel)
    return Plugin::error(ErrorCode::UNKNOWN,
                         "Failed to allocate memory for L0 kernel");

  new (L0Kernel) L0KernelTy(Name);

  return *L0Kernel;
}

uint32_t L0DeviceTy::getMemAllocType(const void *Ptr) const {
  ze_memory_allocation_properties_t properties = {
      ZE_STRUCTURE_TYPE_MEMORY_ALLOCATION_PROPERTIES,
      nullptr,                // extension
      ZE_MEMORY_TYPE_UNKNOWN, // type
      0,                      // id
      0,                      // page size
  };

  ze_result_t rc;
  CALL_ZE(rc, zeMemGetAllocProperties, getZeContext(), Ptr, &properties,
          nullptr);

  if (rc == ZE_RESULT_ERROR_INVALID_ARGUMENT)
    return ZE_MEMORY_TYPE_UNKNOWN;
  else
    return properties.type;
}

interop_spec_t L0DeviceTy::selectInteropPreference(int32_t InteropType,
                                                   int32_t NumPrefers,
                                                   interop_spec_t *Prefers) {
  // no supported preference found, set default to level_zero,
  // non-ordered unless is targetsync
  return interop_spec_t{
      tgt_fr_level_zero,
      {InteropType == kmp_interop_type_targetsync ? true : false /*inorder*/,
       0},
      0};
}

Expected<OmpInteropTy> L0DeviceTy::createInterop(int32_t InteropContext,
                                                 interop_spec_t &InteropSpec) {
  auto Ret =
      new omp_interop_val_t(DeviceId, (kmp_interop_type_t)InteropContext);
  Ret->fr_id = tgt_fr_level_zero;
  Ret->vendor_id = omp_vendor_intel;

  if (InteropContext == kmp_interop_type_target ||
      InteropContext == kmp_interop_type_targetsync) {
    Ret->device_info.Platform = getZeDriver();
    Ret->device_info.Device = getZeDevice();
    Ret->device_info.Context = getZeContext();
  }

  Ret->rtl_property = new L0Interop::Property();
  if (InteropContext == kmp_interop_type_targetsync) {
    Ret->async_info = new __tgt_async_info();
    auto L0 = static_cast<L0Interop::Property *>(Ret->rtl_property);

    bool InOrder = InteropSpec.attrs.inorder;
    Ret->attrs.inorder = InOrder;
    if (useImmForInterop()) {
      auto CmdListOrErr = createImmCmdList(InOrder);
      if (!CmdListOrErr) {
        delete Ret;
        return CmdListOrErr.takeError();
      }
      Ret->async_info->Queue = *CmdListOrErr;
      L0->ImmCmdList = *CmdListOrErr;
    } else {
      auto QueueOrErr = createCommandQueue(InOrder);
      if (!QueueOrErr) {
        delete Ret;
        return QueueOrErr.takeError();
      }
      Ret->async_info->Queue = *QueueOrErr;
      L0->CommandQueue =
          static_cast<ze_command_queue_handle_t>(Ret->async_info->Queue);
    }
  }

  return Ret;
}

Error L0DeviceTy::releaseInterop(OmpInteropTy Interop) {
  const auto DeviceId = getDeviceId();

  if (!Interop || Interop->device_id != (intptr_t)DeviceId) {
    return Plugin::error(ErrorCode::INVALID_ARGUMENT,
                         "Invalid/inconsistent OpenMP interop " DPxMOD "\n",
                         DPxPTR(Interop));
  }
  auto L0 = static_cast<L0Interop::Property *>(Interop->rtl_property);
  if (Interop->async_info && Interop->async_info->Queue) {
    if (useImmForInterop()) {
      auto ImmCmdList = L0->ImmCmdList;
      CALL_ZE_RET_ERROR(zeCommandListDestroy, ImmCmdList);
    } else {
      auto CmdQueue = L0->CommandQueue;
      CALL_ZE_RET_ERROR(zeCommandQueueDestroy, CmdQueue);
    }
  }
  delete L0;
  delete Interop;

  return Plugin::success();
}

Error L0DeviceTy::enqueueMemCopy(void *Dst, const void *Src, size_t Size,
                                 __tgt_async_info *AsyncInfo,
                                 bool UseCopyEngine) {
  ze_command_list_handle_t CmdList = nullptr;
  ze_command_queue_handle_t CmdQueue = nullptr;

  if (useImmForCopy()) {
    auto CmdListOrErr = UseCopyEngine ? getImmCopyCmdList() : getImmCmdList();
    if (!CmdListOrErr)
      return CmdListOrErr.takeError();
    CmdList = *CmdListOrErr;
    CALL_ZE_RET_ERROR(zeCommandListAppendMemoryCopy, CmdList, Dst, Src, Size,
                      nullptr, 0, nullptr);
    CALL_ZE_RET_ERROR(zeCommandListHostSynchronize, CmdList, UINT64_MAX);
  } else {
    if (UseCopyEngine) {
      auto CmdListOrErr = getCopyCmdList();
      if (!CmdListOrErr)
        return CmdListOrErr.takeError();
      CmdList = *CmdListOrErr;
      auto CmdQueueOrErr = getCopyCmdQueue();
      if (!CmdQueueOrErr)
        return CmdQueueOrErr.takeError();
      CmdQueue = *CmdQueueOrErr;
    } else {
      auto CmdListOrErr = getCmdList();
      if (!CmdListOrErr)
        return CmdListOrErr.takeError();
      CmdList = *CmdListOrErr;
      auto CmdQueueOrErr = getCmdQueue();
      if (!CmdQueueOrErr)
        return CmdQueueOrErr.takeError();
      CmdQueue = *CmdQueueOrErr;
    }

    CALL_ZE_RET_ERROR(zeCommandListAppendMemoryCopy, CmdList, Dst, Src, Size,
                      nullptr, 0, nullptr);
    CALL_ZE_RET_ERROR(zeCommandListClose, CmdList);
    CALL_ZE_RET_ERROR_MTX(zeCommandQueueExecuteCommandLists, getMutex(),
                          CmdQueue, 1, &CmdList, nullptr);
    CALL_ZE_RET_ERROR(zeCommandQueueSynchronize, CmdQueue, UINT64_MAX);
    CALL_ZE_RET_ERROR(zeCommandListReset, CmdList);
  }
  return Plugin::success();
}

/// Enqueue non-blocking memory copy. This function is invoked only when IMM is
/// fully enabled and async mode is requested.
Error L0DeviceTy::enqueueMemCopyAsync(void *Dst, const void *Src, size_t Size,
                                      __tgt_async_info *AsyncInfo,
                                      bool CopyTo) {
  const bool Ordered =
      (getPlugin().getOptions().CommandMode == CommandModeTy::AsyncOrdered);
  auto EventOrErr = getEvent();
  if (!EventOrErr)
    return EventOrErr.takeError();
  ze_event_handle_t SignalEvent = *EventOrErr;
  size_t NumWaitEvents = 0;
  ze_event_handle_t *WaitEvents = nullptr;
  AsyncQueueTy *AsyncQueue = reinterpret_cast<AsyncQueueTy *>(AsyncInfo->Queue);
  if (!AsyncQueue->WaitEvents.empty()) {
    // Use a single wait event if events are ordered or a kernel event exists.
    NumWaitEvents = 1;
    if (Ordered)
      WaitEvents = &AsyncQueue->WaitEvents.back();
    else if (AsyncQueue->KernelEvent)
      WaitEvents = &AsyncQueue->KernelEvent;
    else
      NumWaitEvents = 0;
  }
  auto CmdListOrError = getImmCopyCmdList();
  if (!CmdListOrError)
    return CmdListOrError.takeError();
  const auto CmdList = *CmdListOrError;
  CALL_ZE_RET_ERROR(zeCommandListAppendMemoryCopy, CmdList, Dst, Src, Size,
                    SignalEvent, NumWaitEvents, WaitEvents);
  AsyncQueue->WaitEvents.push_back(SignalEvent);
  return Plugin::success();
}

/// Enqueue memory fill
Error L0DeviceTy::enqueueMemFill(void *Ptr, const void *Pattern,
                                 size_t PatternSize, size_t Size) {
  if (useImmForCopy()) {
    auto CmdListOrErr = getImmCopyCmdList();
    if (!CmdListOrErr)
      return CmdListOrErr.takeError();
    const auto CmdList = *CmdListOrErr;
    auto EventOrErr = getEvent();
    if (!EventOrErr)
      return EventOrErr.takeError();
    ze_event_handle_t Event = *EventOrErr;
    CALL_ZE_RET_ERROR(zeCommandListAppendMemoryFill, CmdList, Ptr, Pattern,
                      PatternSize, Size, Event, 0, nullptr);
    CALL_ZE_RET_ERROR(zeEventHostSynchronize, Event, UINT64_MAX);
  } else {
    auto CmdListOrErr = getCopyCmdList();
    if (!CmdListOrErr)
      return CmdListOrErr.takeError();
    auto CmdList = *CmdListOrErr;
    auto CmdQueueOrErr = getCopyCmdQueue();
    if (!CmdQueueOrErr)
      return CmdQueueOrErr.takeError();
    const auto CmdQueue = *CmdQueueOrErr;
    CALL_ZE_RET_ERROR(zeCommandListAppendMemoryFill, CmdList, Ptr, Pattern,
                      PatternSize, Size, nullptr, 0, nullptr);
    CALL_ZE_RET_ERROR(zeCommandListClose, CmdList);
    CALL_ZE_RET_ERROR(zeCommandQueueExecuteCommandLists, CmdQueue, 1, &CmdList,
                      nullptr);
    CALL_ZE_RET_ERROR(zeCommandQueueSynchronize, CmdQueue, UINT64_MAX);
    CALL_ZE_RET_ERROR(zeCommandListReset, CmdList);
  }
  return Plugin::success();
}

Error L0DeviceTy::dataFillImpl(void *TgtPtr, const void *PatternPtr,
                               int64_t PatternSize, int64_t Size,
                               AsyncInfoWrapperTy &AsyncInfoWrapper) {
  // TODO: support async version
  return enqueueMemFill(TgtPtr, PatternPtr, PatternSize, Size);
}

Expected<void *> L0DeviceTy::dataAlloc(size_t Size, size_t Align, int32_t Kind,
                                       intptr_t Offset, bool UserAlloc,
                                       bool DevMalloc, uint32_t MemAdvice,
                                       AllocOptionTy AllocOpt) {

  const bool UseDedicatedPool =
      (AllocOpt == AllocOptionTy::ALLOC_OPT_REDUCTION_SCRATCH) ||
      (AllocOpt == AllocOptionTy::ALLOC_OPT_REDUCTION_COUNTER);
  if (Kind == TARGET_ALLOC_DEFAULT) {
    if (UserAlloc)
      Kind = TARGET_ALLOC_DEVICE;
    else if (AllocOpt == AllocOptionTy::ALLOC_OPT_HOST_MEM)
      Kind = TARGET_ALLOC_HOST;
    else if (UseDedicatedPool)
      Kind = TARGET_ALLOC_DEVICE;
    else
      Kind = getAllocKind();
  }
  auto &Allocator = getMemAllocator(Kind);
  return Allocator.alloc(Size, Align, Kind, Offset, UserAlloc, DevMalloc,
                         MemAdvice, AllocOpt);
}

Error L0DeviceTy::dataDelete(void *Ptr) {
  auto &Allocator = getMemAllocator(Ptr);
  return Allocator.dealloc(Ptr);
}

Error L0DeviceTy::makeMemoryResident(void *Mem, size_t Size) {
  CALL_ZE_RET_ERROR(zeContextMakeMemoryResident, getZeContext(), getZeDevice(),
                    Mem, Size);
  return Plugin::success();
}

// Command queues related functions
/// Create a command list with given ordinal and flags
Expected<ze_command_list_handle_t> L0DeviceTy::createCmdList(
    ze_context_handle_t Context, ze_device_handle_t Device, uint32_t Ordinal,
    ze_command_list_flags_t Flags, const std::string_view DeviceIdStr) {
  ze_command_list_desc_t cmdListDesc = {ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC,
                                        nullptr, // extension
                                        Ordinal, Flags};
  ze_command_list_handle_t cmdList;
  CALL_ZE_RET_ERROR(zeCommandListCreate, Context, Device, &cmdListDesc,
                    &cmdList);
  DP("Created a command list " DPxMOD " (Ordinal: %" PRIu32
     ") for device %s.\n",
     DPxPTR(cmdList), Ordinal, DeviceIdStr.data());
  return cmdList;
}

/// Create a command list with default flags
Expected<ze_command_list_handle_t>
L0DeviceTy::createCmdList(ze_context_handle_t Context,
                          ze_device_handle_t Device, uint32_t Ordinal,
                          const std::string_view DeviceIdStr) {
  return (Ordinal == UINT32_MAX)
             ? nullptr
             : createCmdList(Context, Device, Ordinal, 0, DeviceIdStr);
}

Expected<ze_command_list_handle_t> L0DeviceTy::getCmdList() {
  auto &TLS = getTLS();
  auto CmdList = TLS.getCmdList();
  if (!CmdList) {
    auto CmdListOrErr = createCmdList(getZeContext(), getZeDevice(),
                                      getComputeEngine(), getZeId());
    if (!CmdListOrErr)
      return CmdListOrErr.takeError();
    CmdList = *CmdListOrErr;
    TLS.setCmdList(CmdList);
  }
  return CmdList;
}

/// Create a command queue with given ordinal and flags
Expected<ze_command_queue_handle_t>
L0DeviceTy::createCmdQueue(ze_context_handle_t Context,
                           ze_device_handle_t Device, uint32_t Ordinal,
                           uint32_t Index, ze_command_queue_flags_t Flags,
                           const std::string_view DeviceIdStr) {
  ze_command_queue_desc_t cmdQueueDesc = {ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC,
                                          nullptr, // extension
                                          Ordinal,
                                          Index,
                                          Flags, // flags
                                          ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS,
                                          ZE_COMMAND_QUEUE_PRIORITY_NORMAL};
  ze_command_queue_handle_t cmdQueue;
  CALL_ZE_RET_ERROR(zeCommandQueueCreate, Context, Device, &cmdQueueDesc,
                    &cmdQueue);
  DP("Created a command queue " DPxMOD " (Ordinal: %" PRIu32 ", Index: %" PRIu32
     ", Flags: %" PRIu32 ") for device %s.\n",
     DPxPTR(cmdQueue), Ordinal, Index, Flags, DeviceIdStr.data());
  return cmdQueue;
}

/// Create a command queue with default flags
Expected<ze_command_queue_handle_t> L0DeviceTy::createCmdQueue(
    ze_context_handle_t Context, ze_device_handle_t Device, uint32_t Ordinal,
    uint32_t Index, const std::string_view DeviceIdStr, bool InOrder) {
  ze_command_queue_flags_t Flags = InOrder ? ZE_COMMAND_QUEUE_FLAG_IN_ORDER : 0;
  return (Ordinal == UINT32_MAX) ? nullptr
                                 : createCmdQueue(Context, Device, Ordinal,
                                                  Index, Flags, DeviceIdStr);
}

/// Create a new command queue for the given OpenMP device ID
Expected<ze_command_queue_handle_t>
L0DeviceTy::createCommandQueue(bool InOrder) {
  auto cmdQueue =
      createCmdQueue(getZeContext(), getZeDevice(), getComputeEngine(),
                     getComputeIndex(), getZeId(), InOrder);
  return cmdQueue;
}

/// Create an immediate command list
Expected<ze_command_list_handle_t>
L0DeviceTy::createImmCmdList(uint32_t Ordinal, uint32_t Index, bool InOrder) {
  ze_command_queue_flags_t Flags = InOrder ? ZE_COMMAND_QUEUE_FLAG_IN_ORDER : 0;
  ze_command_queue_desc_t Desc{ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC,
                               nullptr,
                               Ordinal,
                               Index,
                               Flags,
                               ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS,
                               ZE_COMMAND_QUEUE_PRIORITY_NORMAL};
  ze_command_list_handle_t CmdList = nullptr;
  CALL_ZE_RET_ERROR(zeCommandListCreateImmediate, getZeContext(), getZeDevice(),
                    &Desc, &CmdList);
  DP("Created an immediate command list " DPxMOD " (Ordinal: %" PRIu32
     ", Index: %" PRIu32 ", Flags: %" PRIu32 ") for device %s.\n",
     DPxPTR(CmdList), Ordinal, Index, Flags, getZeIdCStr());
  return CmdList;
}

/// Create an immediate command list for copying
Expected<ze_command_list_handle_t> L0DeviceTy::createImmCopyCmdList() {
  uint32_t Ordinal = getMainCopyEngine();
  if (Ordinal == UINT32_MAX)
    Ordinal = getComputeEngine();
  return createImmCmdList(Ordinal, /*Index*/ 0);
}

Expected<ze_command_queue_handle_t> L0DeviceTy::getCmdQueue() {
  auto &TLS = getTLS();
  auto CmdQueue = TLS.getCmdQueue();
  if (!CmdQueue) {
    auto CmdQueueOrErr = createCommandQueue();
    if (!CmdQueueOrErr)
      return CmdQueueOrErr.takeError();
    CmdQueue = *CmdQueueOrErr;
    TLS.setCmdQueue(CmdQueue);
  }
  return CmdQueue;
}

Expected<ze_command_list_handle_t> L0DeviceTy::getCopyCmdList() {
  // Use main copy engine if available
  if (hasMainCopyEngine()) {
    auto &TLS = getTLS();
    auto CmdList = TLS.getCopyCmdList();
    if (!CmdList) {
      auto CmdListOrErr = createCmdList(getZeContext(), getZeDevice(),
                                        getMainCopyEngine(), getZeId());
      if (!CmdListOrErr)
        return CmdListOrErr.takeError();
      CmdList = *CmdListOrErr;
      TLS.setCopyCmdList(CmdList);
    }
    return CmdList;
  }
  // Use compute engine otherwise
  return getCmdList();
}

Expected<ze_command_queue_handle_t> L0DeviceTy::getCopyCmdQueue() {
  // Use main copy engine if available
  if (hasMainCopyEngine()) {
    auto &TLS = getTLS();
    auto CmdQueue = TLS.getCopyCmdQueue();
    if (!CmdQueue) {
      auto CmdQueueOrErr = createCmdQueue(getZeContext(), getZeDevice(),
                                          getMainCopyEngine(), 0, getZeId());
      if (!CmdQueueOrErr)
        return CmdQueueOrErr.takeError();
      CmdQueue = *CmdQueueOrErr;
      TLS.setCopyCmdQueue(CmdQueue);
    }
    return CmdQueue;
  }
  // Use compute engine otherwise
  return getCmdQueue();
}

Expected<ze_command_list_handle_t> L0DeviceTy::getImmCmdList() {
  auto &TLS = getTLS();
  auto CmdList = TLS.getImmCmdList();
  if (!CmdList) {
    auto CmdListOrErr = createImmCmdList();
    if (!CmdListOrErr)
      return CmdListOrErr.takeError();
    CmdList = *CmdListOrErr;
    TLS.setImmCmdList(CmdList);
  }
  return CmdList;
}

Expected<ze_command_list_handle_t> L0DeviceTy::getImmCopyCmdList() {
  auto &TLS = getTLS();
  auto CmdList = TLS.getImmCopyCmdList();
  if (!CmdList) {
    auto CmdListOrErr = createImmCopyCmdList();
    if (!CmdListOrErr)
      return CmdListOrErr.takeError();
    CmdList = *CmdListOrErr;
    TLS.setImmCopyCmdList(CmdList);
  }
  return CmdList;
}

Error L0DeviceTy::dataFence(__tgt_async_info *Async) {
  const bool Ordered =
      (getPlugin().getOptions().CommandMode == CommandModeTy::AsyncOrdered);

  // Nothing to do if everything is ordered
  if (Ordered)
    return Plugin::success();

  ze_command_list_handle_t CmdList = nullptr;
  ze_command_queue_handle_t CmdQueue = nullptr;

  if (useImmForCopy()) {
    auto CmdListOrErr = getImmCopyCmdList();
    if (!CmdListOrErr)
      return CmdListOrErr.takeError();
    auto CmdList = *CmdListOrErr;
    CALL_ZE_RET_ERROR(zeCommandListAppendBarrier, CmdList, nullptr, 0, nullptr);
  } else {
    auto CmdListOrErr = getCopyCmdList();
    if (!CmdListOrErr)
      return CmdListOrErr.takeError();
    auto CmdQueueOrerr = getCopyCmdQueue();
    if (!CmdQueueOrerr)
      return CmdQueueOrerr.takeError();

    CmdList = *CmdListOrErr;
    CmdQueue = *CmdQueueOrerr;
    CALL_ZE_RET_ERROR(zeCommandListAppendBarrier, CmdList, nullptr, 0, nullptr);
    CALL_ZE_RET_ERROR(zeCommandListClose, CmdList);
    CALL_ZE_RET_ERROR(zeCommandQueueExecuteCommandLists, CmdQueue, 1, &CmdList,
                      nullptr);
    CALL_ZE_RET_ERROR(zeCommandListReset, CmdList);
  }

  return Plugin::success();
}

} // namespace llvm::omp::target::plugin
