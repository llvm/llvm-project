//===--- Level Zero Target RTL Implementation -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// GenericDevice instatiation for SPIR-V/Xe machine.
//
//===----------------------------------------------------------------------===//

#include "L0Device.h"
#include "L0Defs.h"
#include "L0Event.h"
#include "L0Interop.h"
#include "L0Plugin.h"
#include "L0Program.h"
#include "L0Trace.h"

#include "GlobalHandler.h"
#include "OffloadAPI.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/Object/ELF.h"

namespace llvm::omp::target::plugin {

// clang-format off
/// Mapping from device arch to GPU runtime's device identifiers.
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
  if (PCIDeviceId == 0) {
    ODBG(OLDT_Device) << "Warning: Cannot decide device arch for " << getName()
                      << ".";
    return DeviceArchTy::DeviceArch_None;
  }

  for (int ArchIndex = 0; ArchIndex < DeviceArchMapSize; ArchIndex++) {
    for (int i = 0;; i++) {
      const auto Id = DeviceArchMap[ArchIndex].ids[i];
      if (Id == PCIIdTy::None)
        break;
      auto maskedId = static_cast<PCIIdTy>(PCIDeviceId & 0xFF00);
      if (maskedId == Id)
        return DeviceArchMap[ArchIndex].arch; // Exact match or prefix match.
    }
  }

  ODBG(OLDT_Device) << "Warning: Cannot decide device arch for " << getName()
                    << ".";
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

/// Get default compute group ordinal. Returns Ordinal-NumQueues pair.
std::pair<uint32_t, uint32_t> L0DeviceTy::findComputeOrdinal() {
  std::pair<uint32_t, uint32_t> Ordinal{MaxOrdinal, 0};
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
    // support cooperative kernels.
    if (Properties[I].flags & ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COMPUTE) {
      Ordinal.first = I;
      Ordinal.second = Properties[I].numQueues;
      break;
    }
  }
  if (Ordinal.first == MaxOrdinal)
    ODBG(OLDT_Device) << "Error: no command queues are found";

  return Ordinal;
}

/// Check if device supports cooperative kernels by checking if any command
/// queue group has the cooperative kernels flag set.
bool L0DeviceTy::checkCooperativeKernelSupport() {
  uint32_t Count = 0;
  const auto zeDevice = getZeDevice();
  CALL_ZE_RET(false, zeDeviceGetCommandQueueGroupProperties, zeDevice, &Count,
              nullptr);

  std::vector<ze_command_queue_group_properties_t> Properties(
      Count,
      {ZE_STRUCTURE_TYPE_COMMAND_QUEUE_GROUP_PROPERTIES, nullptr, 0, 0, 0});
  CALL_ZE_RET(false, zeDeviceGetCommandQueueGroupProperties, zeDevice, &Count,
              Properties.data());

  for (auto &Property : Properties)
    if (Property.flags &
        ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COOPERATIVE_KERNELS)
      return true;

  return false;
}

void L0DeviceTy::reportDeviceInfo() const {
  ODBG_OS(OLDT_Device, [&](llvm::raw_ostream &O) {
    O << "Device " << DeviceId << " information\n"
      << "-- Name                         : " << getName() << "\n"
      << "-- PCI ID                       : "
      << llvm::format("0x%" PRIx32, getPCIId()) << "\n"
      << "-- UUID                         : " << getUuid().data() << "\n"
      << "-- Number of total EUs          : " << getNumEUs() << "\n"
      << "-- Number of threads per EU     : " << getNumThreadsPerEU() << "\n"
      << "-- EU SIMD width                : " << getSIMDWidth() << "\n"
      << "-- Number of EUs per subslice   : " << getNumEUsPerSubslice() << "\n"
      << "-- Number of subslices per slice: " << getNumSubslicesPerSlice()
      << "\n"
      << "-- Number of slices             : " << getNumSlices() << "\n"
      << "-- Local memory size (bytes)    : " << getMaxSharedLocalMemory()
      << "\n"
      << "-- Global memory size (bytes)   : " << getGlobalMemorySize() << "\n"
      << "-- Cache size (bytes)           : " << getCacheSize() << "\n"
      << "-- Max clock frequency (MHz)    : " << getClockRate() << "\n";
  });
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
  CALL_ZE_RET_ERROR(zeDeviceGetModuleProperties, zeDevice, &ModuleProperties);

  DeviceName = std::string(DeviceProperties.name);

  ODBG(OLDT_Device) << "Found a GPU device, Name = " << DeviceProperties.name;

  DeviceArch = computeArch();
  // Default allocation kind for this device.
  AllocKind = isDiscreteDevice() ? TARGET_ALLOC_DEVICE : TARGET_ALLOC_SHARED;

  ze_kernel_indirect_access_flags_t Flags =
      (AllocKind == TARGET_ALLOC_DEVICE)
          ? ZE_KERNEL_INDIRECT_ACCESS_FLAG_DEVICE
          : ZE_KERNEL_INDIRECT_ACCESS_FLAG_SHARED;
  IndirectAccessFlags = Flags;

  // Get the UUID.
  std::string uid;
  for (int n = 0; n < ZE_MAX_DEVICE_UUID_SIZE; n++)
    uid += std::to_string(DeviceProperties.uuid.id[n]);
  DeviceUuid = std::move(uid);

  ComputeOrdinal = findComputeOrdinal();
  QueueCache.setCommandMode(getPlugin().getOptions().CommandMode);

  SupportsCooperativeKernels = checkCooperativeKernelSupport();

  if (auto Err = MemAllocator.initDevicePools(*this, Options))
    return Err;
  l0Context.getHostMemAllocator().updateMaxAllocSize(*this);
  reportDeviceInfo();
  return Plugin::success();
}

Error L0DeviceTy::deinitImpl() {
  for (auto &PGM : Programs)
    if (auto Err = PGM.deinit())
      return Err;
  if (auto Err = QueueCache.deinit())
    return Err;
  return MemAllocator.deinit();
}

Expected<DeviceImageTy *>
L0DeviceTy::loadBinaryImpl(std::unique_ptr<MemoryBuffer> &&TgtImage,
                           int32_t ImageId) {
  auto *PGM = getProgramFromImage(TgtImage->getMemBufferRef());
  if (PGM) {
    // Program already exists.
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

  L0ProgramBuilderTy Builder(*this, std::move(TgtImage));
  if (auto Err = Builder.buildModules(CompilationOptions))
    return std::move(Err);

  auto ProgramOrErr = addProgram(ImageId, Builder);
  if (!ProgramOrErr)
    return ProgramOrErr.takeError();
  auto &Program = *ProgramOrErr;

  if (auto Err = Program.loadModuleKernels())
    return std::move(Err);

  return &Program;
}

Error L0DeviceTy::unloadBinaryImpl(DeviceImageTy *Image) {
  // Ignoring for now.
  // TODO: call properly L0Program unload.
  return Plugin::success();
}

Expected<L0QueueTy *>
L0DeviceTy::getOrCreateQueue(__tgt_async_info *AsyncInfo) {
  L0QueueTy *Queue = static_cast<L0QueueTy *>(AsyncInfo->Queue);
  if (!Queue) {
    auto NewQueueOrErr = QueueCache.getQueue();
    if (!NewQueueOrErr)
      return NewQueueOrErr.takeError();
    Queue = *NewQueueOrErr;
    AsyncInfo->Queue = Queue;
  }
  return Queue;
}

Error L0DeviceTy::synchronizeImpl(__tgt_async_info &AsyncInfo,
                                  bool ReleaseQueue) {

  L0QueueTy *Queue = static_cast<L0QueueTy *>(AsyncInfo.Queue);
  if (!Queue)
    return Plugin::success();

  Error SyncErr = Queue->synchronize();

  if (ReleaseQueue) {
    releaseQueue(Queue);
    getStagingBuffer().reset();
    AsyncInfo.Queue = nullptr;
  }

  return SyncErr;
}

Expected<bool>
L0DeviceTy::hasPendingWorkImpl(AsyncInfoWrapperTy &AsyncInfoWrapper) {
  L0QueueTy *Queue = AsyncInfoWrapper.getQueueAs<L0QueueTy *>();
  if (!Queue)
    return false;
  return Queue->hasPendingWork();
}

Error L0DeviceTy::queryAsyncImpl(__tgt_async_info &AsyncInfo, bool ReleaseQueue,
                                 bool *IsQueueWorkCompleted) {
  L0QueueTy *Queue = static_cast<L0QueueTy *>(AsyncInfo.Queue);
  bool WorkCompleted = true;

  if (Queue) {
    auto PendingWorkOrErr = Queue->hasPendingWork();
    if (!PendingWorkOrErr)
      return PendingWorkOrErr.takeError();
    WorkCompleted = !*PendingWorkOrErr;
  }

  if (IsQueueWorkCompleted)
    *IsQueueWorkCompleted = WorkCompleted;

  if (!WorkCompleted || !Queue)
    return Plugin::success();

  if (ReleaseQueue) {
    releaseQueue(Queue);
    getStagingBuffer().reset();
    AsyncInfo.Queue = nullptr;
  }

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

  __tgt_async_info *AsyncInfo = AsyncInfoWrapper;
  auto AsyncQueueOrErr = getOrCreateQueue(AsyncInfo);
  if (!AsyncQueueOrErr)
    return AsyncQueueOrErr.takeError();
  auto *AsyncQueue = *AsyncQueueOrErr;

  if (auto Err = AsyncQueue->dataSubmit(TgtPtr, HstPtr, Size))
    return Err;
  ODBG(OLDT_DataTransfer) << "Device " << getDeviceId() << ": Submitted "
                          << Size
                          << " bytes from host to device (hst:" << HstPtr
                          << ") -> (tgt:" << TgtPtr << ").";
  return Plugin::success();
}

Error L0DeviceTy::dataRetrieveImpl(void *HstPtr, const void *TgtPtr,
                                   int64_t Size,
                                   AsyncInfoWrapperTy &AsyncInfoWrapper) {
  if (Size == 0)
    return Plugin::success();

  __tgt_async_info *AsyncInfo = AsyncInfoWrapper;
  assert(AsyncInfo && "AsyncInfo must be provided for data retrieval");

  const auto DeviceId = getDeviceId();
  auto QueueOrErr = getOrCreateQueue(AsyncInfo);
  if (!QueueOrErr)
    return QueueOrErr.takeError();
  auto *Queue = *QueueOrErr;

  if (auto Err = Queue->dataRetrieve(HstPtr, TgtPtr, Size))
    return Err;
  ODBG(OLDT_DataTransfer) << "Device " << DeviceId << ": Retrieved " << Size
                          << " bytes from device to host (tgt:" << TgtPtr
                          << ") -> (hst:" << HstPtr << ").";
  return Plugin::success();
}

Error L0DeviceTy::dataExchangeImpl(const void *SrcPtr, GenericDeviceTy &DstDev,
                                   void *DstPtr, int64_t Size,
                                   AsyncInfoWrapperTy &AsyncInfoWrapper) {
  if (auto Err =
          enqueueMemCopy(DstPtr, SrcPtr, Size,
                         static_cast<__tgt_async_info *>(AsyncInfoWrapper)))
    return Err;
  return Plugin::success();
}

Error L0DeviceTy::initAsyncInfoImpl(AsyncInfoWrapperTy &AsyncInfoWrapper) {
  auto QueueOrErr = getOrCreateQueue(AsyncInfoWrapper);
  return QueueOrErr ? Plugin::success() : QueueOrErr.takeError();
}

const char *L0DeviceTy::getArchCStr() const {
  switch (getDeviceArch()) {
  case DeviceArchTy::DeviceArch_Gen:
    return "Intel GPU Xe";
  case DeviceArchTy::DeviceArch_XeLPG:
    return "Intel GPU Xe LPG";
  case DeviceArchTy::DeviceArch_XeHPC:
    return "Intel GPU Xe HPC";
  case DeviceArchTy::DeviceArch_XeHPG:
    return "Intel GPU Xe HPG";
  case DeviceArchTy::DeviceArch_Xe2LP:
    return "Intel GPU Xe2 LP";
  case DeviceArchTy::DeviceArch_Xe2HP:
    return "Intel GPU Xe HP";
  case DeviceArchTy::DeviceArch_x86_64:
    return "Intel X86 64";
  default:
    return "Intel GPU Unknown";
  }
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
  Info.add("Product Name", getArchCStr(), "", DeviceInfo::PRODUCT_NAME);
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
  auto &MaxGroupSize =
      *Info.add("Workgroup Max Size per Dimension", std::monostate{}, "",
                DeviceInfo::MAX_WORK_GROUP_SIZE_PER_DIMENSION);
  MaxGroupSize.add("x", getMaxGroupSizeX());
  MaxGroupSize.add("y", getMaxGroupSizeY());
  MaxGroupSize.add("z", getMaxGroupSizeZ());
  Info.add("Maximum Grid Dimensions", getMaxGroupSize() * getMaxGroupCount(),
           "", DeviceInfo::MAX_WORK_SIZE);
  auto &MaxSize = *Info.add("Grid Size per Dimension", std::monostate{}, "",
                            DeviceInfo::MAX_WORK_SIZE_PER_DIMENSION);
  MaxSize.add("x", getMaxGroupSizeX() * getMaxGroupCountX());
  MaxSize.add("y", getMaxGroupSizeY() * getMaxGroupCountY());
  MaxSize.add("z", getMaxGroupSizeZ() * getMaxGroupCountZ());

  Info.add("Local memory size (bytes)", getMaxSharedLocalMemory(), "",
           DeviceInfo::WORK_GROUP_LOCAL_MEM_SIZE);
  Info.add("Global memory size (bytes)", getGlobalMemorySize(), "",
           DeviceInfo::GLOBAL_MEM_SIZE);
  Info.add("Cache size (bytes)", getCacheSize());
  Info.add("Max Memory Allocation Size (bytes)", getMaxMemAllocSize(), "",
           DeviceInfo::MAX_MEM_ALLOC_SIZE);
  Info.add("Max clock frequency (MHz)", getClockRate(), "",
           DeviceInfo::MAX_CLOCK_FREQUENCY);
  Info.add("Max memory clock frequency (MHz)", getMemoryClockRate(), "",
           DeviceInfo::MEMORY_CLOCK_RATE);
  Info.add("Memory Address Size", uint64_t{64u}, "bits",
           DeviceInfo::ADDRESS_BITS);

  // FP64 (Double precision).
  Info.add("Double FP Support", supportsFP64(), "",
           DeviceInfo::DOUBLE_FP_SUPPORT);
  ol_device_fp_capability_flags_t DoubleFPCapabilities = 0;
  ze_device_fp_flags_t ZeDoubleFPFlags = getFP64Flags();
  if (ZeDoubleFPFlags & ZE_DEVICE_FP_FLAG_DENORM)
    DoubleFPCapabilities |= OL_DEVICE_FP_CAPABILITY_FLAG_DENORM;
  if (ZeDoubleFPFlags & ZE_DEVICE_FP_FLAG_INF_NAN)
    DoubleFPCapabilities |= OL_DEVICE_FP_CAPABILITY_FLAG_INF_NAN;
  if (ZeDoubleFPFlags & ZE_DEVICE_FP_FLAG_ROUND_TO_NEAREST)
    DoubleFPCapabilities |= OL_DEVICE_FP_CAPABILITY_FLAG_ROUND_TO_NEAREST;
  if (ZeDoubleFPFlags & ZE_DEVICE_FP_FLAG_ROUND_TO_ZERO)
    DoubleFPCapabilities |= OL_DEVICE_FP_CAPABILITY_FLAG_ROUND_TO_ZERO;
  if (ZeDoubleFPFlags & ZE_DEVICE_FP_FLAG_ROUND_TO_INF)
    DoubleFPCapabilities |= OL_DEVICE_FP_CAPABILITY_FLAG_ROUND_TO_INF;
  if (ZeDoubleFPFlags & ZE_DEVICE_FP_FLAG_FMA)
    DoubleFPCapabilities |= OL_DEVICE_FP_CAPABILITY_FLAG_FMA;
  if (ZeDoubleFPFlags & ZE_DEVICE_FP_FLAG_ROUNDED_DIVIDE_SQRT)
    DoubleFPCapabilities |=
        OL_DEVICE_FP_CAPABILITY_FLAG_CORRECTLY_ROUNDED_DIVIDE_SQRT;
  Info.add("Double FP Capabilities", DoubleFPCapabilities, "",
           DeviceInfo::DOUBLE_FP_CONFIG);

  // FP16 (Half precision).
  Info.add("Half FP Support", supportsFP16(), "", DeviceInfo::HALF_FP_SUPPORT);
  ol_device_fp_capability_flags_t HalfFPCapabilities = 0;
  ze_device_fp_flags_t ZeHalfFPFlags = getFP16Flags();
  if (ZeHalfFPFlags & ZE_DEVICE_FP_FLAG_DENORM)
    HalfFPCapabilities |= OL_DEVICE_FP_CAPABILITY_FLAG_DENORM;
  if (ZeHalfFPFlags & ZE_DEVICE_FP_FLAG_INF_NAN)
    HalfFPCapabilities |= OL_DEVICE_FP_CAPABILITY_FLAG_INF_NAN;
  if (ZeHalfFPFlags & ZE_DEVICE_FP_FLAG_ROUND_TO_NEAREST)
    HalfFPCapabilities |= OL_DEVICE_FP_CAPABILITY_FLAG_ROUND_TO_NEAREST;
  if (ZeHalfFPFlags & ZE_DEVICE_FP_FLAG_ROUND_TO_ZERO)
    HalfFPCapabilities |= OL_DEVICE_FP_CAPABILITY_FLAG_ROUND_TO_ZERO;
  if (ZeHalfFPFlags & ZE_DEVICE_FP_FLAG_ROUND_TO_INF)
    HalfFPCapabilities |= OL_DEVICE_FP_CAPABILITY_FLAG_ROUND_TO_INF;
  if (ZeHalfFPFlags & ZE_DEVICE_FP_FLAG_FMA)
    HalfFPCapabilities |= OL_DEVICE_FP_CAPABILITY_FLAG_FMA;
  if (ZeHalfFPFlags & ZE_DEVICE_FP_FLAG_ROUNDED_DIVIDE_SQRT)
    HalfFPCapabilities |=
        OL_DEVICE_FP_CAPABILITY_FLAG_CORRECTLY_ROUNDED_DIVIDE_SQRT;
  Info.add("Half FP Capabilities", HalfFPCapabilities, "",
           DeviceInfo::HALF_FP_CONFIG);

  // FP32 (Single FP).
  Info.add("Single FP Support", true, "", DeviceInfo::SINGLE_FP_SUPPORT);
  ol_device_fp_capability_flags_t SingleFPCapabilities = 0;
  ze_device_fp_flags_t ZeSingleFPFlags = getFP32Flags();
  if (ZeSingleFPFlags & ZE_DEVICE_FP_FLAG_DENORM)
    SingleFPCapabilities |= OL_DEVICE_FP_CAPABILITY_FLAG_DENORM;
  if (ZeSingleFPFlags & ZE_DEVICE_FP_FLAG_INF_NAN)
    SingleFPCapabilities |= OL_DEVICE_FP_CAPABILITY_FLAG_INF_NAN;
  if (ZeSingleFPFlags & ZE_DEVICE_FP_FLAG_ROUND_TO_NEAREST)
    SingleFPCapabilities |= OL_DEVICE_FP_CAPABILITY_FLAG_ROUND_TO_NEAREST;
  if (ZeSingleFPFlags & ZE_DEVICE_FP_FLAG_ROUND_TO_ZERO)
    SingleFPCapabilities |= OL_DEVICE_FP_CAPABILITY_FLAG_ROUND_TO_ZERO;
  if (ZeSingleFPFlags & ZE_DEVICE_FP_FLAG_ROUND_TO_INF)
    SingleFPCapabilities |= OL_DEVICE_FP_CAPABILITY_FLAG_ROUND_TO_INF;
  if (ZeSingleFPFlags & ZE_DEVICE_FP_FLAG_FMA)
    SingleFPCapabilities |= OL_DEVICE_FP_CAPABILITY_FLAG_FMA;
  if (ZeSingleFPFlags & ZE_DEVICE_FP_FLAG_ROUNDED_DIVIDE_SQRT)
    SingleFPCapabilities |=
        OL_DEVICE_FP_CAPABILITY_FLAG_CORRECTLY_ROUNDED_DIVIDE_SQRT;
  Info.add("Single FP Capabilities", SingleFPCapabilities, "",
           DeviceInfo::SINGLE_FP_CONFIG);

  Info.add("Cooperative launch support", SupportsCooperativeKernels, "",
           DeviceInfo::COOPERATIVE_LAUNCH_SUPPORT);
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
      nullptr,                // Extension.
      ZE_MEMORY_TYPE_UNKNOWN, // Type.
      0,                      // Id.
      0,                      // Page size.
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
  // non-ordered unless is targetsync.
  return interop_spec_t{
      tgt_fr_level_zero,
      {InteropType == kmp_interop_type_targetsync /*inorder*/, 0},
      0};
}

Expected<OmpInteropTy> L0DeviceTy::createInterop(int32_t InteropContext,
                                                 interop_spec_t &InteropSpec) {
  auto Ret = new omp_interop_val_t(
      DeviceId, static_cast<kmp_interop_type_t>(InteropContext));
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

    // Ensure cleanup on error
    llvm::scope_exit CleanupOnError([&]() {
      if (Ret->async_info)
        delete Ret->async_info;
      if (Ret->rtl_property)
        delete static_cast<L0Interop::Property *>(Ret->rtl_property);
      delete Ret;
    });

    auto L0 = static_cast<L0Interop::Property *>(Ret->rtl_property);

    bool InOrder = InteropSpec.attrs.inorder;
    Ret->attrs.inorder = InOrder;
    auto CmdListOrErr = createImmCmdList(InOrder);
    if (!CmdListOrErr)
      return CmdListOrErr.takeError();
    Ret->async_info->Queue = *CmdListOrErr;
    L0->ImmCmdList = *CmdListOrErr;

    CleanupOnError.release();
  }

  return Ret;
}

Error L0DeviceTy::releaseInterop(OmpInteropTy Interop) {
  const auto DeviceId = getDeviceId();

  if (!Interop || Interop->device_id != static_cast<intptr_t>(DeviceId)) {
    return Plugin::error(ErrorCode::INVALID_ARGUMENT,
                         "Invalid/inconsistent OpenMP interop " DPxMOD "\n",
                         DPxPTR(Interop));
  }
  auto L0 = static_cast<L0Interop::Property *>(Interop->rtl_property);
  if (Interop->async_info && Interop->async_info->Queue) {
    auto ImmCmdList = L0->ImmCmdList;
    CALL_ZE_RET_ERROR(zeCommandListDestroy, ImmCmdList);
  }
  delete L0;
  delete Interop;

  return Plugin::success();
}

/// Enqueue fill command.
Error L0DeviceTy::enqueueMemFill(void *Ptr, const void *Pattern,
                                 size_t PatternSize, size_t Size,
                                 __tgt_async_info *AsyncInfo) {
  auto QueueOrErr = getOrCreateQueue(AsyncInfo);
  if (!QueueOrErr)
    return QueueOrErr.takeError();
  L0QueueTy *AsyncQueue = *QueueOrErr;
  return AsyncQueue->memoryFill(Ptr, Pattern, PatternSize, Size);
}

Error L0DeviceTy::dataFillImpl(void *TgtPtr, const void *PatternPtr,
                               int64_t PatternSize, int64_t Size,
                               AsyncInfoWrapperTy &AsyncInfoWrapper) {
  return enqueueMemFill(TgtPtr, PatternPtr, PatternSize, Size,
                        AsyncInfoWrapper);
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

/// Create an immediate command list.
Expected<ze_command_list_handle_t>
L0DeviceTy::createImmCmdList(uint32_t Ordinal, uint32_t Index, bool InOrder) {
  ze_command_queue_flags_t Flags = InOrder ? ZE_COMMAND_QUEUE_FLAG_IN_ORDER : 0;
  ze_command_queue_desc_t Desc{ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC,
                               nullptr,
                               Ordinal,
                               Index,
                               Flags | ZE_COMMAND_QUEUE_FLAG_COPY_OFFLOAD_HINT,
                               ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS,
                               ZE_COMMAND_QUEUE_PRIORITY_NORMAL};
  ze_command_list_handle_t CmdList = nullptr;
  CALL_ZE_RET_ERROR(zeCommandListCreateImmediate, getZeContext(), getZeDevice(),
                    &Desc, &CmdList);
  ODBG(OLDT_Device) << "Created an immediate command list " << CmdList
                    << " (Ordinal: " << Ordinal << ", Index: " << Index
                    << ", Flags: " << Flags << ") for device " << getZeIdCStr();
  return CmdList;
}

Error L0DeviceTy::dataFence(__tgt_async_info *Async) {
  auto QueueOrErr = getOrCreateQueue(Async);
  if (!QueueOrErr)
    return QueueOrErr.takeError();
  L0QueueTy *Queue = *QueueOrErr;
  return Queue->dataFence();
}

Expected<bool> L0DeviceTy::isAccessiblePtrImpl(const void *Ptr, size_t Size) {
  if (!Ptr || Size == 0)
    return Plugin::error(ErrorCode::INVALID_ARGUMENT,
                         "Invalid input to %s (Ptr = %p, Size = %zu)", __func__,
                         Ptr, Size);
  return getMemAllocator(Ptr).contains(Ptr, Size);
}

Error L0DeviceTy::createEventImpl(void **EventPtrStorage,
                                  bool EnableProfiling) {
  auto EventOrErr = getEventObject();
  if (!EventOrErr)
    return EventOrErr.takeError();
  *EventPtrStorage = *EventOrErr;
  return Plugin::success();
}

Error L0DeviceTy::destroyEventImpl(void *EventPtr, bool EnableProfiling) {
  L0EventTy *Event = static_cast<L0EventTy *>(EventPtr);
  return releaseEventObject(Event);
}

Error L0DeviceTy::recordEventImpl(void *EventPtr,
                                  AsyncInfoWrapperTy &AsyncInfoWrapper,
                                  bool EnableProfiling) {
  auto QueueOrErr = getOrCreateQueue(AsyncInfoWrapper);
  if (!QueueOrErr)
    return QueueOrErr.takeError();
  L0QueueTy *Queue = *QueueOrErr;
  L0EventTy *Event = static_cast<L0EventTy *>(EventPtr);
  Event->setQueue(*Queue);
  return Queue->appendSignalEvent(Event);
}

Error L0DeviceTy::waitEventImpl(void *EventPtr,
                                AsyncInfoWrapperTy &AsyncInfoWrapper) {
  auto QueueOrErr = getOrCreateQueue(AsyncInfoWrapper);
  if (!QueueOrErr)
    return QueueOrErr.takeError();
  L0QueueTy *Queue = *QueueOrErr;
  L0EventTy *Event = static_cast<L0EventTy *>(EventPtr);
  return Queue->appendWaitOnEvent(Event);
}

Error L0DeviceTy::syncEventImpl(void *EventPtr) {
  L0EventTy *Event = static_cast<L0EventTy *>(EventPtr);
  if (!Event->getQueue())
    return Plugin::error(ErrorCode::INVALID_ARGUMENT,
                         "event does not have any associated queue");
  return Event->getQueue()->synchronizeEvent(Event);
}

Expected<bool> L0DeviceTy::isEventCompleteImpl(void *EventPtr,
                                               AsyncInfoWrapperTy &) {
  L0EventTy *Event = static_cast<L0EventTy *>(EventPtr);
  if (!Event->getQueue())
    return Plugin::error(ErrorCode::INVALID_ARGUMENT,
                         "event does not have any associated queue");
  return Event->getQueue()->isEventComplete(Event);
}

Expected<float> L0DeviceTy::getEventElapsedTimeImpl(void *StartEventPtr,
                                                    void *EndEventPtr) {
  return Plugin::error(error::ErrorCode::UNKNOWN, "%s not implemented yet\n",
                       __func__);
}

Error L0DeviceTy::callGlobalConstructors(GenericPluginTy &Plugin,
                                         DeviceImageTy &Image) {
  return callGlobalCtorDtorCommon(Plugin, Image, /*IsCtor=*/true);
}

Error L0DeviceTy::callGlobalDestructors(GenericPluginTy &Plugin,
                                        DeviceImageTy &Image) {
  return callGlobalCtorDtorCommon(Plugin, Image, /*IsCtor=*/false);
}

Error L0DeviceTy::callGlobalCtorDtorCommon(GenericPluginTy &Plugin,
                                           DeviceImageTy &Image, bool IsCtor) {
  const char *KernelName = IsCtor ? "spirv$device$init" : "spirv$device$fini";

  // Check if a kernel was generated to run constructor or destructors.
  // It should be created by the 'spirv-lower-ctor-dtor' pass.
  GenericGlobalHandlerTy &Handler = Plugin.getGlobalHandler();
  if (!Handler.isSymbolInImage(*this, Image, KernelName))
    return Plugin::success();

  // Instead of returning errors directly, we capture them and provide
  // more of context about this routine.
  auto HandleErr = [&](Error Err) {
    std::string Buffer;
    llvm::raw_string_ostream(Buffer)
        << "failed to call global " << (IsCtor ? "constructors" : "destructors")
        << " in the image";
    return Plugin::error(ErrorCode::INVALID_BINARY, std::move(Err),
                         Buffer.c_str());
  };

  // The SPIR-V backend cannot handle creating the ctor / dtor array
  // automatically so we must create it ourselves. The backend will emit
  // several globals that contain function pointers we can call. These are
  // prefixed with a __init_array_object_ or __fini_array_object_.
  auto ELFObjOrErr = Handler.getELFObjectFile(Image);
  if (!ELFObjOrErr)
    return HandleErr(ELFObjOrErr.takeError());

  using FuncNameAndPriority = std::pair<StringRef, uint16_t>;
  SmallVector<FuncNameAndPriority> Funcs;
  for (ELFSymbolRef Sym : (*ELFObjOrErr)->symbols()) {
    auto NameOrErr = Sym.getName();
    if (!NameOrErr)
      return HandleErr(NameOrErr.takeError());

    if (!NameOrErr->starts_with(IsCtor ? "__init_array_object_"
                                       : "__fini_array_object_"))
      continue;

    uint16_t Priority;
    if (NameOrErr->rsplit('_').second.getAsInteger(10, Priority))
      return Plugin::error(
          ErrorCode::INVALID_BINARY,
          "failed to call global %s in the image: invalid priority",
          IsCtor ? "constructors" : "destructors");

    Funcs.emplace_back(*NameOrErr, Priority);
  }

  if (Funcs.empty()) {
    ODBG(OLDT_Module) << KernelName << " found in the image but no "
                      << (IsCtor ? "constructors" : "destructors")
                      << " found in the image.";
    return Plugin::success();
  }

  // Sort the created array to be in priority order.
  llvm::sort(Funcs,
             [](const auto &X, const auto &Y) { return X.second < Y.second; });

  auto BufferOrErr = allocate(Funcs.size() * sizeof(void *),
                              /*HostPtr=*/nullptr, TARGET_ALLOC_DEVICE);
  if (!BufferOrErr)
    return HandleErr(BufferOrErr.takeError());

  void *Buffer = *BufferOrErr;
  if (!Buffer)
    return Plugin::error(
        ErrorCode::OUT_OF_RESOURCES,
        "failed to allocate memory for global buffer to run %s",
        IsCtor ? "constructors" : "destructors");

  auto CleanupBufferAndErr = [&](Error RetErr) {
    if (auto Err = free(Buffer, TARGET_ALLOC_DEVICE)) {
      return joinErrors(std::move(RetErr), std::move(Err));
    }
    return RetErr;
  };

  auto *GlobalPtrStart = reinterpret_cast<uintptr_t *>(Buffer);
  auto *GlobalPtrStop = reinterpret_cast<uintptr_t *>(Buffer) + Funcs.size();

  SmallVector<void *> FunctionPtrs(Funcs.size());
  size_t Idx = 0;
  for (auto [Name, Priority] : Funcs) {
    GlobalTy FunctionAddr(Name.str(), sizeof(void *), &FunctionPtrs[Idx++]);
    if (auto Err = Handler.readGlobalFromDevice(*this, Image, FunctionAddr))
      return CleanupBufferAndErr(std::move(Err));
  }

  if (auto Err = dataSubmit(GlobalPtrStart, FunctionPtrs.data(),
                            FunctionPtrs.size() * sizeof(void *),
                            /*AsyncInfo=*/nullptr))
    return CleanupBufferAndErr(std::move(Err));

  GlobalTy StartGlobal(IsCtor ? "__init_array_start" : "__fini_array_start",
                       sizeof(void *), &GlobalPtrStart);
  if (auto Err = Handler.writeGlobalToDevice(*this, Image, StartGlobal))
    return CleanupBufferAndErr(std::move(Err));

  GlobalTy StopGlobal(IsCtor ? "__init_array_end" : "__fini_array_end",
                      sizeof(void *), &GlobalPtrStop);
  if (auto Err = Handler.writeGlobalToDevice(*this, Image, StopGlobal))
    return CleanupBufferAndErr(std::move(Err));

  // Call the generated kernel to execute the constructors or destructors.
  auto KernelOrErr = constructKernel(KernelName);
  if (!KernelOrErr)
    return CleanupBufferAndErr(KernelOrErr.takeError());

  GenericKernelTy &L0Kernel = *KernelOrErr;
  if (auto Err = L0Kernel.init(*this, Image))
    return CleanupBufferAndErr(std::move(Err));

  AsyncInfoWrapperTy AsyncInfoWrapper(*this, /*AsyncInfoPtr=*/nullptr);

  KernelArgsTy KernelArgs{};
  uint32_t NumBlocksAndThreads[3] = {1u, 1u, 1u};
  auto Err =
      L0Kernel.launchImpl(*this, NumBlocksAndThreads, NumBlocksAndThreads, 0,
                          KernelArgs, KernelLaunchParamsTy{}, AsyncInfoWrapper);

  AsyncInfoWrapper.finalize(Err);
  return CleanupBufferAndErr(std::move(Err));
}

} // namespace llvm::omp::target::plugin
