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

#ifndef OPENMP_LIBOMPTARGET_PLUGINS_NEXTGEN_LEVEL_ZERO_L0DEVICE_H
#define OPENMP_LIBOMPTARGET_PLUGINS_NEXTGEN_LEVEL_ZERO_L0DEVICE_H

#include "llvm/ADT/SmallVector.h"

#include "PerThreadTable.h"

#include "AsyncQueue.h"
#include "L0Context.h"
#include "L0Program.h"
#include "PluginInterface.h"
#include "TLS.h"

namespace llvm {
namespace omp {
namespace target {
namespace plugin {

using OmpInteropTy = omp_interop_val_t *;
class LevelZeroPluginTy;

// clang-format off
enum class PCIIdTy : int32_t {
  None            = 0x0000,
  SKL             = 0x1900,
  KBL             = 0x5900,
  CFL             = 0x3E00,
  CFL_2           = 0x9B00,
  ICX             = 0x8A00,
  TGL             = 0xFF20,
  TGL_2           = 0x9A00,
  DG1             = 0x4900,
  RKL             = 0x4C00,
  ADLS            = 0x4600,
  RTL             = 0xA700,
  MTL             = 0x7D00,
  PVC             = 0x0B00,
  DG2_ATS_M       = 0x4F00,
  DG2_ATS_M_2     = 0x5600,
  LNL             = 0x6400,
  BMG             = 0xE200,
};

/// Device type enumeration common to compiler and runtime
enum class DeviceArchTy : uint64_t {
  DeviceArch_None   = 0,
  DeviceArch_Gen    = 0x0001, // Gen 9, Gen 11 or Xe
  DeviceArch_XeLPG  = 0x0002,
  DeviceArch_XeHPC  = 0x0004,
  DeviceArch_XeHPG  = 0x0008,
  DeviceArch_Xe2LP  = 0x0010,
  DeviceArch_Xe2HP  = 0x0020,
  DeviceArch_x86_64 = 0x0100
};
// clang-format on

struct L0DeviceIdTy {
  ze_device_handle_t zeId;
  int32_t RootId;
  int32_t SubId;
  int32_t CCSId;

  L0DeviceIdTy(ze_device_handle_t Device, int32_t RootId, int32_t SubId = -1,
               int32_t CCSId = -1)
      : zeId(Device), RootId(RootId), SubId(SubId), CCSId(CCSId) {}
};

class L0DeviceTLSTy {
  /// Command list for each device
  ze_command_list_handle_t CmdList = nullptr;

  /// Main copy command list for each device
  ze_command_list_handle_t CopyCmdList = nullptr;

  /// Link copy command list for each device
  ze_command_list_handle_t LinkCopyCmdList = nullptr;

  /// Command queue for each device
  ze_command_queue_handle_t CmdQueue = nullptr;

  /// Main copy command queue for each device
  ze_command_queue_handle_t CopyCmdQueue = nullptr;

  /// Link copy command queues for each device
  ze_command_queue_handle_t LinkCopyCmdQueue = nullptr;

  /// Immediate command list for each device
  ze_command_list_handle_t ImmCmdList = nullptr;

  /// Immediate copy command list for each device
  ze_command_list_handle_t ImmCopyCmdList = nullptr;

public:
  L0DeviceTLSTy() = default;
  ~L0DeviceTLSTy() {
    // assert all fields are nullptr on destruction
    assert(CmdList == nullptr && "CmdList is not nullptr on destruction");
    assert(CopyCmdList == nullptr &&
           "CopyCmdList is not nullptr on destruction");
    assert(LinkCopyCmdList == nullptr &&
           "LinkCopyCmdList is not nullptr on destruction");
    assert(CmdQueue == nullptr && "CmdQueue is not nullptr on destruction");
    assert(CopyCmdQueue == nullptr &&
           "CopyCmdQueue is not nullptr on destruction");
    assert(LinkCopyCmdQueue == nullptr &&
           "LinkCopyCmdQueue is not nullptr on destruction");
    assert(ImmCmdList == nullptr && "ImmCmdList is not nullptr on destruction");
    assert(ImmCopyCmdList == nullptr &&
           "ImmCopyCmdList is not nullptr on destruction");
  }

  L0DeviceTLSTy(const L0DeviceTLSTy &) = delete;
  L0DeviceTLSTy(L0DeviceTLSTy &&Other) {
    CmdList = std::exchange(Other.CmdList, nullptr);
    CopyCmdList = std::exchange(Other.CopyCmdList, nullptr);
    LinkCopyCmdList = std::exchange(Other.LinkCopyCmdList, nullptr);
    CmdQueue = std::exchange(Other.CmdQueue, nullptr);
    CopyCmdQueue = std::exchange(Other.CopyCmdQueue, nullptr);
    LinkCopyCmdQueue = std::exchange(Other.LinkCopyCmdQueue, nullptr);
    ImmCmdList = std::exchange(Other.ImmCmdList, nullptr);
    ImmCopyCmdList = std::exchange(Other.ImmCopyCmdList, nullptr);
  }

  void clear() {
    // destroy all lists and queues
    if (CmdList)
      CALL_ZE_EXIT_FAIL(zeCommandListDestroy, CmdList);
    if (CopyCmdList)
      CALL_ZE_EXIT_FAIL(zeCommandListDestroy, CopyCmdList);
    if (LinkCopyCmdList)
      CALL_ZE_EXIT_FAIL(zeCommandListDestroy, LinkCopyCmdList);
    if (ImmCmdList)
      CALL_ZE_EXIT_FAIL(zeCommandListDestroy, ImmCmdList);
    if (ImmCopyCmdList)
      CALL_ZE_EXIT_FAIL(zeCommandListDestroy, ImmCopyCmdList);
    if (CmdQueue)
      CALL_ZE_EXIT_FAIL(zeCommandQueueDestroy, CmdQueue);
    if (CopyCmdQueue)
      CALL_ZE_EXIT_FAIL(zeCommandQueueDestroy, CopyCmdQueue);
    if (LinkCopyCmdQueue)
      CALL_ZE_EXIT_FAIL(zeCommandQueueDestroy, LinkCopyCmdQueue);

    CmdList = nullptr;
    CopyCmdList = nullptr;
    LinkCopyCmdList = nullptr;
    CmdQueue = nullptr;
    CopyCmdQueue = nullptr;
    LinkCopyCmdQueue = nullptr;
    ImmCmdList = nullptr;
    ImmCopyCmdList = nullptr;
  }

  L0DeviceTLSTy &operator=(const L0DeviceTLSTy &) = delete;
  L0DeviceTLSTy &operator=(L0DeviceTLSTy &&) = delete;

  auto getCmdList() const { return CmdList; }
  void setCmdList(ze_command_list_handle_t _CmdList) { CmdList = _CmdList; }

  auto getCopyCmdList() const { return CopyCmdList; }
  void setCopyCmdList(ze_command_list_handle_t _CopyCmdList) {
    CopyCmdList = _CopyCmdList;
  }

  auto getLinkCopyCmdList() const { return LinkCopyCmdList; }
  void setLinkCopyCmdList(ze_command_list_handle_t _LinkCopyCmdList) {
    LinkCopyCmdList = _LinkCopyCmdList;
  }

  auto getImmCmdList() const { return ImmCmdList; }
  void setImmCmdList(ze_command_list_handle_t _ImmCmdList) {
    ImmCmdList = _ImmCmdList;
  }

  auto getImmCopyCmdList() const { return ImmCopyCmdList; }
  void setImmCopyCmdList(ze_command_list_handle_t _ImmCopyCmdList) {
    ImmCopyCmdList = _ImmCopyCmdList;
  }

  auto getCmdQueue() const { return CmdQueue; }
  void setCmdQueue(ze_command_queue_handle_t _CmdQueue) {
    CmdQueue = _CmdQueue;
  }

  auto getCopyCmdQueue() const { return CopyCmdQueue; }
  void setCopyCmdQueue(ze_command_queue_handle_t _CopyCmdQueue) {
    CopyCmdQueue = _CopyCmdQueue;
  }

  auto getLinkCopyCmdQueue() const { return LinkCopyCmdQueue; }
  void setLinkCopyCmdQueue(ze_command_queue_handle_t _LinkCopyCmdQueue) {
    LinkCopyCmdQueue = _LinkCopyCmdQueue;
  }
};

struct L0DeviceTLSTableTy
    : public PerThreadContainer<std::vector<L0DeviceTLSTy>, 8> {
  void clear() {
    PerThreadTable::clear([](L0DeviceTLSTy &Entry) { Entry.clear(); });
  }
};

class L0DeviceTy final : public GenericDeviceTy {
  // Level Zero Context for this Device
  L0ContextTy &l0Context;

  // Level Zero handle  for this Device
  ze_device_handle_t zeDevice;
  // Device Properties
  ze_device_properties_t DeviceProperties{};
  ze_device_compute_properties_t ComputeProperties{};
  ze_device_memory_properties_t MemoryProperties{};
  ze_device_cache_properties_t CacheProperties{};

  /// Devices' default target allocation kind for internal allocation
  int32_t AllocKind = TARGET_ALLOC_DEVICE;

  DeviceArchTy DeviceArch = DeviceArchTy::DeviceArch_None;

  std::string DeviceName;

  /// Common indirect access flags for this device
  ze_kernel_indirect_access_flags_t IndirectAccessFlags = 0;

  /// Device UUID for toplevel devices only
  std::string DeviceUuid;

  /// L0 Device ID as string
  std::string zeId;

  /// Command queue group ordinals for each device
  std::pair<uint32_t, uint32_t> ComputeOrdinal{UINT32_MAX, 0};
  /// Command queue group ordinals for copying
  std::pair<uint32_t, uint32_t> CopyOrdinal{UINT32_MAX, 0};
  /// Command queue group ordinals and number of queues for link copy engines
  std::pair<uint32_t, uint32_t> LinkCopyOrdinal{UINT32_MAX, 0};

  /// Command queue index for each device
  uint32_t ComputeIndex = 0;

  bool IsAsyncEnabled = false;

  // lock for this device
  std::mutex Mutex;

  /// Contains all modules (possibly from multiple device images) to handle
  /// dynamic link across multiple images
  llvm::SmallVector<ze_module_handle_t> GlobalModules;

  /// L0 programs created for this device
  std::list<L0ProgramTy> Programs;

  /// MemAllocator for this device
  MemAllocatorTy MemAllocator;

  /// The current size of the global device memory pool (managed by us).
  uint64_t HeapSize = 1L << 23L /*8MB=*/;

  int32_t synchronize(__tgt_async_info *AsyncInfo, bool ReleaseQueue = true);
  int32_t submitData(void *TgtPtr, const void *HstPtr, int64_t Size,
                     __tgt_async_info *AsyncInfo);
  int32_t retrieveData(void *HstPtr, const void *TgtPtr, int64_t Size,
                       __tgt_async_info *AsyncInfo);

  bool shouldSetupDeviceMemoryPool() const override { return false; }
  DeviceArchTy computeArch() const;

  /// Get default compute group ordinal. Returns Ordinal-NumQueues pair
  std::pair<uint32_t, uint32_t> findComputeOrdinal();

  /// Get copy command queue group ordinal. Returns Ordinal-NumQueues pair
  std::pair<uint32_t, uint32_t> findCopyOrdinal(bool LinkCopy = false);

  Error internalInit();

public:
  L0DeviceTy(GenericPluginTy &Plugin, int32_t DeviceId, int32_t NumDevices,
             ze_device_handle_t zeDevice, L0ContextTy &DriverInfo,
             const std::string_view zeId, int32_t ComputeIndex)
      : GenericDeviceTy(Plugin, DeviceId, NumDevices, SPIRVGridValues),
        l0Context(DriverInfo), zeDevice(zeDevice), zeId(zeId),
        ComputeIndex(ComputeIndex) {
    DeviceProperties.stype = ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES;
    DeviceProperties.pNext = nullptr;
    ComputeProperties.stype = ZE_STRUCTURE_TYPE_DEVICE_COMPUTE_PROPERTIES;
    ComputeProperties.pNext = nullptr;
    MemoryProperties.stype = ZE_STRUCTURE_TYPE_DEVICE_MEMORY_PROPERTIES;
    MemoryProperties.pNext = nullptr;
    CacheProperties.stype = ZE_STRUCTURE_TYPE_DEVICE_CACHE_PROPERTIES;
    CacheProperties.pNext = nullptr;

    auto Err = internalInit();
    if (Err)
      FATAL_MESSAGE(DeviceId, "Couldn't initialize device: %s\n",
                    toString(std::move(Err)).c_str());
  }

  static L0DeviceTy &makeL0Device(GenericDeviceTy &Device) {
    return static_cast<L0DeviceTy &>(Device);
  }

  auto &getPlugin() { return (LevelZeroPluginTy &)Plugin; }
  L0DeviceTLSTy &getTLS();

  Error setContext() override { return Plugin::success(); }
  Error initImpl(GenericPluginTy &Plugin) override;
  Error deinitImpl() override {
    Programs.clear();
    return Plugin::success();
  }

  auto getZeDevice() const { return zeDevice; }

  const L0ContextTy &getL0Context() const { return l0Context; }
  L0ContextTy &getL0Context() { return l0Context; }

  const std::string_view getName() const { return DeviceName; }
  const char *getNameCStr() const { return DeviceName.c_str(); }

  const std::string_view getZeId() const { return zeId; }
  const char *getZeIdCStr() const { return zeId.c_str(); }

  std::mutex &getMutex() { return Mutex; }

  auto getComputeIndex() const { return ComputeIndex; }
  auto getIndirectFlags() const { return IndirectAccessFlags; }

  auto getNumGlobalModules() const { return GlobalModules.size(); }
  void addGlobalModule(ze_module_handle_t Module) {
    GlobalModules.push_back(Module);
  }
  auto getGlobalModulesArray() { return GlobalModules.data(); }

  L0ProgramTy *getProgramFromImage(MemoryBufferRef Image) {
    for (auto &PGM : Programs)
      if (PGM.getMemoryBuffer() == Image)
        return &PGM;
    return nullptr;
  }

  int32_t buildAllKernels() {
    for (auto &PGM : Programs) {
      int32_t RC = PGM.loadModuleKernels();
      if (RC != OFFLOAD_SUCCESS)
        return RC;
    }
    return OFFLOAD_SUCCESS;
  }

  // add a new program to the device. Return a reference to the new program
  auto &addProgram(int32_t ImageId, std::unique_ptr<MemoryBuffer> &&Image) {
    Programs.emplace_back(ImageId, *this, std::move(Image));
    return Programs.back();
  }

  const auto &getLastProgram() const { return Programs.back(); }
  auto &getLastProgram() { return Programs.back(); }
  // Device properties getters
  auto getVendorId() const { return DeviceProperties.vendorId; }
  bool isGPU() const { return DeviceProperties.type == ZE_DEVICE_TYPE_GPU; }

  auto getPCIId() const { return DeviceProperties.deviceId; }
  auto getNumThreadsPerEU() const { return DeviceProperties.numThreadsPerEU; }
  auto getSIMDWidth() const { return DeviceProperties.physicalEUSimdWidth; }
  auto getNumEUsPerSubslice() const {
    return DeviceProperties.numEUsPerSubslice;
  }
  auto getNumSubslicesPerSlice() const {
    return DeviceProperties.numSubslicesPerSlice;
  }
  auto getNumSlices() const { return DeviceProperties.numSlices; }
  auto getNumSubslices() const {
    return DeviceProperties.numSubslicesPerSlice * DeviceProperties.numSlices;
  }
  uint32_t getNumEUs() const {
    return DeviceProperties.numEUsPerSubslice * getNumSubslices();
  }
  auto getTotalThreads() const {
    return DeviceProperties.numThreadsPerEU * getNumEUs();
  }
  auto getNumThreadsPerSubslice() const {
    return getNumEUsPerSubslice() * getNumThreadsPerEU();
  }
  auto getClockRate() const { return DeviceProperties.coreClockRate; }

  auto getMaxSharedLocalMemory() const {
    return ComputeProperties.maxSharedLocalMemory;
  }
  auto getMaxGroupSize() const { return ComputeProperties.maxTotalGroupSize; }
  auto getGlobalMemorySize() const { return MemoryProperties.totalSize; }
  auto getCacheSize() const { return CacheProperties.cacheSize; }
  auto getMaxMemAllocSize() const { return DeviceProperties.maxMemAllocSize; }

  int32_t getAllocKind() const { return AllocKind; }
  DeviceArchTy getDeviceArch() const { return DeviceArch; }
  bool isDeviceArch(DeviceArchTy Arch) const { return DeviceArch == Arch; }

  static bool isDiscrete(uint32_t PCIId) {
    switch (static_cast<PCIIdTy>(PCIId & 0xFF00)) {
    case PCIIdTy::BMG:
      return true;
    default:
      return false;
    }
  }

  static bool isDiscrete(ze_device_handle_t Device) {
    ze_device_properties_t PR{};
    PR.stype = ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES;
    PR.pNext = nullptr;
    CALL_ZE_RET(false, zeDeviceGetProperties, Device, &PR);
    return isDiscrete(PR.deviceId);
  }

  bool isDiscreteDevice() { return isDiscrete(getPCIId()); }
  bool isDeviceIPorNewer(uint32_t Version) const;

  const std::string_view getUuid() const { return DeviceUuid; }

  uint32_t getComputeEngine() const { return ComputeOrdinal.first; }
  uint32_t getNumComputeQueues() const { return ComputeOrdinal.second; }

  bool hasMainCopyEngine() const { return CopyOrdinal.first != UINT32_MAX; }
  uint32_t getMainCopyEngine() const { return CopyOrdinal.first; }

  uint32_t getLinkCopyEngine() const { return LinkCopyOrdinal.first; }
  uint32_t getNumLinkCopyQueues() const { return LinkCopyOrdinal.second; }
  bool hasLinkCopyEngine() const { return getNumLinkCopyQueues() > 0; }

  bool deviceRequiresImmCmdList() const {
    return isDeviceIPorNewer(0x05004000);
  }
  bool asyncEnabled() const { return IsAsyncEnabled; }
  bool useImmForCompute() const { return true; }
  bool useImmForCopy() const { return true; }
  bool useImmForInterop() const { return true; }

  void reportDeviceInfo() const;

  // Command queues related functions
  /// Create a command list with given ordinal and flags
  ze_command_list_handle_t createCmdList(ze_context_handle_t Context,
                                         ze_device_handle_t Device,
                                         uint32_t Ordinal,
                                         ze_command_list_flags_t Flags,
                                         const std::string_view DeviceIdStr);

  /// Create a command list with default flags
  ze_command_list_handle_t createCmdList(ze_context_handle_t Context,
                                         ze_device_handle_t Device,
                                         uint32_t Ordinal,
                                         const std::string_view DeviceIdStr);

  ze_command_list_handle_t getCmdList();

  /// Create a command queue with given ordinal and flags
  ze_command_queue_handle_t createCmdQueue(ze_context_handle_t Context,
                                           ze_device_handle_t Device,
                                           uint32_t Ordinal, uint32_t Index,
                                           ze_command_queue_flags_t Flags,
                                           const std::string_view DeviceIdStr);

  /// Create a command queue with default flags
  ze_command_queue_handle_t createCmdQueue(ze_context_handle_t Context,
                                           ze_device_handle_t Device,
                                           uint32_t Ordinal, uint32_t Index,
                                           const std::string_view DeviceIdStr,
                                           bool InOrder = false);

  /// Create a new command queue for the given OpenMP device ID
  ze_command_queue_handle_t createCommandQueue(bool InOrder = false);

  /// Create an immediate command list
  ze_command_list_handle_t createImmCmdList(uint32_t Ordinal, uint32_t Index,
                                            bool InOrder = false);

  /// Create an immediate command list for computing
  ze_command_list_handle_t createImmCmdList(bool InOrder = false) {
    return createImmCmdList(getComputeEngine(), getComputeIndex(), InOrder);
  }

  /// Create an immediate command list for copying
  ze_command_list_handle_t createImmCopyCmdList();
  ze_command_queue_handle_t getCmdQueue();
  ze_command_list_handle_t getCopyCmdList();
  ze_command_queue_handle_t getCopyCmdQueue();
  ze_command_list_handle_t getLinkCopyCmdList();
  ze_command_queue_handle_t getLinkCopyCmdQueue();
  ze_command_list_handle_t getImmCmdList();
  ze_command_list_handle_t getImmCopyCmdList();

  /// Enqueue copy command
  int32_t enqueueMemCopy(void *Dst, const void *Src, size_t Size,
                         __tgt_async_info *AsyncInfo = nullptr,
                         bool UseCopyEngine = true);

  /// Enqueue asynchronous copy command
  int32_t enqueueMemCopyAsync(void *Dst, const void *Src, size_t Size,
                              __tgt_async_info *AsyncInfo, bool CopyTo = true);

  /// Enqueue fill command
  int32_t enqueueMemFill(void *Ptr, const void *Pattern, size_t PatternSize,
                         size_t Size);

  /// Driver related functions

  /// Reurn the driver handle for this device
  ze_driver_handle_t getZeDriver() const { return l0Context.getZeDriver(); }

  /// Return context for this device
  ze_context_handle_t getZeContext() const { return l0Context.getZeContext(); }

  /// Return driver API version for this device
  ze_api_version_t getDriverAPIVersion() const {
    return l0Context.getDriverAPIVersion();
  }

  /// Return an event from the driver associated to this device
  ze_event_handle_t getEvent() { return l0Context.getEventPool().getEvent(); }

  /// Release event to the pool associated to this device
  void releaseEvent(ze_event_handle_t Event) {
    l0Context.getEventPool().releaseEvent(Event, *this);
  }

  StagingBufferTy &getStagingBuffer() { return l0Context.getStagingBuffer(); }

  bool supportsLargeMem() const { return l0Context.supportsLargeMem(); }

  // Allocation related routines

  /// Data alloc
  Expected<void *>
  dataAlloc(size_t Size, size_t Align, int32_t Kind, intptr_t Offset,
            bool UserAlloc, bool DevMalloc = false,
            uint32_t MemAdvice = UINT32_MAX,
            AllocOptionTy AllocOpt = AllocOptionTy::ALLOC_OPT_NONE);

  /// Data delete
  Error dataDelete(void *Ptr);

  /// Return the memory allocation type for the specified memory location.
  uint32_t getMemAllocType(const void *Ptr) const;

  const MemAllocatorTy &getDeviceMemAllocator() const { return MemAllocator; }
  MemAllocatorTy &getDeviceMemAllocator() { return MemAllocator; }

  MemAllocatorTy &getMemAllocator(int32_t Kind) {
    if (Kind == TARGET_ALLOC_HOST)
      return l0Context.getHostMemAllocator();
    return getDeviceMemAllocator();
  }

  MemAllocatorTy &getMemAllocator(const void *Ptr) {
    bool IsHostMem = (ZE_MEMORY_TYPE_HOST == getMemAllocType(Ptr));
    if (IsHostMem)
      return l0Context.getHostMemAllocator();
    return getDeviceMemAllocator();
  }

  int32_t makeMemoryResident(void *Mem, size_t Size);

  // Generic device interface implementation
  Expected<DeviceImageTy *>
  loadBinaryImpl(std::unique_ptr<MemoryBuffer> &&TgtImage,
                 int32_t ImageId) override;
  Error unloadBinaryImpl(DeviceImageTy *Image) override;
  Expected<void *> allocate(size_t Size, void *HstPtr,
                            TargetAllocTy Kind) override;
  Error free(void *TgtPtr, TargetAllocTy Kind = TARGET_ALLOC_DEFAULT) override;

  Expected<void *> dataLockImpl(void *HstPtr, int64_t Size) override {
    return Plugin::error(error::ErrorCode::UNKNOWN,
                         "dataLockImpl not supported");
  }
  Error dataUnlockImpl(void *HstPtr) override { return Plugin::success(); }

  Expected<bool> isPinnedPtrImpl(void *, void *&, void *&,
                                 size_t &) const override {
    // Don't need to do anything, this is handled by the driver.
    return false;
  }

  Error dataFence(__tgt_async_info *Async) override;
  Error dataFillImpl(void *TgtPtr, const void *PatternPtr, int64_t PatternSize,
                     int64_t Size,
                     AsyncInfoWrapperTy &AsyncInfoWrapper) override;
  Error synchronizeImpl(__tgt_async_info &AsyncInfo,
                        bool ReleaseQueue) override;
  Error queryAsyncImpl(__tgt_async_info &AsyncInfo) override;
  Error dataSubmitImpl(void *TgtPtr, const void *HstPtr, int64_t Size,
                       AsyncInfoWrapperTy &AsyncInfoWrapper) override;
  Error dataRetrieveImpl(void *HstPtr, const void *TgtPtr, int64_t Size,
                         AsyncInfoWrapperTy &AsyncInfoWrapper) override;
  Error dataExchangeImpl(const void *SrcPtr, GenericDeviceTy &DstDev,
                         void *DstPtr, int64_t Size,
                         AsyncInfoWrapperTy &AsyncInfoWrapper) override;
  Error initAsyncInfoImpl(AsyncInfoWrapperTy &AsyncInfoWrapper) override;
  Error initDeviceInfoImpl(__tgt_device_info *Info) override;
  Expected<bool>
  hasPendingWorkImpl(AsyncInfoWrapperTy &AsyncInfoWrapper) override;

  Error enqueueHostCallImpl(void (*Callback)(void *), void *UserData,
                            AsyncInfoWrapperTy &AsyncInfo) override{
      L0_UNIMPLEMENTED_ERR}

  /* Event routines are used to ensure ordering between dataTransfers. Instead
   * of adding extra events in the queues, we make sure they're ordered by
   * using the events from the data submission APIs so we don't need to support
   * these routines.
   * They still need to report succes to indicate the event are handled
   * somewhere waitEvent and syncEvent should remain unimplemented
   */
  Expected<bool> isEventCompleteImpl(void *EventPtr,
                                     AsyncInfoWrapperTy &) override {
    return true;
  }

  Error createEventImpl(void **EventPtrStorage) override {
    return Plugin::success();
  }
  Error destroyEventImpl(void *EventPtr) override { return Plugin::success(); }
  Error recordEventImpl(void *EventPtr,
                        AsyncInfoWrapperTy &AsyncInfoWrapper) override {
    return Plugin::success();
  }

  Error waitEventImpl(void *EventPtr,
                      AsyncInfoWrapperTy &AsyncInfoWrapper) override {
    return Plugin::error(error::ErrorCode::UNKNOWN, "%s not implemented yet\n",
                         __func__);
  }

  Error syncEventImpl(void *EventPtr) override {
    return Plugin::error(error::ErrorCode::UNKNOWN, "%s not implemented yet\n",
                         __func__);
  }

  Expected<InfoTreeNode> obtainInfoImpl() override;

  Error getDeviceStackSize(uint64_t &V) override {
    V = 0;
    return Plugin::success();
  }
  Expected<GenericKernelTy &> constructKernel(const char *Name) override;

  Error setDeviceStackSize(uint64_t V) override { return Plugin::success(); }
  Error getDeviceHeapSize(uint64_t &V) override {
    V = HeapSize;
    return Plugin::success();
  }
  Error setDeviceHeapSize(uint64_t V) override {
    HeapSize = V;
    return Plugin::success();
  }

  Expected<omp_interop_val_t *>
  createInterop(int32_t InteropType, interop_spec_t &InteropSpec) override;
  Error releaseInterop(omp_interop_val_t *Interop) override;

  interop_spec_t selectInteropPreference(int32_t InteropType,
                                         int32_t NumPrefers,
                                         interop_spec_t *Prefers) override;
};

} // namespace plugin
} // namespace target
} // namespace omp
} // namespace llvm
#endif // OPENMP_LIBOMPTARGET_PLUGINS_NEXTGEN_LEVEL_ZERO_L0DEVICE_H
