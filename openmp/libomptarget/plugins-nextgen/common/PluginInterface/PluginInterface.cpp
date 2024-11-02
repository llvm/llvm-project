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
#include "Debug.h"
#include "GlobalHandler.h"
#include "elf_common.h"
#include "omptarget.h"
#include "omptargetplugin.h"

#include <cstdint>
#include <limits>

using namespace llvm;
using namespace omp;
using namespace target;
using namespace plugin;

uint32_t GenericPluginTy::NumActiveInstances = 0;

AsyncInfoWrapperTy::~AsyncInfoWrapperTy() {
  // If we used a local async info object we want synchronous behavior.
  // In that case, and assuming the current status code is OK, we will
  // synchronize explicitly when the object is deleted.
  if (AsyncInfoPtr == &LocalAsyncInfo && !Err)
    Err = Device.synchronize(&LocalAsyncInfo);
}

Error GenericKernelTy::init(GenericDeviceTy &GenericDevice,
                            DeviceImageTy &Image) {
  PreferredNumThreads = getDefaultNumThreads(GenericDevice);
  if (isGenericMode())
    PreferredNumThreads += GenericDevice.getWarpSize();

  MaxNumThreads = GenericDevice.getThreadLimit();

  DynamicMemorySize = GenericDevice.getDynamicMemorySize();

  return initImpl(GenericDevice, Image);
}

Error GenericKernelTy::launch(GenericDeviceTy &GenericDevice, void **ArgPtrs,
                              ptrdiff_t *ArgOffsets, int32_t NumArgs,
                              uint64_t NumTeamsClause,
                              uint32_t ThreadLimitClause,
                              uint64_t LoopTripCount,
                              AsyncInfoWrapperTy &AsyncInfoWrapper) const {
  llvm::SmallVector<void *, 16> Args;
  llvm::SmallVector<void *, 16> Ptrs;

  void *KernelArgsPtr = prepareArgs(GenericDevice, ArgPtrs, ArgOffsets, NumArgs,
                                    Args, Ptrs, AsyncInfoWrapper);

  uint32_t NumThreads = getNumThreads(GenericDevice, ThreadLimitClause);
  uint64_t NumBlocks =
      getNumBlocks(GenericDevice, NumTeamsClause, LoopTripCount, NumThreads);

  INFO(OMP_INFOTYPE_PLUGIN_KERNEL, GenericDevice.getDeviceId(),
       "Launching kernel %s with %" PRIu64
       " blocks and %d threads in %s mode\n",
       getName(), NumBlocks, NumThreads, getExecutionModeName());

  return launchImpl(GenericDevice, NumThreads, NumBlocks, DynamicMemorySize,
                    NumArgs, KernelArgsPtr, AsyncInfoWrapper);
}

void *GenericKernelTy::prepareArgs(GenericDeviceTy &GenericDevice,
                                   void **ArgPtrs, ptrdiff_t *ArgOffsets,
                                   int32_t NumArgs,
                                   llvm::SmallVectorImpl<void *> &Args,
                                   llvm::SmallVectorImpl<void *> &Ptrs,
                                   AsyncInfoWrapperTy &AsyncInfoWrapper) const {
  Args.resize(NumArgs);
  Ptrs.resize(NumArgs);

  if (NumArgs == 0)
    return nullptr;

  for (int I = 0; I < NumArgs; ++I) {
    Ptrs[I] = (void *)((intptr_t)ArgPtrs[I] + ArgOffsets[I]);
    Args[I] = &Ptrs[I];
  }
  return &Args[0];
}

uint32_t GenericKernelTy::getNumThreads(GenericDeviceTy &GenericDevice,
                                        uint32_t ThreadLimitClause) const {
  return std::min(MaxNumThreads, (ThreadLimitClause > 0) ? ThreadLimitClause
                                                         : PreferredNumThreads);
}

uint64_t GenericKernelTy::getNumBlocks(GenericDeviceTy &GenericDevice,
                                       uint64_t NumTeamsClause,
                                       uint64_t LoopTripCount,
                                       uint32_t NumThreads) const {
  uint64_t PreferredNumBlocks = getDefaultNumBlocks(GenericDevice);
  if (NumTeamsClause > 0) {
    PreferredNumBlocks = NumTeamsClause;
  } else if (LoopTripCount > 0) {
    if (isSPMDMode()) {
      // We have a combined construct, i.e. `target teams distribute
      // parallel for [simd]`. We launch so many teams so that each thread
      // will execute one iteration of the loop. round up to the nearest
      // integer
      PreferredNumBlocks = ((LoopTripCount - 1) / NumThreads) + 1;
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
      PreferredNumBlocks = LoopTripCount;
    }
  }
  return std::min(PreferredNumBlocks, GenericDevice.getBlockLimit());
}

GenericDeviceTy::GenericDeviceTy(int32_t DeviceId, int32_t NumDevices,
                                 const llvm::omp::GV &OMPGridValues)
    : OMP_TeamLimit("OMP_TEAM_LIMIT"), OMP_NumTeams("OMP_NUM_TEAMS"),
      OMP_TeamsThreadLimit("OMP_TEAMS_THREAD_LIMIT"),
      OMPX_DebugKind("LIBOMPTARGET_DEVICE_RTL_DEBUG"),
      OMPX_SharedMemorySize("LIBOMPTARGET_SHARED_MEMORY_SIZE"),
      // Do not initialize the following two envars since they depend on the
      // device initialization. These cannot be consulted until the device is
      // initialized correctly. We intialize them in GenericDeviceTy::init().
      OMPX_TargetStackSize(), OMPX_TargetHeapSize(), MemoryManager(nullptr),
      DeviceId(DeviceId), GridValues(OMPGridValues),
      PeerAccesses(NumDevices, PeerAccessState::PENDING), PeerAccessesLock() {
  if (OMP_NumTeams > 0)
    GridValues.GV_Max_Teams =
        std::min(GridValues.GV_Max_Teams, uint32_t(OMP_NumTeams));

  if (OMP_TeamsThreadLimit > 0)
    GridValues.GV_Max_WG_Size =
        std::min(GridValues.GV_Max_WG_Size, uint32_t(OMP_TeamsThreadLimit));
};

Error GenericDeviceTy::init(GenericPluginTy &Plugin) {
  if (auto Err = initImpl(Plugin))
    return Err;

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

  // Enable the memory manager if required.
  auto [ThresholdMM, EnableMM] = MemoryManagerTy::getSizeThresholdFromEnv();
  if (EnableMM)
    MemoryManager = new MemoryManagerTy(*this, ThresholdMM);

  return Plugin::success();
}

Error GenericDeviceTy::deinit() {
  // Delete the memory manager before deinitilizing the device. Otherwise,
  // we may delete device allocations after the device is deinitialized.
  if (MemoryManager)
    delete MemoryManager;
  MemoryManager = nullptr;

  return deinitImpl();
}

Expected<__tgt_target_table *>
GenericDeviceTy::loadBinary(GenericPluginTy &Plugin,
                            const __tgt_device_image *TgtImage) {
  DP("Load data from image " DPxMOD "\n", DPxPTR(TgtImage->ImageStart));

  // Load the binary and allocate the image object. Use the next available id
  // for the image id, which is the number of previously loaded images.
  auto ImageOrErr = loadBinaryImpl(TgtImage, LoadedImages.size());
  if (!ImageOrErr)
    return ImageOrErr.takeError();

  DeviceImageTy *Image = *ImageOrErr;
  assert(Image != nullptr && "Invalid image");

  // Add the image to list.
  LoadedImages.push_back(Image);

  // Setup the device environment if needed.
  if (auto Err = setupDeviceEnvironment(Plugin, *Image))
    return std::move(Err);

  // Register all offload entries of the image.
  if (auto Err = registerOffloadEntries(*Image))
    return std::move(Err);

  // Return the pointer to the table of entries.
  return Image->getOffloadEntryTable();
}

Error GenericDeviceTy::setupDeviceEnvironment(GenericPluginTy &Plugin,
                                              DeviceImageTy &Image) {
  // There are some plugins that do not need this step.
  if (!shouldSetupDeviceEnvironment())
    return Plugin::success();

  DeviceEnvironmentTy DeviceEnvironment;
  DeviceEnvironment.DebugKind = OMPX_DebugKind;
  DeviceEnvironment.NumDevices = Plugin.getNumDevices();
  // TODO: The device ID used here is not the real device ID used by OpenMP.
  DeviceEnvironment.DeviceNum = DeviceId;
  DeviceEnvironment.DynamicMemSize = OMPX_SharedMemorySize;

  // Create the metainfo of the device environment global.
  GlobalTy DeviceEnvGlobal("omptarget_device_environment",
                           sizeof(DeviceEnvironmentTy), &DeviceEnvironment);

  // Write device environment values to the device.
  GenericGlobalHandlerTy &GlobalHandler = Plugin.getGlobalHandler();
  return GlobalHandler.writeGlobalToDevice(*this, Image, DeviceEnvGlobal);
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
    if (auto Err = GHandler.writeGlobalToDevice(*this, Image, HostGlobal,
                                                DeviceGlobal))
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
  auto KernelOrErr = constructKernelEntry(KernelEntry, Image);
  if (!KernelOrErr)
    return KernelOrErr.takeError();

  GenericKernelTy *Kernel = *KernelOrErr;
  assert(Kernel != nullptr && "Invalid kernel");

  // Initialize the kernel.
  if (auto Err = Kernel->init(*this, Image))
    return Err;

  // Set the device entry address to the kernel address and store the entry on
  // the entry table.
  DeviceEntry.addr = (void *)Kernel;
  Image.getOffloadEntryTable().addEntry(DeviceEntry);

  return Plugin::success();
}

Error GenericDeviceTy::synchronize(__tgt_async_info *AsyncInfo) {
  if (!AsyncInfo || !AsyncInfo->Queue)
    return Plugin::error("Invalid async info queue");

  return synchronizeImpl(*AsyncInfo);
}

Expected<void *> GenericDeviceTy::dataAlloc(int64_t Size, void *HostPtr,
                                            TargetAllocTy Kind) {
  void *Alloc = nullptr;

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

  // Sucessful and valid allocation.
  if (Alloc)
    return Alloc;

  // At this point means that we did not tried to allocate from the memory
  // manager nor the device allocator.
  return Plugin::error("Invalid target data allocation kind or requested "
                       "allocator not implemented yet");
}

Error GenericDeviceTy::dataDelete(void *TgtPtr, TargetAllocTy Kind) {
  int Res;
  if (MemoryManager)
    Res = MemoryManager->free(TgtPtr);
  else
    Res = free(TgtPtr, Kind);

  if (Res)
    return Plugin::error("Failure to deallocate device pointer %p", TgtPtr);

  return Plugin::success();
}

Error GenericDeviceTy::dataSubmit(void *TgtPtr, const void *HstPtr,
                                  int64_t Size, __tgt_async_info *AsyncInfo) {
  auto Err = Plugin::success();
  AsyncInfoWrapperTy AsyncInfoWrapper(Err, *this, AsyncInfo);
  Err = dataSubmitImpl(TgtPtr, HstPtr, Size, AsyncInfoWrapper);
  return Err;
}

Error GenericDeviceTy::dataRetrieve(void *HstPtr, const void *TgtPtr,
                                    int64_t Size, __tgt_async_info *AsyncInfo) {
  auto Err = Plugin::success();
  AsyncInfoWrapperTy AsyncInfoWrapper(Err, *this, AsyncInfo);
  Err = dataRetrieveImpl(HstPtr, TgtPtr, Size, AsyncInfoWrapper);
  return Err;
}

Error GenericDeviceTy::dataExchange(const void *SrcPtr, GenericDeviceTy &DstDev,
                                    void *DstPtr, int64_t Size,
                                    __tgt_async_info *AsyncInfo) {
  auto Err = Plugin::success();
  AsyncInfoWrapperTy AsyncInfoWrapper(Err, *this, AsyncInfo);
  Err = dataExchangeImpl(SrcPtr, DstDev, DstPtr, Size, AsyncInfoWrapper);
  return Err;
}

Error GenericDeviceTy::runTargetTeamRegion(
    void *EntryPtr, void **ArgPtrs, ptrdiff_t *ArgOffsets, int32_t NumArgs,
    uint64_t NumTeamsClause, uint32_t ThreadLimitClause, uint64_t LoopTripCount,
    __tgt_async_info *AsyncInfo) {
  auto Err = Plugin::success();
  AsyncInfoWrapperTy AsyncInfoWrapper(Err, *this, AsyncInfo);

  GenericKernelTy &GenericKernel =
      *reinterpret_cast<GenericKernelTy *>(EntryPtr);

  Err =
      GenericKernel.launch(*this, ArgPtrs, ArgOffsets, NumArgs, NumTeamsClause,
                           ThreadLimitClause, LoopTripCount, AsyncInfoWrapper);
  return Err;
}

Error GenericDeviceTy::initAsyncInfo(__tgt_async_info **AsyncInfoPtr) {
  assert(AsyncInfoPtr && "Invalid async info");

  *AsyncInfoPtr = new __tgt_async_info();

  auto Err = Plugin::success();
  AsyncInfoWrapperTy AsyncInfoWrapper(Err, *this, *AsyncInfoPtr);
  Err = initAsyncInfoImpl(AsyncInfoWrapper);
  return Err;
}

Error GenericDeviceTy::initDeviceInfo(__tgt_device_info *DeviceInfo) {
  assert(DeviceInfo && "Invalid device info");

  return initDeviceInfoImpl(DeviceInfo);
}

Error GenericPluginTy::initDevice(int32_t DeviceId) {
  assert(!Devices[DeviceId] && "Device already initialized");

  // Create the device and save the reference.
  GenericDeviceTy &Device = createDevice(DeviceId);
  Devices[DeviceId] = &Device;

  // Initialize the device and its resources.
  return Device.init(*this);
}

Error GenericPluginTy::deinitDevice(int32_t DeviceId) {
  // The device may be already deinitialized.
  if (Devices[DeviceId] == nullptr)
    return Plugin::success();

  // Deinitialize the device and release its resources.
  if (auto Err = Devices[DeviceId]->deinit())
    return Err;

  // Delete the device and invalidate its reference.
  delete Devices[DeviceId];
  Devices[DeviceId] = nullptr;

  return Plugin::success();
}

Error GenericDeviceTy::printInfo() {
  // TODO: Print generic information here
  return printInfoImpl();
}

Error GenericDeviceTy::createEvent(void **EventPtrStorage) {
  return createEventImpl(EventPtrStorage);
}

Error GenericDeviceTy::destroyEvent(void *EventPtr) {
  return destroyEventImpl(EventPtr);
}

Error GenericDeviceTy::recordEvent(void *EventPtr,
                                   __tgt_async_info *AsyncInfo) {
  auto Err = Plugin::success();
  AsyncInfoWrapperTy AsyncInfoWrapper(Err, *this, AsyncInfo);
  Err = recordEventImpl(EventPtr, AsyncInfoWrapper);
  return Err;
}

Error GenericDeviceTy::waitEvent(void *EventPtr, __tgt_async_info *AsyncInfo) {
  auto Err = Plugin::success();
  AsyncInfoWrapperTy AsyncInfoWrapper(Err, *this, AsyncInfo);
  Err = waitEventImpl(EventPtr, AsyncInfoWrapper);
  return Err;
}

Error GenericDeviceTy::syncEvent(void *EventPtr) {
  return syncEventImpl(EventPtr);
}

/// Exposed library API function, basically wrappers around the GenericDeviceTy
/// functionality with the same name. All non-async functions are redirected
/// to the async versions right away with a NULL AsyncInfoPtr.
#ifdef __cplusplus
extern "C" {
#endif

int32_t __tgt_rtl_init_plugin() {
  auto Err = Plugin::init();
  if (Err)
    REPORT("Failure to initialize plugin " GETNAME(TARGET_NAME) ": %s\n",
           toString(std::move(Err)).data());

  return (bool)Err;
}

int32_t __tgt_rtl_deinit_plugin() {
  auto Err = Plugin::deinit();
  if (Err)
    REPORT("Failure to deinitialize plugin " GETNAME(TARGET_NAME) ": %s\n",
           toString(std::move(Err)).data());

  return (bool)Err;
}

int32_t __tgt_rtl_is_valid_binary(__tgt_device_image *TgtImage) {
  if (!Plugin::isActive())
    return false;

  return elf_check_machine(TgtImage, Plugin::get().getMagicElfBits());
}

int32_t __tgt_rtl_is_valid_binary_info(__tgt_device_image *TgtImage,
                                       __tgt_image_info *Info) {
  if (!Plugin::isActive())
    return false;

  if (!__tgt_rtl_is_valid_binary(TgtImage))
    return false;

  // A subarchitecture was not specified. Assume it is compatible.
  if (!Info->Arch)
    return true;

  // Check the compatibility with all the available devices. Notice the
  // devices may not be initialized yet.
  auto CompatibleOrErr = Plugin::get().isImageCompatible(Info);
  if (!CompatibleOrErr) {
    // This error should not abort the execution, so we just inform the user
    // through the debug system.
    std::string ErrString = toString(CompatibleOrErr.takeError());
    DP("Failure to check whether image %p is valid: %s\n", TgtImage,
       ErrString.data());
    return false;
  }

  bool Compatible = *CompatibleOrErr;
  DP("Image is %scompatible with current environment: %s\n",
     (Compatible) ? "" : "not", Info->Arch);

  return Compatible;
}

int32_t __tgt_rtl_supports_empty_images() {
  return Plugin::get().supportsEmptyImages();
}

int32_t __tgt_rtl_init_device(int32_t DeviceId) {
  auto Err = Plugin::get().initDevice(DeviceId);
  if (Err)
    REPORT("Failure to initialize device %d: %s\n", DeviceId,
           toString(std::move(Err)).data());

  return (bool)Err;
}

int32_t __tgt_rtl_deinit_device(int32_t DeviceId) {
  auto Err = Plugin::get().deinitDevice(DeviceId);
  if (Err)
    REPORT("Failure to deinitialize device %d: %s\n", DeviceId,
           toString(std::move(Err)).data());

  return (bool)Err;
}

int32_t __tgt_rtl_number_of_devices() { return Plugin::get().getNumDevices(); }

int64_t __tgt_rtl_init_requires(int64_t RequiresFlags) {
  Plugin::get().setRequiresFlag(RequiresFlags);
  return RequiresFlags;
}

int32_t __tgt_rtl_is_data_exchangable(int32_t SrcDeviceId,
                                      int32_t DstDeviceId) {
  return Plugin::get().isDataExchangable(SrcDeviceId, DstDeviceId);
}

__tgt_target_table *__tgt_rtl_load_binary(int32_t DeviceId,
                                          __tgt_device_image *TgtImage) {
  GenericPluginTy &Plugin = Plugin::get();
  auto TableOrErr = Plugin.getDevice(DeviceId).loadBinary(Plugin, TgtImage);
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
  if (Err)
    REPORT("Failure to deallocate device pointer %p: %s\n", TgtPtr,
           toString(std::move(Err)).data());

  return (bool)Err;
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
  if (Err)
    REPORT("Failure to copy data from host to device. Pointers: host "
           "= " DPxMOD ", device = " DPxMOD ", size = %" PRId64 ": %s\n",
           DPxPTR(HstPtr), DPxPTR(TgtPtr), Size,
           toString(std::move(Err)).data());

  return (bool)Err;
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
  if (Err)
    REPORT("Faliure to copy data from device to host. Pointers: host "
           "= " DPxMOD ", device = " DPxMOD ", size = %" PRId64 ": %s\n",
           DPxPTR(HstPtr), DPxPTR(TgtPtr), Size,
           toString(std::move(Err)).data());

  return (bool)Err;
}

int32_t __tgt_rtl_data_exchange(int32_t SrcDeviceId, void *SrcPtr,
                                int32_t DstDeviceId, void *DstPtr,
                                int64_t Size) {
  return __tgt_rtl_data_exchange_async(SrcDeviceId, SrcPtr, DstDeviceId, DstPtr,
                                       Size, /* AsyncInfoPtr */ nullptr);
}

int32_t __tgt_rtl_data_exchange_async(int32_t SrcDeviceId, void *SrcPtr,
                                      int DstDeviceId, void *DstPtr,
                                      int64_t Size,
                                      __tgt_async_info *AsyncInfo) {
  GenericDeviceTy &SrcDevice = Plugin::get().getDevice(SrcDeviceId);
  GenericDeviceTy &DstDevice = Plugin::get().getDevice(DstDeviceId);
  auto Err = SrcDevice.dataExchange(SrcPtr, DstDevice, DstPtr, Size, AsyncInfo);
  if (Err)
    REPORT("Failure to copy data from device (%d) to device (%d). Pointers: "
           "host = " DPxMOD ", device = " DPxMOD ", size = %" PRId64 ": %s\n",
           SrcDeviceId, DstDeviceId, DPxPTR(SrcPtr), DPxPTR(DstPtr), Size,
           toString(std::move(Err)).data());

  return (bool)Err;
}

int32_t __tgt_rtl_run_target_team_region(int32_t DeviceId, void *TgtEntryPtr,
                                         void **TgtArgs, ptrdiff_t *TgtOffsets,
                                         int32_t NumArgs, int32_t NumTeams,
                                         int32_t ThreadLimit,
                                         uint64_t LoopTripCount) {
  return __tgt_rtl_run_target_team_region_async(DeviceId, TgtEntryPtr, TgtArgs,
                                                TgtOffsets, NumArgs, NumTeams,
                                                ThreadLimit, LoopTripCount,
                                                /* AsyncInfoPtr */ nullptr);
}

int32_t __tgt_rtl_run_target_team_region_async(
    int32_t DeviceId, void *TgtEntryPtr, void **TgtArgs, ptrdiff_t *TgtOffsets,
    int32_t NumArgs, int32_t NumTeams, int32_t ThreadLimit,
    uint64_t LoopTripCount, __tgt_async_info *AsyncInfoPtr) {
  auto Err = Plugin::get().getDevice(DeviceId).runTargetTeamRegion(
      TgtEntryPtr, TgtArgs, TgtOffsets, NumArgs, NumTeams, ThreadLimit,
      LoopTripCount, AsyncInfoPtr);
  if (Err)
    REPORT("Failure to run target region " DPxMOD " in device %d: %s\n",
           DPxPTR(TgtEntryPtr), DeviceId, toString(std::move(Err)).data());

  return (bool)Err;
}

int32_t __tgt_rtl_synchronize(int32_t DeviceId,
                              __tgt_async_info *AsyncInfoPtr) {
  auto Err = Plugin::get().getDevice(DeviceId).synchronize(AsyncInfoPtr);
  if (Err)
    REPORT("Failure to synchronize stream %p: %s\n", AsyncInfoPtr->Queue,
           toString(std::move(Err)).data());

  return (bool)Err;
}

int32_t __tgt_rtl_run_target_region(int32_t DeviceId, void *TgtEntryPtr,
                                    void **TgtArgs, ptrdiff_t *TgtOffsets,
                                    int32_t NumArgs) {
  return __tgt_rtl_run_target_region_async(DeviceId, TgtEntryPtr, TgtArgs,
                                           TgtOffsets, NumArgs,
                                           /* AsyncInfoPtr */ nullptr);
}

int32_t __tgt_rtl_run_target_region_async(int32_t DeviceId, void *TgtEntryPtr,
                                          void **TgtArgs, ptrdiff_t *TgtOffsets,
                                          int32_t NumArgs,
                                          __tgt_async_info *AsyncInfoPtr) {
  return __tgt_rtl_run_target_team_region_async(
      DeviceId, TgtEntryPtr, TgtArgs, TgtOffsets, NumArgs,
      /* team num*/ 1, /* thread limit */ 1, /* loop tripcount */ 0,
      AsyncInfoPtr);
}

void __tgt_rtl_print_device_info(int32_t DeviceId) {
  if (auto Err = Plugin::get().getDevice(DeviceId).printInfo())
    REPORT("Failure to print device %d info: %s\n", DeviceId,
           toString(std::move(Err)).data());
}

int32_t __tgt_rtl_create_event(int32_t DeviceId, void **EventPtr) {
  auto Err = Plugin::get().getDevice(DeviceId).createEvent(EventPtr);
  if (Err)
    REPORT("Failure to create event: %s\n", toString(std::move(Err)).data());

  return (bool)Err;
}

int32_t __tgt_rtl_record_event(int32_t DeviceId, void *EventPtr,
                               __tgt_async_info *AsyncInfoPtr) {
  auto Err =
      Plugin::get().getDevice(DeviceId).recordEvent(EventPtr, AsyncInfoPtr);
  if (Err)
    REPORT("Failure to record event %p: %s\n", EventPtr,
           toString(std::move(Err)).data());

  return (bool)Err;
}

int32_t __tgt_rtl_wait_event(int32_t DeviceId, void *EventPtr,
                             __tgt_async_info *AsyncInfoPtr) {
  auto Err =
      Plugin::get().getDevice(DeviceId).waitEvent(EventPtr, AsyncInfoPtr);
  if (Err)
    REPORT("Failure to wait event %p: %s\n", EventPtr,
           toString(std::move(Err)).data());

  return (bool)Err;
}

int32_t __tgt_rtl_sync_event(int32_t DeviceId, void *EventPtr) {
  auto Err = Plugin::get().getDevice(DeviceId).syncEvent(EventPtr);
  if (Err)
    REPORT("Failure to synchronize event %p: %s\n", EventPtr,
           toString(std::move(Err)).data());

  return (bool)Err;
}

int32_t __tgt_rtl_destroy_event(int32_t DeviceId, void *EventPtr) {
  auto Err = Plugin::get().getDevice(DeviceId).destroyEvent(EventPtr);
  if (Err)
    REPORT("Failure to destroy event %p: %s\n", EventPtr,
           toString(std::move(Err)).data());

  return (bool)Err;
}

void __tgt_rtl_set_info_flag(uint32_t NewInfoLevel) {
  std::atomic<uint32_t> &InfoLevel = getInfoLevelInternal();
  InfoLevel.store(NewInfoLevel);
}

int32_t __tgt_rtl_init_async_info(int32_t DeviceId,
                                  __tgt_async_info **AsyncInfoPtr) {
  assert(AsyncInfoPtr && "Invalid async info");

  auto Err = Plugin::get().getDevice(DeviceId).initAsyncInfo(AsyncInfoPtr);
  if (Err)
    REPORT("Failure to initialize async info at " DPxMOD " on device %d: %s\n",
           DPxPTR(*AsyncInfoPtr), DeviceId, toString(std::move(Err)).data());

  return (bool)Err;
}

int32_t __tgt_rtl_init_device_info(int32_t DeviceId,
                                   __tgt_device_info *DeviceInfo,
                                   const char **ErrStr) {
  *ErrStr = "";

  auto Err = Plugin::get().getDevice(DeviceId).initDeviceInfo(DeviceInfo);
  if (Err)
    REPORT("Failure to initialize device info at " DPxMOD " on device %d: %s\n",
           DPxPTR(DeviceInfo), DeviceId, toString(std::move(Err)).data());

  return (bool)Err;
}

#ifdef __cplusplus
}
#endif
