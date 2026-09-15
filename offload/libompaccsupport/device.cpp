//===--------- device.cpp - Target independent OpenMP target RTL ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Functionality for managing devices that are handled by RTL plugins.
//
//===----------------------------------------------------------------------===//

#include "device.h"
#include "OffloadEntry.h"
#include "OpenMP/Mapping.h"
#include "OpenMP/OMPT/Callback.h"
#include "OpenMP/OMPT/Interface.h"
#include "PluginManager.h"
#include "Shared/APITypes.h"
#include "Shared/Debug.h"
#include "omptarget.h"
#include "rtl.h"

#include "Shared/EnvironmentVar.h"
#include "llvm/Frontend/OpenMP/OMPConstants.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/MathExtras.h"

#include <algorithm>
#include <cassert>
#include <climits>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <mutex>
#include <string>
#include <thread>

#ifdef OMPT_SUPPORT
using namespace llvm::omp::target::ompt;
#endif

using namespace llvm::omp::target::plugin;
using namespace llvm::omp::target::debug;

int HostDataToTargetTy::addEventIfNecessary(DeviceTy &Device,
                                            AsyncInfoTy &AsyncInfo) const {
  // First, check if the user disabled atomic map transfer/malloc/dealloc.
  if (!MappingConfig::get().UseEventsForAtomicTransfers)
    return OFFLOAD_SUCCESS;

  void *Event = getEvent();
  bool NeedNewEvent = Event == nullptr;
  if (NeedNewEvent && Device.createEvent(&Event) != OFFLOAD_SUCCESS) {
    REPORT() << "Failed to create event";
    return OFFLOAD_FAIL;
  }

  // We cannot assume the event should not be nullptr because we don't
  // know if the target support event. But if a target doesn't,
  // recordEvent should always return success.
  if (Device.recordEvent(Event, AsyncInfo) != OFFLOAD_SUCCESS) {
    REPORT() << "Failed to set dependence on event " << Event;
    return OFFLOAD_FAIL;
  }

  if (NeedNewEvent)
    setEvent(Event);

  return OFFLOAD_SUCCESS;
}

DeviceTy::DeviceTy(GenericPluginTy *RTL, int32_t DeviceID, int32_t RTLDeviceID)
    : DeviceID(DeviceID), RTL(RTL), RTLDeviceID(RTLDeviceID),
      MappingInfo(*this) {}

DeviceTy::~DeviceTy() {
  if (DeviceID == -1 || !(getInfoLevel() & OMP_INFOTYPE_DUMP_TABLE))
    return;

  ident_t Loc = {0, 0, 0, 0, ";libomptarget;libomptarget;0;0;;"};
  dumpTargetPointerMappings(&Loc, *this);
}

llvm::Error DeviceTy::init() {
  int32_t Ret = RTL->init_device(RTLDeviceID);
  if (Ret != OFFLOAD_SUCCESS)
    return error::createOffloadError(error::ErrorCode::BACKEND_FAILURE,
                                     "failed to initialize device %d\n",
                                     DeviceID);

  // Enables recording kernels if set.
  BoolEnvar OMPX_RecordKernel("LIBOMPTARGET_RECORD", false);
  if (OMPX_RecordKernel) {
    BoolEnvar OMPX_RecordOutput("LIBOMPTARGET_RECORD_OUTPUT", true);
    Int64Envar OMPX_RecordMemSize("LIBOMPTARGET_RECORD_MEMSIZE",
                                  8 * 1024 * 1024 * 1024ULL);
    Int32Envar OMPX_RecordDevice("LIBOMPTARGET_RECORD_DEVICE", 0);
    StringEnvar OMPX_RecordOutputDir("LIBOMPTARGET_RECORD_DIR", "");
    BoolEnvar OMPX_EmitRecordReport("LIBOMPTARGET_RECORD_REPORT", false);
    StringEnvar OMPX_RecordReportFilename("LIBOMPTARGET_RECORD_REPORT_FILENAME",
                                          "");
    if (OMPX_RecordDevice != RTLDeviceID)
      return llvm::Error::success();

    // Print report if it was enabled explicitly or a report file was indicated.
    bool EmitReport =
        OMPX_EmitRecordReport || !OMPX_RecordReportFilename.get().empty();

    Ret = RTL->initialize_record_replay(
        RTLDeviceID, OMPX_RecordMemSize, nullptr,
        /*IsRecord=*/true, /*IsNative=*/true, OMPX_RecordOutput, EmitReport,
        OMPX_RecordReportFilename.get().c_str(),
        OMPX_RecordOutputDir.get().c_str());
    if (Ret != OFFLOAD_SUCCESS)
      return error::createOffloadError(error::ErrorCode::BACKEND_FAILURE,
                                       "failed to initialize RR in device %d\n",
                                       DeviceID);
  }

  return llvm::Error::success();
}

// Extract the mapping of host function pointers to device function pointers
// from the entry table. Functions marked as 'indirect' in OpenMP will have
// offloading entries generated for them which map the host's function pointer
// to a global containing the corresponding function pointer on the device.
static llvm::Expected<std::pair<void *, uint64_t>>
setupIndirectCallTable(DeviceTy &Device, __tgt_device_image *Image,
                       __tgt_device_binary Binary) {
  AsyncInfoTy AsyncInfo(Device);
  llvm::ArrayRef<llvm::offloading::EntryTy> Entries(Image->EntriesBegin,
                                                    Image->EntriesEnd);
  llvm::SmallVector<std::pair<void *, void *>> IndirectCallTable;
  for (const auto &Entry : Entries) {
    if (Entry.Kind != llvm::object::OffloadKind::OFK_OpenMP ||
        Entry.Size == 0 ||
        (!(Entry.Flags & OMP_DECLARE_TARGET_INDIRECT) &&
         !(Entry.Flags & OMP_DECLARE_TARGET_INDIRECT_VTABLE)))
      continue;

    size_t PtrSize = sizeof(void *);
    if (Entry.Flags & OMP_DECLARE_TARGET_INDIRECT_VTABLE) {
      // This is a VTable entry, the current entry is the first index of the
      // VTable and Entry.Size is the total size of the VTable. Unlike the
      // indirect function case below, the Global is not of size Entry.Size and
      // is instead of size PtrSize (sizeof(void*)).
      void *Vtable;
      void *res;
      if (Device.RTL->get_global(Binary, PtrSize, Entry.SymbolName, &Vtable))
        return error::createOffloadError(error::ErrorCode::INVALID_BINARY,
                                         "failed to load %s", Entry.SymbolName);

      // HstPtr = Entry.Address;
      if (Device.retrieveData(&res, Vtable, PtrSize, AsyncInfo))
        return error::createOffloadError(error::ErrorCode::INVALID_BINARY,
                                         "failed to load %s", Entry.SymbolName);
      if (Device.synchronize(AsyncInfo))
        return error::createOffloadError(
            error::ErrorCode::INVALID_BINARY,
            "failed to synchronize after retrieving %s", Entry.SymbolName);
      // Calculate and emplace entire Vtable from first Vtable byte
      for (uint64_t i = 0; i < Entry.Size / PtrSize; ++i) {
        auto &[HstPtr, DevPtr] = IndirectCallTable.emplace_back();
        HstPtr = reinterpret_cast<void *>(
            reinterpret_cast<uintptr_t>(Entry.Address) + i * PtrSize);
        DevPtr = reinterpret_cast<void *>(reinterpret_cast<uintptr_t>(res) +
                                          i * PtrSize);
      }
    } else {
      // Indirect function case: Entry.Size should equal PtrSize since we're
      // dealing with a single function pointer (not a VTable)
      assert(Entry.Size == PtrSize && "Global not a function pointer?");
      auto &[HstPtr, DevPtr] = IndirectCallTable.emplace_back();
      void *Ptr;
      if (Device.RTL->get_global(Binary, Entry.Size, Entry.SymbolName, &Ptr))
        return error::createOffloadError(error::ErrorCode::INVALID_BINARY,
                                         "failed to load %s", Entry.SymbolName);

      HstPtr = Entry.Address;
      if (Device.retrieveData(&DevPtr, Ptr, Entry.Size, AsyncInfo))
        return error::createOffloadError(error::ErrorCode::INVALID_BINARY,
                                         "failed to load %s", Entry.SymbolName);
    }
    if (Device.synchronize(AsyncInfo))
      return error::createOffloadError(
          error::ErrorCode::INVALID_BINARY,
          "failed to synchronize after retrieving %s", Entry.SymbolName);
  }

  // If we do not have any indirect globals we exit early.
  if (IndirectCallTable.empty())
    return std::pair{nullptr, 0};

  // Sort the array to allow for more efficient lookup of device pointers.
  llvm::sort(IndirectCallTable,
             [](const auto &x, const auto &y) { return x.first < y.first; });

  uint64_t TableSize =
      IndirectCallTable.size() * sizeof(std::pair<void *, void *>);
  void *DevicePtr = Device.allocData(TableSize, nullptr, TARGET_ALLOC_DEVICE);
  if (Device.submitData(DevicePtr, IndirectCallTable.data(), TableSize,
                        AsyncInfo))
    return error::createOffloadError(error::ErrorCode::INVALID_BINARY,
                                     "failed to copy data");
  // The IndirectCallTable is on the stack, so we must synchronize to ensure
  // the data is copied before we return.
  if (Device.synchronize(AsyncInfo))
    return error::createOffloadError(
        error::ErrorCode::INVALID_BINARY,
        "failed to synchronize after copying data");

  return std::pair<void *, uint64_t>(DevicePtr, IndirectCallTable.size());
}

// Load binary to device and perform global initialization if needed.
llvm::Expected<__tgt_device_binary>
DeviceTy::loadBinary(__tgt_device_image *Img) {
  __tgt_device_binary Binary;

  if (RTL->load_binary(RTLDeviceID, Img, &Binary) != OFFLOAD_SUCCESS)
    return error::createOffloadError(error::ErrorCode::INVALID_BINARY,
                                     "failed to load binary %p", Img);

  // This symbol is optional.
  void *DeviceEnvironmentPtr;
  if (RTL->get_global(Binary, sizeof(DeviceEnvironmentTy),
                      "__omp_rtl_device_environment", &DeviceEnvironmentPtr))
    return Binary;

  // Obtain a table mapping host function pointers to device function pointers.
  auto CallTablePairOrErr = setupIndirectCallTable(*this, Img, Binary);
  if (!CallTablePairOrErr)
    return CallTablePairOrErr.takeError();

  GenericDeviceTy &GenericDevice = RTL->getDevice(RTLDeviceID);
  DeviceEnvironmentTy DeviceEnvironment;
  DeviceEnvironment.DeviceDebugKind = GenericDevice.getDebugKind();
  DeviceEnvironment.NumDevices = RTL->getNumDevices();
  // TODO: The device ID used here is not the real device ID used by OpenMP.
  DeviceEnvironment.DeviceNum = RTLDeviceID;
  DeviceEnvironment.DynamicMemSize = 0;
  DeviceEnvironment.ClockFrequency = GenericDevice.getClockFrequency();
  DeviceEnvironment.IndirectCallTable =
      reinterpret_cast<uintptr_t>(CallTablePairOrErr->first);
  DeviceEnvironment.IndirectCallTableSize = CallTablePairOrErr->second;
  DeviceEnvironment.HardwareParallelism =
      GenericDevice.getHardwareParallelism();

  AsyncInfoTy AsyncInfo(*this);
  if (submitData(DeviceEnvironmentPtr, &DeviceEnvironment,
                 sizeof(DeviceEnvironment), AsyncInfo))
    return error::createOffloadError(error::ErrorCode::INVALID_BINARY,
                                     "failed to copy data");

  return Binary;
}

void *DeviceTy::allocData(int64_t Size, void *HstPtr, int32_t Kind) {
  /// RAII to establish tool anchors before and after data allocation
  void *TargetPtr = nullptr;
  OMPT_IF_BUILT(InterfaceRAII TargetDataAllocRAII(
                    RegionInterface.getCallbacks<ompt_target_data_alloc>(),
                    DeviceID, HstPtr, &TargetPtr, Size,
                    /*CodePtr=*/OMPT_GET_RETURN_ADDRESS);)

  TargetPtr = RTL->data_alloc(RTLDeviceID, Size, HstPtr, Kind);
  return TargetPtr;
}

int32_t DeviceTy::deleteData(void *TgtAllocBegin, int32_t Kind) {
  /// RAII to establish tool anchors before and after data deletion
  OMPT_IF_BUILT(InterfaceRAII TargetDataDeleteRAII(
                    RegionInterface.getCallbacks<ompt_target_data_delete>(),
                    DeviceID, TgtAllocBegin,
                    /*CodePtr=*/OMPT_GET_RETURN_ADDRESS);)

  return RTL->data_delete(RTLDeviceID, TgtAllocBegin, Kind);
}

// Submit data to device
int32_t DeviceTy::submitData(void *TgtPtrBegin, void *HstPtrBegin, int64_t Size,
                             AsyncInfoTy &AsyncInfo, HostDataToTargetTy *Entry,
                             MappingInfoTy::HDTTMapAccessorTy *HDTTMapPtr) {
  if (getInfoLevel() & OMP_INFOTYPE_DATA_TRANSFER)
    MappingInfo.printCopyInfo(TgtPtrBegin, HstPtrBegin, Size, /*H2D=*/true,
                              Entry, HDTTMapPtr);

  /// RAII to establish tool anchors before and after data submit
  OMPT_IF_BUILT(
      InterfaceRAII TargetDataSubmitRAII(
          RegionInterface.getCallbacks<ompt_target_data_transfer_to_device>(),
          omp_initial_device, HstPtrBegin, DeviceID, TgtPtrBegin, Size,
          /*CodePtr=*/OMPT_GET_RETURN_ADDRESS);)

  return RTL->data_submit_async(RTLDeviceID, TgtPtrBegin, HstPtrBegin, Size,
                                AsyncInfo);
}

// Retrieve data from device
int32_t DeviceTy::retrieveData(void *HstPtrBegin, void *TgtPtrBegin,
                               int64_t Size, AsyncInfoTy &AsyncInfo,
                               HostDataToTargetTy *Entry,
                               MappingInfoTy::HDTTMapAccessorTy *HDTTMapPtr) {
  if (getInfoLevel() & OMP_INFOTYPE_DATA_TRANSFER)
    MappingInfo.printCopyInfo(TgtPtrBegin, HstPtrBegin, Size, /*H2D=*/false,
                              Entry, HDTTMapPtr);

  /// RAII to establish tool anchors before and after data retrieval
  OMPT_IF_BUILT(
      InterfaceRAII TargetDataRetrieveRAII(
          RegionInterface.getCallbacks<ompt_target_data_transfer_from_device>(),
          DeviceID, TgtPtrBegin, omp_initial_device, HstPtrBegin, Size,
          /*CodePtr=*/OMPT_GET_RETURN_ADDRESS);)

  return RTL->data_retrieve_async(RTLDeviceID, HstPtrBegin, TgtPtrBegin, Size,
                                  AsyncInfo);
}

// Copy data from current device to destination device directly
int32_t DeviceTy::dataExchange(void *SrcPtr, DeviceTy &DstDev, void *DstPtr,
                               int64_t Size, AsyncInfoTy &AsyncInfo) {
  /// RAII to establish tool anchors before and after data exchange
  /// Note: Despite the fact that this is a data exchange, we use 'from_device'
  ///       operation enum (w.r.t. ompt_target_data_op_t) as there is currently
  ///       no better alternative. It is still possible to distinguish this
  ///       scenario from a real data retrieve by checking if both involved
  ///       device numbers are less than omp_get_num_devices().
  OMPT_IF_BUILT(
      InterfaceRAII TargetDataExchangeRAII(
          RegionInterface.getCallbacks<ompt_target_data_transfer_from_device>(),
          RTLDeviceID, SrcPtr, DstDev.RTLDeviceID, DstPtr, Size,
          /*CodePtr=*/OMPT_GET_RETURN_ADDRESS);)
  if (!AsyncInfo) {
    return RTL->data_exchange(RTLDeviceID, SrcPtr, DstDev.RTLDeviceID, DstPtr,
                              Size);
  }
  return RTL->data_exchange_async(RTLDeviceID, SrcPtr, DstDev.RTLDeviceID,
                                  DstPtr, Size, AsyncInfo);
}

int32_t DeviceTy::dataFence(AsyncInfoTy &AsyncInfo) {
  return RTL->data_fence(RTLDeviceID, AsyncInfo);
}

int32_t DeviceTy::notifyDataMapped(void *HstPtr, int64_t Size) {
  ODBG(ODT_Mapping) << "Notifying about new mapping: HstPtr=" << HstPtr
                    << ", Size=" << Size;

  if (RTL->data_notify_mapped(RTLDeviceID, HstPtr, Size)) {
    REPORT() << "Notifying about data mapping failed.";
    return OFFLOAD_FAIL;
  }
  return OFFLOAD_SUCCESS;
}

int32_t DeviceTy::notifyDataUnmapped(void *HstPtr) {
  ODBG(ODT_Mapping) << "Notifying about an unmapping: HstPtr=" << HstPtr;

  if (RTL->data_notify_unmapped(RTLDeviceID, HstPtr)) {
    REPORT() << "Notifying about data unmapping failed.";
    return OFFLOAD_FAIL;
  }
  return OFFLOAD_SUCCESS;
}

/// Resolve \p NumArgs (base pointer, offset) pairs into a flattened array of
/// argument-value pointers suitable for a kernel launch, writing the result
/// into \p LaunchArgs.NumArgs/Args.
static void resolveKernelLaunchParams(void **const TgtArgs,
                                      ptrdiff_t *const TgtOffsets,
                                      uint32_t NumArgs,
                                      llvm::SmallVector<void *> &Args,
                                      llvm::SmallVector<void *> &Ptrs,
                                      KernelLaunchArgsTy &LaunchArgs) {
  LaunchArgs.NumArgs = NumArgs;
  Args.resize(NumArgs);
  Ptrs.resize(NumArgs);

  if (NumArgs == 0)
    return;

  for (uint32_t I = 0; I < NumArgs; ++I) {
    Args[I] = reinterpret_cast<void *>(reinterpret_cast<intptr_t>(TgtArgs[I]) +
                                       TgtOffsets[I]);
    Ptrs[I] = &Args[I];
  }

  LaunchArgs.Args = &Ptrs[0];
}

namespace {
/// Configuration of dynamic block memory needed for launching a kernel.
struct DynBlockMemConfTy {
  /// The size of the dynamic block memory buffer.
  uint32_t Size = 0;
  /// The size of dynamic shared memory natively provided by the device.
  uint32_t NativeSize = 0;
  /// The fallback that was triggered (if any).
  DynCGroupMemFallbackType Fallback = DynCGroupMemFallbackType::None;
  /// The fallback pointer if global memory was used as alternative.
  void *FallbackPtr = nullptr;
};
} // namespace

/// Prepare the block memory buffer requested for the kernel and execute the
/// specified fallback if necessary.
static llvm::Expected<DynBlockMemConfTy>
prepareBlockMemory(GenericDeviceTy &GenericDevice,
                   const KernelLaunchInfoTy &KernelEnv, uint32_t DynCGroupMem,
                   DynCGroupMemFallbackType DynCGroupMemFallback,
                   uint32_t NumBlocks) {
  uint32_t MaxBlockMemSize = GenericDevice.getMaxBlockSharedMemSize();
  uint32_t DynBlockMemSize = DynCGroupMem;
  uint32_t TotalBlockMemSize = KernelEnv.StaticBlockMemSize + DynBlockMemSize;
  uint32_t DynNativeBlockMemSize = DynBlockMemSize;
  void *DynFallbackPtr = nullptr;

  // No enough block memory to cover the static one. Cannot run the kernel.
  if (KernelEnv.StaticBlockMemSize > MaxBlockMemSize)
    return error::createOffloadError(
        error::ErrorCode::INVALID_ARGUMENT,
        "Static block memory size exceeds maximum");
  // No enough block memory to cover dynamic one, and the fallback is aborting.
  if (DynCGroupMemFallback == DynCGroupMemFallbackType::Abort &&
      TotalBlockMemSize > MaxBlockMemSize)
    return error::createOffloadError(
        error::ErrorCode::INVALID_ARGUMENT,
        "Requested block memory size (static + dynamic) exceeds maximum");

  DynCGroupMemFallbackType DynFallback = DynCGroupMemFallbackType::None;
  if (DynBlockMemSize && TotalBlockMemSize > MaxBlockMemSize) {
    // Launch without native dynamic block memory.
    DynNativeBlockMemSize = 0;
    DynFallback = DynCGroupMemFallback;
    if (DynFallback != DynCGroupMemFallbackType::DefaultMem) {
      // Do not provide any memory as fallback.
      DynBlockMemSize = 0;
    } else {
      // Get global memory as fallback.
      auto AllocOrErr = GenericDevice.dataAlloc(
          NumBlocks * DynBlockMemSize,
          /*HostPtr=*/nullptr, TARGET_ALLOC_DEVICE, /*Alignment=*/0);
      if (!AllocOrErr)
        return AllocOrErr.takeError();
      DynFallbackPtr = *AllocOrErr;
    }
  }
  return DynBlockMemConfTy{DynBlockMemSize, DynNativeBlockMemSize, DynFallback,
                           DynFallbackPtr};
}

static void freeAfterSynchronization(GenericDeviceTy &GenericDevice,
                                     AsyncInfoTy &AsyncInfo, void *Ptr,
                                     TargetAllocTy Kind) {
  AsyncInfo.addPostProcessingFunction([&GenericDevice, Ptr, Kind]() -> int {
    if (auto Err = GenericDevice.dataDelete(Ptr, Kind)) {
      REPORT() << "Failure to free device memory " << Ptr << ": "
               << toString(std::move(Err));
      return OFFLOAD_FAIL;
    }
    return OFFLOAD_SUCCESS;
  });
}

/// Return a device pointer to a new kernel launch environment, or null if
/// this launch has no reserved dyn_ptr slot to store one in. \p NumBlocks0 is
/// the number of blocks for this launch and is used to size the reduction
/// buffer.
static llvm::Expected<KernelLaunchEnvironmentTy *> getKernelLaunchEnvironment(
    GenericDeviceTy &GenericDevice, const KernelLaunchArgsTy &LaunchArgs,
    const KernelLaunchInfoTy &KernelEnv,
    const DynBlockMemConfTy &DynBlockMemConf, uint32_t DynCGroupMem,
    void **DynPtrSlot, AsyncInfoTy &AsyncInfo, uint32_t NumBlocks0) {
  // Ctor/Dtor have no arguments, replaying uses the original kernel launch
  // environment, and launches with no reserved dyn_ptr slot (e.g. older
  // compiler versions, or non-OpenMP launches) have nowhere to store one.
  if ((GenericDevice.getRecordReplay() &&
       GenericDevice.getRecordReplay()->isReplaying()) ||
      !DynPtrSlot)
    return nullptr;

  const bool NeedsReductionBuffer = KernelEnv.ReductionDataSize != 0;
  if (NeedsReductionBuffer && LaunchArgs.OmpABIVersion < OMP_KERNEL_ARG_VERSION)
    return error::createOffloadError(
        error::ErrorCode::INVALID_BINARY,
        "kernel was built against an older OpenMP kernel-launch-environment "
        "ABI (v%u); current runtime requires v%u for cross-team reductions",
        LaunchArgs.OmpABIVersion, OMP_KERNEL_ARG_VERSION);
  if (!NeedsReductionBuffer && !DynCGroupMem)
    return reinterpret_cast<KernelLaunchEnvironmentTy *>(~0);

  auto AllocOrErr = GenericDevice.dataAlloc(
      sizeof(KernelLaunchEnvironmentTy),
      /*HostPtr=*/nullptr, TARGET_ALLOC_DEVICE, /*Alignment=*/0);
  if (!AllocOrErr)
    return AllocOrErr.takeError();

  // Remember to free the memory later.
  freeAfterSynchronization(GenericDevice, AsyncInfo, *AllocOrErr,
                           TARGET_ALLOC_DEVICE);

  // Use the KLE in the __tgt_async_info to ensure a stable address for the
  // async data transfer.
  auto &LocalKLE =
      static_cast<__tgt_async_info *>(AsyncInfo)->KernelLaunchEnvironment;
  LocalKLE = KernelLaunchEnvironmentTy{};
  LocalKLE.DynCGroupMemSize = DynBlockMemConf.Size;
  LocalKLE.DynCGroupMemFbPtr = DynBlockMemConf.FallbackPtr;
  LocalKLE.DynCGroupMemFb = DynBlockMemConf.Fallback;
  LocalKLE.ReductionBuffer = nullptr;

  if (NeedsReductionBuffer) {
    // Use number of teams many buffer elements.
    auto ReductionAllocOrErr = GenericDevice.dataAlloc(
        uint64_t(KernelEnv.ReductionDataSize) * NumBlocks0,
        /*HostPtr=*/nullptr, TARGET_ALLOC_DEVICE, /*Alignment=*/0);
    if (!ReductionAllocOrErr)
      return ReductionAllocOrErr.takeError();
    LocalKLE.ReductionBuffer = *ReductionAllocOrErr;
    // Remember to free the memory later.
    freeAfterSynchronization(GenericDevice, AsyncInfo, *ReductionAllocOrErr,
                             TARGET_ALLOC_DEVICE);
  }

  INFO(OMP_INFOTYPE_DATA_TRANSFER, GenericDevice.getDeviceId(),
       "Copying data from host to device, HstPtr=" DPxMOD ", TgtPtr=" DPxMOD
       ", Size=%" PRId64 ", Name=KernelLaunchEnv\n",
       DPxPTR(&LocalKLE), DPxPTR(*AllocOrErr),
       sizeof(KernelLaunchEnvironmentTy));

  if (auto Err = GenericDevice.dataSubmit(
          *AllocOrErr, &LocalKLE, sizeof(KernelLaunchEnvironmentTy), AsyncInfo))
    return Err;
  return static_cast<KernelLaunchEnvironmentTy *>(*AllocOrErr);
}

/// Get the effective number of threads for the kernel based on the
/// user-defined number of threads.
static uint32_t getEffectiveNumThreads(GenericDeviceTy &GenericDevice,
                                       uint32_t UserThreadLimit,
                                       const KernelLaunchInfoTy &KernelEnv) {
  assert(!KernelEnv.isBareMode() &&
         "bare kernel should not call this function");

  if (UserThreadLimit > 0 && KernelEnv.isGenericMode())
    UserThreadLimit += GenericDevice.getWarpSize();

  return std::min(KernelEnv.MaxNumThreads, (UserThreadLimit > 0)
                                               ? UserThreadLimit
                                               : KernelEnv.PreferredNumThreads);
}

/// Get the effective number of blocks for the kernel based on the
/// user-defined number of blocks and the loop trip count.
/// The number of threads \p EffectiveNumThreads can be adjusted by this
/// method. \p IsNumThreadsFromUser is true if \p EffectiveNumThreads is
/// defined by the user via the thread_limit clause.
static uint32_t
getEffectiveNumBlocks(GenericDeviceTy &GenericDevice, uint32_t UserNumBlocks,
                      uint64_t LoopTripCount, uint32_t &EffectiveNumThreads,
                      bool IsNumThreadsStrict, bool IsNumThreadsFromUser,
                      const KernelLaunchInfoTy &KernelEnv) {
  assert(!KernelEnv.isBareMode() &&
         "bare kernel should not call this function");

  // NOTE: This clamps the user-requested number of blocks to the device limit
  // rather than honoring it exactly, which is non-standard behavior. Truly
  // honoring an arbitrary value would require launching multiple kernels or
  // reusing blocks until the requested count has been served.
  if (UserNumBlocks > 0)
    return std::min(UserNumBlocks,
                    GenericDevice.getBlockLimit(EffectiveNumThreads));

  // Return the number of blocks required to cover the loop iterations.
  if (KernelEnv.isNoLoopMode())
    return LoopTripCount > 0 ? (((LoopTripCount - 1) / EffectiveNumThreads) + 1)
                             : 1;

  uint64_t DefaultNumBlocks = GenericDevice.getDefaultNumBlocks();
  uint64_t TripCountNumBlocks = std::numeric_limits<uint64_t>::max();
  if (LoopTripCount > 0) {
    if (KernelEnv.isSPMDMode()) {
      // We have a combined construct, i.e. `target teams distribute
      // parallel for [simd]`. We launch so many blocks so that each thread
      // will execute one iteration of the loop; rounded up to the nearest
      // integer. However, if that results in too few blocks, we artificially
      // reduce the thread count per block to increase the outer parallelism.
      auto MinThreads = GenericDevice.getMinThreadsForLowTripCountLoop();
      MinThreads = std::min(MinThreads, EffectiveNumThreads);

      // Honor the thread_limit clause; only lower the number of threads.
      [[maybe_unused]] auto OldNumThreads = EffectiveNumThreads;
      if (LoopTripCount >= DefaultNumBlocks * EffectiveNumThreads ||
          IsNumThreadsFromUser || IsNumThreadsStrict) {
        // Enough parallelism for blocks and threads.
        TripCountNumBlocks = ((LoopTripCount - 1) / EffectiveNumThreads) + 1;
        assert(IsNumThreadsFromUser ||
               TripCountNumBlocks >= DefaultNumBlocks &&
                   "Expected sufficient outer parallelism.");
      } else if (LoopTripCount >= DefaultNumBlocks * MinThreads) {
        // Enough parallelism for blocks, limit threads.

        // This case is hard; for now, we force "full warps":
        // First, compute a thread count assuming DefaultNumBlocks.
        auto NumThreadsDefaultBlocks =
            (LoopTripCount + DefaultNumBlocks - 1) / DefaultNumBlocks;
        // Now get a power of two that is larger or equal.
        auto NumThreadsDefaultBlocksP2 =
            llvm::PowerOf2Ceil(NumThreadsDefaultBlocks);
        // Do not increase a thread limit given be the user.
        EffectiveNumThreads =
            std::min(EffectiveNumThreads, uint32_t(NumThreadsDefaultBlocksP2));
        assert(EffectiveNumThreads >= MinThreads &&
               "Expected sufficient inner parallelism.");
        TripCountNumBlocks = ((LoopTripCount - 1) / EffectiveNumThreads) + 1;
      } else {
        // Not enough parallelism for blocks and threads, limit both.
        EffectiveNumThreads = std::min(EffectiveNumThreads, MinThreads);
        TripCountNumBlocks = ((LoopTripCount - 1) / EffectiveNumThreads) + 1;
      }

      assert(EffectiveNumThreads * TripCountNumBlocks >= LoopTripCount &&
             "Expected sufficient parallelism");
      assert(OldNumThreads >= EffectiveNumThreads &&
             "Number of threads cannot be increased!");
    } else {
      assert((KernelEnv.isGenericMode() || KernelEnv.isGenericSPMDMode()) &&
             "Unexpected execution mode!");
      // If we reach this point, then we have a non-combined construct, i.e.
      // `teams distribute` with a nested `parallel for` and each block is
      // assigned one iteration of the `distribute` loop. E.g.:
      //
      // #pragma omp target teams distribute
      // for(...loop_tripcount...) {
      //   #pragma omp parallel for
      //   for(...) {}
      // }
      //
      // Threads within a block will execute the iterations of the `parallel`
      // loop.
      TripCountNumBlocks = LoopTripCount;
    }
  }

  uint32_t PreferredNumBlocks = TripCountNumBlocks;
  // If the loops are long running we rather reuse blocks than spawn too many.
  if (GenericDevice.getReuseBlocksForHighTripCount())
    PreferredNumBlocks = std::min(TripCountNumBlocks, DefaultNumBlocks);
  return std::min(PreferredNumBlocks,
                  GenericDevice.getBlockLimit(EffectiveNumThreads));
}

/// Build the base KernelLaunchArgsTy for a launch from the public
/// KernelArgsTy and the kernel's cached launch-geometry properties.
static KernelLaunchArgsTy buildLaunchArgs(const KernelArgsTy &KernelArgs,
                                          KernelReplayOutcomeTy *ReplayOutcome,
                                          const KernelLaunchInfoTy &KernelEnv) {
  KernelLaunchArgsTy LaunchArgs;
  LaunchArgs.OmpABIVersion = KernelArgs.Version;
  LaunchArgs.ReplayOutcome = ReplayOutcome;
  LaunchArgs.ArgSizes = KernelArgs.ArgSizes;
  LaunchArgs.Tripcount = KernelArgs.Tripcount;
  llvm::copy(KernelArgs.UserNumBlocks, LaunchArgs.UserNumBlocks);
  llvm::copy(KernelArgs.UserThreadLimit, LaunchArgs.UserThreadLimit);
  LaunchArgs.Flags.Cooperative = KernelArgs.Flags.Cooperative;
  LaunchArgs.MaxNumThreads = KernelEnv.MaxNumThreads;
  return LaunchArgs;
}

/// Assert the launch geometry invariants expected by the plugin layer.
static void checkLaunchInvariants(const KernelLaunchArgsTy &LaunchArgs,
                                  const KernelArgsTy &KernelArgs,
                                  const KernelLaunchInfoTy &KernelEnv) {
  // Multidimensional is only supported with bare mode for now.
  assert(KernelEnv.isBareMode() ||
         LaunchArgs.UserThreadLimit[1] == 1 &&
             LaunchArgs.UserThreadLimit[2] == 1 &&
             LaunchArgs.UserNumBlocks[1] == 1 &&
             LaunchArgs.UserNumBlocks[2] == 1 &&
             "Non-bare mode should only use the first thread and block "
             "dimensions");

  assert(!KernelArgs.Flags.StrictBlocks ||
         LaunchArgs.UserNumBlocks[0] > 0 && LaunchArgs.UserNumBlocks[1] > 0 &&
             LaunchArgs.UserNumBlocks[2] > 0 &&
             "Strict requires number of blocks greater than zero");
  assert(!KernelArgs.Flags.StrictThreads ||
         LaunchArgs.UserThreadLimit[0] > 0 &&
             LaunchArgs.UserThreadLimit[1] > 0 &&
             LaunchArgs.UserThreadLimit[2] > 0 &&
             "Strict requires number of threads greater than zero");
}

/// Calculate or adjust, in place, the effective number of threads and blocks
/// for the first dimension, unless the caller requested strict counts.
static void adjustEffectiveGeometry(GenericDeviceTy &GenericDevice,
                                    KernelLaunchArgsTy &LaunchArgs,
                                    const KernelArgsTy &KernelArgs,
                                    const KernelLaunchInfoTy &KernelEnv) {
  const bool StrictBlocks = KernelArgs.Flags.StrictBlocks;
  const bool StrictThreads = KernelArgs.Flags.StrictThreads;
  if (StrictThreads && StrictBlocks)
    return;

  assert(!KernelEnv.isBareMode() &&
         "bare kernel launches must request strict thread/block counts");

  // Record whether the user actually requested a thread limit (thread_limit
  // clause) before possibly overwriting UserThreadLimit[0] below with the
  // computed effective value.
  const bool ThreadLimitFromUser = LaunchArgs.UserThreadLimit[0] > 0;

  uint32_t EffectiveNumThreads = LaunchArgs.UserThreadLimit[0];
  if (!StrictThreads)
    EffectiveNumThreads =
        getEffectiveNumThreads(GenericDevice, EffectiveNumThreads, KernelEnv);

  if (!StrictBlocks)
    LaunchArgs.UserNumBlocks[0] = getEffectiveNumBlocks(
        GenericDevice, LaunchArgs.UserNumBlocks[0], LaunchArgs.Tripcount,
        EffectiveNumThreads, StrictThreads, ThreadLimitFromUser, KernelEnv);

  LaunchArgs.UserThreadLimit[0] = EffectiveNumThreads;
}

/// Flatten the kernel arguments into \p LaunchArgs.Args. Returns the address
/// of the element reserved for the kernel launch environment (dyn_ptr), or
/// null if this launch has no such slot.
static void **resolveArgsAndDynPtrSlot(KernelArgsTy &KernelArgs,
                                       void **TgtVarsPtr, ptrdiff_t *TgtOffsets,
                                       llvm::SmallVector<void *> &Args,
                                       llvm::SmallVector<void *> &Ptrs,
                                       llvm::SmallVector<int64_t> &ArgSizes,
                                       KernelLaunchArgsTy &LaunchArgs) {
  if (KernelArgs.Flags.IsCUDA) {
    // Kernel languages (CUDA/HIP) pass an already-flattened argument-pointer
    // array through KernelArgs.ArgPtrs instead of using the OpenMP
    // base-pointer/offset argument scheme.
    auto *LaunchParams =
        reinterpret_cast<KernelLaunchParamsTy *>(KernelArgs.ArgPtrs);
    LaunchArgs.NumArgs = LaunchParams->NumArgs;
    LaunchArgs.Args = LaunchParams->Args;
    return nullptr;
  }

  resolveKernelLaunchParams(TgtVarsPtr, TgtOffsets, KernelArgs.NumArgs, Args,
                            Ptrs, LaunchArgs);

  if (KernelArgs.NumArgs == 0 ||
      KernelArgs.Version < OMP_KERNEL_ARG_MIN_VERSION_WITH_DYN_PTR)
    return nullptr;

  // The dyn_ptr slot is reserved by the host (version >= 4) or by
  // upgradeKernelArgs (version 3) as the last element of the argument array.
  // Version 3 device kernels expect it first instead, so rotate it to the
  // front to match that ABI.
  if (KernelArgs.Version != OMP_KERNEL_ARG_MIN_VERSION_WITH_DYN_PTR)
    return &Args[KernelArgs.NumArgs - 1];

  std::rotate(Args.begin(), Args.end() - 1, Args.end());

  // Keep ArgSizes in sync with the rotated Args, if present.
  if (LaunchArgs.ArgSizes) {
    ArgSizes.assign(LaunchArgs.ArgSizes,
                    LaunchArgs.ArgSizes + KernelArgs.NumArgs);
    std::rotate(ArgSizes.begin(), ArgSizes.end() - 1, ArgSizes.end());
    LaunchArgs.ArgSizes = ArgSizes.data();
  }
  return &Args[0];
}

/// Compute the dynamic block-memory configuration for this launch, filling in
/// \p LaunchArgs.DynCGroupMem with the native size to request, and, if this
/// launch has a reserved dyn_ptr slot (\p DynPtrSlot), the device-side kernel
/// launch environment.
static llvm::Error
prepareDynamicLaunchState(GenericDeviceTy &GenericDevice,
                          const KernelLaunchInfoTy &KernelEnv,
                          KernelLaunchArgsTy &LaunchArgs, uint32_t DynCGroupMem,
                          DynCGroupMemFallbackType DynCGroupMemFallback,
                          void **DynPtrSlot, AsyncInfoTy &AsyncInfo) {
  uint32_t NumBlocksTotal = LaunchArgs.UserNumBlocks[0] *
                            LaunchArgs.UserNumBlocks[1] *
                            LaunchArgs.UserNumBlocks[2];
  auto DynBlockMemConfOrErr =
      prepareBlockMemory(GenericDevice, KernelEnv, DynCGroupMem,
                         DynCGroupMemFallback, NumBlocksTotal);
  if (!DynBlockMemConfOrErr)
    return DynBlockMemConfOrErr.takeError();

  DynBlockMemConfTy &DynBlockMemConf = *DynBlockMemConfOrErr;
  LaunchArgs.DynCGroupMem = DynBlockMemConf.NativeSize;
  if (DynBlockMemConf.FallbackPtr)
    freeAfterSynchronization(GenericDevice, AsyncInfo,
                             DynBlockMemConf.FallbackPtr, TARGET_ALLOC_DEVICE);

  auto KernelLaunchEnvOrErr = getKernelLaunchEnvironment(
      GenericDevice, LaunchArgs, KernelEnv, DynBlockMemConf, DynCGroupMem,
      DynPtrSlot, AsyncInfo, LaunchArgs.UserNumBlocks[0]);
  if (!KernelLaunchEnvOrErr)
    return KernelLaunchEnvOrErr.takeError();

  // Fill in the kernel launch environment (dyn_ptr) if this launch has a
  // reserved slot for it. When replaying, getKernelLaunchEnvironment()
  // returns null so the recorded value already in the slot is preserved.
  if (DynPtrSlot && *KernelLaunchEnvOrErr)
    *DynPtrSlot = *KernelLaunchEnvOrErr;

  return llvm::Error::success();
}

// Run region on device
int32_t DeviceTy::launchKernel(void *TgtEntryPtr, void **TgtVarsPtr,
                               ptrdiff_t *TgtOffsets, KernelArgsTy &KernelArgs,
                               KernelReplayOutcomeTy *ReplayOutcome,
                               AsyncInfoTy &AsyncInfo) {
  llvm::SmallVector<void *> Args, Ptrs;
  llvm::SmallVector<int64_t> ArgSizes;

  GenericDeviceTy &GenericDevice = RTL->getDevice(RTLDeviceID);
  KernelLaunchInfoTy KernelEnv = getKernelLaunchInfo(TgtEntryPtr);
  KernelLaunchArgsTy LaunchArgs =
      buildLaunchArgs(KernelArgs, ReplayOutcome, KernelEnv);

  checkLaunchInvariants(LaunchArgs, KernelArgs, KernelEnv);
  adjustEffectiveGeometry(GenericDevice, LaunchArgs, KernelArgs, KernelEnv);

  void **DynPtrSlot = resolveArgsAndDynPtrSlot(
      KernelArgs, TgtVarsPtr, TgtOffsets, Args, Ptrs, ArgSizes, LaunchArgs);

  auto DynCGroupMemFallback = static_cast<DynCGroupMemFallbackType>(
      KernelArgs.Flags.DynCGroupMemFallback);
  if (auto Err = prepareDynamicLaunchState(
          GenericDevice, KernelEnv, LaunchArgs, KernelArgs.DynCGroupMem,
          DynCGroupMemFallback, DynPtrSlot, AsyncInfo)) {
    REPORT() << "Failure to prepare launch state for kernel " << TgtEntryPtr
             << ": " << toString(std::move(Err));
    return OFFLOAD_FAIL;
  }

  auto *Kernel = reinterpret_cast<GenericKernelTy *>(TgtEntryPtr);
  INFO(OMP_INFOTYPE_PLUGIN_KERNEL, GenericDevice.getDeviceId(),
       "Launching kernel %s with [%u,%u,%u] blocks and [%u,%u,%u] threads in "
       "%s mode\n",
       Kernel->getName(), LaunchArgs.UserNumBlocks[0],
       LaunchArgs.UserNumBlocks[1], LaunchArgs.UserNumBlocks[2],
       LaunchArgs.UserThreadLimit[0], LaunchArgs.UserThreadLimit[1],
       LaunchArgs.UserThreadLimit[2], KernelEnv.getExecutionModeName());

  return RTL->launch_kernel(RTLDeviceID, TgtEntryPtr, LaunchArgs, AsyncInfo);
}

// Run region on device
bool DeviceTy::printDeviceInfo() {
  RTL->print_device_info(RTLDeviceID);
  return true;
}

// Whether data can be copied to DstDevice directly
bool DeviceTy::isDataExchangable(const DeviceTy &DstDevice) {
  if (RTL != DstDevice.RTL)
    return false;

  if (RTL->is_data_exchangable(RTLDeviceID, DstDevice.RTLDeviceID))
    return true;
  return false;
}

int32_t DeviceTy::synchronize(AsyncInfoTy &AsyncInfo) {
  return RTL->synchronize(RTLDeviceID, AsyncInfo);
}

int32_t DeviceTy::queryAsync(AsyncInfoTy &AsyncInfo) {
  return RTL->query_async(RTLDeviceID, AsyncInfo);
}

int32_t DeviceTy::createEvent(void **Event) {
  return RTL->create_event(RTLDeviceID, Event);
}

int32_t DeviceTy::recordEvent(void *Event, AsyncInfoTy &AsyncInfo) {
  return RTL->record_event(RTLDeviceID, Event, AsyncInfo);
}

int32_t DeviceTy::waitEvent(void *Event, AsyncInfoTy &AsyncInfo) {
  return RTL->wait_event(RTLDeviceID, Event, AsyncInfo);
}

int32_t DeviceTy::syncEvent(void *Event) {
  return RTL->sync_event(RTLDeviceID, Event);
}

int32_t DeviceTy::destroyEvent(void *Event) {
  return RTL->destroy_event(RTLDeviceID, Event);
}

void DeviceTy::dumpOffloadEntries() {
  fprintf(stderr, "Device %i offload entries:\n", DeviceID);
  for (auto &It : *DeviceOffloadEntries.getExclusiveAccessor()) {
    const char *Kind = "kernel";
    if (It.second.isLink())
      Kind = "link";
    else if (It.second.isGlobal())
      Kind = "global var.";
    fprintf(stderr, "  %11s: %s\n", Kind, It.second.getNameAsCStr());
  }
}

bool DeviceTy::useAutoZeroCopy() {
  if (PM->getRequirements() & OMP_REQ_UNIFIED_SHARED_MEMORY)
    return false;
  return RTL->use_auto_zero_copy(RTLDeviceID);
}

bool DeviceTy::isAccessiblePtr(const void *Ptr, size_t Size) {
  return RTL->is_accessible_ptr(RTLDeviceID, Ptr, Size);
}
