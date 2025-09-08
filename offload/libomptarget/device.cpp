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
#include "private.h"
#include "rtl.h"

#include "Shared/EnvironmentVar.h"
#include "llvm/Support/Error.h"

#include <cassert>
#include <climits>
#include <cstdint>
#include <cstdio>
#include <mutex>
#include <string>
#include <thread>

#ifdef OMPT_SUPPORT
using namespace llvm::omp::target::ompt;
#endif

using namespace llvm::omp::target::plugin;

int HostDataToTargetTy::addEventIfNecessary(DeviceTy &Device,
                                            AsyncInfoTy &AsyncInfo) const {
  // First, check if the user disabled atomic map transfer/malloc/dealloc.
  if (!MappingConfig::get().UseEventsForAtomicTransfers)
    return OFFLOAD_SUCCESS;

  void *Event = getEvent();
  bool NeedNewEvent = Event == nullptr;
  if (NeedNewEvent && Device.createEvent(&Event) != OFFLOAD_SUCCESS) {
    REPORT("Failed to create event\n");
    return OFFLOAD_FAIL;
  }

  // We cannot assume the event should not be nullptr because we don't
  // know if the target support event. But if a target doesn't,
  // recordEvent should always return success.
  if (Device.recordEvent(Event, AsyncInfo) != OFFLOAD_SUCCESS) {
    REPORT("Failed to set dependence on event " DPxMOD "\n", DPxPTR(Event));
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
    // Enables saving the device memory kernel output post execution if set.
    BoolEnvar OMPX_ReplaySaveOutput("LIBOMPTARGET_RR_SAVE_OUTPUT", false);

    uint64_t ReqPtrArgOffset;
    RTL->initialize_record_replay(RTLDeviceID, 0, nullptr, true,
                                  OMPX_ReplaySaveOutput, ReqPtrArgOffset);
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
        Entry.Size == 0 || !(Entry.Flags & OMP_DECLARE_TARGET_INDIRECT))
      continue;

    assert(Entry.Size == sizeof(void *) && "Global not a function pointer?");
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
  DeviceEnvironment.DynamicMemSize = GenericDevice.getDynamicMemorySize();
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
          omp_get_initial_device(), HstPtrBegin, DeviceID, TgtPtrBegin, Size,
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
          DeviceID, TgtPtrBegin, omp_get_initial_device(), HstPtrBegin, Size,
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
  DP("Notifying about new mapping: HstPtr=" DPxMOD ", Size=%" PRId64 "\n",
     DPxPTR(HstPtr), Size);

  if (RTL->data_notify_mapped(RTLDeviceID, HstPtr, Size)) {
    REPORT("Notifying about data mapping failed.\n");
    return OFFLOAD_FAIL;
  }
  return OFFLOAD_SUCCESS;
}

int32_t DeviceTy::notifyDataUnmapped(void *HstPtr) {
  DP("Notifying about an unmapping: HstPtr=" DPxMOD "\n", DPxPTR(HstPtr));

  if (RTL->data_notify_unmapped(RTLDeviceID, HstPtr)) {
    REPORT("Notifying about data unmapping failed.\n");
    return OFFLOAD_FAIL;
  }
  return OFFLOAD_SUCCESS;
}

// Run region on device
int32_t DeviceTy::launchKernel(void *TgtEntryPtr, void **TgtVarsPtr,
                               ptrdiff_t *TgtOffsets, KernelArgsTy &KernelArgs,
                               AsyncInfoTy &AsyncInfo) {
  return RTL->launch_kernel(RTLDeviceID, TgtEntryPtr, TgtVarsPtr, TgtOffsets,
                            &KernelArgs, AsyncInfo);
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
