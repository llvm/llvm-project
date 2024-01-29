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

DeviceTy::DeviceTy(PluginAdaptorTy *RTL, int32_t DeviceID, int32_t RTLDeviceID)
    : DeviceID(DeviceID), RTL(RTL), RTLDeviceID(RTLDeviceID),
      PendingCtorsDtors(), PendingGlobalsMtx(), MappingInfo(*this) {}

DeviceTy::~DeviceTy() {
  if (DeviceID == -1 || !(getInfoLevel() & OMP_INFOTYPE_DUMP_TABLE))
    return;

  ident_t Loc = {0, 0, 0, 0, ";libomptarget;libomptarget;0;0;;"};
  dumpTargetPointerMappings(&Loc, *this);
}

llvm::Error DeviceTy::init() {
  // Make call to init_requires if it exists for this plugin.
  int32_t Ret = 0;
  if (RTL->init_requires)
    Ret = RTL->init_requires(PM->getRequirements());
  if (Ret != OFFLOAD_SUCCESS)
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "Failed to initialize requirements for device %d\n", DeviceID);

  Ret = RTL->init_device(RTLDeviceID);
  if (Ret != OFFLOAD_SUCCESS)
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "Failed to initialize device %d\n",
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

// Load binary to device.
llvm::Expected<__tgt_device_binary>
DeviceTy::loadBinary(__tgt_device_image *Img) {
  __tgt_device_binary Binary;

  if (RTL->load_binary(RTLDeviceID, Img, &Binary) != OFFLOAD_SUCCESS)
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "Failed to load binary %p", Img);
  return Binary;
}

void *DeviceTy::allocData(int64_t Size, void *HstPtr, int32_t Kind) {
  /// RAII to establish tool anchors before and after data allocation
  void *TargetPtr = nullptr;
  OMPT_IF_BUILT(InterfaceRAII TargetDataAllocRAII(
                    RegionInterface.getCallbacks<ompt_target_data_alloc>(),
                    DeviceID, HstPtr, &TargetPtr, Size,
                    /*CodePtr=*/OMPT_GET_RETURN_ADDRESS(0));)

  TargetPtr = RTL->data_alloc(RTLDeviceID, Size, HstPtr, Kind);
  return TargetPtr;
}

int32_t DeviceTy::deleteData(void *TgtAllocBegin, int32_t Kind) {
  /// RAII to establish tool anchors before and after data deletion
  OMPT_IF_BUILT(InterfaceRAII TargetDataDeleteRAII(
                    RegionInterface.getCallbacks<ompt_target_data_delete>(),
                    DeviceID, TgtAllocBegin,
                    /*CodePtr=*/OMPT_GET_RETURN_ADDRESS(0));)

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
          DeviceID, TgtPtrBegin, HstPtrBegin, Size,
          /*CodePtr=*/OMPT_GET_RETURN_ADDRESS(0));)

  if (!AsyncInfo || !RTL->data_submit_async || !RTL->synchronize)
    return RTL->data_submit(RTLDeviceID, TgtPtrBegin, HstPtrBegin, Size);
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
          DeviceID, HstPtrBegin, TgtPtrBegin, Size,
          /*CodePtr=*/OMPT_GET_RETURN_ADDRESS(0));)

  if (!RTL->data_retrieve_async || !RTL->synchronize)
    return RTL->data_retrieve(RTLDeviceID, HstPtrBegin, TgtPtrBegin, Size);
  return RTL->data_retrieve_async(RTLDeviceID, HstPtrBegin, TgtPtrBegin, Size,
                                  AsyncInfo);
}

// Copy data from current device to destination device directly
int32_t DeviceTy::dataExchange(void *SrcPtr, DeviceTy &DstDev, void *DstPtr,
                               int64_t Size, AsyncInfoTy &AsyncInfo) {
  if (!AsyncInfo || !RTL->data_exchange_async || !RTL->synchronize) {
    assert(RTL->data_exchange && "RTL->data_exchange is nullptr");
    return RTL->data_exchange(RTLDeviceID, SrcPtr, DstDev.RTLDeviceID, DstPtr,
                              Size);
  }
  return RTL->data_exchange_async(RTLDeviceID, SrcPtr, DstDev.RTLDeviceID,
                                  DstPtr, Size, AsyncInfo);
}

int32_t DeviceTy::notifyDataMapped(void *HstPtr, int64_t Size) {
  if (!RTL->data_notify_mapped)
    return OFFLOAD_SUCCESS;

  DP("Notifying about new mapping: HstPtr=" DPxMOD ", Size=%" PRId64 "\n",
     DPxPTR(HstPtr), Size);

  if (RTL->data_notify_mapped(RTLDeviceID, HstPtr, Size)) {
    REPORT("Notifiying about data mapping failed.\n");
    return OFFLOAD_FAIL;
  }
  return OFFLOAD_SUCCESS;
}

int32_t DeviceTy::notifyDataUnmapped(void *HstPtr) {
  if (!RTL->data_notify_unmapped)
    return OFFLOAD_SUCCESS;

  DP("Notifying about an unmapping: HstPtr=" DPxMOD "\n", DPxPTR(HstPtr));

  if (RTL->data_notify_unmapped(RTLDeviceID, HstPtr)) {
    REPORT("Notifiying about data unmapping failed.\n");
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
  if (!RTL->print_device_info)
    return false;
  RTL->print_device_info(RTLDeviceID);
  return true;
}

// Whether data can be copied to DstDevice directly
bool DeviceTy::isDataExchangable(const DeviceTy &DstDevice) {
  if (RTL != DstDevice.RTL || !RTL->is_data_exchangable)
    return false;

  if (RTL->is_data_exchangable(RTLDeviceID, DstDevice.RTLDeviceID))
    return (RTL->data_exchange != nullptr) ||
           (RTL->data_exchange_async != nullptr);

  return false;
}

int32_t DeviceTy::synchronize(AsyncInfoTy &AsyncInfo) {
  if (RTL->synchronize)
    return RTL->synchronize(RTLDeviceID, AsyncInfo);
  return OFFLOAD_SUCCESS;
}

int32_t DeviceTy::queryAsync(AsyncInfoTy &AsyncInfo) {
  if (RTL->query_async)
    return RTL->query_async(RTLDeviceID, AsyncInfo);

  return synchronize(AsyncInfo);
}

int32_t DeviceTy::createEvent(void **Event) {
  if (RTL->create_event)
    return RTL->create_event(RTLDeviceID, Event);

  return OFFLOAD_SUCCESS;
}

int32_t DeviceTy::recordEvent(void *Event, AsyncInfoTy &AsyncInfo) {
  if (RTL->record_event)
    return RTL->record_event(RTLDeviceID, Event, AsyncInfo);

  return OFFLOAD_SUCCESS;
}

int32_t DeviceTy::waitEvent(void *Event, AsyncInfoTy &AsyncInfo) {
  if (RTL->wait_event)
    return RTL->wait_event(RTLDeviceID, Event, AsyncInfo);

  return OFFLOAD_SUCCESS;
}

int32_t DeviceTy::syncEvent(void *Event) {
  if (RTL->sync_event)
    return RTL->sync_event(RTLDeviceID, Event);

  return OFFLOAD_SUCCESS;
}

int32_t DeviceTy::destroyEvent(void *Event) {
  if (RTL->create_event)
    return RTL->destroy_event(RTLDeviceID, Event);

  return OFFLOAD_SUCCESS;
}

void DeviceTy::addOffloadEntry(const OffloadEntryTy &Entry) {
  std::lock_guard<decltype(PendingGlobalsMtx)> Lock(PendingGlobalsMtx);
  DeviceOffloadEntries.getExclusiveAccessor()->insert({Entry.getName(), Entry});
  if (Entry.isGlobal())
    return;

  if (Entry.isCTor()) {
    DP("Adding ctor " DPxMOD " to the pending list.\n",
       DPxPTR(Entry.getAddress()));
    MESSAGE("WARNING: Calling deprecated constructor for entry %s will be "
            "removed in a future release \n",
            Entry.getNameAsCStr());
    PendingCtorsDtors[Entry.getBinaryDescription()].PendingCtors.push_back(
        Entry.getAddress());
  } else if (Entry.isDTor()) {
    // Dtors are pushed in reverse order so they are executed from end
    // to beginning when unregistering the library!
    DP("Adding dtor " DPxMOD " to the pending list.\n",
       DPxPTR(Entry.getAddress()));
    MESSAGE("WARNING: Calling deprecated destructor for entry %s will be "
            "removed in a future release \n",
            Entry.getNameAsCStr());
    PendingCtorsDtors[Entry.getBinaryDescription()].PendingDtors.push_front(
        Entry.getAddress());
  }

  if (Entry.isLink()) {
    MESSAGE(
        "WARNING: The \"link\" attribute is not yet supported for entry: %s!\n",
        Entry.getNameAsCStr());
  }
}

void DeviceTy::dumpOffloadEntries() {
  fprintf(stderr, "Device %i offload entries:\n", DeviceID);
  for (auto &It : *DeviceOffloadEntries.getExclusiveAccessor()) {
    const char *Kind = "kernel";
    if (It.second.isCTor())
      Kind = "constructor";
    else if (It.second.isDTor())
      Kind = "destructor";
    else if (It.second.isLink())
      Kind = "link";
    else if (It.second.isGlobal())
      Kind = "global var.";
    fprintf(stderr, "  %11s: %s\n", Kind, It.second.getNameAsCStr());
  }
}

bool DeviceTy::useAutoZeroCopy() {
  if (RTL->use_auto_zero_copy)
    return RTL->use_auto_zero_copy(RTLDeviceID);
  return false;
}
