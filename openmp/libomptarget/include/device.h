//===----------- device.h - Target independent OpenMP target RTL ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Declarations for managing devices that are handled by RTL plugins.
//
//===----------------------------------------------------------------------===//

#ifndef _OMPTARGET_DEVICE_H
#define _OMPTARGET_DEVICE_H

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <list>
#include <map>
#include <mutex>
#include <set>

#include "ExclusiveAccess.h"
#include "omptarget.h"
#include "rtl.h"

#include "OpenMP/Mapping.h"

#include "llvm/ADT/SmallVector.h"

// Forward declarations.
struct RTLInfoTy;
struct __tgt_bin_desc;
struct __tgt_target_table;

// enum for OMP_TARGET_OFFLOAD; keep in sync with kmp.h definition
enum kmp_target_offload_kind {
  tgt_disabled = 0,
  tgt_default = 1,
  tgt_mandatory = 2
};
typedef enum kmp_target_offload_kind kmp_target_offload_kind_t;

//
struct PendingCtorDtorListsTy {
  std::list<void *> PendingCtors;
  std::list<void *> PendingDtors;
};
typedef std::map<__tgt_bin_desc *, PendingCtorDtorListsTy>
    PendingCtorsDtorsPerLibrary;

struct DeviceTy {
  int32_t DeviceID;
  RTLInfoTy *RTL;
  int32_t RTLDeviceID;
  /// The physical number of processors that may concurrently execute a team
  /// For cuda, this is number of SMs, for amdgcn, this is number of CUs.
  /// This field is used by ompx_get_team_procs(devid).
  int32_t TeamProcs;

  bool IsInit;
  std::once_flag InitFlag;
  bool HasPendingGlobals;

  /// Host data to device map type with a wrapper key indirection that allows
  /// concurrent modification of the entries without invalidating the underlying
  /// entries.
  using HostDataToTargetListTy =
      std::set<HostDataToTargetMapKeyTy, std::less<>>;

  /// The HDTTMap is a protected object that can only be accessed by one thread
  /// at a time.
  ProtectedObj<HostDataToTargetListTy> HostDataToTargetMap;

  /// The type used to access the HDTT map.
  using HDTTMapAccessorTy = decltype(HostDataToTargetMap)::AccessorTy;

  PendingCtorsDtorsPerLibrary PendingCtorsDtors;

  std::mutex PendingGlobalsMtx;

  /// Flag to force synchronous data transfers
  /// Controlled via environment flag OMPX_FORCE_SYNC_REGIONS
  bool ForceSynchronousTargetRegions;

  DeviceTy(RTLInfoTy *RTL);
  // DeviceTy is not copyable
  DeviceTy(const DeviceTy &D) = delete;
  DeviceTy &operator=(const DeviceTy &D) = delete;

  ~DeviceTy();

  // Return true if data can be copied to DstDevice directly
  bool isDataExchangable(const DeviceTy &DstDevice);

  /// Lookup the mapping of \p HstPtrBegin in \p HDTTMap. The accessor ensures
  /// exclusive access to the HDTT map.
  LookupResult lookupMapping(HDTTMapAccessorTy &HDTTMap, void *HstPtrBegin,
                             int64_t Size,
                             HostDataToTargetTy *OwnedTPR = nullptr);

  /// Get the target pointer based on host pointer begin and base. If the
  /// mapping already exists, the target pointer will be returned directly. In
  /// addition, if required, the memory region pointed by \p HstPtrBegin of size
  /// \p Size will also be transferred to the device. If the mapping doesn't
  /// exist, and if unified shared memory is not enabled, a new mapping will be
  /// created and the data will also be transferred accordingly. nullptr will be
  /// returned because of any of following reasons:
  /// - Data allocation failed;
  /// - The user tried to do an illegal mapping;
  /// - Data transfer issue fails.
  TargetPointerResultTy getTargetPointer(
      HDTTMapAccessorTy &HDTTMap, void *HstPtrBegin, void *HstPtrBase,
      int64_t TgtPadding, int64_t Size, map_var_info_t HstPtrName,
      bool HasFlagTo, bool HasFlagAlways, bool IsImplicit, bool UpdateRefCount,
      bool HasCloseModifier, bool HasPresentModifier, bool HasHoldModifier,
      AsyncInfoTy &AsyncInfo, HostDataToTargetTy *OwnedTPR = nullptr,
      bool ReleaseHDTTMap = true);

  /// Return the target pointer for \p HstPtrBegin in \p HDTTMap. The accessor
  /// ensures exclusive access to the HDTT map.
  void *getTgtPtrBegin(HDTTMapAccessorTy &HDTTMap, void *HstPtrBegin,
                       int64_t Size);

  /// Return the target pointer begin (where the data will be moved).
  /// Used by targetDataBegin, targetDataEnd, targetDataUpdate and target.
  /// - \p UpdateRefCount and \p UseHoldRefCount controls which and if the entry
  /// reference counters will be decremented.
  /// - \p MustContain enforces that the query must not extend beyond an already
  /// mapped entry to be valid.
  /// - \p ForceDelete deletes the entry regardless of its reference counting
  /// (unless it is infinite).
  /// - \p FromDataEnd tracks the number of threads referencing the entry at
  /// targetDataEnd for delayed deletion purpose.
  [[nodiscard]] TargetPointerResultTy
  getTgtPtrBegin(void *HstPtrBegin, int64_t Size, bool UpdateRefCount,
                 bool UseHoldRefCount, bool MustContain = false,
                 bool ForceDelete = false, bool FromDataEnd = false);

  /// Remove the \p Entry from the data map. Expect the entry's total reference
  /// count to be zero and the caller thread to be the last one using it. \p
  /// HDTTMap ensure the caller holds exclusive access and can modify the map.
  /// Return \c OFFLOAD_SUCCESS if the map entry existed, and return \c
  /// OFFLOAD_FAIL if not. It is the caller's responsibility to skip calling
  /// this function if the map entry is not expected to exist because \p
  /// HstPtrBegin uses shared memory.
  [[nodiscard]] int eraseMapEntry(HDTTMapAccessorTy &HDTTMap,
                                  HostDataToTargetTy *Entry, int64_t Size);

  /// Deallocate the \p Entry from the device memory and delete it. Return \c
  /// OFFLOAD_SUCCESS if the deallocation operations executed successfully, and
  /// return \c OFFLOAD_FAIL otherwise.
  [[nodiscard]] int deallocTgtPtrAndEntry(HostDataToTargetTy *Entry,
                                          int64_t Size);

  int associatePtr(void *HstPtrBegin, void *TgtPtrBegin, int64_t Size);
  int disassociatePtr(void *HstPtrBegin);

  // calls to RTL
  int32_t initOnce();
  __tgt_target_table *loadBinary(void *Img);

  // device memory allocation/deallocation routines
  /// Allocates \p Size bytes on the device, host or shared memory space
  /// (depending on \p Kind) and returns the address/nullptr when
  /// succeeds/fails. \p HstPtr is an address of the host data which the
  /// allocated target data will be associated with. If it is unknown, the
  /// default value of \p HstPtr is nullptr. Note: this function doesn't do
  /// pointer association. Actually, all the __tgt_rtl_data_alloc
  /// implementations ignore \p HstPtr. \p Kind dictates what allocator should
  /// be used (host, shared, device).
  void *allocData(int64_t Size, void *HstPtr = nullptr,
                  int32_t Kind = TARGET_ALLOC_DEFAULT);
  /// Deallocates memory which \p TgtPtrBegin points at and returns
  /// OFFLOAD_SUCCESS/OFFLOAD_FAIL when succeeds/fails. p Kind dictates what
  /// allocator should be used (host, shared, device).
  int32_t deleteData(void *TgtPtrBegin, int32_t Kind = TARGET_ALLOC_DEFAULT);

  // Data transfer. When AsyncInfo is nullptr, the transfer will be
  // synchronous.
  // Copy data from host to device
  int32_t submitData(void *TgtPtrBegin, void *HstPtrBegin, int64_t Size,
                     AsyncInfoTy &AsyncInfo,
                     HostDataToTargetTy *Entry = nullptr);
  // Copy data from device back to host
  int32_t retrieveData(void *HstPtrBegin, void *TgtPtrBegin, int64_t Size,
                       AsyncInfoTy &AsyncInfo,
                       HostDataToTargetTy *Entry = nullptr);
  // Copy data from current device to destination device directly
  int32_t dataExchange(void *SrcPtr, DeviceTy &DstDev, void *DstPtr,
                       int64_t Size, AsyncInfoTy &AsyncInfo);

  /// Notify the plugin about a new mapping starting at the host address
  /// \p HstPtr and \p Size bytes.
  int32_t notifyDataMapped(void *HstPtr, int64_t Size);

  /// Notify the plugin about an existing mapping being unmapped starting at
  /// the host address \p HstPtr.
  int32_t notifyDataUnmapped(void *HstPtr);

  // Launch the kernel identified by \p TgtEntryPtr with the given arguments.
  int32_t launchKernel(void *TgtEntryPtr, void **TgtVarsPtr,
                       ptrdiff_t *TgtOffsets, const KernelArgsTy &KernelArgs,
                       AsyncInfoTy &AsyncInfo);

  /// Synchronize device/queue/event based on \p AsyncInfo and return
  /// OFFLOAD_SUCCESS/OFFLOAD_FAIL when succeeds/fails.
  int32_t synchronize(AsyncInfoTy &AsyncInfo);

  /// Query for device/queue/event based completion on \p AsyncInfo in a
  /// non-blocking manner and return OFFLOAD_SUCCESS/OFFLOAD_FAIL when
  /// succeeds/fails. Must be called multiple times until AsyncInfo is
  /// completed and AsyncInfo.isDone() returns true.
  int32_t queryAsync(AsyncInfoTy &AsyncInfo);

  /// Calls the corresponding print in the \p RTLDEVID
  /// device RTL to obtain the information of the specific device.
  bool printDeviceInfo(int32_t RTLDevID);

  /// Event related interfaces.
  /// {
  /// Create an event.
  int32_t createEvent(void **Event);

  /// Record the event based on status in AsyncInfo->Queue at the moment the
  /// function is called.
  int32_t recordEvent(void *Event, AsyncInfoTy &AsyncInfo);

  /// Wait for an event. This function can be blocking or non-blocking,
  /// depending on the implmentation. It is expected to set a dependence on the
  /// event such that corresponding operations shall only start once the event
  /// is fulfilled.
  int32_t waitEvent(void *Event, AsyncInfoTy &AsyncInfo);

  /// Synchronize the event. It is expected to block the thread.
  int32_t syncEvent(void *Event);

  /// Destroy the event.
  int32_t destroyEvent(void *Event);

  void setTeamProcs(int32_t num_team_procs) { TeamProcs = num_team_procs; }
  int32_t getTeamProcs() { return TeamProcs; }
  /// }

private:
  // Call to RTL
  void init(); // To be called only via DeviceTy::initOnce()

  /// Deinitialize the device (and plugin).
  void deinit();
};

extern bool deviceIsReady(int DeviceNum);

/// Struct for the data required to handle plugins
struct PluginManager {
  PluginManager(bool UseEventsForAtomicTransfers)
      : UseEventsForAtomicTransfers(UseEventsForAtomicTransfers) {}

  /// RTLs identified on the host
  RTLsTy RTLs;

  /// Executable images and information extracted from the input images passed
  /// to the runtime.
  std::list<std::pair<__tgt_device_image, __tgt_image_info>> Images;

  /// Devices associated with RTLs
  std::vector<std::unique_ptr<DeviceTy>> Devices;
  std::mutex RTLsMtx; ///< For RTLs and Devices

  /// Translation table retreived from the binary
  HostEntriesBeginToTransTableTy HostEntriesBeginToTransTable;
  std::mutex TrlTblMtx; ///< For Translation Table
  /// Host offload entries in order of image registration
  std::vector<__tgt_offload_entry *> HostEntriesBeginRegistrationOrder;

  /// Map from ptrs on the host to an entry in the Translation Table
  HostPtrToTableMapTy HostPtrToTableMap;
  std::mutex TblMapMtx; ///< For HostPtrToTableMap

  // Store target policy (disabled, mandatory, default)
  kmp_target_offload_kind_t TargetOffloadPolicy = tgt_default;
  std::mutex TargetOffloadMtx; ///< For TargetOffloadPolicy

  /// Flag to indicate if we use events to ensure the atomicity of
  /// map clauses or not. Can be modified with an environment variable.
  const bool UseEventsForAtomicTransfers;

  // Work around for plugins that call dlopen on shared libraries that call
  // tgt_register_lib during their initialisation. Stash the pointers in a
  // vector until the plugins are all initialised and then register them.
  bool delayRegisterLib(__tgt_bin_desc *Desc) {
    if (RTLsLoaded)
      return false;
    DelayedBinDesc.push_back(Desc);
    return true;
  }

  void registerDelayedLibraries() {
    // Only called by libomptarget constructor
    RTLsLoaded = true;
    for (auto *Desc : DelayedBinDesc)
      __tgt_register_lib(Desc);
    DelayedBinDesc.clear();
  }

private:
  bool RTLsLoaded = false;
  llvm::SmallVector<__tgt_bin_desc *> DelayedBinDesc;
};

extern PluginManager *PM;

#endif
