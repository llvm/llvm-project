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
#include "omptarget.h"
#include "private.h"
#include "rtl.h"

#include <cassert>
#include <climits>
#include <cstdio>
#include <string>

int HostDataToTargetTy::addEventIfNecessary(DeviceTy &Device,
                                            AsyncInfoTy &AsyncInfo) const {
  // First, check if the user disabled atomic map transfer/malloc/dealloc.
  if (!PM->UseEventsForAtomicTransfers)
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

DeviceTy::DeviceTy(RTLInfoTy *RTL)
    : DeviceID(-1), RTL(RTL), RTLDeviceID(-1), IsInit(false), InitFlag(),
      HasPendingGlobals(false), HostDataToTargetMap(), PendingCtorsDtors(),
      ShadowPtrMap(), DataMapMtx(), PendingGlobalsMtx(), ShadowMtx() {}

DeviceTy::~DeviceTy() {
  if (DeviceID == -1 || !(getInfoLevel() & OMP_INFOTYPE_DUMP_TABLE))
    return;

  ident_t loc = {0, 0, 0, 0, ";libomptarget;libomptarget;0;0;;"};
  dumpTargetPointerMappings(&loc, *this);
}

int DeviceTy::associatePtr(void *HstPtrBegin, void *TgtPtrBegin, int64_t Size) {
  std::lock_guard<decltype(DataMapMtx)> LG(DataMapMtx);

  // Check if entry exists
  auto search = HostDataToTargetMap.find(HstPtrBeginTy{(uintptr_t)HstPtrBegin});
  if (search != HostDataToTargetMap.end()) {
    // Mapping already exists
    bool isValid = search->HstPtrEnd == (uintptr_t)HstPtrBegin + Size &&
                   search->TgtPtrBegin == (uintptr_t)TgtPtrBegin;
    if (isValid) {
      DP("Attempt to re-associate the same device ptr+offset with the same "
         "host ptr, nothing to do\n");
      return OFFLOAD_SUCCESS;
    } else {
      REPORT("Not allowed to re-associate a different device ptr+offset with "
             "the same host ptr\n");
      return OFFLOAD_FAIL;
    }
  }

  // Mapping does not exist, allocate it with refCount=INF
  const HostDataToTargetTy &newEntry =
      *HostDataToTargetMap
           .emplace(
               /*HstPtrBase=*/(uintptr_t)HstPtrBegin,
               /*HstPtrBegin=*/(uintptr_t)HstPtrBegin,
               /*HstPtrEnd=*/(uintptr_t)HstPtrBegin + Size,
               /*TgtPtrBegin=*/(uintptr_t)TgtPtrBegin,
               /*UseHoldRefCount=*/false, /*Name=*/nullptr,
               /*IsRefCountINF=*/true)
           .first;
  DP("Creating new map entry: HstBase=" DPxMOD ", HstBegin=" DPxMOD
     ", HstEnd=" DPxMOD ", TgtBegin=" DPxMOD ", DynRefCount=%s, "
     "HoldRefCount=%s\n",
     DPxPTR(newEntry.HstPtrBase), DPxPTR(newEntry.HstPtrBegin),
     DPxPTR(newEntry.HstPtrEnd), DPxPTR(newEntry.TgtPtrBegin),
     newEntry.dynRefCountToStr().c_str(), newEntry.holdRefCountToStr().c_str());
  (void)newEntry;

  return OFFLOAD_SUCCESS;
}

int DeviceTy::disassociatePtr(void *HstPtrBegin) {
  std::lock_guard<decltype(DataMapMtx)> LG(DataMapMtx);

  auto search = HostDataToTargetMap.find(HstPtrBeginTy{(uintptr_t)HstPtrBegin});
  if (search != HostDataToTargetMap.end()) {
    // Mapping exists
    if (search->getHoldRefCount()) {
      // This is based on OpenACC 3.1, sec 3.2.33 "acc_unmap_data", L3656-3657:
      // "It is an error to call acc_unmap_data if the structured reference
      // count for the pointer is not zero."
      REPORT("Trying to disassociate a pointer with a non-zero hold reference "
             "count\n");
    } else if (search->isDynRefCountInf()) {
      DP("Association found, removing it\n");
      void *Event = search->getEvent();
      if (Event)
        destroyEvent(Event);
      HostDataToTargetMap.erase(search);
      return OFFLOAD_SUCCESS;
    } else {
      REPORT("Trying to disassociate a pointer which was not mapped via "
             "omp_target_associate_ptr\n");
    }
  } else {
    REPORT("Association not found\n");
  }

  // Mapping not found
  return OFFLOAD_FAIL;
}

LookupResult DeviceTy::lookupMapping(void *HstPtrBegin, int64_t Size) {
  uintptr_t hp = (uintptr_t)HstPtrBegin;
  LookupResult lr;

  DP("Looking up mapping(HstPtrBegin=" DPxMOD ", Size=%" PRId64 ")...\n",
     DPxPTR(hp), Size);

  if (HostDataToTargetMap.empty())
    return lr;

  auto upper = HostDataToTargetMap.upper_bound(hp);
  // check the left bin
  if (upper != HostDataToTargetMap.begin()) {
    lr.Entry = std::prev(upper);
    auto &HT = *lr.Entry;
    // Is it contained?
    lr.Flags.IsContained = hp >= HT.HstPtrBegin && hp < HT.HstPtrEnd &&
                           (hp + Size) <= HT.HstPtrEnd;
    // Does it extend beyond the mapped region?
    lr.Flags.ExtendsAfter = hp < HT.HstPtrEnd && (hp + Size) > HT.HstPtrEnd;
  }

  // check the right bin
  if (!(lr.Flags.IsContained || lr.Flags.ExtendsAfter) &&
      upper != HostDataToTargetMap.end()) {
    lr.Entry = upper;
    auto &HT = *lr.Entry;
    // Does it extend into an already mapped region?
    lr.Flags.ExtendsBefore =
        hp < HT.HstPtrBegin && (hp + Size) > HT.HstPtrBegin;
    // Does it extend beyond the mapped region?
    lr.Flags.ExtendsAfter = hp < HT.HstPtrEnd && (hp + Size) > HT.HstPtrEnd;
  }

  if (lr.Flags.ExtendsBefore) {
    DP("WARNING: Pointer is not mapped but section extends into already "
       "mapped data\n");
  }
  if (lr.Flags.ExtendsAfter) {
    DP("WARNING: Pointer is already mapped but section extends beyond mapped "
       "region\n");
  }

  return lr;
}

TargetPointerResultTy DeviceTy::getTargetPointer(
    void *HstPtrBegin, void *HstPtrBase, int64_t Size,
    map_var_info_t HstPtrName, bool HasFlagTo, bool HasFlagAlways,
    bool IsImplicit, bool UpdateRefCount, bool HasCloseModifier,
    bool HasPresentModifier, bool HasHoldModifier, AsyncInfoTy &AsyncInfo) {
  void *TargetPointer = nullptr;
  bool IsHostPtr = false;
  bool IsNew = false;

  DataMapMtx.lock();

  LookupResult LR = lookupMapping(HstPtrBegin, Size);
  auto Entry = LR.Entry;

  // Check if the pointer is contained.
  // If a variable is mapped to the device manually by the user - which would
  // lead to the IsContained flag to be true - then we must ensure that the
  // device address is returned even under unified memory conditions.
  if (LR.Flags.IsContained ||
      ((LR.Flags.ExtendsBefore || LR.Flags.ExtendsAfter) && IsImplicit)) {
    auto &HT = *LR.Entry;
    const char *RefCountAction;
    assert(HT.getTotalRefCount() > 0 && "expected existing RefCount > 0");
    if (UpdateRefCount) {
      // After this, RefCount > 1.
      HT.incRefCount(HasHoldModifier);
      RefCountAction = " (incremented)";
    } else {
      // It might have been allocated with the parent, but it's still new.
      IsNew = HT.getTotalRefCount() == 1;
      RefCountAction = " (update suppressed)";
    }
    const char *DynRefCountAction = HasHoldModifier ? "" : RefCountAction;
    const char *HoldRefCountAction = HasHoldModifier ? RefCountAction : "";
    uintptr_t Ptr = HT.TgtPtrBegin + ((uintptr_t)HstPtrBegin - HT.HstPtrBegin);
    INFO(OMP_INFOTYPE_MAPPING_EXISTS, DeviceID,
         "Mapping exists%s with HstPtrBegin=" DPxMOD ", TgtPtrBegin=" DPxMOD
         ", Size=%" PRId64 ", DynRefCount=%s%s, HoldRefCount=%s%s, Name=%s\n",
         (IsImplicit ? " (implicit)" : ""), DPxPTR(HstPtrBegin), DPxPTR(Ptr),
         Size, HT.dynRefCountToStr().c_str(), DynRefCountAction,
         HT.holdRefCountToStr().c_str(), HoldRefCountAction,
         (HstPtrName) ? getNameFromMapping(HstPtrName).c_str() : "unknown");
    TargetPointer = (void *)Ptr;
  } else if ((LR.Flags.ExtendsBefore || LR.Flags.ExtendsAfter) && !IsImplicit) {
    // Explicit extension of mapped data - not allowed.
    MESSAGE("explicit extension not allowed: host address specified is " DPxMOD
            " (%" PRId64
            " bytes), but device allocation maps to host at " DPxMOD
            " (%" PRId64 " bytes)",
            DPxPTR(HstPtrBegin), Size, DPxPTR(Entry->HstPtrBegin),
            Entry->HstPtrEnd - Entry->HstPtrBegin);
    if (HasPresentModifier)
      MESSAGE("device mapping required by 'present' map type modifier does not "
              "exist for host address " DPxMOD " (%" PRId64 " bytes)",
              DPxPTR(HstPtrBegin), Size);
  } else if (PM->RTLs.RequiresFlags & OMP_REQ_UNIFIED_SHARED_MEMORY &&
             !HasCloseModifier) {
    // If unified shared memory is active, implicitly mapped variables that are
    // not privatized use host address. Any explicitly mapped variables also use
    // host address where correctness is not impeded. In all other cases maps
    // are respected.
    // In addition to the mapping rules above, the close map modifier forces the
    // mapping of the variable to the device.
    if (Size) {
      DP("Return HstPtrBegin " DPxMOD " Size=%" PRId64 " for unified shared "
         "memory\n",
         DPxPTR((uintptr_t)HstPtrBegin), Size);
      IsHostPtr = true;
      TargetPointer = HstPtrBegin;
    }
  } else if (HasPresentModifier) {
    DP("Mapping required by 'present' map type modifier does not exist for "
       "HstPtrBegin=" DPxMOD ", Size=%" PRId64 "\n",
       DPxPTR(HstPtrBegin), Size);
    MESSAGE("device mapping required by 'present' map type modifier does not "
            "exist for host address " DPxMOD " (%" PRId64 " bytes)",
            DPxPTR(HstPtrBegin), Size);
  } else if (Size) {
    // If it is not contained and Size > 0, we should create a new entry for it.
    IsNew = true;
    uintptr_t Ptr = (uintptr_t)allocData(Size, HstPtrBegin);
    Entry = HostDataToTargetMap
                .emplace((uintptr_t)HstPtrBase, (uintptr_t)HstPtrBegin,
                         (uintptr_t)HstPtrBegin + Size, Ptr, HasHoldModifier,
                         HstPtrName)
                .first;
    INFO(OMP_INFOTYPE_MAPPING_CHANGED, DeviceID,
         "Creating new map entry with "
         "HstPtrBegin=" DPxMOD ", TgtPtrBegin=" DPxMOD ", Size=%ld, "
         "DynRefCount=%s, HoldRefCount=%s, Name=%s\n",
         DPxPTR(HstPtrBegin), DPxPTR(Ptr), Size,
         Entry->dynRefCountToStr().c_str(), Entry->holdRefCountToStr().c_str(),
         (HstPtrName) ? getNameFromMapping(HstPtrName).c_str() : "unknown");
    TargetPointer = (void *)Ptr;
  }

  // If the target pointer is valid, and we need to transfer data, issue the
  // data transfer.
  if (TargetPointer && !IsHostPtr && HasFlagTo && (IsNew || HasFlagAlways)) {
    // Lock the entry before releasing the mapping table lock such that another
    // thread that could issue data movement will get the right result.
    std::lock_guard<decltype(*Entry)> LG(*Entry);
    // Release the mapping table lock right after the entry is locked.
    DataMapMtx.unlock();

    DP("Moving %" PRId64 " bytes (hst:" DPxMOD ") -> (tgt:" DPxMOD ")\n", Size,
       DPxPTR(HstPtrBegin), DPxPTR(TargetPointer));

    int Ret = submitData(TargetPointer, HstPtrBegin, Size, AsyncInfo);
    if (Ret != OFFLOAD_SUCCESS) {
      REPORT("Copying data to device failed.\n");
      // We will also return nullptr if the data movement fails because that
      // pointer points to a corrupted memory region so it doesn't make any
      // sense to continue to use it.
      TargetPointer = nullptr;
    } else if (Entry->addEventIfNecessary(*this, AsyncInfo) != OFFLOAD_SUCCESS)
      return {{false /* IsNewEntry */, false /* IsHostPointer */},
              {} /* MapTableEntry */,
              nullptr /* TargetPointer */};
  } else {
    // Release the mapping table lock directly.
    DataMapMtx.unlock();
    // If not a host pointer and no present modifier, we need to wait for the
    // event if it exists.
    // Note: Entry might be nullptr because of zero length array section.
    if (Entry != HostDataToTargetListTy::iterator() && !IsHostPtr &&
        !HasPresentModifier) {
      std::lock_guard<decltype(*Entry)> LG(*Entry);
      void *Event = Entry->getEvent();
      if (Event) {
        int Ret = waitEvent(Event, AsyncInfo);
        if (Ret != OFFLOAD_SUCCESS) {
          // If it fails to wait for the event, we need to return nullptr in
          // case of any data race.
          REPORT("Failed to wait for event " DPxMOD ".\n", DPxPTR(Event));
          return {{false /* IsNewEntry */, false /* IsHostPointer */},
                  {} /* MapTableEntry */,
                  nullptr /* TargetPointer */};
        }
      }
    }
  }

  return {{IsNew, IsHostPtr}, Entry, TargetPointer};
}

// Used by targetDataBegin, targetDataEnd, targetDataUpdate and target.
// Return the target pointer begin (where the data will be moved).
// Decrement the reference counter if called from targetDataEnd.
TargetPointerResultTy
DeviceTy::getTgtPtrBegin(void *HstPtrBegin, int64_t Size, bool &IsLast,
                         bool UpdateRefCount, bool UseHoldRefCount,
                         bool &IsHostPtr, bool MustContain, bool ForceDelete) {
  void *TargetPointer = NULL;
  bool IsNew = false;
  IsHostPtr = false;
  IsLast = false;
  std::lock_guard<decltype(DataMapMtx)> LG(DataMapMtx);
  LookupResult lr = lookupMapping(HstPtrBegin, Size);

  if (lr.Flags.IsContained ||
      (!MustContain && (lr.Flags.ExtendsBefore || lr.Flags.ExtendsAfter))) {
    auto &HT = *lr.Entry;
    // We do not zero the total reference count here.  deallocTgtPtr does that
    // atomically with removing the mapping.  Otherwise, before this thread
    // removed the mapping in deallocTgtPtr, another thread could retrieve the
    // mapping, increment and decrement back to zero, and then both threads
    // would try to remove the mapping, resulting in a double free.
    IsLast = HT.decShouldRemove(UseHoldRefCount, ForceDelete);
    const char *RefCountAction;
    if (!UpdateRefCount) {
      RefCountAction = " (update suppressed)";
    } else if (ForceDelete) {
      HT.resetRefCount(UseHoldRefCount);
      assert(IsLast == HT.decShouldRemove(UseHoldRefCount) &&
             "expected correct IsLast prediction for reset");
      if (IsLast)
        RefCountAction = " (reset, deferred final decrement)";
      else {
        HT.decRefCount(UseHoldRefCount);
        RefCountAction = " (reset)";
      }
    } else if (IsLast) {
      RefCountAction = " (deferred final decrement)";
    } else {
      HT.decRefCount(UseHoldRefCount);
      RefCountAction = " (decremented)";
    }
    const char *DynRefCountAction = UseHoldRefCount ? "" : RefCountAction;
    const char *HoldRefCountAction = UseHoldRefCount ? RefCountAction : "";
    uintptr_t tp = HT.TgtPtrBegin + ((uintptr_t)HstPtrBegin - HT.HstPtrBegin);
    INFO(OMP_INFOTYPE_MAPPING_EXISTS, DeviceID,
         "Mapping exists with HstPtrBegin=" DPxMOD ", TgtPtrBegin=" DPxMOD ", "
         "Size=%" PRId64 ", DynRefCount=%s%s, HoldRefCount=%s%s\n",
         DPxPTR(HstPtrBegin), DPxPTR(tp), Size, HT.dynRefCountToStr().c_str(),
         DynRefCountAction, HT.holdRefCountToStr().c_str(), HoldRefCountAction);
    TargetPointer = (void *)tp;
  } else if (PM->RTLs.RequiresFlags & OMP_REQ_UNIFIED_SHARED_MEMORY) {
    // If the value isn't found in the mapping and unified shared memory
    // is on then it means we have stumbled upon a value which we need to
    // use directly from the host.
    DP("Get HstPtrBegin " DPxMOD " Size=%" PRId64 " for unified shared "
       "memory\n",
       DPxPTR((uintptr_t)HstPtrBegin), Size);
    IsHostPtr = true;
    TargetPointer = HstPtrBegin;
  }

  return {{IsNew, IsHostPtr}, lr.Entry, TargetPointer};
}

// Return the target pointer begin (where the data will be moved).
// Lock-free version called when loading global symbols from the fat binary.
void *DeviceTy::getTgtPtrBegin(void *HstPtrBegin, int64_t Size) {
  uintptr_t hp = (uintptr_t)HstPtrBegin;
  LookupResult lr = lookupMapping(HstPtrBegin, Size);
  if (lr.Flags.IsContained || lr.Flags.ExtendsBefore || lr.Flags.ExtendsAfter) {
    auto &HT = *lr.Entry;
    uintptr_t tp = HT.TgtPtrBegin + (hp - HT.HstPtrBegin);
    return (void *)tp;
  }

  return NULL;
}

int DeviceTy::deallocTgtPtr(void *HstPtrBegin, int64_t Size,
                            bool HasHoldModifier) {
  std::lock_guard<decltype(DataMapMtx)> LG(DataMapMtx);

  // Check if the pointer is contained in any sub-nodes.
  int Ret = OFFLOAD_SUCCESS;
  LookupResult lr = lookupMapping(HstPtrBegin, Size);
  if (lr.Flags.IsContained || lr.Flags.ExtendsBefore || lr.Flags.ExtendsAfter) {
    auto &HT = *lr.Entry;
    if (HT.decRefCount(HasHoldModifier) == 0) {
      DP("Deleting tgt data " DPxMOD " of size %" PRId64 "\n",
         DPxPTR(HT.TgtPtrBegin), Size);
      deleteData((void *)HT.TgtPtrBegin);
      INFO(OMP_INFOTYPE_MAPPING_CHANGED, DeviceID,
           "Removing map entry with HstPtrBegin=" DPxMOD ", TgtPtrBegin=" DPxMOD
           ", Size=%" PRId64 ", Name=%s\n",
           DPxPTR(HT.HstPtrBegin), DPxPTR(HT.TgtPtrBegin), Size,
           (HT.HstPtrName) ? getNameFromMapping(HT.HstPtrName).c_str()
                           : "unknown");
      void *Event = lr.Entry->getEvent();
      HostDataToTargetMap.erase(lr.Entry);
      if (Event && destroyEvent(Event) != OFFLOAD_SUCCESS) {
        REPORT("Failed to destroy event " DPxMOD "\n", DPxPTR(Event));
        Ret = OFFLOAD_FAIL;
      }
    }
  } else {
    REPORT("Section to delete (hst addr " DPxMOD ") does not exist in the"
           " allocated memory\n",
           DPxPTR(HstPtrBegin));
    Ret = OFFLOAD_FAIL;
  }

  return Ret;
}

/// Init device, should not be called directly.
void DeviceTy::init() {
  // Make call to init_requires if it exists for this plugin.
  if (RTL->init_requires)
    RTL->init_requires(PM->RTLs.RequiresFlags);
  int32_t Ret = RTL->init_device(RTLDeviceID);
  if (Ret != OFFLOAD_SUCCESS)
    return;

  IsInit = true;
}

/// Thread-safe method to initialize the device only once.
int32_t DeviceTy::initOnce() {
  std::call_once(InitFlag, &DeviceTy::init, this);

  // At this point, if IsInit is true, then either this thread or some other
  // thread in the past successfully initialized the device, so we can return
  // OFFLOAD_SUCCESS. If this thread executed init() via call_once() and it
  // failed, return OFFLOAD_FAIL. If call_once did not invoke init(), it means
  // that some other thread already attempted to execute init() and if IsInit
  // is still false, return OFFLOAD_FAIL.
  if (IsInit)
    return OFFLOAD_SUCCESS;
  else
    return OFFLOAD_FAIL;
}

void DeviceTy::deinit() {
  if (RTL->deinit_device)
    RTL->deinit_device(RTLDeviceID);
}

// Load binary to device.
__tgt_target_table *DeviceTy::load_binary(void *Img) {
  std::lock_guard<decltype(RTL->Mtx)> LG(RTL->Mtx);
  __tgt_target_table *rc = RTL->load_binary(RTLDeviceID, Img);
  return rc;
}

void *DeviceTy::allocData(int64_t Size, void *HstPtr, int32_t Kind) {
  return RTL->data_alloc(RTLDeviceID, Size, HstPtr, Kind);
}

int32_t DeviceTy::deleteData(void *TgtPtrBegin) {
  return RTL->data_delete(RTLDeviceID, TgtPtrBegin);
}

// Submit data to device
int32_t DeviceTy::submitData(void *TgtPtrBegin, void *HstPtrBegin, int64_t Size,
                             AsyncInfoTy &AsyncInfo) {
  if (getInfoLevel() & OMP_INFOTYPE_DATA_TRANSFER) {
    LookupResult LR = lookupMapping(HstPtrBegin, Size);
    auto *HT = &*LR.Entry;

    INFO(OMP_INFOTYPE_DATA_TRANSFER, DeviceID,
         "Copying data from host to device, HstPtr=" DPxMOD ", TgtPtr=" DPxMOD
         ", Size=%" PRId64 ", Name=%s\n",
         DPxPTR(HstPtrBegin), DPxPTR(TgtPtrBegin), Size,
         (HT && HT->HstPtrName) ? getNameFromMapping(HT->HstPtrName).c_str()
                                : "unknown");
  }

  if (!AsyncInfo || !RTL->data_submit_async || !RTL->synchronize)
    return RTL->data_submit(RTLDeviceID, TgtPtrBegin, HstPtrBegin, Size);
  else
    return RTL->data_submit_async(RTLDeviceID, TgtPtrBegin, HstPtrBegin, Size,
                                  AsyncInfo);
}

// Retrieve data from device
int32_t DeviceTy::retrieveData(void *HstPtrBegin, void *TgtPtrBegin,
                               int64_t Size, AsyncInfoTy &AsyncInfo) {
  if (getInfoLevel() & OMP_INFOTYPE_DATA_TRANSFER) {
    LookupResult LR = lookupMapping(HstPtrBegin, Size);
    auto *HT = &*LR.Entry;
    INFO(OMP_INFOTYPE_DATA_TRANSFER, DeviceID,
         "Copying data from device to host, TgtPtr=" DPxMOD ", HstPtr=" DPxMOD
         ", Size=%" PRId64 ", Name=%s\n",
         DPxPTR(TgtPtrBegin), DPxPTR(HstPtrBegin), Size,
         (HT && HT->HstPtrName) ? getNameFromMapping(HT->HstPtrName).c_str()
                                : "unknown");
  }

  if (!RTL->data_retrieve_async || !RTL->synchronize)
    return RTL->data_retrieve(RTLDeviceID, HstPtrBegin, TgtPtrBegin, Size);
  else
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
  } else
    return RTL->data_exchange_async(RTLDeviceID, SrcPtr, DstDev.RTLDeviceID,
                                    DstPtr, Size, AsyncInfo);
}

// Run region on device
int32_t DeviceTy::runRegion(void *TgtEntryPtr, void **TgtVarsPtr,
                            ptrdiff_t *TgtOffsets, int32_t TgtVarsSize,
                            AsyncInfoTy &AsyncInfo) {
  if (!RTL->run_region || !RTL->synchronize)
    return RTL->run_region(RTLDeviceID, TgtEntryPtr, TgtVarsPtr, TgtOffsets,
                           TgtVarsSize);
  else
    return RTL->run_region_async(RTLDeviceID, TgtEntryPtr, TgtVarsPtr,
                                 TgtOffsets, TgtVarsSize, AsyncInfo);
}

// Run region on device
bool DeviceTy::printDeviceInfo(int32_t RTLDevId) {
  if (!RTL->print_device_info)
    return false;
  RTL->print_device_info(RTLDevId);
  return true;
}

// Run team region on device.
int32_t DeviceTy::runTeamRegion(void *TgtEntryPtr, void **TgtVarsPtr,
                                ptrdiff_t *TgtOffsets, int32_t TgtVarsSize,
                                int32_t NumTeams, int32_t ThreadLimit,
                                uint64_t LoopTripCount,
                                AsyncInfoTy &AsyncInfo) {
  if (!RTL->run_team_region_async || !RTL->synchronize)
    return RTL->run_team_region(RTLDeviceID, TgtEntryPtr, TgtVarsPtr,
                                TgtOffsets, TgtVarsSize, NumTeams, ThreadLimit,
                                LoopTripCount);
  else
    return RTL->run_team_region_async(RTLDeviceID, TgtEntryPtr, TgtVarsPtr,
                                      TgtOffsets, TgtVarsSize, NumTeams,
                                      ThreadLimit, LoopTripCount, AsyncInfo);
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

/// Check whether a device has an associated RTL and initialize it if it's not
/// already initialized.
bool device_is_ready(int device_num) {
  DP("Checking whether device %d is ready.\n", device_num);
  // Devices.size() can only change while registering a new
  // library, so try to acquire the lock of RTLs' mutex.
  size_t DevicesSize;
  {
    std::lock_guard<decltype(PM->RTLsMtx)> LG(PM->RTLsMtx);
    DevicesSize = PM->Devices.size();
  }
  if (DevicesSize <= (size_t)device_num) {
    DP("Device ID  %d does not have a matching RTL\n", device_num);
    return false;
  }

  // Get device info
  DeviceTy &Device = *PM->Devices[device_num];

  DP("Is the device %d (local ID %d) initialized? %d\n", device_num,
     Device.RTLDeviceID, Device.IsInit);

  // Init the device if not done before
  if (!Device.IsInit && Device.initOnce() != OFFLOAD_SUCCESS) {
    DP("Failed to init device %d\n", device_num);
    return false;
  }

  DP("Device %d is ready to use.\n", device_num);

  return true;
}
