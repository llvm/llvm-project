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
#include "OmptCallback.h"
#include "OmptInterface.h"
#include "omptarget.h"
#include "private.h"
#include "rtl.h"

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
      HasPendingGlobals(false), PendingCtorsDtors(), PendingGlobalsMtx() {}

DeviceTy::~DeviceTy() {
  if (DeviceID == -1 || !(getInfoLevel() & OMP_INFOTYPE_DUMP_TABLE))
    return;

  ident_t Loc = {0, 0, 0, 0, ";libomptarget;libomptarget;0;0;;"};
  dumpTargetPointerMappings(&Loc, *this);
}

int DeviceTy::associatePtr(void *HstPtrBegin, void *TgtPtrBegin, int64_t Size) {
  HDTTMapAccessorTy HDTTMap = HostDataToTargetMap.getExclusiveAccessor();

  // Check if entry exists
  auto It = HDTTMap->find(HstPtrBegin);
  if (It != HDTTMap->end()) {
    HostDataToTargetTy &HDTT = *It->HDTT;
    std::lock_guard<HostDataToTargetTy> LG(HDTT);
    // Mapping already exists
    bool IsValid = HDTT.HstPtrEnd == (uintptr_t)HstPtrBegin + Size &&
                   HDTT.TgtPtrBegin == (uintptr_t)TgtPtrBegin;
    if (IsValid) {
      DP("Attempt to re-associate the same device ptr+offset with the same "
         "host ptr, nothing to do\n");
      return OFFLOAD_SUCCESS;
    }
    REPORT("Not allowed to re-associate a different device ptr+offset with "
           "the same host ptr\n");
    return OFFLOAD_FAIL;
  }

  // Mapping does not exist, allocate it with refCount=INF
  const HostDataToTargetTy &NewEntry =
      *HDTTMap
           ->emplace(new HostDataToTargetTy(
               /*HstPtrBase=*/(uintptr_t)HstPtrBegin,
               /*HstPtrBegin=*/(uintptr_t)HstPtrBegin,
               /*HstPtrEnd=*/(uintptr_t)HstPtrBegin + Size,
               /*TgtAllocBegin=*/(uintptr_t)TgtPtrBegin,
               /*TgtPtrBegin=*/(uintptr_t)TgtPtrBegin,
               /*UseHoldRefCount=*/false, /*Name=*/nullptr,
               /*IsRefCountINF=*/true))
           .first->HDTT;
  DP("Creating new map entry: HstBase=" DPxMOD ", HstBegin=" DPxMOD
     ", HstEnd=" DPxMOD ", TgtBegin=" DPxMOD ", DynRefCount=%s, "
     "HoldRefCount=%s\n",
     DPxPTR(NewEntry.HstPtrBase), DPxPTR(NewEntry.HstPtrBegin),
     DPxPTR(NewEntry.HstPtrEnd), DPxPTR(NewEntry.TgtPtrBegin),
     NewEntry.dynRefCountToStr().c_str(), NewEntry.holdRefCountToStr().c_str());
  (void)NewEntry;

  // Notify the plugin about the new mapping.
  return notifyDataMapped(HstPtrBegin, Size);
}

int DeviceTy::disassociatePtr(void *HstPtrBegin) {
  HDTTMapAccessorTy HDTTMap = HostDataToTargetMap.getExclusiveAccessor();

  auto It = HDTTMap->find(HstPtrBegin);
  if (It == HDTTMap->end()) {
    REPORT("Association not found\n");
    return OFFLOAD_FAIL;
  }
  // Mapping exists
  HostDataToTargetTy &HDTT = *It->HDTT;
  std::lock_guard<HostDataToTargetTy> LG(HDTT);

  if (HDTT.getHoldRefCount()) {
    // This is based on OpenACC 3.1, sec 3.2.33 "acc_unmap_data", L3656-3657:
    // "It is an error to call acc_unmap_data if the structured reference
    // count for the pointer is not zero."
    REPORT("Trying to disassociate a pointer with a non-zero hold reference "
           "count\n");
    return OFFLOAD_FAIL;
  }

  if (HDTT.isDynRefCountInf()) {
    DP("Association found, removing it\n");
    void *Event = HDTT.getEvent();
    delete &HDTT;
    if (Event)
      destroyEvent(Event);
    HDTTMap->erase(It);
    return notifyDataUnmapped(HstPtrBegin);
  }

  REPORT("Trying to disassociate a pointer which was not mapped via "
         "omp_target_associate_ptr\n");
  return OFFLOAD_FAIL;
}

LookupResult DeviceTy::lookupMapping(HDTTMapAccessorTy &HDTTMap,
                                     void *HstPtrBegin, int64_t Size,
                                     HostDataToTargetTy *OwnedTPR) {

  uintptr_t HP = (uintptr_t)HstPtrBegin;
  LookupResult LR;

  DP("Looking up mapping(HstPtrBegin=" DPxMOD ", Size=%" PRId64 ")...\n",
     DPxPTR(HP), Size);

  if (HDTTMap->empty())
    return LR;

  auto Upper = HDTTMap->upper_bound(HP);

  if (Size == 0) {
    // specification v5.1 Pointer Initialization for Device Data Environments
    // upper_bound satisfies
    //   std::prev(upper)->HDTT.HstPtrBegin <= hp < upper->HDTT.HstPtrBegin
    if (Upper != HDTTMap->begin()) {
      LR.TPR.setEntry(std::prev(Upper)->HDTT, OwnedTPR);
      // the left side of extended address range is satisified.
      // hp >= LR.TPR.getEntry()->HstPtrBegin || hp >=
      // LR.TPR.getEntry()->HstPtrBase
      LR.Flags.IsContained = HP < LR.TPR.getEntry()->HstPtrEnd ||
                             HP < LR.TPR.getEntry()->HstPtrBase;
    }

    if (!LR.Flags.IsContained && Upper != HDTTMap->end()) {
      LR.TPR.setEntry(Upper->HDTT, OwnedTPR);
      // the right side of extended address range is satisified.
      // hp < LR.TPR.getEntry()->HstPtrEnd || hp < LR.TPR.getEntry()->HstPtrBase
      LR.Flags.IsContained = HP >= LR.TPR.getEntry()->HstPtrBase;
    }
  } else {
    // check the left bin
    if (Upper != HDTTMap->begin()) {
      LR.TPR.setEntry(std::prev(Upper)->HDTT, OwnedTPR);
      // Is it contained?
      LR.Flags.IsContained = HP >= LR.TPR.getEntry()->HstPtrBegin &&
                             HP < LR.TPR.getEntry()->HstPtrEnd &&
                             (HP + Size) <= LR.TPR.getEntry()->HstPtrEnd;
      // Does it extend beyond the mapped region?
      LR.Flags.ExtendsAfter = HP < LR.TPR.getEntry()->HstPtrEnd &&
                              (HP + Size) > LR.TPR.getEntry()->HstPtrEnd;
    }

    // check the right bin
    if (!(LR.Flags.IsContained || LR.Flags.ExtendsAfter) &&
        Upper != HDTTMap->end()) {
      LR.TPR.setEntry(Upper->HDTT, OwnedTPR);
      // Does it extend into an already mapped region?
      LR.Flags.ExtendsBefore = HP < LR.TPR.getEntry()->HstPtrBegin &&
                               (HP + Size) > LR.TPR.getEntry()->HstPtrBegin;
      // Does it extend beyond the mapped region?
      LR.Flags.ExtendsAfter = HP < LR.TPR.getEntry()->HstPtrEnd &&
                              (HP + Size) > LR.TPR.getEntry()->HstPtrEnd;
    }

    if (LR.Flags.ExtendsBefore) {
      DP("WARNING: Pointer is not mapped but section extends into already "
         "mapped data\n");
    }
    if (LR.Flags.ExtendsAfter) {
      DP("WARNING: Pointer is already mapped but section extends beyond mapped "
         "region\n");
    }
  }

  return LR;
}

TargetPointerResultTy DeviceTy::getTargetPointer(
    HDTTMapAccessorTy &HDTTMap, void *HstPtrBegin, void *HstPtrBase,
    int64_t TgtPadding, int64_t Size, map_var_info_t HstPtrName, bool HasFlagTo,
    bool HasFlagAlways, bool IsImplicit, bool UpdateRefCount,
    bool HasCloseModifier, bool HasPresentModifier, bool HasHoldModifier,
    AsyncInfoTy &AsyncInfo, HostDataToTargetTy *OwnedTPR, bool ReleaseHDTTMap) {

  LookupResult LR = lookupMapping(HDTTMap, HstPtrBegin, Size, OwnedTPR);
  LR.TPR.Flags.IsPresent = true;

  // Release the mapping table lock only after the entry is locked by
  // attaching it to TPR. Once TPR is destroyed it will release the lock
  // on entry. If it is returned the lock will move to the returned object.
  // If LR.Entry is already owned/locked we avoid trying to lock it again.

  // Check if the pointer is contained.
  // If a variable is mapped to the device manually by the user - which would
  // lead to the IsContained flag to be true - then we must ensure that the
  // device address is returned even under unified memory conditions.
  if (LR.Flags.IsContained ||
      ((LR.Flags.ExtendsBefore || LR.Flags.ExtendsAfter) && IsImplicit)) {
    const char *RefCountAction;
    if (UpdateRefCount) {
      // After this, reference count >= 1. If the reference count was 0 but the
      // entry was still there we can reuse the data on the device and avoid a
      // new submission.
      LR.TPR.getEntry()->incRefCount(HasHoldModifier);
      RefCountAction = " (incremented)";
    } else {
      // It might have been allocated with the parent, but it's still new.
      LR.TPR.Flags.IsNewEntry = LR.TPR.getEntry()->getTotalRefCount() == 1;
      RefCountAction = " (update suppressed)";
    }
    const char *DynRefCountAction = HasHoldModifier ? "" : RefCountAction;
    const char *HoldRefCountAction = HasHoldModifier ? RefCountAction : "";
    uintptr_t Ptr = LR.TPR.getEntry()->TgtPtrBegin +
                    ((uintptr_t)HstPtrBegin - LR.TPR.getEntry()->HstPtrBegin);
    INFO(OMP_INFOTYPE_MAPPING_EXISTS, DeviceID,
         "Mapping exists%s with HstPtrBegin=" DPxMOD ", TgtPtrBegin=" DPxMOD
         ", Size=%" PRId64 ", DynRefCount=%s%s, HoldRefCount=%s%s, Name=%s\n",
         (IsImplicit ? " (implicit)" : ""), DPxPTR(HstPtrBegin), DPxPTR(Ptr),
         Size, LR.TPR.getEntry()->dynRefCountToStr().c_str(), DynRefCountAction,
         LR.TPR.getEntry()->holdRefCountToStr().c_str(), HoldRefCountAction,
         (HstPtrName) ? getNameFromMapping(HstPtrName).c_str() : "unknown");
    LR.TPR.TargetPointer = (void *)Ptr;
  } else if ((LR.Flags.ExtendsBefore || LR.Flags.ExtendsAfter) && !IsImplicit) {
    // Explicit extension of mapped data - not allowed.
    MESSAGE("explicit extension not allowed: host address specified is " DPxMOD
            " (%" PRId64
            " bytes), but device allocation maps to host at " DPxMOD
            " (%" PRId64 " bytes)",
            DPxPTR(HstPtrBegin), Size, DPxPTR(LR.TPR.getEntry()->HstPtrBegin),
            LR.TPR.getEntry()->HstPtrEnd - LR.TPR.getEntry()->HstPtrBegin);
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
      LR.TPR.Flags.IsPresent = false;
      LR.TPR.Flags.IsHostPointer = true;
      LR.TPR.TargetPointer = HstPtrBegin;
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
    LR.TPR.Flags.IsNewEntry = true;
    uintptr_t TgtAllocBegin =
        (uintptr_t)allocData(TgtPadding + Size, HstPtrBegin);
    uintptr_t TgtPtrBegin = TgtAllocBegin + TgtPadding;
    // Release the mapping table lock only after the entry is locked by
    // attaching it to TPR.
    LR.TPR.setEntry(HDTTMap
                        ->emplace(new HostDataToTargetTy(
                            (uintptr_t)HstPtrBase, (uintptr_t)HstPtrBegin,
                            (uintptr_t)HstPtrBegin + Size, TgtAllocBegin,
                            TgtPtrBegin, HasHoldModifier, HstPtrName))
                        .first->HDTT);
    INFO(OMP_INFOTYPE_MAPPING_CHANGED, DeviceID,
         "Creating new map entry with HstPtrBase=" DPxMOD
         ", HstPtrBegin=" DPxMOD ", TgtAllocBegin=" DPxMOD
         ", TgtPtrBegin=" DPxMOD
         ", Size=%ld, DynRefCount=%s, HoldRefCount=%s, Name=%s\n",
         DPxPTR(HstPtrBase), DPxPTR(HstPtrBegin), DPxPTR(TgtAllocBegin),
         DPxPTR(TgtPtrBegin), Size,
         LR.TPR.getEntry()->dynRefCountToStr().c_str(),
         LR.TPR.getEntry()->holdRefCountToStr().c_str(),
         (HstPtrName) ? getNameFromMapping(HstPtrName).c_str() : "unknown");
    LR.TPR.TargetPointer = (void *)TgtPtrBegin;

    // Notify the plugin about the new mapping.
    if (notifyDataMapped(HstPtrBegin, Size))
      return {{false /* IsNewEntry */, false /* IsHostPointer */},
              nullptr /* Entry */,
              nullptr /* TargetPointer */};
  } else {
    // This entry is not present and we did not create a new entry for it.
    LR.TPR.Flags.IsPresent = false;
  }

  // All mapping table modifications have been made. If the user requested it we
  // give up the lock.
  if (ReleaseHDTTMap)
    HDTTMap.destroy();

  // If the target pointer is valid, and we need to transfer data, issue the
  // data transfer.
  if (LR.TPR.TargetPointer && !LR.TPR.Flags.IsHostPointer && HasFlagTo &&
      (LR.TPR.Flags.IsNewEntry || HasFlagAlways) && Size != 0) {
    DP("Moving %" PRId64 " bytes (hst:" DPxMOD ") -> (tgt:" DPxMOD ")\n", Size,
       DPxPTR(HstPtrBegin), DPxPTR(LR.TPR.TargetPointer));

    int Ret = submitData(LR.TPR.TargetPointer, HstPtrBegin, Size, AsyncInfo,
                         LR.TPR.getEntry());
    if (Ret != OFFLOAD_SUCCESS) {
      REPORT("Copying data to device failed.\n");
      // We will also return nullptr if the data movement fails because that
      // pointer points to a corrupted memory region so it doesn't make any
      // sense to continue to use it.
      LR.TPR.TargetPointer = nullptr;
    } else if (LR.TPR.getEntry()->addEventIfNecessary(*this, AsyncInfo) !=
               OFFLOAD_SUCCESS)
      return {{false /* IsNewEntry */, false /* IsHostPointer */},
              nullptr /* Entry */,
              nullptr /* TargetPointer */};
  } else {
    // If not a host pointer and no present modifier, we need to wait for the
    // event if it exists.
    // Note: Entry might be nullptr because of zero length array section.
    if (LR.TPR.getEntry() && !LR.TPR.Flags.IsHostPointer &&
        !HasPresentModifier) {
      void *Event = LR.TPR.getEntry()->getEvent();
      if (Event) {
        int Ret = waitEvent(Event, AsyncInfo);
        if (Ret != OFFLOAD_SUCCESS) {
          // If it fails to wait for the event, we need to return nullptr in
          // case of any data race.
          REPORT("Failed to wait for event " DPxMOD ".\n", DPxPTR(Event));
          return {{false /* IsNewEntry */, false /* IsHostPointer */},
                  nullptr /* Entry */,
                  nullptr /* TargetPointer */};
        }
      }
    }
  }

  return std::move(LR.TPR);
}

TargetPointerResultTy
DeviceTy::getTgtPtrBegin(void *HstPtrBegin, int64_t Size, bool UpdateRefCount,
                         bool UseHoldRefCount, bool MustContain,
                         bool ForceDelete, bool FromDataEnd) {
  HDTTMapAccessorTy HDTTMap = HostDataToTargetMap.getExclusiveAccessor();

  LookupResult LR = lookupMapping(HDTTMap, HstPtrBegin, Size);

  LR.TPR.Flags.IsPresent = true;

  if (LR.Flags.IsContained ||
      (!MustContain && (LR.Flags.ExtendsBefore || LR.Flags.ExtendsAfter))) {
    LR.TPR.Flags.IsLast =
        LR.TPR.getEntry()->decShouldRemove(UseHoldRefCount, ForceDelete);

    if (ForceDelete) {
      LR.TPR.getEntry()->resetRefCount(UseHoldRefCount);
      assert(LR.TPR.Flags.IsLast ==
                 LR.TPR.getEntry()->decShouldRemove(UseHoldRefCount) &&
             "expected correct IsLast prediction for reset");
    }

    // Increment the number of threads that is using the entry on a
    // targetDataEnd, tracking the number of possible "deleters". A thread may
    // come to own the entry deletion even if it was not the last one querying
    // for it. Thus, we must track every query on targetDataEnds to ensure only
    // the last thread that holds a reference to an entry actually deletes it.
    if (FromDataEnd)
      LR.TPR.getEntry()->incDataEndThreadCount();

    const char *RefCountAction;
    if (!UpdateRefCount) {
      RefCountAction = " (update suppressed)";
    } else if (LR.TPR.Flags.IsLast) {
      LR.TPR.getEntry()->decRefCount(UseHoldRefCount);
      assert(LR.TPR.getEntry()->getTotalRefCount() == 0 &&
             "Expected zero reference count when deletion is scheduled");
      if (ForceDelete)
        RefCountAction = " (reset, delayed deletion)";
      else
        RefCountAction = " (decremented, delayed deletion)";
    } else {
      LR.TPR.getEntry()->decRefCount(UseHoldRefCount);
      RefCountAction = " (decremented)";
    }
    const char *DynRefCountAction = UseHoldRefCount ? "" : RefCountAction;
    const char *HoldRefCountAction = UseHoldRefCount ? RefCountAction : "";
    uintptr_t TP = LR.TPR.getEntry()->TgtPtrBegin +
                   ((uintptr_t)HstPtrBegin - LR.TPR.getEntry()->HstPtrBegin);
    INFO(OMP_INFOTYPE_MAPPING_EXISTS, DeviceID,
         "Mapping exists with HstPtrBegin=" DPxMOD ", TgtPtrBegin=" DPxMOD ", "
         "Size=%" PRId64 ", DynRefCount=%s%s, HoldRefCount=%s%s\n",
         DPxPTR(HstPtrBegin), DPxPTR(TP), Size,
         LR.TPR.getEntry()->dynRefCountToStr().c_str(), DynRefCountAction,
         LR.TPR.getEntry()->holdRefCountToStr().c_str(), HoldRefCountAction);
    LR.TPR.TargetPointer = (void *)TP;
  } else if (PM->RTLs.RequiresFlags & OMP_REQ_UNIFIED_SHARED_MEMORY) {
    // If the value isn't found in the mapping and unified shared memory
    // is on then it means we have stumbled upon a value which we need to
    // use directly from the host.
    DP("Get HstPtrBegin " DPxMOD " Size=%" PRId64 " for unified shared "
       "memory\n",
       DPxPTR((uintptr_t)HstPtrBegin), Size);
    LR.TPR.Flags.IsPresent = false;
    LR.TPR.Flags.IsHostPointer = true;
    LR.TPR.TargetPointer = HstPtrBegin;
  } else {
    // OpenMP Specification v5.2: if a matching list item is not found, the
    // pointer retains its original value as per firstprivate semantics.
    LR.TPR.Flags.IsPresent = false;
    LR.TPR.Flags.IsHostPointer = false;
    LR.TPR.TargetPointer = HstPtrBegin;
  }

  return std::move(LR.TPR);
}

// Return the target pointer begin (where the data will be moved).
void *DeviceTy::getTgtPtrBegin(HDTTMapAccessorTy &HDTTMap, void *HstPtrBegin,
                               int64_t Size) {
  uintptr_t HP = (uintptr_t)HstPtrBegin;
  LookupResult LR = lookupMapping(HDTTMap, HstPtrBegin, Size);
  if (LR.Flags.IsContained || LR.Flags.ExtendsBefore || LR.Flags.ExtendsAfter) {
    uintptr_t TP =
        LR.TPR.getEntry()->TgtPtrBegin + (HP - LR.TPR.getEntry()->HstPtrBegin);
    return (void *)TP;
  }

  return NULL;
}

int DeviceTy::eraseMapEntry(HDTTMapAccessorTy &HDTTMap,
                            HostDataToTargetTy *Entry, int64_t Size) {
  assert(Entry && "Trying to delete a null entry from the HDTT map.");
  assert(Entry->getTotalRefCount() == 0 && Entry->getDataEndThreadCount() == 0 &&
         "Trying to delete entry that is in use or owned by another thread.");

  INFO(OMP_INFOTYPE_MAPPING_CHANGED, DeviceID,
       "Removing map entry with HstPtrBegin=" DPxMOD ", TgtPtrBegin=" DPxMOD
       ", Size=%" PRId64 ", Name=%s\n",
       DPxPTR(Entry->HstPtrBegin), DPxPTR(Entry->TgtPtrBegin), Size,
       (Entry->HstPtrName) ? getNameFromMapping(Entry->HstPtrName).c_str()
                           : "unknown");

  if (HDTTMap->erase(Entry) == 0) {
    REPORT("Trying to remove a non-existent map entry\n");
    return OFFLOAD_FAIL;
  }

  return OFFLOAD_SUCCESS;
}

int DeviceTy::deallocTgtPtrAndEntry(HostDataToTargetTy *Entry, int64_t Size) {
  assert(Entry && "Trying to deallocate a null entry.");

  DP("Deleting tgt data " DPxMOD " of size %" PRId64 " by freeing allocation "
     "starting at " DPxMOD "\n",
     DPxPTR(Entry->TgtPtrBegin), Size, DPxPTR(Entry->TgtAllocBegin));

  void *Event = Entry->getEvent();
  if (Event && destroyEvent(Event) != OFFLOAD_SUCCESS) {
    REPORT("Failed to destroy event " DPxMOD "\n", DPxPTR(Event));
    return OFFLOAD_FAIL;
  }

  int Ret = deleteData((void *)Entry->TgtAllocBegin);

  // Notify the plugin about the unmapped memory.
  Ret |= notifyDataUnmapped((void *)Entry->HstPtrBegin);

  delete Entry;

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
  return OFFLOAD_FAIL;
}

void DeviceTy::deinit() {
  if (RTL->deinit_device)
    RTL->deinit_device(RTLDeviceID);
}

// Load binary to device.
__tgt_target_table *DeviceTy::loadBinary(void *Img) {
  std::lock_guard<decltype(RTL->Mtx)> LG(RTL->Mtx);
  return RTL->load_binary(RTLDeviceID, Img);
}

void *DeviceTy::allocData(int64_t Size, void *HstPtr, int32_t Kind) {
  /// RAII to establish tool anchors before and after data allocation
  void *TargetPtr = nullptr;
  OMPT_IF_BUILT(InterfaceRAII TargetDataAllocRAII(
                    RegionInterface.getCallbacks<ompt_target_data_alloc>(),
                    RTLDeviceID, HstPtr, &TargetPtr, Size,
                    /* CodePtr */ OMPT_GET_RETURN_ADDRESS(0));)

  TargetPtr = RTL->data_alloc(RTLDeviceID, Size, HstPtr, Kind);
  return TargetPtr;
}

int32_t DeviceTy::deleteData(void *TgtAllocBegin, int32_t Kind) {
  /// RAII to establish tool anchors before and after data deletion
  OMPT_IF_BUILT(InterfaceRAII TargetDataDeleteRAII(
                    RegionInterface.getCallbacks<ompt_target_data_delete>(),
                    RTLDeviceID, TgtAllocBegin,
                    /* CodePtr */ OMPT_GET_RETURN_ADDRESS(0));)

  return RTL->data_delete(RTLDeviceID, TgtAllocBegin, Kind);
}

static void printCopyInfo(int DeviceId, bool H2D, void *SrcPtrBegin,
                          void *DstPtrBegin, int64_t Size,
                          HostDataToTargetTy *HT) {

  INFO(OMP_INFOTYPE_DATA_TRANSFER, DeviceId,
       "Copying data from %s to %s, %sPtr=" DPxMOD ", %sPtr=" DPxMOD
       ", Size=%" PRId64 ", Name=%s\n",
       H2D ? "host" : "device", H2D ? "device" : "host", H2D ? "Hst" : "Tgt",
       DPxPTR(SrcPtrBegin), H2D ? "Tgt" : "Hst", DPxPTR(DstPtrBegin), Size,
       (HT && HT->HstPtrName) ? getNameFromMapping(HT->HstPtrName).c_str()
                              : "unknown");
}

// Submit data to device
int32_t DeviceTy::submitData(void *TgtPtrBegin, void *HstPtrBegin, int64_t Size,
                             AsyncInfoTy &AsyncInfo,
                             HostDataToTargetTy *Entry) {
  if (getInfoLevel() & OMP_INFOTYPE_DATA_TRANSFER) {
    HDTTMapAccessorTy HDTTMap = HostDataToTargetMap.getExclusiveAccessor(Entry);
    LookupResult LR;
    if (!Entry) {
      LR = lookupMapping(HDTTMap, HstPtrBegin, Size);
      Entry = LR.TPR.getEntry();
    }
    printCopyInfo(DeviceID, /* H2D */ true, HstPtrBegin, TgtPtrBegin, Size,
                  Entry);
  }

  /// RAII to establish tool anchors before and after data submit
  OMPT_IF_BUILT(
      InterfaceRAII TargetDataSubmitRAII(
          RegionInterface.getCallbacks<ompt_target_data_transfer_to_device>(),
          RTLDeviceID, TgtPtrBegin, HstPtrBegin, Size,
          /* CodePtr */ OMPT_GET_RETURN_ADDRESS(0));)

  if (!AsyncInfo || !RTL->data_submit_async || !RTL->synchronize)
    return RTL->data_submit(RTLDeviceID, TgtPtrBegin, HstPtrBegin, Size);
  return RTL->data_submit_async(RTLDeviceID, TgtPtrBegin, HstPtrBegin, Size,
                                AsyncInfo);
}

// Retrieve data from device
int32_t DeviceTy::retrieveData(void *HstPtrBegin, void *TgtPtrBegin,
                               int64_t Size, AsyncInfoTy &AsyncInfo,
                               HostDataToTargetTy *Entry) {
  if (getInfoLevel() & OMP_INFOTYPE_DATA_TRANSFER) {
    HDTTMapAccessorTy HDTTMap = HostDataToTargetMap.getExclusiveAccessor(Entry);
    LookupResult LR;
    if (!Entry) {
      LR = lookupMapping(HDTTMap, HstPtrBegin, Size);
      Entry = LR.TPR.getEntry();
    }
    printCopyInfo(DeviceID, /* H2D */ false, TgtPtrBegin, HstPtrBegin, Size,
                  Entry);
  }

  /// RAII to establish tool anchors before and after data retrieval
  OMPT_IF_BUILT(
      InterfaceRAII TargetDataRetrieveRAII(
          RegionInterface.getCallbacks<ompt_target_data_transfer_from_device>(),
          RTLDeviceID, HstPtrBegin, TgtPtrBegin, Size,
          /* CodePtr */ OMPT_GET_RETURN_ADDRESS(0));)

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
                               ptrdiff_t *TgtOffsets,
                               const KernelArgsTy &KernelArgs,
                               AsyncInfoTy &AsyncInfo) {
  return RTL->launch_kernel(RTLDeviceID, TgtEntryPtr, TgtVarsPtr, TgtOffsets,
                            &KernelArgs, AsyncInfo);
}

// Run region on device
bool DeviceTy::printDeviceInfo(int32_t RTLDevId) {
  if (!RTL->print_device_info)
    return false;
  RTL->print_device_info(RTLDevId);
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

/// Check whether a device has an associated RTL and initialize it if it's not
/// already initialized.
bool deviceIsReady(int DeviceNum) {
  DP("Checking whether device %d is ready.\n", DeviceNum);
  // Devices.size() can only change while registering a new
  // library, so try to acquire the lock of RTLs' mutex.
  size_t DevicesSize;
  {
    std::lock_guard<decltype(PM->RTLsMtx)> LG(PM->RTLsMtx);
    DevicesSize = PM->Devices.size();
  }
  if (DevicesSize <= (size_t)DeviceNum) {
    DP("Device ID  %d does not have a matching RTL\n", DeviceNum);
    return false;
  }

  // Get device info
  DeviceTy &Device = *PM->Devices[DeviceNum];

  DP("Is the device %d (local ID %d) initialized? %d\n", DeviceNum,
     Device.RTLDeviceID, Device.IsInit);

  // Init the device if not done before
  if (!Device.IsInit && Device.initOnce() != OFFLOAD_SUCCESS) {
    DP("Failed to init device %d\n", DeviceNum);
    return false;
  }

  DP("Device %d is ready to use.\n", DeviceNum);

  return true;
}
