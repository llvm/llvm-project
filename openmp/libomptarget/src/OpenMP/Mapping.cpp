//===-- OpenMP/Mapping.cpp - OpenMP/OpenACC pointer mapping impl. ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#include "OpenMP/Mapping.h"

#include "PluginManager.h"
#include "Shared/Debug.h"
#include "Shared/Requirements.h"
#include "device.h"

/// Dump a table of all the host-target pointer pairs on failure
void dumpTargetPointerMappings(const ident_t *Loc, DeviceTy &Device) {
  MappingInfoTy::HDTTMapAccessorTy HDTTMap =
      Device.getMappingInfo().HostDataToTargetMap.getExclusiveAccessor();
  if (HDTTMap->empty())
    return;

  SourceInfo Kernel(Loc);
  INFO(OMP_INFOTYPE_ALL, Device.DeviceID,
       "OpenMP Host-Device pointer mappings after block at %s:%d:%d:\n",
       Kernel.getFilename(), Kernel.getLine(), Kernel.getColumn());
  INFO(OMP_INFOTYPE_ALL, Device.DeviceID, "%-18s %-18s %s %s %s %s\n",
       "Host Ptr", "Target Ptr", "Size (B)", "DynRefCount", "HoldRefCount",
       "Declaration");
  for (const auto &It : *HDTTMap) {
    HostDataToTargetTy &HDTT = *It.HDTT;
    SourceInfo Info(HDTT.HstPtrName);
    INFO(OMP_INFOTYPE_ALL, Device.DeviceID,
         DPxMOD " " DPxMOD " %-8" PRIuPTR " %-11s %-12s %s at %s:%d:%d\n",
         DPxPTR(HDTT.HstPtrBegin), DPxPTR(HDTT.TgtPtrBegin),
         HDTT.HstPtrEnd - HDTT.HstPtrBegin, HDTT.dynRefCountToStr().c_str(),
         HDTT.holdRefCountToStr().c_str(), Info.getName(), Info.getFilename(),
         Info.getLine(), Info.getColumn());
  }
}

int MappingInfoTy::associatePtr(void *HstPtrBegin, void *TgtPtrBegin,
                                int64_t Size) {
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
  return Device.notifyDataMapped(HstPtrBegin, Size);
}

int MappingInfoTy::disassociatePtr(void *HstPtrBegin) {
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
      Device.destroyEvent(Event);
    HDTTMap->erase(It);
    return Device.notifyDataUnmapped(HstPtrBegin);
  }

  REPORT("Trying to disassociate a pointer which was not mapped via "
         "omp_target_associate_ptr\n");
  return OFFLOAD_FAIL;
}

LookupResult MappingInfoTy::lookupMapping(HDTTMapAccessorTy &HDTTMap,
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

TargetPointerResultTy MappingInfoTy::getTargetPointer(
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
    INFO(OMP_INFOTYPE_MAPPING_EXISTS, Device.DeviceID,
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
  } else if (PM->getRequirements() & OMP_REQ_UNIFIED_SHARED_MEMORY &&
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
        (uintptr_t)Device.allocData(TgtPadding + Size, HstPtrBegin);
    uintptr_t TgtPtrBegin = TgtAllocBegin + TgtPadding;
    // Release the mapping table lock only after the entry is locked by
    // attaching it to TPR.
    LR.TPR.setEntry(HDTTMap
                        ->emplace(new HostDataToTargetTy(
                            (uintptr_t)HstPtrBase, (uintptr_t)HstPtrBegin,
                            (uintptr_t)HstPtrBegin + Size, TgtAllocBegin,
                            TgtPtrBegin, HasHoldModifier, HstPtrName))
                        .first->HDTT);
    INFO(OMP_INFOTYPE_MAPPING_CHANGED, Device.DeviceID,
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
    if (Device.notifyDataMapped(HstPtrBegin, Size))
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

    int Ret = Device.submitData(LR.TPR.TargetPointer, HstPtrBegin, Size,
                                AsyncInfo, LR.TPR.getEntry());
    if (Ret != OFFLOAD_SUCCESS) {
      REPORT("Copying data to device failed.\n");
      // We will also return nullptr if the data movement fails because that
      // pointer points to a corrupted memory region so it doesn't make any
      // sense to continue to use it.
      LR.TPR.TargetPointer = nullptr;
    } else if (LR.TPR.getEntry()->addEventIfNecessary(Device, AsyncInfo) !=
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
        int Ret = Device.waitEvent(Event, AsyncInfo);
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

TargetPointerResultTy MappingInfoTy::getTgtPtrBegin(
    void *HstPtrBegin, int64_t Size, bool UpdateRefCount, bool UseHoldRefCount,
    bool MustContain, bool ForceDelete, bool FromDataEnd) {
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
    INFO(OMP_INFOTYPE_MAPPING_EXISTS, Device.DeviceID,
         "Mapping exists with HstPtrBegin=" DPxMOD ", TgtPtrBegin=" DPxMOD ", "
         "Size=%" PRId64 ", DynRefCount=%s%s, HoldRefCount=%s%s\n",
         DPxPTR(HstPtrBegin), DPxPTR(TP), Size,
         LR.TPR.getEntry()->dynRefCountToStr().c_str(), DynRefCountAction,
         LR.TPR.getEntry()->holdRefCountToStr().c_str(), HoldRefCountAction);
    LR.TPR.TargetPointer = (void *)TP;
  } else if (PM->getRequirements() & OMP_REQ_UNIFIED_SHARED_MEMORY) {
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
void *MappingInfoTy::getTgtPtrBegin(HDTTMapAccessorTy &HDTTMap,
                                    void *HstPtrBegin, int64_t Size) {
  uintptr_t HP = (uintptr_t)HstPtrBegin;
  LookupResult LR = lookupMapping(HDTTMap, HstPtrBegin, Size);
  if (LR.Flags.IsContained || LR.Flags.ExtendsBefore || LR.Flags.ExtendsAfter) {
    uintptr_t TP =
        LR.TPR.getEntry()->TgtPtrBegin + (HP - LR.TPR.getEntry()->HstPtrBegin);
    return (void *)TP;
  }

  return NULL;
}

int MappingInfoTy::eraseMapEntry(HDTTMapAccessorTy &HDTTMap,
                                 HostDataToTargetTy *Entry, int64_t Size) {
  assert(Entry && "Trying to delete a null entry from the HDTT map.");
  assert(Entry->getTotalRefCount() == 0 &&
         Entry->getDataEndThreadCount() == 0 &&
         "Trying to delete entry that is in use or owned by another thread.");

  INFO(OMP_INFOTYPE_MAPPING_CHANGED, Device.DeviceID,
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

int MappingInfoTy::deallocTgtPtrAndEntry(HostDataToTargetTy *Entry,
                                         int64_t Size) {
  assert(Entry && "Trying to deallocate a null entry.");

  DP("Deleting tgt data " DPxMOD " of size %" PRId64 " by freeing allocation "
     "starting at " DPxMOD "\n",
     DPxPTR(Entry->TgtPtrBegin), Size, DPxPTR(Entry->TgtAllocBegin));

  void *Event = Entry->getEvent();
  if (Event && Device.destroyEvent(Event) != OFFLOAD_SUCCESS) {
    REPORT("Failed to destroy event " DPxMOD "\n", DPxPTR(Event));
    return OFFLOAD_FAIL;
  }

  int Ret = Device.deleteData((void *)Entry->TgtAllocBegin);

  // Notify the plugin about the unmapped memory.
  Ret |= Device.notifyDataUnmapped((void *)Entry->HstPtrBegin);

  delete Entry;

  return Ret;
}

static void printCopyInfoImpl(int DeviceId, bool H2D, void *SrcPtrBegin,
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

void MappingInfoTy::printCopyInfo(
    void *TgtPtrBegin, void *HstPtrBegin, int64_t Size, bool H2D,
    HostDataToTargetTy *Entry, MappingInfoTy::HDTTMapAccessorTy *HDTTMapPtr) {
  auto HDTTMap =
      HostDataToTargetMap.getExclusiveAccessor(!!Entry || !!HDTTMapPtr);
  LookupResult LR;
  if (!Entry) {
    LR = lookupMapping(HDTTMapPtr ? *HDTTMapPtr : HDTTMap, HstPtrBegin, Size);
    Entry = LR.TPR.getEntry();
  }
  printCopyInfoImpl(Device.DeviceID, H2D, HstPtrBegin, TgtPtrBegin, Size,
                    Entry);
}
