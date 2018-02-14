//===--------- device.cpp - Target independent OpenMP target RTL ----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.txt for details.
//
//===----------------------------------------------------------------------===//
//
// Functionality for managing devices that are handled by RTL plugins.
//
//===----------------------------------------------------------------------===//

#include "device.h"
#include "private.h"
#include "rtl.h"

#include <cassert>
#include <climits>
#include <string>

/// Map between Device ID (i.e. openmp device id) and its DeviceTy.
DevicesTy Devices;

int DeviceTy::associatePtr(void *HstPtrBegin, void *TgtPtrBegin, int64_t Size) {
  DataMapMtx.lock();

  // Check if entry exists
  for (auto &HT : HostDataToTargetMap) {
    if ((uintptr_t)HstPtrBegin == HT.HstPtrBegin) {
      // Mapping already exists
      bool isValid = HT.HstPtrBegin == (uintptr_t) HstPtrBegin &&
                     HT.HstPtrEnd == (uintptr_t) HstPtrBegin + Size &&
                     HT.TgtPtrBegin == (uintptr_t) TgtPtrBegin;
      DataMapMtx.unlock();
      if (isValid) {
        DP("Attempt to re-associate the same device ptr+offset with the same "
            "host ptr, nothing to do\n");
        return OFFLOAD_SUCCESS;
      } else {
        DP("Not allowed to re-associate a different device ptr+offset with the "
            "same host ptr\n");
        return OFFLOAD_FAIL;
      }
    }
  }

  // Mapping does not exist, allocate it
  HostDataToTargetTy newEntry;

  // Set up missing fields
  newEntry.HstPtrBase = (uintptr_t) HstPtrBegin;
  newEntry.HstPtrBegin = (uintptr_t) HstPtrBegin;
  newEntry.HstPtrEnd = (uintptr_t) HstPtrBegin + Size;
  newEntry.TgtPtrBegin = (uintptr_t) TgtPtrBegin;
  // refCount must be infinite
  newEntry.RefCount = INF_REF_CNT;

  DP("Creating new map entry: HstBase=" DPxMOD ", HstBegin=" DPxMOD ", HstEnd="
      DPxMOD ", TgtBegin=" DPxMOD "\n", DPxPTR(newEntry.HstPtrBase),
      DPxPTR(newEntry.HstPtrBegin), DPxPTR(newEntry.HstPtrEnd),
      DPxPTR(newEntry.TgtPtrBegin));
  HostDataToTargetMap.push_front(newEntry);

  DataMapMtx.unlock();

  return OFFLOAD_SUCCESS;
}

int DeviceTy::disassociatePtr(void *HstPtrBegin) {
  DataMapMtx.lock();

  // Check if entry exists
  for (HostDataToTargetListTy::iterator ii = HostDataToTargetMap.begin();
      ii != HostDataToTargetMap.end(); ++ii) {
    if ((uintptr_t)HstPtrBegin == ii->HstPtrBegin) {
      // Mapping exists
      if (CONSIDERED_INF(ii->RefCount)) {
        DP("Association found, removing it\n");
        HostDataToTargetMap.erase(ii);
        DataMapMtx.unlock();
        return OFFLOAD_SUCCESS;
      } else {
        DP("Trying to disassociate a pointer which was not mapped via "
            "omp_target_associate_ptr\n");
        break;
      }
    }
  }

  // Mapping not found
  DataMapMtx.unlock();
  DP("Association not found\n");
  return OFFLOAD_FAIL;
}

// Get ref count of map entry containing HstPtrBegin
long DeviceTy::getMapEntryRefCnt(void *HstPtrBegin) {
  uintptr_t hp = (uintptr_t)HstPtrBegin;
  long RefCnt = -1;

  DataMapMtx.lock();
  for (auto &HT : HostDataToTargetMap) {
    if (hp >= HT.HstPtrBegin && hp < HT.HstPtrEnd) {
      DP("DeviceTy::getMapEntry: requested entry found\n");
      RefCnt = HT.RefCount;
      break;
    }
  }
  DataMapMtx.unlock();

  if (RefCnt < 0) {
    DP("DeviceTy::getMapEntry: requested entry not found\n");
  }

  return RefCnt;
}

LookupResult DeviceTy::lookupMapping(void *HstPtrBegin, int64_t Size) {
  uintptr_t hp = (uintptr_t)HstPtrBegin;
  LookupResult lr;

  DP("Looking up mapping(HstPtrBegin=" DPxMOD ", Size=%ld)...\n", DPxPTR(hp),
      Size);
  for (lr.Entry = HostDataToTargetMap.begin();
      lr.Entry != HostDataToTargetMap.end(); ++lr.Entry) {
    auto &HT = *lr.Entry;
    // Is it contained?
    lr.Flags.IsContained = hp >= HT.HstPtrBegin && hp < HT.HstPtrEnd &&
        (hp+Size) <= HT.HstPtrEnd;
    // Does it extend into an already mapped region?
    lr.Flags.ExtendsBefore = hp < HT.HstPtrBegin && (hp+Size) > HT.HstPtrBegin;
    // Does it extend beyond the mapped region?
    lr.Flags.ExtendsAfter = hp < HT.HstPtrEnd && (hp+Size) > HT.HstPtrEnd;

    if (lr.Flags.IsContained || lr.Flags.ExtendsBefore ||
        lr.Flags.ExtendsAfter) {
      break;
    }
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

// Used by target_data_begin
// Return the target pointer begin (where the data will be moved).
// Allocate memory if this is the first occurrence if this mapping.
// Increment the reference counter.
// If NULL is returned, then either data allocation failed or the user tried
// to do an illegal mapping.
void *DeviceTy::getOrAllocTgtPtr(void *HstPtrBegin, void *HstPtrBase,
    int64_t Size, bool &IsNew, bool IsImplicit, bool UpdateRefCount) {
  void *rc = NULL;
  DataMapMtx.lock();
  LookupResult lr = lookupMapping(HstPtrBegin, Size);

  // Check if the pointer is contained.
  if (lr.Flags.IsContained ||
      ((lr.Flags.ExtendsBefore || lr.Flags.ExtendsAfter) && IsImplicit)) {
    auto &HT = *lr.Entry;
    IsNew = false;

    if (UpdateRefCount)
      ++HT.RefCount;

    uintptr_t tp = HT.TgtPtrBegin + ((uintptr_t)HstPtrBegin - HT.HstPtrBegin);
    DP("Mapping exists%s with HstPtrBegin=" DPxMOD ", TgtPtrBegin=" DPxMOD ", "
        "Size=%ld,%s RefCount=%s\n", (IsImplicit ? " (implicit)" : ""),
        DPxPTR(HstPtrBegin), DPxPTR(tp), Size,
        (UpdateRefCount ? " updated" : ""),
        (CONSIDERED_INF(HT.RefCount)) ? "INF" :
            std::to_string(HT.RefCount).c_str());
    rc = (void *)tp;
  } else if ((lr.Flags.ExtendsBefore || lr.Flags.ExtendsAfter) && !IsImplicit) {
    // Explicit extension of mapped data - not allowed.
    DP("Explicit extension of mapping is not allowed.\n");
  } else if (Size) {
    // If it is not contained and Size > 0 we should create a new entry for it.
    IsNew = true;
    uintptr_t tp = (uintptr_t)RTL->data_alloc(RTLDeviceID, Size, HstPtrBegin);
    DP("Creating new map entry: HstBase=" DPxMOD ", HstBegin=" DPxMOD ", "
        "HstEnd=" DPxMOD ", TgtBegin=" DPxMOD "\n", DPxPTR(HstPtrBase),
        DPxPTR(HstPtrBegin), DPxPTR((uintptr_t)HstPtrBegin + Size), DPxPTR(tp));
    HostDataToTargetMap.push_front(HostDataToTargetTy((uintptr_t)HstPtrBase,
        (uintptr_t)HstPtrBegin, (uintptr_t)HstPtrBegin + Size, tp));
    rc = (void *)tp;
  }

  DataMapMtx.unlock();
  return rc;
}

// Used by target_data_begin, target_data_end, target_data_update and target.
// Return the target pointer begin (where the data will be moved).
// Decrement the reference counter if called from target_data_end.
void *DeviceTy::getTgtPtrBegin(void *HstPtrBegin, int64_t Size, bool &IsLast,
    bool UpdateRefCount) {
  void *rc = NULL;
  DataMapMtx.lock();
  LookupResult lr = lookupMapping(HstPtrBegin, Size);

  if (lr.Flags.IsContained || lr.Flags.ExtendsBefore || lr.Flags.ExtendsAfter) {
    auto &HT = *lr.Entry;
    IsLast = !(HT.RefCount > 1);

    if (HT.RefCount > 1 && UpdateRefCount)
      --HT.RefCount;

    uintptr_t tp = HT.TgtPtrBegin + ((uintptr_t)HstPtrBegin - HT.HstPtrBegin);
    DP("Mapping exists with HstPtrBegin=" DPxMOD ", TgtPtrBegin=" DPxMOD ", "
        "Size=%ld,%s RefCount=%s\n", DPxPTR(HstPtrBegin), DPxPTR(tp), Size,
        (UpdateRefCount ? " updated" : ""),
        (CONSIDERED_INF(HT.RefCount)) ? "INF" :
            std::to_string(HT.RefCount).c_str());
    rc = (void *)tp;
  } else {
    IsLast = false;
  }

  DataMapMtx.unlock();
  return rc;
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

int DeviceTy::deallocTgtPtr(void *HstPtrBegin, int64_t Size, bool ForceDelete) {
  // Check if the pointer is contained in any sub-nodes.
  int rc;
  DataMapMtx.lock();
  LookupResult lr = lookupMapping(HstPtrBegin, Size);
  if (lr.Flags.IsContained || lr.Flags.ExtendsBefore || lr.Flags.ExtendsAfter) {
    auto &HT = *lr.Entry;
    if (ForceDelete)
      HT.RefCount = 1;
    if (--HT.RefCount <= 0) {
      assert(HT.RefCount == 0 && "did not expect a negative ref count");
      DP("Deleting tgt data " DPxMOD " of size %ld\n",
          DPxPTR(HT.TgtPtrBegin), Size);
      RTL->data_delete(RTLDeviceID, (void *)HT.TgtPtrBegin);
      DP("Removing%s mapping with HstPtrBegin=" DPxMOD ", TgtPtrBegin=" DPxMOD
          ", Size=%ld\n", (ForceDelete ? " (forced)" : ""),
          DPxPTR(HT.HstPtrBegin), DPxPTR(HT.TgtPtrBegin), Size);
      HostDataToTargetMap.erase(lr.Entry);
    }
    rc = OFFLOAD_SUCCESS;
  } else {
    DP("Section to delete (hst addr " DPxMOD ") does not exist in the allocated"
       " memory\n", DPxPTR(HstPtrBegin));
    rc = OFFLOAD_FAIL;
  }

  DataMapMtx.unlock();
  return rc;
}

/// Init device, should not be called directly.
void DeviceTy::init() {
  int32_t rc = RTL->init_device(RTLDeviceID);
  if (rc == OFFLOAD_SUCCESS) {
    IsInit = true;
  }
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

// Load binary to device.
__tgt_target_table *DeviceTy::load_binary(void *Img) {
  RTL->Mtx.lock();
  __tgt_target_table *rc = RTL->load_binary(RTLDeviceID, Img);
  RTL->Mtx.unlock();
  return rc;
}

// Submit data to device.
int32_t DeviceTy::data_submit(void *TgtPtrBegin, void *HstPtrBegin,
    int64_t Size) {
  return RTL->data_submit(RTLDeviceID, TgtPtrBegin, HstPtrBegin, Size);
}

// Retrieve data from device.
int32_t DeviceTy::data_retrieve(void *HstPtrBegin, void *TgtPtrBegin,
    int64_t Size) {
  return RTL->data_retrieve(RTLDeviceID, HstPtrBegin, TgtPtrBegin, Size);
}

// Run region on device
int32_t DeviceTy::run_region(void *TgtEntryPtr, void **TgtVarsPtr,
    ptrdiff_t *TgtOffsets, int32_t TgtVarsSize) {
  return RTL->run_region(RTLDeviceID, TgtEntryPtr, TgtVarsPtr, TgtOffsets,
      TgtVarsSize);
}

// Run team region on device.
int32_t DeviceTy::run_team_region(void *TgtEntryPtr, void **TgtVarsPtr,
    ptrdiff_t *TgtOffsets, int32_t TgtVarsSize, int32_t NumTeams,
    int32_t ThreadLimit, uint64_t LoopTripCount) {
  return RTL->run_team_region(RTLDeviceID, TgtEntryPtr, TgtVarsPtr, TgtOffsets,
      TgtVarsSize, NumTeams, ThreadLimit, LoopTripCount);
}

/// Check whether a device has an associated RTL and initialize it if it's not
/// already initialized.
bool device_is_ready(int device_num) {
  DP("Checking whether device %d is ready.\n", device_num);
  // Devices.size() can only change while registering a new
  // library, so try to acquire the lock of RTLs' mutex.
  RTLsMtx.lock();
  size_t Devices_size = Devices.size();
  RTLsMtx.unlock();
  if (Devices_size <= (size_t)device_num) {
    DP("Device ID  %d does not have a matching RTL\n", device_num);
    return false;
  }

  // Get device info
  DeviceTy &Device = Devices[device_num];

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
