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

#include <cstddef>
#include <climits>
#include <list>
#include <map>
#include <mutex>
#include <vector>

// Forward declarations.
struct RTLInfoTy;
struct __tgt_bin_desc;
struct __tgt_target_table;

#define INF_REF_CNT (LONG_MAX>>1) // leave room for additions/subtractions
#define CONSIDERED_INF(x) (x > (INF_REF_CNT>>1))

/// Map between host data and target data.
struct HostDataToTargetTy {
  uintptr_t HstPtrBase; // host info.
  uintptr_t HstPtrBegin;
  uintptr_t HstPtrEnd; // non-inclusive.

  uintptr_t TgtPtrBegin; // target info.

  long RefCount;

  HostDataToTargetTy()
      : HstPtrBase(0), HstPtrBegin(0), HstPtrEnd(0),
        TgtPtrBegin(0), RefCount(0) {}
  HostDataToTargetTy(uintptr_t BP, uintptr_t B, uintptr_t E, uintptr_t TB)
      : HstPtrBase(BP), HstPtrBegin(B), HstPtrEnd(E),
        TgtPtrBegin(TB), RefCount(1) {}
  HostDataToTargetTy(uintptr_t BP, uintptr_t B, uintptr_t E, uintptr_t TB,
      long RF)
      : HstPtrBase(BP), HstPtrBegin(B), HstPtrEnd(E),
        TgtPtrBegin(TB), RefCount(RF) {}
};

typedef std::list<HostDataToTargetTy> HostDataToTargetListTy;

struct LookupResult {
  struct {
    unsigned IsContained   : 1;
    unsigned ExtendsBefore : 1;
    unsigned ExtendsAfter  : 1;
  } Flags;

  HostDataToTargetListTy::iterator Entry;

  LookupResult() : Flags({0,0,0}), Entry() {}
};

/// Map for shadow pointers
struct ShadowPtrValTy {
  void *HstPtrVal;
  void *TgtPtrAddr;
  void *TgtPtrVal;
};
typedef std::map<void *, ShadowPtrValTy> ShadowPtrListTy;

///
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

  bool IsInit;
  std::once_flag InitFlag;
  bool HasPendingGlobals;

  HostDataToTargetListTy HostDataToTargetMap;
  PendingCtorsDtorsPerLibrary PendingCtorsDtors;

  ShadowPtrListTy ShadowPtrMap;

  std::mutex DataMapMtx, PendingGlobalsMtx, ShadowMtx;

  uint64_t loopTripCnt;

  int64_t RTLRequiresFlags;

  DeviceTy(RTLInfoTy *RTL)
      : DeviceID(-1), RTL(RTL), RTLDeviceID(-1), IsInit(false), InitFlag(),
        HasPendingGlobals(false), HostDataToTargetMap(),
        PendingCtorsDtors(), ShadowPtrMap(), DataMapMtx(), PendingGlobalsMtx(),
        ShadowMtx(), loopTripCnt(0), RTLRequiresFlags(0) {}

  // The existence of mutexes makes DeviceTy non-copyable. We need to
  // provide a copy constructor and an assignment operator explicitly.
  DeviceTy(const DeviceTy &d)
      : DeviceID(d.DeviceID), RTL(d.RTL), RTLDeviceID(d.RTLDeviceID),
        IsInit(d.IsInit), InitFlag(), HasPendingGlobals(d.HasPendingGlobals),
        HostDataToTargetMap(d.HostDataToTargetMap),
        PendingCtorsDtors(d.PendingCtorsDtors), ShadowPtrMap(d.ShadowPtrMap),
        DataMapMtx(), PendingGlobalsMtx(),
        ShadowMtx(), loopTripCnt(d.loopTripCnt),
        RTLRequiresFlags(d.RTLRequiresFlags) {}

  DeviceTy& operator=(const DeviceTy &d) {
    DeviceID = d.DeviceID;
    RTL = d.RTL;
    RTLDeviceID = d.RTLDeviceID;
    IsInit = d.IsInit;
    HasPendingGlobals = d.HasPendingGlobals;
    HostDataToTargetMap = d.HostDataToTargetMap;
    PendingCtorsDtors = d.PendingCtorsDtors;
    ShadowPtrMap = d.ShadowPtrMap;
    loopTripCnt = d.loopTripCnt;
    RTLRequiresFlags = d.RTLRequiresFlags;

    return *this;
  }

  long getMapEntryRefCnt(void *HstPtrBegin);
  LookupResult lookupMapping(void *HstPtrBegin, int64_t Size);
  void *getOrAllocTgtPtr(void *HstPtrBegin, void *HstPtrBase, int64_t Size,
      bool &IsNew, bool IsImplicit, bool UpdateRefCount = true);
  void *getTgtPtrBegin(void *HstPtrBegin, int64_t Size);
  void *getTgtPtrBegin(void *HstPtrBegin, int64_t Size, bool &IsLast,
      bool UpdateRefCount);
  int deallocTgtPtr(void *TgtPtrBegin, int64_t Size, bool ForceDelete);
  int associatePtr(void *HstPtrBegin, void *TgtPtrBegin, int64_t Size);
  int disassociatePtr(void *HstPtrBegin);

  // calls to RTL
  int32_t initOnce();
  __tgt_target_table *load_binary(void *Img);

  int32_t data_submit(void *TgtPtrBegin, void *HstPtrBegin, int64_t Size);
  int32_t data_retrieve(void *HstPtrBegin, void *TgtPtrBegin, int64_t Size);

  int32_t run_region(void *TgtEntryPtr, void **TgtVarsPtr,
      ptrdiff_t *TgtOffsets, int32_t TgtVarsSize);
  int32_t run_team_region(void *TgtEntryPtr, void **TgtVarsPtr,
      ptrdiff_t *TgtOffsets, int32_t TgtVarsSize, int32_t NumTeams,
      int32_t ThreadLimit, uint64_t LoopTripCount);

private:
  // Call to RTL
  void init(); // To be called only via DeviceTy::initOnce()
};

/// Map between Device ID (i.e. openmp device id) and its DeviceTy.
typedef std::vector<DeviceTy> DevicesTy;
extern DevicesTy Devices;

extern bool device_is_ready(int device_num);

#endif
