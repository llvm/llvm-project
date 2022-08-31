//===--------------------- rtl.cpp - Remote RTL Plugin --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// RTL for Host.
//
//===----------------------------------------------------------------------===//

#include <cstddef>
#include <memory>
#include <string>

#include "Client.h"
#include "Utils.h"
#include "omptarget.h"
#include "omptargetplugin.h"

#define TARGET_NAME RPC
#define DEBUG_PREFIX "Target " GETNAME(TARGET_NAME) " RTL"

RemoteClientManager *Manager;

__attribute__((constructor(101))) void initRPC() {
  DP("Init RPC library!\n");

  Manager = new RemoteClientManager();
}

__attribute__((destructor(101))) void deinitRPC() {
  Manager->shutdown(); // TODO: Error handle shutting down
  DP("Deinit RPC library!\n");
  delete Manager;
}

// Exposed library API function
#ifdef __cplusplus
extern "C" {
#endif

int32_t __tgt_rtl_register_lib(__tgt_bin_desc *Desc) {
  return Manager->registerLib(Desc);
}

int32_t __tgt_rtl_unregister_lib(__tgt_bin_desc *Desc) {
  return Manager->unregisterLib(Desc);
}

int32_t __tgt_rtl_is_valid_binary(__tgt_device_image *Image) {
  return Manager->isValidBinary(Image);
}

int32_t __tgt_rtl_number_of_devices() { return Manager->getNumberOfDevices(); }

int32_t __tgt_rtl_init_device(int32_t DeviceId) {
  return Manager->initDevice(DeviceId);
}

int64_t __tgt_rtl_init_requires(int64_t RequiresFlags) {
  return Manager->initRequires(RequiresFlags);
}

__tgt_target_table *__tgt_rtl_load_binary(int32_t DeviceId,
                                          __tgt_device_image *Image) {
  return Manager->loadBinary(DeviceId, (__tgt_device_image *)Image);
}

int32_t __tgt_rtl_is_data_exchangable(int32_t SrcDevId, int32_t DstDevId) {
  return Manager->isDataExchangeable(SrcDevId, DstDevId);
}

void *__tgt_rtl_data_alloc(int32_t DeviceId, int64_t Size, void *HstPtr,
                           int32_t Kind) {
  if (Kind != TARGET_ALLOC_DEFAULT) {
    REPORT("Invalid target data allocation kind or requested allocator not "
           "implemented yet\n");
    return NULL;
  }

  return Manager->dataAlloc(DeviceId, Size, HstPtr);
}

int32_t __tgt_rtl_data_submit(int32_t DeviceId, void *TgtPtr, void *HstPtr,
                              int64_t Size) {
  return Manager->dataSubmit(DeviceId, TgtPtr, HstPtr, Size);
}

int32_t __tgt_rtl_data_retrieve(int32_t DeviceId, void *HstPtr, void *TgtPtr,
                                int64_t Size) {
  return Manager->dataRetrieve(DeviceId, HstPtr, TgtPtr, Size);
}

int32_t __tgt_rtl_data_delete(int32_t DeviceId, void *TgtPtr, int32_t) {
  return Manager->dataDelete(DeviceId, TgtPtr);
}

int32_t __tgt_rtl_data_exchange(int32_t SrcDevId, void *SrcPtr,
                                int32_t DstDevId, void *DstPtr, int64_t Size) {
  return Manager->dataExchange(SrcDevId, SrcPtr, DstDevId, DstPtr, Size);
}


int32_t __tgt_rtl_run_target_region(int32_t DeviceId, void *TgtEntryPtr,
                                    void **TgtArgs, ptrdiff_t *TgtOffsets,
                                    int32_t ArgNum) {
  return Manager->runTargetRegion(DeviceId, TgtEntryPtr, TgtArgs, TgtOffsets,
                                  ArgNum);
}

int32_t __tgt_rtl_run_target_team_region(int32_t DeviceId, void *TgtEntryPtr,
                                         void **TgtArgs, ptrdiff_t *TgtOffsets,
                                         int32_t ArgNum, int32_t TeamNum,
                                         int32_t ThreadLimit,
                                         uint64_t LoopTripCount) {
  return Manager->runTargetTeamRegion(DeviceId, TgtEntryPtr, TgtArgs,
                                      TgtOffsets, ArgNum, TeamNum, ThreadLimit,
                                      LoopTripCount);
}

// Exposed library API function
#ifdef __cplusplus
}
#endif
