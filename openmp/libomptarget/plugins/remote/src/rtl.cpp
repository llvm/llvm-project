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

  RPCConfig Config;
  parseEnvironment(Config);

  int Timeout = 5;
  if (const char *Env1 = std::getenv("LIBOMPTARGET_RPC_LATENCY"))
    Timeout = std::stoi(Env1);

  Manager = new RemoteClientManager(Config.ServerAddresses, Timeout,
                                    Config.MaxSize, Config.BlockSize);
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

int32_t __tgt_rtl_synchronize(int32_t DeviceId, __tgt_async_info *AsyncInfo) {
  return Manager->synchronize(DeviceId, AsyncInfo);
}

int32_t __tgt_rtl_is_data_exchangable(int32_t SrcDevId, int32_t DstDevId) {
  return Manager->isDataExchangeable(SrcDevId, DstDevId);
}

void *__tgt_rtl_data_alloc(int32_t DeviceId, int64_t Size, void *HstPtr,
                           int32_t kind) {
  if (kind != TARGET_ALLOC_DEFAULT) {
    REPORT("Invalid target data allocation kind or requested allocator not "
           "implemented yet\n");
    return NULL;
  }

  return Manager->dataAlloc(DeviceId, Size, HstPtr);
}

int32_t __tgt_rtl_data_submit(int32_t DeviceId, void *TgtPtr, void *HstPtr,
                              int64_t Size) {
  return Manager->dataSubmitAsync(DeviceId, TgtPtr, HstPtr, Size, nullptr);
}

int32_t __tgt_rtl_data_submit_async(int32_t DeviceId, void *TgtPtr,
                                    void *HstPtr, int64_t Size,
                                    __tgt_async_info *AsyncInfo) {
  return Manager->dataSubmitAsync(DeviceId, TgtPtr, HstPtr, Size, AsyncInfo);
}

int32_t __tgt_rtl_data_retrieve(int32_t DeviceId, void *HstPtr, void *TgtPtr,
                                int64_t Size) {
  return Manager->dataRetrieveAsync(DeviceId, HstPtr, TgtPtr, Size, nullptr);
}

int32_t __tgt_rtl_data_retrieve_async(int32_t DeviceId, void *HstPtr,
                                      void *TgtPtr, int64_t Size,
                                      __tgt_async_info *AsyncInfo) {
  return Manager->dataRetrieveAsync(DeviceId, HstPtr, TgtPtr, Size, AsyncInfo);
}

int32_t __tgt_rtl_data_delete(int32_t DeviceId, void *TgtPtr) {
  return Manager->dataDelete(DeviceId, TgtPtr);
}

int32_t __tgt_rtl_data_exchange(int32_t SrcDevId, void *SrcPtr,
                                int32_t DstDevId, void *DstPtr, int64_t Size) {
  return Manager->dataExchangeAsync(SrcDevId, SrcPtr, DstDevId, DstPtr, Size,
                                    nullptr);
}

int32_t __tgt_rtl_data_exchange_async(int32_t SrcDevId, void *SrcPtr,
                                      int32_t DstDevId, void *DstPtr,
                                      int64_t Size,
                                      __tgt_async_info *AsyncInfo) {
  return Manager->dataExchangeAsync(SrcDevId, SrcPtr, DstDevId, DstPtr, Size,
                                    AsyncInfo);
}

int32_t __tgt_rtl_run_target_region(int32_t DeviceId, void *TgtEntryPtr,
                                    void **TgtArgs, ptrdiff_t *TgtOffsets,
                                    int32_t ArgNum) {
  return Manager->runTargetRegionAsync(DeviceId, TgtEntryPtr, TgtArgs,
                                       TgtOffsets, ArgNum, nullptr);
}

int32_t __tgt_rtl_run_target_region_async(int32_t DeviceId, void *TgtEntryPtr,
                                          void **TgtArgs, ptrdiff_t *TgtOffsets,
                                          int32_t ArgNum,
                                          __tgt_async_info *AsyncInfo) {
  return Manager->runTargetRegionAsync(DeviceId, TgtEntryPtr, TgtArgs,
                                       TgtOffsets, ArgNum, AsyncInfo);
}

int32_t __tgt_rtl_run_target_team_region(int32_t DeviceId, void *TgtEntryPtr,
                                         void **TgtArgs, ptrdiff_t *TgtOffsets,
                                         int32_t ArgNum, int32_t TeamNum,
                                         int32_t ThreadLimit,
                                         uint64_t LoopTripCount) {
  return Manager->runTargetTeamRegionAsync(DeviceId, TgtEntryPtr, TgtArgs,
                                           TgtOffsets, ArgNum, TeamNum,
                                           ThreadLimit, LoopTripCount, nullptr);
}

int32_t __tgt_rtl_run_target_team_region_async(
    int32_t DeviceId, void *TgtEntryPtr, void **TgtArgs, ptrdiff_t *TgtOffsets,
    int32_t ArgNum, int32_t TeamNum, int32_t ThreadLimit,
    uint64_t LoopTripCount, __tgt_async_info *AsyncInfo) {
  return Manager->runTargetTeamRegionAsync(
      DeviceId, TgtEntryPtr, TgtArgs, TgtOffsets, ArgNum, TeamNum, ThreadLimit,
      LoopTripCount, AsyncInfo);
}

// Exposed library API function
#ifdef __cplusplus
}
#endif
