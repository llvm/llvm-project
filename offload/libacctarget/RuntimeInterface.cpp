//===- AccEntryCommonImpl.cpp -----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "DeviceManager.h"
#include "Interface.h"
#include "Private.h"
#include "QueueManager.h"
#include "Shared/Debug.h"
#include "include/openacc.h"

using namespace llvm::acc::target;

extern "C" {
int acc_get_num_devices(acc_device_t DevType) {
  return DM->getNumDevices(DevType);
}
int acc_get_num_devices_(acc_device_t *DevType) {
  return acc_get_num_devices(*DevType);
}

int acc_get_device_num(acc_device_t DevType) {
  return DM->getDeviceId(DevType);
}
int acc_get_device_num_(acc_device_t *DevType) {
  return acc_get_device_num(*DevType);
}

void acc_set_device_num(int DevNum, acc_device_t DevType) {
  __tgt_acc_set_device_num(nullptr, 0, DevType, DevNum);
}
void acc_set_device_num_(int *DevNum, acc_device_t *DevType) {
  acc_set_device_num(*DevNum, *DevType);
}

void acc_set_device_type(acc_device_t DevType) {
  __tgt_acc_set_device_type(nullptr, 0, DevType);
}
void acc_set_device_type_(acc_device_t *DevType) {
  acc_set_device_type(*DevType);
}

void acc_set_device(acc_device_t DevType) {
  __tgt_acc_set_device_type(nullptr, 0, DevType);
}
void acc_set_device_(acc_device_t *DevType) { acc_set_device(*DevType); }

acc_device_t acc_get_device_type(void) { return DM->getDeviceType(); }
acc_device_t acc_get_device_type_(void) { return acc_get_device_type(); }

acc_device_t acc_get_device(void) { return DM->getDeviceType(); }
acc_device_t acc_get_device_(void) { return acc_get_device(); }

size_t acc_get_property(int DevNum, acc_device_t DevType,
                        acc_device_property_t Prop) {
  return DM->getDeviceProperty(DevNum, DevType, Prop);
}
size_t acc_get_property_(int *DevNum, acc_device_t *DevType,
                         acc_device_property_t *Prop) {
  return acc_get_property(*DevNum, *DevType, *Prop);
}

const char *acc_get_property_string(int DevNum, acc_device_t DevType,
                                    acc_device_property_t Prop) {
  return DM->getDevicePropertyString(DevNum, DevType, Prop);
}
const char *acc_get_property_string_(int *DevNum, acc_device_t *DevType,
                                     acc_device_property_t *Prop) {
  return acc_get_property_string(*DevNum, *DevType, *Prop);
}

void acc_async_wait(int WaitArg) {
  accAsyncWait(nullptr, DM->getPMDeviceId(), WaitArg);
}
void acc_async_wait_(int *WaitArg) { acc_async_wait(*WaitArg); }

void acc_wait_async(int WaitArg) {
  accAsyncWait(nullptr, DM->getPMDeviceId(), WaitArg);
}
void acc_wait_async_(int *WaitArg) { acc_wait_async(*WaitArg); }

void acc_wait(int WaitArg) {
  accAsyncWait(nullptr, DM->getPMDeviceId(), WaitArg);
}
void acc_wait_(int *WaitArg) { acc_wait(*WaitArg); }

void acc_wait_device(int WaitArg, int DevNum) {
  accAsyncWait(nullptr, DevNum, WaitArg);
}
void acc_wait_device_(int *WaitArg, int *DevNum) {
  acc_wait_device(*WaitArg, *DevNum);
}

void acc_wait_all_async() { accAsyncWaitAll(nullptr); }
void acc_wait_all_async_() { acc_wait_all_async(); }

void acc_async_wait_all() { accAsyncWaitAll(nullptr); }
void acc_async_wait_all_() { acc_async_wait_all(); }

void acc_wait_all() { accAsyncWaitAll(nullptr); }
void acc_wait_all_() { acc_wait_all(); }

void acc_wait_all_device(int DevNum) { accAsyncWaitAll(nullptr, DevNum); }
void acc_wait_all_device_(int *DevNum) { acc_wait_all_device(*DevNum); }

int acc_wait_any(int Count, int *WaitNum) {
  REPORT_FATAL() << "acc_wait_any not yet implemented.";
  return 0;
}
int acc_wait_any_(int *Count, int **WaitNum) {
  return acc_wait_any(*Count, *WaitNum);
}

int acc_wait_any_device(int Count, int *WaitNum, int DevNum) {
  REPORT_FATAL() << "acc_wait_any_device not yet implemented.";
  return 0;
}
int acc_wait_any_device_(int *Count, int **WaitNum, int *DevNum) {
  return acc_wait_any_device(*Count, *WaitNum, *DevNum);
}

void acc_set_default_async(int Async) {
  __tgt_acc_set_default_async(nullptr, Async);
}
void acc_set_default_async_(int *Async) { acc_set_default_async(*Async); }

int acc_get_default_async(void) { return icv::AccDefaultAsyncVar; }
int acc_get_default_async_(void) { return acc_get_default_async(); }

int acc_async_test(int TestArg) {
  return !accAsyncTest(nullptr, DM->getPMDeviceId(), TestArg);
};
int acc_async_test_(int *WaitArg) { return acc_async_test(*WaitArg); }

int acc_async_test_device(int DevNum, int WaitArg) {
  return !accAsyncTest(nullptr, DevNum, WaitArg);
}
int acc_async_test_device_(int *DevNum, int *WaitArg) {
  return acc_async_test_device(*DevNum, *WaitArg);
}

int acc_async_test_all(void) { return !accAsyncTestAll(nullptr); }
int acc_async_test_all_(void) { return acc_async_test_all(); }

int acc_async_test_all_device(int DevNum) {
  return !accAsyncTestAll(nullptr, DevNum);
}
int acc_async_test_all_device_(int *DevNum) {
  return acc_async_test_all_device(*DevNum);
}

void acc_init(acc_device_t DevType) { __tgt_acc_init(nullptr, 0, DevType, -1); }
void acc_init_(acc_device_t *DevType) { acc_init(*DevType); }

void acc_init_device(int DevNum, acc_device_t DevType) {
  __tgt_acc_init(nullptr, 0, DevType, DevNum);
}
void acc_init_device_(int *DevNum, acc_device_t *DevType) {
  acc_init_device(*DevNum, *DevType);
}

void acc_shutdown(acc_device_t DevType) {
  __tgt_acc_shutdown(nullptr, 0, DevType, -1);
}
void acc_shutdown_(acc_device_t *DevType) { acc_shutdown(*DevType); }

void acc_shutdown_device(int DevNum, acc_device_t DevType) {
  __tgt_acc_shutdown(nullptr, 0, DevType, DevNum);
}
void acc_shutdown_device_(int *DevNum, acc_device_t *DevType) {
  acc_shutdown_device(*DevNum, *DevType);
}

void acc_free(void *DataDev) { accFree(DataDev); }
void acc_free_(void **DataDev) { acc_free(*DataDev); }

void *acc_malloc(size_t Bytes) { return accAlloc(Bytes); }
void *acc_malloc_(size_t *Bytes) { return acc_malloc(*Bytes); }

void acc_map_data(void *DataArg, void *DataDev, size_t Bytes) {
  accMapData(DataArg, DataDev, Bytes);
}
void acc_map_data_(void **DataArg, void **DataDev, size_t *Bytes) {
  acc_map_data(*DataArg, *DataDev, *Bytes);
}

void acc_unmap_data(void *DataArg) { accUnmapData(DataArg); }
void acc_unmap_data_(void **DataArg) { acc_unmap_data(*DataArg); }

void *acc_deviceptr(void *DataArg) {
  return __tgt_acc_get_deviceptr(nullptr, DataArg, 0, DataArg);
}
void *acc_deviceptr_(void **DataArg) { return acc_deviceptr(*DataArg); }

void *acc_hostptr(void *DataDev) {
  REPORT_FATAL() << "acc_hostptr not yet implemented";
  return nullptr;
}
void *acc_hostptr_(void **DataDev) { return acc_hostptr(*DataDev); }

void acc_memcpy_from_device(void *DataHostDest, void *DataDevSrc,
                            size_t Bytes) {
  accMemcpyFromDevice(DataHostDest, DataDevSrc, Bytes);
}
void acc_memcpy_from_device_(void **DataHostDest, void **DataDevSrc,
                             size_t *Bytes) {
  acc_memcpy_from_device(*DataHostDest, *DataDevSrc, *Bytes);
}

void acc_memcpy_to_device(void *DataDevDest, void *DataHostSrc, size_t Bytes) {
  accMemcpyToDevice(DataDevDest, DataHostSrc, Bytes);
}
void acc_memcpy_to_device_(void **DataDevDest, void **DataHostSrc,
                           size_t *Bytes) {
  acc_memcpy_to_device(*DataDevDest, *DataHostSrc, *Bytes);
}

void acc_memcpy_d2d(void *DataDevDest, void *DataHostSrc, size_t Bytes,
                    int DevNumDest, int DevNumSrc) {
  accMemcpyD2D(DataDevDest, DataHostSrc, Bytes, DevNumDest, DevNumSrc);
}
void acc_memcpy_d2d_(void **DataDevDest, void **DataHostSrc, size_t *Bytes,
                     int *DevNumDest, int *DevNumSrc) {
  acc_memcpy_d2d(*DataDevDest, *DataHostSrc, *Bytes, *DevNumDest, *DevNumSrc);
}

int acc_on_device(acc_device_t DevType) { return DevType == acc_device_host; }
int acc_on_device_(acc_device_t *DevType) { return acc_on_device(*DevType); }

void acc_present_dump_all() {
  REPORT_WARN() << "acc_present_dump_all not yet implemented";
}
void acc_present_dump_all_() { acc_present_dump_all(); }

void acc_attach_dump_all() {
  REPORT_WARN() << "acc_attach_dump_all not yet implemented";
}
void acc_attach_dump_all_() { acc_attach_dump_all(); }

void acc_attach_dump() {
  REPORT_WARN() << "acc_attach_dump not yet implemented";
}
void acc_attach_dump_() { acc_attach_dump(); }
}
