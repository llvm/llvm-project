//===- AccAPIImpl.cpp - OpenACC runtime API implementation -------- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implementation of the OpenACC runtime routines.
//
//===----------------------------------------------------------------------===//

#include "Debug.h"
#include "DeviceManager.h"

#include "ISO_Fortran_binding.h"
#include "openacc.h"

#include <cstdint>

namespace llvm::acc::impl::f {
#include "AccFImplDecls.inc"
} // namespace llvm::acc::impl::f

using namespace llvm::acc::target;

/// Report that this runtime API routine is not yet implemented.
#define ACC_API_NYI() ACC_NYI() << __FUNCTION__

namespace llvm::acc::impl::f {

bool acc_async_test(int64_t wait_arg) { ACC_API_NYI(); }

bool acc_async_test_all(void) { ACC_API_NYI(); }

bool acc_async_test_all_device(int dev_num) { ACC_API_NYI(); }

bool acc_async_test_device(int64_t wait_arg, int dev_num) { ACC_API_NYI(); }

void acc_copyin(void *data_arg, size_t bytes) { ACC_API_NYI(); }

void acc_copyin_async(void *data_arg, size_t bytes, int64_t async_arg) {
  ACC_API_NYI();
}

void acc_copyout(void *data_arg, size_t bytes) { ACC_API_NYI(); }

void acc_copyout_async(void *data_arg, size_t bytes, int64_t async_arg) {
  ACC_API_NYI();
}

void acc_copyout_finalize(void *data_arg, size_t bytes) { ACC_API_NYI(); }

void acc_copyout_finalize_async(void *data_arg, size_t bytes,
                                int64_t async_arg) {
  ACC_API_NYI();
}

void acc_create(void *data_arg, size_t bytes) { ACC_API_NYI(); }

void acc_create_async(void *data_arg, size_t bytes, int64_t async_arg) {
  ACC_API_NYI();
}

void acc_delete(void *data_arg, size_t bytes) { ACC_API_NYI(); }

void acc_delete_async(void *data_arg, size_t bytes, int64_t async_arg) {
  ACC_API_NYI();
}

void acc_delete_finalize(void *data_arg, size_t bytes) { ACC_API_NYI(); }

void acc_delete_finalize_async(void *data_arg, size_t bytes,
                               int64_t async_arg) {
  ACC_API_NYI();
}

void *acc_deviceptr(void *data_arg) { ACC_API_NYI(); }

void acc_free(void *data_dev) { ACC_API_NYI(); }

int64_t acc_get_default_async(void) { ACC_API_NYI(); }

int acc_get_device_num(acc_device_t dev_type) {
  return DM->getDeviceNum(dev_type);
}

acc_device_t acc_get_device_type(void) { return DM->getDeviceType(); }

int acc_get_num_devices(acc_device_t dev_type) {
  return DM->getNumDevices(dev_type);
}

size_t acc_get_property(int dev_num, acc_device_t dev_type,
                        acc_device_property_t property) {
  ACC_API_NYI();
}

void acc_get_property_string(int dev_num, acc_device_t dev_type,
                             acc_device_property_t property,
                             CFI_cdesc_t *string) {
  ACC_API_NYI();
}

void *acc_hostptr(void *data_dev) { ACC_API_NYI(); }

void acc_init(acc_device_t dev_type) { ACC_API_NYI(); }

void acc_init_device(int dev_num, acc_device_t dev_type) { ACC_API_NYI(); }

bool acc_is_present(void *data_arg, size_t bytes) { ACC_API_NYI(); }

void *acc_malloc(size_t bytes) { ACC_API_NYI(); }

void acc_map_data(void *data_arg, void *data_dev, size_t bytes) {
  ACC_API_NYI();
}

void acc_memcpy_d2d(void *data_arg_dest, void *data_arg_src, size_t bytes,
                    int dev_num_dest, int dev_num_src) {
  ACC_API_NYI();
}

void acc_memcpy_d2d_async(void *data_arg_dest, void *data_arg_src, size_t bytes,
                          int dev_num_dest, int dev_num_src,
                          int64_t async_arg_src) {
  ACC_API_NYI();
}

void acc_memcpy_device(void *data_dev_dest, void *data_dev_src, size_t bytes) {
  ACC_API_NYI();
}

void acc_memcpy_device_async(void *data_dev_dest, void *data_dev_src,
                             size_t bytes, int64_t async_arg) {
  ACC_API_NYI();
}

void acc_memcpy_from_device(void *data_host_dest, void *data_dev_src,
                            size_t bytes) {
  ACC_API_NYI();
}

void acc_memcpy_from_device_async(void *data_host_dest, void *data_dev_src,
                                  size_t bytes, int64_t async_arg) {
  ACC_API_NYI();
}

void acc_memcpy_to_device(void *data_dev_dest, void *data_host_src,
                          size_t bytes) {
  ACC_API_NYI();
}

void acc_memcpy_to_device_async(void *data_dev_dest, void *data_host_src,
                                size_t bytes, int64_t async_arg) {
  ACC_API_NYI();
}

bool acc_on_device(acc_device_t dev_type) { ACC_API_NYI(); }

void acc_set_default_async(int64_t async_arg) { ACC_API_NYI(); }

void acc_set_device_num(int dev_num, acc_device_t dev_type) {
  DM->setDeviceNum(dev_type, dev_num);
}

void acc_set_device_type(acc_device_t dev_type) { DM->setDeviceType(dev_type); }

void acc_shutdown(acc_device_t dev_type) { ACC_API_NYI(); }

void acc_shutdown_device(int dev_num, acc_device_t dev_type) { ACC_API_NYI(); }

void acc_unmap_data(void *data_arg) { ACC_API_NYI(); }

void acc_update_device(void *data_arg, size_t bytes) { ACC_API_NYI(); }

void acc_update_device_async(void *data_arg, size_t bytes, int64_t async_arg) {
  ACC_API_NYI();
}

void acc_update_self(void *data_arg, size_t bytes) { ACC_API_NYI(); }

void acc_update_self_async(void *data_arg, size_t bytes, int64_t async_arg) {
  ACC_API_NYI();
}

void acc_wait(int64_t wait_arg) { ACC_API_NYI(); }

void acc_wait_all(void) { ACC_API_NYI(); }

void acc_wait_all_async(int64_t async_arg) { ACC_API_NYI(); }

void acc_wait_all_device(int dev_num) { ACC_API_NYI(); }

void acc_wait_all_device_async(int64_t async_arg, int dev_num) {
  ACC_API_NYI();
}

int acc_wait_any(int count, int64_t *wait_arg) { ACC_API_NYI(); }

int acc_wait_any_device(int count, int64_t *wait_arg, int dev_num) {
  ACC_API_NYI();
}

void acc_wait_async(int64_t wait_arg, int64_t async_arg) { ACC_API_NYI(); }

void acc_wait_device(int64_t wait_arg, int dev_num) { ACC_API_NYI(); }

void acc_wait_device_async(int64_t wait_arg, int64_t async_arg, int dev_num) {
  ACC_API_NYI();
}

void cfi_acc_attach(const CFI_cdesc_t *ptr_addr) { ACC_API_NYI(); }

void cfi_acc_attach_async(const CFI_cdesc_t *ptr_addr, int64_t async_arg) {
  ACC_API_NYI();
}

void cfi_acc_copyin(const CFI_cdesc_t *data_arg) { ACC_API_NYI(); }

void cfi_acc_copyin_async(const CFI_cdesc_t *data_arg, int64_t async_arg) {
  ACC_API_NYI();
}

void cfi_acc_copyout(const CFI_cdesc_t *data_arg) { ACC_API_NYI(); }

void cfi_acc_copyout_async(const CFI_cdesc_t *data_arg, int64_t async_arg) {
  ACC_API_NYI();
}

void cfi_acc_copyout_finalize(const CFI_cdesc_t *data_arg) { ACC_API_NYI(); }

void cfi_acc_copyout_finalize_async(const CFI_cdesc_t *data_arg,
                                    int64_t async_arg) {
  ACC_API_NYI();
}

void cfi_acc_create(const CFI_cdesc_t *data_arg) { ACC_API_NYI(); }

void cfi_acc_create_async(const CFI_cdesc_t *data_arg, int64_t async_arg) {
  ACC_API_NYI();
}

void cfi_acc_delete(const CFI_cdesc_t *data_arg) { ACC_API_NYI(); }

void cfi_acc_delete_async(const CFI_cdesc_t *data_arg, int64_t async_arg) {
  ACC_API_NYI();
}

void cfi_acc_delete_finalize(const CFI_cdesc_t *data_arg) { ACC_API_NYI(); }

void cfi_acc_delete_finalize_async(const CFI_cdesc_t *data_arg,
                                   int64_t async_arg) {
  ACC_API_NYI();
}

void cfi_acc_detach(const CFI_cdesc_t *ptr_addr) { ACC_API_NYI(); }

void cfi_acc_detach_async(const CFI_cdesc_t *ptr_addr, int64_t async_arg) {
  ACC_API_NYI();
}

void cfi_acc_detach_finalize(const CFI_cdesc_t *ptr_addr) { ACC_API_NYI(); }

void cfi_acc_detach_finalize_async(const CFI_cdesc_t *ptr_addr,
                                   int64_t async_arg) {
  ACC_API_NYI();
}

bool cfi_acc_is_present(const CFI_cdesc_t *data_arg) { ACC_API_NYI(); }

void cfi_acc_update_device(const CFI_cdesc_t *data_arg) { ACC_API_NYI(); }

void cfi_acc_update_device_async(const CFI_cdesc_t *data_arg,
                                 int64_t async_arg) {
  ACC_API_NYI();
}

void cfi_acc_update_self(const CFI_cdesc_t *data_arg) { ACC_API_NYI(); }

void cfi_acc_update_self_async(const CFI_cdesc_t *data_arg, int64_t async_arg) {
  ACC_API_NYI();
}

} // namespace llvm::acc::impl::f
