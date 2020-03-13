//===----------- api.cpp - Target independent OpenMP target RTL -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implementation of OpenMP API interface functions.
//
//===----------------------------------------------------------------------===//

#include <omptarget.h>

#include "device.h"
#include "private.h"
#include "rtl.h"

#include <climits>
#include <cstring>
#include <cstdlib>

EXTERN int omp_get_num_devices(void) {
  RTLsMtx->lock();
  size_t Devices_size = Devices.size();
  RTLsMtx->unlock();

  DP("Call to omp_get_num_devices returning %zd\n", Devices_size);

  return Devices_size;
}

EXTERN int omp_get_initial_device(void) {
  DP("Call to omp_get_initial_device returning %d\n", HOST_DEVICE);
  return HOST_DEVICE;
}

EXTERN void *omp_target_alloc(size_t size, int device_num) {
  DP("Call to omp_target_alloc for device %d requesting %zu bytes\n",
      device_num, size);

  if (size <= 0) {
    DP("Call to omp_target_alloc with non-positive length\n");
    return NULL;
  }

  void *rc = NULL;

  if (device_num == omp_get_initial_device()) {
    rc = malloc(size);
    DP("omp_target_alloc returns host ptr " DPxMOD "\n", DPxPTR(rc));
    return rc;
  }

  if (!device_is_ready(device_num)) {
    DP("omp_target_alloc returns NULL ptr\n");
    return NULL;
  }

  DeviceTy &Device = Devices[device_num];
  rc = Device.RTL->data_alloc(Device.RTLDeviceID, size, NULL);
  DP("omp_target_alloc returns device ptr " DPxMOD "\n", DPxPTR(rc));
  return rc;
}

EXTERN void omp_target_free(void *device_ptr, int device_num) {
  DP("Call to omp_target_free for device %d and address " DPxMOD "\n",
      device_num, DPxPTR(device_ptr));

  if (!device_ptr) {
    DP("Call to omp_target_free with NULL ptr\n");
    return;
  }

  if (device_num == omp_get_initial_device()) {
    free(device_ptr);
    DP("omp_target_free deallocated host ptr\n");
    return;
  }

  if (!device_is_ready(device_num)) {
    DP("omp_target_free returns, nothing to do\n");
    return;
  }

  DeviceTy &Device = Devices[device_num];
  Device.RTL->data_delete(Device.RTLDeviceID, (void *)device_ptr);
  DP("omp_target_free deallocated device ptr\n");
}

EXTERN int omp_target_is_present(void *ptr, int device_num) {
  DP("Call to omp_target_is_present for device %d and address " DPxMOD "\n",
      device_num, DPxPTR(ptr));

  if (!ptr) {
    DP("Call to omp_target_is_present with NULL ptr, returning false\n");
    return false;
  }

  if (device_num == omp_get_initial_device()) {
    DP("Call to omp_target_is_present on host, returning true\n");
    return true;
  }

  RTLsMtx->lock();
  size_t Devices_size = Devices.size();
  RTLsMtx->unlock();
  if (Devices_size <= (size_t)device_num) {
    DP("Call to omp_target_is_present with invalid device ID, returning "
        "false\n");
    return false;
  }

  DeviceTy& Device = Devices[device_num];
  bool IsLast; // not used
  bool IsHostPtr;
  void *TgtPtr = Device.getTgtPtrBegin(ptr, 0, IsLast, false, IsHostPtr);
  int rc = (TgtPtr != NULL);
  // Under unified memory the host pointer can be returned by the
  // getTgtPtrBegin() function which means that there is no device
  // corresponding point for ptr. This function should return false
  // in that situation.
  if (RTLs->RequiresFlags & OMP_REQ_UNIFIED_SHARED_MEMORY)
    rc = !IsHostPtr;
  DP("Call to omp_target_is_present returns %d\n", rc);
  return rc;
}

EXTERN int omp_target_memcpy(void *dst, void *src, size_t length,
    size_t dst_offset, size_t src_offset, int dst_device, int src_device) {
  DP("Call to omp_target_memcpy, dst device %d, src device %d, "
      "dst addr " DPxMOD ", src addr " DPxMOD ", dst offset %zu, "
      "src offset %zu, length %zu\n", dst_device, src_device, DPxPTR(dst),
      DPxPTR(src), dst_offset, src_offset, length);

  if (!dst || !src || length <= 0) {
    DP("Call to omp_target_memcpy with invalid arguments\n");
    return OFFLOAD_FAIL;
  }

  if (src_device != omp_get_initial_device() && !device_is_ready(src_device)) {
      DP("omp_target_memcpy returns OFFLOAD_FAIL\n");
      return OFFLOAD_FAIL;
  }

  if (dst_device != omp_get_initial_device() && !device_is_ready(dst_device)) {
      DP("omp_target_memcpy returns OFFLOAD_FAIL\n");
      return OFFLOAD_FAIL;
  }

  int rc = OFFLOAD_SUCCESS;
  void *srcAddr = (char *)src + src_offset;
  void *dstAddr = (char *)dst + dst_offset;

  if (src_device == omp_get_initial_device() &&
      dst_device == omp_get_initial_device()) {
    DP("copy from host to host\n");
    const void *p = memcpy(dstAddr, srcAddr, length);
    if (p == NULL)
      rc = OFFLOAD_FAIL;
  } else if (src_device == omp_get_initial_device()) {
    DP("copy from host to device\n");
    DeviceTy& DstDev = Devices[dst_device];
    rc = DstDev.data_submit(dstAddr, srcAddr, length);
  } else if (dst_device == omp_get_initial_device()) {
    DP("copy from device to host\n");
    DeviceTy& SrcDev = Devices[src_device];
    rc = SrcDev.data_retrieve(dstAddr, srcAddr, length);
  } else {
    DP("copy from device to device\n");
    void *buffer = malloc(length);
    DeviceTy& SrcDev = Devices[src_device];
    DeviceTy& DstDev = Devices[dst_device];
    rc = SrcDev.data_retrieve(buffer, srcAddr, length);
    if (rc == OFFLOAD_SUCCESS)
      rc = DstDev.data_submit(dstAddr, buffer, length);
    free(buffer);
  }

  DP("omp_target_memcpy returns %d\n", rc);
  return rc;
}

EXTERN int omp_target_memcpy_rect(void *dst, void *src, size_t element_size,
    int num_dims, const size_t *volume, const size_t *dst_offsets,
    const size_t *src_offsets, const size_t *dst_dimensions,
    const size_t *src_dimensions, int dst_device, int src_device) {
  DP("Call to omp_target_memcpy_rect, dst device %d, src device %d, "
      "dst addr " DPxMOD ", src addr " DPxMOD ", dst offsets " DPxMOD ", "
      "src offsets " DPxMOD ", dst dims " DPxMOD ", src dims " DPxMOD ", "
      "volume " DPxMOD ", element size %zu, num_dims %d\n", dst_device,
      src_device, DPxPTR(dst), DPxPTR(src), DPxPTR(dst_offsets),
      DPxPTR(src_offsets), DPxPTR(dst_dimensions), DPxPTR(src_dimensions),
      DPxPTR(volume), element_size, num_dims);

  if (!(dst || src)) {
    DP("Call to omp_target_memcpy_rect returns max supported dimensions %d\n",
        INT_MAX);
    return INT_MAX;
  }

  if (!dst || !src || element_size < 1 || num_dims < 1 || !volume ||
      !dst_offsets || !src_offsets || !dst_dimensions || !src_dimensions) {
    DP("Call to omp_target_memcpy_rect with invalid arguments\n");
    return OFFLOAD_FAIL;
  }

  int rc;
  if (num_dims == 1) {
    rc = omp_target_memcpy(dst, src, element_size * volume[0],
        element_size * dst_offsets[0], element_size * src_offsets[0],
        dst_device, src_device);
  } else {
    size_t dst_slice_size = element_size;
    size_t src_slice_size = element_size;
    for (int i=1; i<num_dims; ++i) {
      dst_slice_size *= dst_dimensions[i];
      src_slice_size *= src_dimensions[i];
    }

    size_t dst_off = dst_offsets[0] * dst_slice_size;
    size_t src_off = src_offsets[0] * src_slice_size;
    for (size_t i=0; i<volume[0]; ++i) {
      rc = omp_target_memcpy_rect((char *) dst + dst_off + dst_slice_size * i,
          (char *) src + src_off + src_slice_size * i, element_size,
          num_dims - 1, volume + 1, dst_offsets + 1, src_offsets + 1,
          dst_dimensions + 1, src_dimensions + 1, dst_device, src_device);

      if (rc) {
        DP("Recursive call to omp_target_memcpy_rect returns unsuccessfully\n");
        return rc;
      }
    }
  }

  DP("omp_target_memcpy_rect returns %d\n", rc);
  return rc;
}

EXTERN int omp_target_associate_ptr(void *host_ptr, void *device_ptr,
    size_t size, size_t device_offset, int device_num) {
  DP("Call to omp_target_associate_ptr with host_ptr " DPxMOD ", "
      "device_ptr " DPxMOD ", size %zu, device_offset %zu, device_num %d\n",
      DPxPTR(host_ptr), DPxPTR(device_ptr), size, device_offset, device_num);

  if (!host_ptr || !device_ptr || size <= 0) {
    DP("Call to omp_target_associate_ptr with invalid arguments\n");
    return OFFLOAD_FAIL;
  }

  if (device_num == omp_get_initial_device()) {
    DP("omp_target_associate_ptr: no association possible on the host\n");
    return OFFLOAD_FAIL;
  }

  if (!device_is_ready(device_num)) {
    DP("omp_target_associate_ptr returns OFFLOAD_FAIL\n");
    return OFFLOAD_FAIL;
  }

  DeviceTy& Device = Devices[device_num];
  void *device_addr = (void *)((uint64_t)device_ptr + (uint64_t)device_offset);
  int rc = Device.associatePtr(host_ptr, device_addr, size);
  DP("omp_target_associate_ptr returns %d\n", rc);
  return rc;
}

EXTERN int omp_target_disassociate_ptr(void *host_ptr, int device_num) {
  DP("Call to omp_target_disassociate_ptr with host_ptr " DPxMOD ", "
      "device_num %d\n", DPxPTR(host_ptr), device_num);

  if (!host_ptr) {
    DP("Call to omp_target_associate_ptr with invalid host_ptr\n");
    return OFFLOAD_FAIL;
  }

  if (device_num == omp_get_initial_device()) {
    DP("omp_target_disassociate_ptr: no association possible on the host\n");
    return OFFLOAD_FAIL;
  }

  if (!device_is_ready(device_num)) {
    DP("omp_target_disassociate_ptr returns OFFLOAD_FAIL\n");
    return OFFLOAD_FAIL;
  }

  DeviceTy& Device = Devices[device_num];
  int rc = Device.disassociatePtr(host_ptr);
  DP("omp_target_disassociate_ptr returns %d\n", rc);
  return rc;
}
