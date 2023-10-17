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

#include "device.h"
#include "omptarget.h"
#include "private.h"
#include "rtl.h"

#include "llvm/ADT/SmallVector.h"

#include <climits>
#include <cstdlib>
#include <cstring>
#include <mutex>

EXTERN int omp_get_num_devices(void) {
  TIMESCOPE();
  PM->RTLsMtx.lock();
  size_t DevicesSize = PM->Devices.size();
  PM->RTLsMtx.unlock();

  DP("Call to omp_get_num_devices returning %zd\n", DevicesSize);

  return DevicesSize;
}

EXTERN int omp_get_device_num(void) {
  TIMESCOPE();
  int HostDevice = omp_get_initial_device();

  DP("Call to omp_get_device_num returning %d\n", HostDevice);

  return HostDevice;
}

EXTERN int omp_get_initial_device(void) {
  TIMESCOPE();
  int HostDevice = omp_get_num_devices();
  DP("Call to omp_get_initial_device returning %d\n", HostDevice);
  return HostDevice;
}

EXTERN void *omp_target_alloc(size_t Size, int DeviceNum) {
  return targetAllocExplicit(Size, DeviceNum, TARGET_ALLOC_DEFAULT, __func__);
}

EXTERN void *llvm_omp_target_alloc_device(size_t Size, int DeviceNum) {
  return targetAllocExplicit(Size, DeviceNum, TARGET_ALLOC_DEVICE, __func__);
}

EXTERN void *llvm_omp_target_alloc_host(size_t Size, int DeviceNum) {
  return targetAllocExplicit(Size, DeviceNum, TARGET_ALLOC_HOST, __func__);
}

EXTERN void *llvm_omp_target_alloc_shared(size_t Size, int DeviceNum) {
  return targetAllocExplicit(Size, DeviceNum, TARGET_ALLOC_SHARED, __func__);
}

EXTERN void omp_target_free(void *Ptr, int DeviceNum) {
  return targetFreeExplicit(Ptr, DeviceNum, TARGET_ALLOC_DEFAULT, __func__);
}

EXTERN void llvm_omp_target_free_device(void *Ptr, int DeviceNum) {
  return targetFreeExplicit(Ptr, DeviceNum, TARGET_ALLOC_DEVICE, __func__);
}

EXTERN void llvm_omp_target_free_host(void *Ptr, int DeviceNum) {
  return targetFreeExplicit(Ptr, DeviceNum, TARGET_ALLOC_HOST, __func__);
}

EXTERN void llvm_omp_target_free_shared(void *Ptre, int DeviceNum) {
  return targetFreeExplicit(Ptre, DeviceNum, TARGET_ALLOC_SHARED, __func__);
}

EXTERN void *llvm_omp_target_dynamic_shared_alloc() { return nullptr; }
EXTERN void *llvm_omp_get_dynamic_shared() { return nullptr; }

EXTERN [[nodiscard]] void *llvm_omp_target_lock_mem(void *Ptr, size_t Size,
                                                    int DeviceNum) {
  return targetLockExplicit(Ptr, Size, DeviceNum, __func__);
}

EXTERN void llvm_omp_target_unlock_mem(void *Ptr, int DeviceNum) {
  targetUnlockExplicit(Ptr, DeviceNum, __func__);
}

EXTERN int omp_target_is_present(const void *Ptr, int DeviceNum) {
  TIMESCOPE();
  DP("Call to omp_target_is_present for device %d and address " DPxMOD "\n",
     DeviceNum, DPxPTR(Ptr));

  if (!Ptr) {
    DP("Call to omp_target_is_present with NULL ptr, returning false\n");
    return false;
  }

  if (DeviceNum == omp_get_initial_device()) {
    DP("Call to omp_target_is_present on host, returning true\n");
    return true;
  }

  PM->RTLsMtx.lock();
  size_t DevicesSize = PM->Devices.size();
  PM->RTLsMtx.unlock();
  if (DevicesSize <= (size_t)DeviceNum) {
    DP("Call to omp_target_is_present with invalid device ID, returning "
       "false\n");
    return false;
  }

  DeviceTy &Device = *PM->Devices[DeviceNum];
  // omp_target_is_present tests whether a host pointer refers to storage that
  // is mapped to a given device. However, due to the lack of the storage size,
  // only check 1 byte. Cannot set size 0 which checks whether the pointer (zero
  // lengh array) is mapped instead of the referred storage.
  TargetPointerResultTy TPR = Device.getTgtPtrBegin(const_cast<void *>(Ptr), 1,
                                                    /*UpdateRefCount=*/false,
                                                    /*UseHoldRefCount=*/false);
  int Rc = TPR.isPresent();
  DP("Call to omp_target_is_present returns %d\n", Rc);
  return Rc;
}

EXTERN int omp_target_memcpy(void *Dst, const void *Src, size_t Length,
                             size_t DstOffset, size_t SrcOffset, int DstDevice,
                             int SrcDevice) {
  TIMESCOPE();
  DP("Call to omp_target_memcpy, dst device %d, src device %d, "
     "dst addr " DPxMOD ", src addr " DPxMOD ", dst offset %zu, "
     "src offset %zu, length %zu\n",
     DstDevice, SrcDevice, DPxPTR(Dst), DPxPTR(Src), DstOffset, SrcOffset,
     Length);

  if (!Dst || !Src || Length <= 0) {
    if (Length == 0) {
      DP("Call to omp_target_memcpy with zero length, nothing to do\n");
      return OFFLOAD_SUCCESS;
    }

    REPORT("Call to omp_target_memcpy with invalid arguments\n");
    return OFFLOAD_FAIL;
  }

  if (SrcDevice != omp_get_initial_device() && !deviceIsReady(SrcDevice)) {
    REPORT("omp_target_memcpy returns OFFLOAD_FAIL\n");
    return OFFLOAD_FAIL;
  }

  if (DstDevice != omp_get_initial_device() && !deviceIsReady(DstDevice)) {
    REPORT("omp_target_memcpy returns OFFLOAD_FAIL\n");
    return OFFLOAD_FAIL;
  }

  int Rc = OFFLOAD_SUCCESS;
  void *SrcAddr = (char *)const_cast<void *>(Src) + SrcOffset;
  void *DstAddr = (char *)Dst + DstOffset;

  if (SrcDevice == omp_get_initial_device() &&
      DstDevice == omp_get_initial_device()) {
    DP("copy from host to host\n");
    const void *P = memcpy(DstAddr, SrcAddr, Length);
    if (P == NULL)
      Rc = OFFLOAD_FAIL;
  } else if (SrcDevice == omp_get_initial_device()) {
    DP("copy from host to device\n");
    DeviceTy &DstDev = *PM->Devices[DstDevice];
    AsyncInfoTy AsyncInfo(DstDev);
    Rc = DstDev.submitData(DstAddr, SrcAddr, Length, AsyncInfo);
  } else if (DstDevice == omp_get_initial_device()) {
    DP("copy from device to host\n");
    DeviceTy &SrcDev = *PM->Devices[SrcDevice];
    AsyncInfoTy AsyncInfo(SrcDev);
    Rc = SrcDev.retrieveData(DstAddr, SrcAddr, Length, AsyncInfo);
  } else {
    DP("copy from device to device\n");
    DeviceTy &SrcDev = *PM->Devices[SrcDevice];
    DeviceTy &DstDev = *PM->Devices[DstDevice];
    // First try to use D2D memcpy which is more efficient. If fails, fall back
    // to unefficient way.
    if (SrcDev.isDataExchangable(DstDev)) {
      AsyncInfoTy AsyncInfo(SrcDev);
      Rc = SrcDev.dataExchange(SrcAddr, DstDev, DstAddr, Length, AsyncInfo);
      if (Rc == OFFLOAD_SUCCESS)
        return OFFLOAD_SUCCESS;
    }

    void *Buffer = malloc(Length);
    {
      AsyncInfoTy AsyncInfo(SrcDev);
      Rc = SrcDev.retrieveData(Buffer, SrcAddr, Length, AsyncInfo);
    }
    if (Rc == OFFLOAD_SUCCESS) {
      AsyncInfoTy AsyncInfo(DstDev);
      Rc = DstDev.submitData(DstAddr, Buffer, Length, AsyncInfo);
    }
    free(Buffer);
  }

  DP("omp_target_memcpy returns %d\n", Rc);
  return Rc;
}

// The helper function that calls omp_target_memcpy or omp_target_memcpy_rect
static int libomp_target_memcpy_async_helper(kmp_int32 Gtid, kmp_task_t *Task) {
  if (Task == nullptr)
    return OFFLOAD_FAIL;

  TargetMemcpyArgsTy *Args = (TargetMemcpyArgsTy *)Task->shareds;

  if (Args == nullptr)
    return OFFLOAD_FAIL;

  // Call blocked version
  int Rc = OFFLOAD_SUCCESS;
  if (Args->IsRectMemcpy) {
    Rc = omp_target_memcpy_rect(
        Args->Dst, Args->Src, Args->ElementSize, Args->NumDims, Args->Volume,
        Args->DstOffsets, Args->SrcOffsets, Args->DstDimensions,
        Args->SrcDimensions, Args->DstDevice, Args->SrcDevice);

    DP("omp_target_memcpy_rect returns %d\n", Rc);
  } else {
    Rc = omp_target_memcpy(Args->Dst, Args->Src, Args->Length, Args->DstOffset,
                           Args->SrcOffset, Args->DstDevice, Args->SrcDevice);

    DP("omp_target_memcpy returns %d\n", Rc);
  }

  // Release the arguments object
  delete Args;

  return Rc;
}

// Allocate and launch helper task
static int libomp_helper_task_creation(TargetMemcpyArgsTy *Args,
                                       int DepObjCount,
                                       omp_depend_t *DepObjList) {
  // Create global thread ID
  int Gtid = __kmpc_global_thread_num(nullptr);
  int (*Fn)(kmp_int32, kmp_task_t *) = &libomp_target_memcpy_async_helper;

  // Setup the hidden helper flags;
  kmp_int32 Flags = 0;
  kmp_tasking_flags_t *InputFlags = (kmp_tasking_flags_t *)&Flags;
  InputFlags->hidden_helper = 1;

  // Alloc helper task
  kmp_task_t *Ptr = __kmpc_omp_target_task_alloc(nullptr, Gtid, Flags,
                                                 sizeof(kmp_task_t), 0, Fn, -1);

  if (Ptr == nullptr) {
    // Task allocation failed, delete the argument object
    delete Args;

    return OFFLOAD_FAIL;
  }

  // Setup the arguments passed to helper task
  Ptr->shareds = Args;

  // Convert the type of depend objects
  llvm::SmallVector<kmp_depend_info_t> DepObjs;
  for (int i = 0; i < DepObjCount; i++) {
    omp_depend_t DepObj = DepObjList[i];
    DepObjs.push_back(*((kmp_depend_info_t *)DepObj));
  }

  // Launch the helper task
  int Rc = __kmpc_omp_task_with_deps(nullptr, Gtid, Ptr, DepObjCount,
                                     DepObjs.data(), 0, nullptr);

  return Rc;
}

EXTERN int omp_target_memcpy_async(void *Dst, const void *Src, size_t Length,
                                   size_t DstOffset, size_t SrcOffset,
                                   int DstDevice, int SrcDevice,
                                   int DepObjCount, omp_depend_t *DepObjList) {
  TIMESCOPE();
  DP("Call to omp_target_memcpy_async, dst device %d, src device %d, "
     "dst addr " DPxMOD ", src addr " DPxMOD ", dst offset %zu, "
     "src offset %zu, length %zu\n",
     DstDevice, SrcDevice, DPxPTR(Dst), DPxPTR(Src), DstOffset, SrcOffset,
     Length);

  // Check the source and dest address
  if (Dst == nullptr || Src == nullptr)
    return OFFLOAD_FAIL;

  // Create task object
  TargetMemcpyArgsTy *Args = new TargetMemcpyArgsTy(
      Dst, Src, Length, DstOffset, SrcOffset, DstDevice, SrcDevice);

  // Create and launch helper task
  int Rc = libomp_helper_task_creation(Args, DepObjCount, DepObjList);

  DP("omp_target_memcpy_async returns %d\n", Rc);
  return Rc;
}

EXTERN int
omp_target_memcpy_rect(void *Dst, const void *Src, size_t ElementSize,
                       int NumDims, const size_t *Volume,
                       const size_t *DstOffsets, const size_t *SrcOffsets,
                       const size_t *DstDimensions, const size_t *SrcDimensions,
                       int DstDevice, int SrcDevice) {
  TIMESCOPE();
  DP("Call to omp_target_memcpy_rect, dst device %d, src device %d, "
     "dst addr " DPxMOD ", src addr " DPxMOD ", dst offsets " DPxMOD ", "
     "src offsets " DPxMOD ", dst dims " DPxMOD ", src dims " DPxMOD ", "
     "volume " DPxMOD ", element size %zu, num_dims %d\n",
     DstDevice, SrcDevice, DPxPTR(Dst), DPxPTR(Src), DPxPTR(DstOffsets),
     DPxPTR(SrcOffsets), DPxPTR(DstDimensions), DPxPTR(SrcDimensions),
     DPxPTR(Volume), ElementSize, NumDims);

  if (!(Dst || Src)) {
    DP("Call to omp_target_memcpy_rect returns max supported dimensions %d\n",
       INT_MAX);
    return INT_MAX;
  }

  if (!Dst || !Src || ElementSize < 1 || NumDims < 1 || !Volume ||
      !DstOffsets || !SrcOffsets || !DstDimensions || !SrcDimensions) {
    REPORT("Call to omp_target_memcpy_rect with invalid arguments\n");
    return OFFLOAD_FAIL;
  }

  int Rc;
  if (NumDims == 1) {
    Rc = omp_target_memcpy(Dst, Src, ElementSize * Volume[0],
                           ElementSize * DstOffsets[0],
                           ElementSize * SrcOffsets[0], DstDevice, SrcDevice);
  } else {
    size_t DstSliceSize = ElementSize;
    size_t SrcSliceSize = ElementSize;
    for (int I = 1; I < NumDims; ++I) {
      DstSliceSize *= DstDimensions[I];
      SrcSliceSize *= SrcDimensions[I];
    }

    size_t DstOff = DstOffsets[0] * DstSliceSize;
    size_t SrcOff = SrcOffsets[0] * SrcSliceSize;
    for (size_t I = 0; I < Volume[0]; ++I) {
      Rc = omp_target_memcpy_rect(
          (char *)Dst + DstOff + DstSliceSize * I,
          (char *)const_cast<void *>(Src) + SrcOff + SrcSliceSize * I,
          ElementSize, NumDims - 1, Volume + 1, DstOffsets + 1, SrcOffsets + 1,
          DstDimensions + 1, SrcDimensions + 1, DstDevice, SrcDevice);

      if (Rc) {
        DP("Recursive call to omp_target_memcpy_rect returns unsuccessfully\n");
        return Rc;
      }
    }
  }

  DP("omp_target_memcpy_rect returns %d\n", Rc);
  return Rc;
}

EXTERN int omp_target_memcpy_rect_async(
    void *Dst, const void *Src, size_t ElementSize, int NumDims,
    const size_t *Volume, const size_t *DstOffsets, const size_t *SrcOffsets,
    const size_t *DstDimensions, const size_t *SrcDimensions, int DstDevice,
    int SrcDevice, int DepObjCount, omp_depend_t *DepObjList) {
  TIMESCOPE();
  DP("Call to omp_target_memcpy_rect_async, dst device %d, src device %d, "
     "dst addr " DPxMOD ", src addr " DPxMOD ", dst offsets " DPxMOD ", "
     "src offsets " DPxMOD ", dst dims " DPxMOD ", src dims " DPxMOD ", "
     "volume " DPxMOD ", element size %zu, num_dims %d\n",
     DstDevice, SrcDevice, DPxPTR(Dst), DPxPTR(Src), DPxPTR(DstOffsets),
     DPxPTR(SrcOffsets), DPxPTR(DstDimensions), DPxPTR(SrcDimensions),
     DPxPTR(Volume), ElementSize, NumDims);

  // Need to check this first to not return OFFLOAD_FAIL instead
  if (!Dst && !Src) {
    DP("Call to omp_target_memcpy_rect returns max supported dimensions %d\n",
       INT_MAX);
    return INT_MAX;
  }

  // Check the source and dest address
  if (Dst == nullptr || Src == nullptr)
    return OFFLOAD_FAIL;

  // Create task object
  TargetMemcpyArgsTy *Args = new TargetMemcpyArgsTy(
      Dst, Src, ElementSize, NumDims, Volume, DstOffsets, SrcOffsets,
      DstDimensions, SrcDimensions, DstDevice, SrcDevice);

  // Create and launch helper task
  int Rc = libomp_helper_task_creation(Args, DepObjCount, DepObjList);

  DP("omp_target_memcpy_rect_async returns %d\n", Rc);
  return Rc;
}

EXTERN int omp_target_associate_ptr(const void *HostPtr, const void *DevicePtr,
                                    size_t Size, size_t DeviceOffset,
                                    int DeviceNum) {
  TIMESCOPE();
  DP("Call to omp_target_associate_ptr with host_ptr " DPxMOD ", "
     "device_ptr " DPxMOD ", size %zu, device_offset %zu, device_num %d\n",
     DPxPTR(HostPtr), DPxPTR(DevicePtr), Size, DeviceOffset, DeviceNum);

  if (!HostPtr || !DevicePtr || Size <= 0) {
    REPORT("Call to omp_target_associate_ptr with invalid arguments\n");
    return OFFLOAD_FAIL;
  }

  if (DeviceNum == omp_get_initial_device()) {
    REPORT("omp_target_associate_ptr: no association possible on the host\n");
    return OFFLOAD_FAIL;
  }

  if (!deviceIsReady(DeviceNum)) {
    REPORT("omp_target_associate_ptr returns OFFLOAD_FAIL\n");
    return OFFLOAD_FAIL;
  }

  DeviceTy &Device = *PM->Devices[DeviceNum];
  void *DeviceAddr = (void *)((uint64_t)DevicePtr + (uint64_t)DeviceOffset);
  int Rc = Device.associatePtr(const_cast<void *>(HostPtr),
                               const_cast<void *>(DeviceAddr), Size);
  DP("omp_target_associate_ptr returns %d\n", Rc);
  return Rc;
}

EXTERN int omp_target_disassociate_ptr(const void *HostPtr, int DeviceNum) {
  TIMESCOPE();
  DP("Call to omp_target_disassociate_ptr with host_ptr " DPxMOD ", "
     "device_num %d\n",
     DPxPTR(HostPtr), DeviceNum);

  if (!HostPtr) {
    REPORT("Call to omp_target_associate_ptr with invalid host_ptr\n");
    return OFFLOAD_FAIL;
  }

  if (DeviceNum == omp_get_initial_device()) {
    REPORT(
        "omp_target_disassociate_ptr: no association possible on the host\n");
    return OFFLOAD_FAIL;
  }

  if (!deviceIsReady(DeviceNum)) {
    REPORT("omp_target_disassociate_ptr returns OFFLOAD_FAIL\n");
    return OFFLOAD_FAIL;
  }

  DeviceTy &Device = *PM->Devices[DeviceNum];
  int Rc = Device.disassociatePtr(const_cast<void *>(HostPtr));
  DP("omp_target_disassociate_ptr returns %d\n", Rc);
  return Rc;
}

EXTERN void *omp_get_mapped_ptr(const void *Ptr, int DeviceNum) {
  TIMESCOPE();
  DP("Call to omp_get_mapped_ptr with ptr " DPxMOD ", device_num %d.\n",
     DPxPTR(Ptr), DeviceNum);

  if (!Ptr) {
    REPORT("Call to omp_get_mapped_ptr with nullptr.\n");
    return nullptr;
  }

  if (DeviceNum == omp_get_initial_device()) {
    REPORT("Device %d is initial device, returning Ptr " DPxMOD ".\n",
           DeviceNum, DPxPTR(Ptr));
    return const_cast<void *>(Ptr);
  }

  int DevicesSize = omp_get_initial_device();
  {
    std::lock_guard<std::mutex> LG(PM->RTLsMtx);
    DevicesSize = PM->Devices.size();
  }
  if (DevicesSize <= DeviceNum) {
    DP("DeviceNum %d is invalid, returning nullptr.\n", DeviceNum);
    return nullptr;
  }

  if (!deviceIsReady(DeviceNum)) {
    REPORT("Device %d is not ready, returning nullptr.\n", DeviceNum);
    return nullptr;
  }

  auto &Device = *PM->Devices[DeviceNum];
  TargetPointerResultTy TPR = Device.getTgtPtrBegin(const_cast<void *>(Ptr), 1,
                                                    /*UpdateRefCount=*/false,
                                                    /*UseHoldRefCount=*/false);
  if (!TPR.isPresent()) {
    DP("Ptr " DPxMOD "is not present on device %d, returning nullptr.\n",
       DPxPTR(Ptr), DeviceNum);
    return nullptr;
  }

  DP("omp_get_mapped_ptr returns " DPxMOD ".\n", DPxPTR(TPR.TargetPointer));

  return TPR.TargetPointer;
}
