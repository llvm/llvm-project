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

#include "OpenMP/OMPT/OmptCommonDefs.h"
#include "PluginManager.h"
#include "device.h"
#include "omptarget.h"
#include "rtl.h"

#include "OpenMP/InternalTypes.h"
#include "OpenMP/InteropAPI.h"
#include "OpenMP/Mapping.h"
#include "OpenMP/OMPT/Interface.h"
#include "OpenMP/omp.h"
#include "Shared/Profile.h"

#include "llvm/ADT/SmallVector.h"

#include <climits>
#include <cstdlib>
#include <cstring>
#include <mutex>

EXTERN int ompx_get_team_procs(int DeviceNum) {
  TIMESCOPE();
  auto DeviceOrErr = PM->getDevice(DeviceNum);
  if (!DeviceOrErr)
    FATAL_MESSAGE(DeviceNum, "%s", toString(DeviceOrErr.takeError()).c_str());
  int TeamProcs = DeviceOrErr->getTeamProcs();
  DP("Call to ompx_get_team_procs returning %d\n", TeamProcs);
  return TeamProcs;
}

EXTERN void ompx_dump_mapping_tables() {
  ident_t Loc = {0, 0, 0, 0, ";libomptarget;libomptarget;0;0;;"};
  auto ExclusiveDevicesAccessor = PM->getExclusiveDevicesAccessor();
  for (auto &Device : PM->devices(ExclusiveDevicesAccessor))
    dumpTargetPointerMappings(&Loc, Device, true);
}

#ifdef OMPT_SUPPORT
using namespace llvm::omp::target::ompt;
#endif
using namespace llvm::omp::target::debug;

using GenericDeviceTy = llvm::omp::target::plugin::GenericDeviceTy;

void *targetAllocExplicit(size_t Size, int DeviceNum, int Kind,
                          const char *Name);
void targetFreeExplicit(void *DevicePtr, int DeviceNum, int Kind,
                        const char *Name);
void *targetLockExplicit(void *HostPtr, size_t Size, int DeviceNum,
                         const char *Name);
void targetUnlockExplicit(void *HostPtr, int DeviceNum, const char *Name);

EXTERN int omp_get_num_devices(void) {
  TIMESCOPE();
  OMPT_IF_BUILT(ReturnAddressSetterRAII RA(__builtin_return_address(0)));
  size_t NumDevices = PM->getNumDevices();

  ODBG(ODT_Interface) << "Call to " << __func__ << " returning " << NumDevices;

  return NumDevices;
}

EXTERN int omp_get_DeviceNum(void) {
  TIMESCOPE();
  OMPT_IF_BUILT(ReturnAddressSetterRAII RA(__builtin_return_address(0)));
  int HostDevice = omp_get_initial_device();

  ODBG(ODT_Interface) << "Call to " << __func__ << " returning " << HostDevice;

  return HostDevice;
}

static inline bool is_initial_device_uid(const char *DeviceUid) {
  return strcmp(DeviceUid, GenericPluginTy::getHostDeviceUid()) == 0;
}

EXTERN int omp_get_device_from_uid(const char *DeviceUid) {
  TIMESCOPE();
  OMPT_IF_BUILT(ReturnAddressSetterRAII RA(__builtin_return_address(0)));

  if (!DeviceUid) {
    ODBG(ODT_Interface) << "Call to " << __func__
                        << " returning omp_invalid_device";
    return omp_invalid_device;
  }
  if (is_initial_device_uid(DeviceUid)) {
    ODBG(ODT_Interface) << "Call to " << __func__
                        << " returning initial device number "
                        << omp_get_initial_device();
    return omp_get_initial_device();
  }

  int DeviceNum = omp_invalid_device;

  auto ExclusiveDevicesAccessor = PM->getExclusiveDevicesAccessor();
  for (const DeviceTy &Device : PM->devices(ExclusiveDevicesAccessor)) {
    const char *Uid = Device.RTL->getDevice(Device.RTLDeviceID).getDeviceUid();
    if (Uid && strcmp(DeviceUid, Uid) == 0) {
      DeviceNum = Device.DeviceID;
      break;
    }
  }

  ODBG(ODT_Interface) << "Call to " << __func__ << " returning " << DeviceNum;
  return DeviceNum;
}

EXTERN const char *omp_get_uid_from_device(int DeviceNum) {
  TIMESCOPE();
  OMPT_IF_BUILT(ReturnAddressSetterRAII RA(__builtin_return_address(0)));

  if (DeviceNum == omp_invalid_device) {
    ODBG(ODT_Interface) << "Call to " << __func__ << " returning nullptr";
    return nullptr;
  }
  if (DeviceNum == omp_get_initial_device()) {
    ODBG(ODT_Interface) << "Call to " << __func__
                        << " returning initial device UID";
    return GenericPluginTy::getHostDeviceUid();
  }

  auto DeviceOrErr = PM->getDevice(DeviceNum);
  if (!DeviceOrErr)
    FATAL_MESSAGE(DeviceNum, "%s", toString(DeviceOrErr.takeError()).c_str());

  const char *Uid =
      DeviceOrErr->RTL->getDevice(DeviceOrErr->RTLDeviceID).getDeviceUid();
  ODBG(ODT_Interface) << "Call to " << __func__ << " returning " << Uid;
  return Uid;
}

EXTERN int omp_get_initial_device(void) {
  TIMESCOPE();
  OMPT_IF_BUILT(ReturnAddressSetterRAII RA(__builtin_return_address(0)));
  int HostDevice = omp_get_num_devices();
  ODBG(ODT_Interface) << "Call to " << __func__ << " returning " << HostDevice;
  return HostDevice;
}

EXTERN void *omp_target_alloc(size_t Size, int DeviceNum) {
  TIMESCOPE_WITH_DETAILS("dst_dev=" + std::to_string(DeviceNum) +
                         ";size=" + std::to_string(Size));
  OMPT_IF_BUILT(ReturnAddressSetterRAII RA(__builtin_return_address(0)));
  return targetAllocExplicit(Size, DeviceNum, TARGET_ALLOC_DEFAULT, __func__);
}

EXTERN void *llvm_omp_target_alloc_device(size_t Size, int DeviceNum) {
  OMPT_IF_BUILT(ReturnAddressSetterRAII RA(__builtin_return_address(0)));
  return targetAllocExplicit(Size, DeviceNum, TARGET_ALLOC_DEVICE, __func__);
}

EXTERN void *llvm_omp_target_alloc_host(size_t Size, int DeviceNum) {
  OMPT_IF_BUILT(ReturnAddressSetterRAII RA(__builtin_return_address(0)));
  return targetAllocExplicit(Size, DeviceNum, TARGET_ALLOC_HOST, __func__);
}

EXTERN void *llvm_omp_target_alloc_shared(size_t Size, int DeviceNum) {
  OMPT_IF_BUILT(ReturnAddressSetterRAII RA(__builtin_return_address(0)));
  return targetAllocExplicit(Size, DeviceNum, TARGET_ALLOC_SHARED, __func__);
}

EXTERN void *llvm_omp_target_alloc_multi_devices(size_t size, int num_devices,
                                                 int DeviceNums[]) {
  if (num_devices < 1)
    return nullptr;

  DeviceTy &Device = *PM->getDevice(DeviceNums[0]);
  if (!Device.RTL->is_system_supporting_managed_memory(Device.DeviceID))
    return nullptr;

  // disregard device ids for now and allocate shared memory that can be
  // accessed by any device and host under xnack+ mode
  void *ptr =
      targetAllocExplicit(size, DeviceNums[0], TARGET_ALLOC_DEFAULT, __func__);
  // TODO: not implemented yet
  // if (Device.RTL->enable_access_to_all_agents)
  //   Device.RTL->enable_access_to_all_agents(DeviceNums[0], ptr);
  return ptr;
}

EXTERN void omp_target_free(void *Ptr, int DeviceNum) {
  TIMESCOPE();
  OMPT_IF_BUILT(ReturnAddressSetterRAII RA(__builtin_return_address(0)));
  return targetFreeExplicit(Ptr, DeviceNum, TARGET_ALLOC_DEFAULT, __func__);
}

EXTERN void llvm_omp_target_free_device(void *Ptr, int DeviceNum) {
  OMPT_IF_BUILT(ReturnAddressSetterRAII RA(__builtin_return_address(0)));
  return targetFreeExplicit(Ptr, DeviceNum, TARGET_ALLOC_DEVICE, __func__);
}

EXTERN void llvm_omp_target_free_host(void *Ptr, int DeviceNum) {
  OMPT_IF_BUILT(ReturnAddressSetterRAII RA(__builtin_return_address(0)));
  return targetFreeExplicit(Ptr, DeviceNum, TARGET_ALLOC_HOST, __func__);
}

EXTERN void llvm_omp_target_free_shared(void *Ptre, int DeviceNum) {
  OMPT_IF_BUILT(ReturnAddressSetterRAII RA(__builtin_return_address(0)));
  return targetFreeExplicit(Ptre, DeviceNum, TARGET_ALLOC_SHARED, __func__);
}

EXTERN void *llvm_omp_target_dynamic_shared_alloc() {
  OMPT_IF_BUILT(ReturnAddressSetterRAII RA(__builtin_return_address(0)));
  return nullptr;
}

EXTERN void *llvm_omp_get_dynamic_shared() {
  OMPT_IF_BUILT(ReturnAddressSetterRAII RA(__builtin_return_address(0)));
  return nullptr;
}

EXTERN [[nodiscard]] void *llvm_omp_target_lock_mem(void *Ptr, size_t Size,
                                                    int DeviceNum) {
  OMPT_IF_BUILT(ReturnAddressSetterRAII RA(__builtin_return_address(0)));
  return targetLockExplicit(Ptr, Size, DeviceNum, __func__);
}

EXTERN void llvm_omp_target_unlock_mem(void *Ptr, int DeviceNum) {
  OMPT_IF_BUILT(ReturnAddressSetterRAII RA(__builtin_return_address(0)));
  targetUnlockExplicit(Ptr, DeviceNum, __func__);
}

EXTERN int omp_target_is_present(const void *Ptr, int DeviceNum) {
  TIMESCOPE();
  OMPT_IF_BUILT(ReturnAddressSetterRAII RA(__builtin_return_address(0)));
  ODBG(ODT_Interface) << "Call to " << __func__ << " for device " << DeviceNum
                      << " and address " << Ptr;

  if (!Ptr) {
    ODBG(ODT_Interface) << "Call to " << __func__
                        << " with NULL ptr, returning false";
    return false;
  }

  if (DeviceNum == omp_get_initial_device()) {
    ODBG(ODT_Interface) << "Call to " << __func__ << " on host, returning true";
    return true;
  }

  auto DeviceOrErr = PM->getDevice(DeviceNum);
  if (!DeviceOrErr)
    FATAL_MESSAGE(DeviceNum, "%s", toString(DeviceOrErr.takeError()).c_str());

  // omp_target_is_present tests whether a host pointer refers to storage that
  // is mapped to a given device. However, due to the lack of the storage size,
  // only check 1 byte. Cannot set size 0 which checks whether the pointer (zero
  // length array) is mapped instead of the referred storage.
  TargetPointerResultTy TPR =
      DeviceOrErr->getMappingInfo().getTgtPtrBegin(const_cast<void *>(Ptr), 1,
                                                   /*UpdateRefCount=*/false,
                                                   /*UseHoldRefCount=*/false);
  int Rc = TPR.isPresent();
  ODBG(ODT_Interface) << "Call to " << __func__ << " returns " << Rc;
  return Rc;
}

/// Check whether a pointer is accessible from a device.
/// Returns true when accessibility is guaranteed otherwise returns false.
EXTERN int omp_target_is_accessible(const void *Ptr, size_t Size,
                                    int DeviceNum) {
  TIMESCOPE();
  OMPT_IF_BUILT(ReturnAddressSetterRAII RA(__builtin_return_address(0)));
  ODBG(ODT_Interface) << "Call to " << __func__ << " for device " << DeviceNum
                      << ", address " << Ptr << ", size " << Size;

  if (!Ptr) {
    ODBG(ODT_Interface) << "Call to " << __func__
                        << " with NULL ptr returning false";
    return false;
  }

  if (DeviceNum == omp_get_initial_device() || DeviceNum == -1) {
    ODBG(ODT_Interface) << "Call to " << __func__ << " on host, returning true";
    return true;
  }

  // The device number must refer to a valid device
  auto DeviceOrErr = PM->getDevice(DeviceNum);
  if (!DeviceOrErr)
    FATAL_MESSAGE(DeviceNum, "%s", toString(DeviceOrErr.takeError()).c_str());

  return DeviceOrErr->isAccessiblePtr(Ptr, Size);
}

EXTERN int omp_target_memcpy(void *Dst, const void *Src, size_t Length,
                             size_t DstOffset, size_t SrcOffset, int DstDevice,
                             int SrcDevice) {
  TIMESCOPE_WITH_DETAILS("dst_dev=" + std::to_string(DstDevice) +
                         ";src_dev=" + std::to_string(SrcDevice) +
                         ";size=" + std::to_string(Length));
  OMPT_IF_BUILT(ReturnAddressSetterRAII RA(__builtin_return_address(0)));
  ODBG(ODT_Interface) << "Call to " << __func__ << ", dst device " << DstDevice
                      << ", src device " << SrcDevice << ", dst addr " << Dst
                      << ", src addr " << Src << ", dst offset " << DstOffset
                      << ", src offset " << SrcOffset << ", length " << Length;

  if (!Dst || !Src || Length <= 0) {
    if (Length == 0) {
      ODBG(ODT_Interface) << "Call to " << __func__
                          << " with zero length, nothing to do";
      return OFFLOAD_SUCCESS;
    }

    REPORT() << "Call to " << __func__ << " with invalid arguments";
    return OFFLOAD_FAIL;
  }

  int Rc = OFFLOAD_SUCCESS;
  void *SrcAddr = (char *)const_cast<void *>(Src) + SrcOffset;
  void *DstAddr = (char *)Dst + DstOffset;

  if (SrcDevice == omp_get_initial_device() &&
      DstDevice == omp_get_initial_device()) {
    ODBG(ODT_Interface) << "copy from host to host";
    const void *P = memcpy(DstAddr, SrcAddr, Length);
    if (P == NULL)
      Rc = OFFLOAD_FAIL;
  } else if (SrcDevice == omp_get_initial_device()) {
    ODBG(ODT_Interface) << "copy from host to device";
    auto DstDeviceOrErr = PM->getDevice(DstDevice);
    if (!DstDeviceOrErr)
      FATAL_MESSAGE(DstDevice, "%s",
                    toString(DstDeviceOrErr.takeError()).c_str());
    AsyncInfoTy AsyncInfo(*DstDeviceOrErr);
    Rc = DstDeviceOrErr->submitData(DstAddr, SrcAddr, Length, AsyncInfo);
  } else if (DstDevice == omp_get_initial_device()) {
    ODBG(ODT_Interface) << "copy from device to host";
    auto SrcDeviceOrErr = PM->getDevice(SrcDevice);
    if (!SrcDeviceOrErr)
      FATAL_MESSAGE(SrcDevice, "%s",
                    toString(SrcDeviceOrErr.takeError()).c_str());
    AsyncInfoTy AsyncInfo(*SrcDeviceOrErr);
    Rc = SrcDeviceOrErr->retrieveData(DstAddr, SrcAddr, Length, AsyncInfo);
  } else {
    ODBG(ODT_Interface) << "copy from device to device";
    auto SrcDeviceOrErr = PM->getDevice(SrcDevice);
    if (!SrcDeviceOrErr)
      FATAL_MESSAGE(SrcDevice, "%s",
                    toString(SrcDeviceOrErr.takeError()).c_str());
    AsyncInfoTy AsyncInfo(*SrcDeviceOrErr);
    auto DstDeviceOrErr = PM->getDevice(DstDevice);
    if (!DstDeviceOrErr)
      FATAL_MESSAGE(DstDevice, "%s",
                    toString(DstDeviceOrErr.takeError()).c_str());
    // First try to use D2D memcpy which is more efficient. If fails, fall back
    // to inefficient way.
    if (SrcDeviceOrErr->isDataExchangable(*DstDeviceOrErr)) {
      AsyncInfoTy AsyncInfo(*SrcDeviceOrErr);
      Rc = SrcDeviceOrErr->dataExchange(SrcAddr, *DstDeviceOrErr, DstAddr,
                                        Length, AsyncInfo);
      if (Rc == OFFLOAD_SUCCESS)
        return OFFLOAD_SUCCESS;
    }

    void *Buffer = malloc(Length);
    {
      AsyncInfoTy AsyncInfo(*SrcDeviceOrErr);
      Rc = SrcDeviceOrErr->retrieveData(Buffer, SrcAddr, Length, AsyncInfo);
    }
    if (Rc == OFFLOAD_SUCCESS) {
      AsyncInfoTy AsyncInfo(*DstDeviceOrErr);
      Rc = DstDeviceOrErr->submitData(DstAddr, Buffer, Length, AsyncInfo);
    }
    free(Buffer);
  }

  ODBG(ODT_Interface) << __func__ << " returns " << Rc;
  return Rc;
}

// The helper function that calls omp_target_memcpy or omp_target_memcpy_rect
static int libomp_target_memcpy_async_task(int32_t Gtid, kmp_task_t *Task) {
  OMPT_IF_BUILT(ReturnAddressSetterRAII RA(__builtin_return_address(0)));
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

    ODBG(ODT_Interface) << " omp_target_memcpy_rect returns " << Rc;
  } else {
    Rc = omp_target_memcpy(Args->Dst, Args->Src, Args->Length, Args->DstOffset,
                           Args->SrcOffset, Args->DstDevice, Args->SrcDevice);

    ODBG(ODT_Interface) << " omp_target_memcpy returns " << Rc;
  }

  // Release the arguments object
  delete Args;

  return Rc;
}

static int libomp_target_memset_async_task(int32_t Gtid, kmp_task_t *Task) {
  OMPT_IF_BUILT(ReturnAddressSetterRAII RA(__builtin_return_address(0)));
  if (!Task)
    return OFFLOAD_FAIL;

  auto *Args = reinterpret_cast<TargetMemsetArgsTy *>(Task->shareds);
  if (!Args)
    return OFFLOAD_FAIL;

  // call omp_target_memset()
  omp_target_memset(Args->Ptr, Args->C, Args->N, Args->DeviceNum);

  delete Args;

  return OFFLOAD_SUCCESS;
}

static inline void
convertDepObjVector(llvm::SmallVector<kmp_depend_info_t> &Vec, int DepObjCount,
                    omp_depend_t *DepObjList) {
  for (int i = 0; i < DepObjCount; ++i) {
    omp_depend_t DepObj = DepObjList[i];
    Vec.push_back(*((kmp_depend_info_t *)DepObj));
  }
}

template <class T>
static inline int
libomp_helper_task_creation(T *Args, int (*Fn)(int32_t, kmp_task_t *),
                            int DepObjCount, omp_depend_t *DepObjList) {
  OMPT_IF_BUILT(ReturnAddressSetterRAII RA(__builtin_return_address(0)));
  // Create global thread ID
  int Gtid = __kmpc_global_thread_num(nullptr);

  // Setup the hidden helper flags
  int32_t Flags = 0;
  kmp_tasking_flags_t *InputFlags = (kmp_tasking_flags_t *)&Flags;
  InputFlags->hidden_helper = 1;

  // Alloc the helper task
  kmp_task_t *Task = __kmpc_omp_target_task_alloc(
      nullptr, Gtid, Flags, sizeof(kmp_task_t), 0, Fn, -1);
  if (!Task) {
    delete Args;
    return OFFLOAD_FAIL;
  }

  // Setup the arguments for the helper task
  Task->shareds = Args;

  // Convert types of depend objects
  llvm::SmallVector<kmp_depend_info_t> DepObjs;
  convertDepObjVector(DepObjs, DepObjCount, DepObjList);

  // Launch the helper task
  int Rc = __kmpc_omp_task_with_deps(nullptr, Gtid, Task, DepObjCount,
                                     DepObjs.data(), 0, nullptr);

  return Rc;
}

EXTERN void *omp_target_memset(void *Ptr, int ByteVal, size_t NumBytes,
                               int DeviceNum) {
  TIMESCOPE();
  OMPT_IF_BUILT(ReturnAddressSetterRAII RA(__builtin_return_address(0)));
  ODBG(ODT_Interface) << "Call to " << __func__ << ", device " << DeviceNum
                      << ", device pointer " << Ptr << ", size " << NumBytes;

  // Behave as a no-op if N==0 or if Ptr is nullptr (as a useful implementation
  // of unspecified behavior, see OpenMP spec).
  if (!Ptr || NumBytes == 0) {
    return Ptr;
  }

  if (DeviceNum == omp_get_initial_device()) {
    ODBG(ODT_Interface) << "filling memory on host via memset";
    memset(Ptr, ByteVal, NumBytes); // ignore return value, memset() cannot fail
  } else {
    // TODO: replace the omp_target_memset() slow path with the fast path.
    // That will require the ability to execute a kernel from within
    // libomptarget.so (which we do not have at the moment).

    // This is a very slow path: create a filled array on the host and upload
    // it to the GPU device.
    int InitialDevice = omp_get_initial_device();
    void *Shadow = omp_target_alloc(NumBytes, InitialDevice);
    if (Shadow) {
      (void)memset(Shadow, ByteVal, NumBytes);
      (void)omp_target_memcpy(Ptr, Shadow, NumBytes, 0, 0, DeviceNum,
                              InitialDevice);
      (void)omp_target_free(Shadow, InitialDevice);
    } else {
      // If the omp_target_alloc has failed, let's just not do anything.
      // omp_target_memset does not have any good way to fail, so we
      // simply avoid a catastrophic failure of the process for now.
      ODBG(ODT_Interface)
          << __func__
          << " failed to fill memory due to error with omp_target_alloc";
    }
  }

  ODBG(ODT_Interface) << __func__ << " returns " << Ptr;
  return Ptr;
}

EXTERN void *omp_target_memset_async(void *Ptr, int ByteVal, size_t NumBytes,
                                     int DeviceNum, int DepObjCount,
                                     omp_depend_t *DepObjList) {
  OMPT_IF_BUILT(ReturnAddressSetterRAII RA(__builtin_return_address(0)));
  ODBG(ODT_Interface) << "Call to " << __func__ << ", device " << DeviceNum
                      << ", device pointer " << Ptr << ", size " << NumBytes;

  // Behave as a no-op if N==0 or if Ptr is nullptr (as a useful implementation
  // of unspecified behavior, see OpenMP spec).
  if (!Ptr || NumBytes == 0)
    return Ptr;

  // Create the task object to deal with the async invocation
  auto *Args = new TargetMemsetArgsTy{Ptr, ByteVal, NumBytes, DeviceNum};

  // omp_target_memset_async() cannot fail via a return code, so ignore the
  // return code of the helper function
  (void)libomp_helper_task_creation(Args, &libomp_target_memset_async_task,
                                    DepObjCount, DepObjList);

  return Ptr;
}

EXTERN int omp_target_memcpy_async(void *Dst, const void *Src, size_t Length,
                                   size_t DstOffset, size_t SrcOffset,
                                   int DstDevice, int SrcDevice,
                                   int DepObjCount, omp_depend_t *DepObjList) {
  TIMESCOPE_WITH_DETAILS("dst_dev=" + std::to_string(DstDevice) +
                         ";src_dev=" + std::to_string(SrcDevice) +
                         ";size=" + std::to_string(Length));
  OMPT_IF_BUILT(ReturnAddressSetterRAII RA(__builtin_return_address(0)));
  ODBG(ODT_Interface) << "Call to " << __func__ << ", dst device " << DstDevice
                      << ", src device " << SrcDevice << ", dst addr " << Dst
                      << ", src addr " << Src << ", dst offset " << DstOffset
                      << ", src offset " << SrcOffset << ", length " << Length;

  // Check the source and dest address
  if (Dst == nullptr || Src == nullptr)
    return OFFLOAD_FAIL;

  // Create task object
  TargetMemcpyArgsTy *Args = new TargetMemcpyArgsTy(
      Dst, Src, Length, DstOffset, SrcOffset, DstDevice, SrcDevice);

  // Create and launch helper task
  int Rc = libomp_helper_task_creation(Args, &libomp_target_memcpy_async_task,
                                       DepObjCount, DepObjList);

  ODBG(ODT_Interface) << __func__ << " returns " << Rc;
  return Rc;
}

EXTERN int
omp_target_memcpy_rect(void *Dst, const void *Src, size_t ElementSize,
                       int NumDims, const size_t *Volume,
                       const size_t *DstOffsets, const size_t *SrcOffsets,
                       const size_t *DstDimensions, const size_t *SrcDimensions,
                       int DstDevice, int SrcDevice) {
  OMPT_IF_BUILT(ReturnAddressSetterRAII RA(__builtin_return_address(0)));
  ODBG(ODT_Interface) << "Call to " << __func__ << ", dst device " << DstDevice
                      << ", src device " << SrcDevice << ", dst addr " << Dst
                      << ", src addr " << Src << ", dst offsets " << DstOffsets
                      << ", src offsets " << SrcOffsets << ", dst dims "
                      << DstDimensions << ", src dims " << SrcDimensions
                      << ", volume " << Volume << ", element size "
                      << ElementSize << ", num_dims " << NumDims;

  if (!(Dst || Src)) {
    ODBG(ODT_Interface) << "Call to " << __func__
                        << " returns max supported dimensions " << INT_MAX;
    return INT_MAX;
  }

  if (!Dst || !Src || ElementSize < 1 || NumDims < 1 || !Volume ||
      !DstOffsets || !SrcOffsets || !DstDimensions || !SrcDimensions) {
    REPORT() << "Call to " << __func__ << " with invalid arguments";
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
        ODBG(ODT_Interface)
            << "Recursive call to " << __func__ << " returns unsuccessfully";
        return Rc;
      }
    }
  }

  ODBG(ODT_Interface) << " returns " << Rc;
  return Rc;
}

EXTERN int omp_target_memcpy_rect_async(
    void *Dst, const void *Src, size_t ElementSize, int NumDims,
    const size_t *Volume, const size_t *DstOffsets, const size_t *SrcOffsets,
    const size_t *DstDimensions, const size_t *SrcDimensions, int DstDevice,
    int SrcDevice, int DepObjCount, omp_depend_t *DepObjList) {
  TIMESCOPE_WITH_DETAILS("dst_dev=" + std::to_string(DstDevice) +
                         ";src_dev=" + std::to_string(SrcDevice) +
                         ";size=" + std::to_string(ElementSize) +
                         ";num_dims=" + std::to_string(NumDims));
  OMPT_IF_BUILT(ReturnAddressSetterRAII RA(__builtin_return_address(0)));
  ODBG(ODT_Interface) << "Call to " << __func__ << ", dst device " << DstDevice
                      << ", src device " << SrcDevice << ", dst addr " << Dst
                      << ", src addr " << Src << ", dst offsets " << DstOffsets
                      << ", src offsets " << SrcOffsets << ", dst dims "
                      << DstDimensions << ", src dims " << SrcDimensions
                      << ", volume " << Volume << ", element size "
                      << ElementSize << ", num_dims " << NumDims;

  // Need to check this first to not return OFFLOAD_FAIL instead
  if (!Dst && !Src) {
    ODBG(ODT_Interface) << "Call to " << __func__
                        << " returns max supported dimensions " << INT_MAX;
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
  int Rc = libomp_helper_task_creation(Args, &libomp_target_memcpy_async_task,
                                       DepObjCount, DepObjList);

  ODBG(ODT_Interface) << __func__ << " returns " << Rc;
  return Rc;
}

EXTERN int omp_target_associate_ptr(const void *HostPtr, const void *DevicePtr,
                                    size_t Size, size_t DeviceOffset,
                                    int DeviceNum) {
  TIMESCOPE();
  OMPT_IF_BUILT(ReturnAddressSetterRAII RA(__builtin_return_address(0)));
  ODBG(ODT_Interface) << "Call to " << __func__ << " with host_ptr " << HostPtr
                      << ", device_ptr " << DevicePtr << ", size " << Size
                      << ", device_offset " << DeviceOffset << ", device_num "
                      << DeviceNum;

  if (!HostPtr || !DevicePtr || Size <= 0) {
    REPORT() << "Call to " << __func__ << " with invalid arguments";
    return OFFLOAD_FAIL;
  }

  if (DeviceNum == omp_get_initial_device()) {
    REPORT() << __func__ << ": no association possible on the host";
    return OFFLOAD_FAIL;
  }

  auto DeviceOrErr = PM->getDevice(DeviceNum);
  if (!DeviceOrErr)
    FATAL_MESSAGE(DeviceNum, "%s", toString(DeviceOrErr.takeError()).c_str());

  void *DeviceAddr = (void *)((uint64_t)DevicePtr + (uint64_t)DeviceOffset);

  OMPT_IF_BUILT(InterfaceRAII(
      RegionInterface.getCallbacks<ompt_target_data_associate>(), DeviceNum,
      const_cast<void *>(HostPtr), const_cast<void *>(DevicePtr), Size,
      __builtin_return_address(0)));

  int Rc = DeviceOrErr->getMappingInfo().associatePtr(
      const_cast<void *>(HostPtr), const_cast<void *>(DeviceAddr), Size);
  ODBG(ODT_Interface) << __func__ << " returns " << Rc;
  return Rc;
}

EXTERN int omp_target_disassociate_ptr(const void *HostPtr, int DeviceNum) {
  TIMESCOPE();
  OMPT_IF_BUILT(ReturnAddressSetterRAII RA(__builtin_return_address(0)));
  ODBG(ODT_Interface) << "Call to " << __func__ << " with host_ptr " << HostPtr
                      << ", device_num " << DeviceNum;

  if (!HostPtr) {
    REPORT() << "Call to " << __func__ << " with invalid host_ptr";
    return OFFLOAD_FAIL;
  }

  if (DeviceNum == omp_get_initial_device()) {
    REPORT() << __func__ << ": no association possible on the host";
    return OFFLOAD_FAIL;
  }

  auto DeviceOrErr = PM->getDevice(DeviceNum);
  if (!DeviceOrErr)
    FATAL_MESSAGE(DeviceNum, "%s", toString(DeviceOrErr.takeError()).c_str());

  OMPT_IF_BUILT(InterfaceRAII(
      RegionInterface.getCallbacks<ompt_target_data_disassociate>(), DeviceNum,
      const_cast<void *>(HostPtr),
      /*DevicePtr=*/nullptr, /*Size=*/0, __builtin_return_address(0)));

  int Rc = DeviceOrErr->getMappingInfo().disassociatePtr(
      const_cast<void *>(HostPtr));
  ODBG(ODT_Interface) << __func__ << " returns " << Rc;
  return Rc;
}

EXTERN int omp_is_coarse_grain_mem_region(void *ptr, size_t size) {
  if (!(PM->getRequirements() & OMP_REQ_UNIFIED_SHARED_MEMORY))
    return 0;
  auto DeviceOrErr = PM->getDevice(omp_get_default_device());
  if (!DeviceOrErr)
    FATAL_MESSAGE(omp_get_default_device(), "%s",
                  toString(DeviceOrErr.takeError()).c_str());

  return DeviceOrErr->RTL->query_coarse_grain_mem_region(
      omp_get_default_device(), ptr, size);
}

// This user-callable function allows host overlays of HIP mem alloc functions
// to register memory as coarse grain in the openmp runtime. This will
// prevent duplicate HSA memory registration when OpenMP sees same memory
// in map clauses.
EXTERN void omp_register_coarse_grain_mem(void *ptr, size_t size, int setattr) {
  if (!(PM->getRequirements() & OMP_REQ_UNIFIED_SHARED_MEMORY))
    return;
  auto DeviceOrErr = PM->getDevice(omp_get_default_device());
  if (!DeviceOrErr)
    FATAL_MESSAGE(omp_get_default_device(), "%s",
                  toString(DeviceOrErr.takeError()).c_str());

  if (!(DeviceOrErr->RTL->is_gfx90a(omp_get_default_device()) &&
        DeviceOrErr->RTL->is_gfx90a_coarse_grain_usm_map_enabled(
            omp_get_default_device())))
    return;

  bool set_attr = (setattr == 1) ? true : false;
  DeviceOrErr->RTL->set_coarse_grain_mem(omp_get_default_device(), ptr, size,
                                         set_attr);
  return;
}

EXTERN void *omp_get_mapped_ptr(const void *Ptr, int DeviceNum) {
  TIMESCOPE();
  OMPT_IF_BUILT(ReturnAddressSetterRAII RA(__builtin_return_address(0)));
  ODBG(ODT_Interface) << "Call to " << __func__ << " with ptr " << Ptr
                      << ", device_num " << DeviceNum;

  if (!Ptr) {
    REPORT() << "Call to " << __func__ << " with nullptr.";
    return nullptr;
  }

  int NumDevices = omp_get_initial_device();
  if (DeviceNum == NumDevices) {
    ODBG(ODT_Interface) << "Device " << DeviceNum
                        << " is initial device, returning Ptr " << Ptr;
    return const_cast<void *>(Ptr);
  }

  if (NumDevices <= DeviceNum) {
    ODBG(ODT_Interface) << "DeviceNum " << DeviceNum
                        << " is invalid, returning nullptr.";
    return nullptr;
  }

  auto DeviceOrErr = PM->getDevice(DeviceNum);
  if (!DeviceOrErr)
    FATAL_MESSAGE(DeviceNum, "%s", toString(DeviceOrErr.takeError()).c_str());

  TargetPointerResultTy TPR =
      DeviceOrErr->getMappingInfo().getTgtPtrBegin(const_cast<void *>(Ptr), 1,
                                                   /*UpdateRefCount=*/false,
                                                   /*UseHoldRefCount=*/false);
  if (!TPR.isPresent()) {
    ODBG(ODT_Interface) << "Ptr " << Ptr
                        << "is not present on device %d, returning nullptr.";
    return nullptr;
  }

  ODBG(ODT_Interface) << __func__ << " returns " << TPR.TargetPointer << ".";

  return TPR.TargetPointer;
}

// This routine gets called from the Host RTL at sync points (taskwait, barrier,
// ...) so we can synchronize the necessary objects from the offload side.
EXTERN void __tgt_target_sync(ident_t *loc_ref, int gtid, void *current_task,
                              void *event) {
  if (!RTLAlive)
    return;

  RTLOngoingSyncs++;
  if (!RTLAlive) {
    RTLOngoingSyncs--;
    return;
  }

  syncImplicitInterops(gtid, event);

  RTLOngoingSyncs--;
}
