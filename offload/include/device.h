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

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <list>
#include <map>
#include <memory>
#include <mutex>
#include <set>

#include "ExclusiveAccess.h"
#include "OffloadEntry.h"
#include "omptarget.h"
#include "rtl.h"

#include "OpenMP/Mapping.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"

#include "GlobalHandler.h"
#include "OffloadAPI.h"
#include "PluginInterface.h"
#include "Program.h"

using GenericPluginTy = llvm::omp::target::plugin::GenericPluginTy;

// Forward declarations.
struct __tgt_bin_desc;
struct __tgt_target_table;

/// Kernel launch-geometry properties.
struct KernelLaunchInfoTy {
  const char *Name = nullptr;
  llvm::MemoryBufferRef Image;
  llvm::MemoryBufferRef IRImage;
  uint32_t MaxNumThreads = 0;
  uint32_t PreferredNumThreads = 0;
  uint32_t ReductionDataSize = 0;
  uint32_t StaticBlockMemSize = 0;
  llvm::omp::OMPTgtExecModeFlags Mode = llvm::omp::OMP_TGT_EXEC_MODE_BARE;

  bool isBareMode() const { return Mode == llvm::omp::OMP_TGT_EXEC_MODE_BARE; }
  bool isGenericMode() const {
    return Mode == llvm::omp::OMP_TGT_EXEC_MODE_GENERIC;
  }
  bool isGenericSPMDMode() const {
    return Mode == llvm::omp::OMP_TGT_EXEC_MODE_GENERIC_SPMD;
  }
  bool isSPMDMode() const { return Mode == llvm::omp::OMP_TGT_EXEC_MODE_SPMD; }
  bool isNoLoopMode() const {
    return Mode == llvm::omp::OMP_TGT_EXEC_MODE_SPMD_NO_LOOP;
  }
  bool isGridStrideLoopMode() const {
    return Mode == llvm::omp::OMP_TGT_EXEC_MODE_SPMD_GRID_STRIDE_LOOP;
  }

  static const char *getExecutionModeName(llvm::omp::OMPTgtExecModeFlags Mode) {
    switch (Mode) {
    case llvm::omp::OMP_TGT_EXEC_MODE_BARE:
      return "BARE";
    case llvm::omp::OMP_TGT_EXEC_MODE_SPMD:
      return "SPMD";
    case llvm::omp::OMP_TGT_EXEC_MODE_GENERIC:
      return "Generic";
    case llvm::omp::OMP_TGT_EXEC_MODE_GENERIC_SPMD:
      return "Generic-SPMD";
    case llvm::omp::OMP_TGT_EXEC_MODE_SPMD_NO_LOOP:
      return "SPMD-No-Loop";
    case llvm::omp::OMP_TGT_EXEC_MODE_SPMD_GRID_STRIDE_LOOP:
      return "SPMD-Grid-Stride-Loop";
    }
    return "Unknown";
  }

  const char *getExecutionModeName() const {
    return getExecutionModeName(Mode);
  }
};

/// Device properties that drive the computation of the launch geometry of a
/// kernel.
struct DeviceGeometryPropsTy {
  /// Number of threads in a warp/wavefront.
  uint32_t WarpSize = 0;
  /// Maximum number of threads a block can be launched with.
  uint32_t ThreadLimit = 0;
  /// Number of threads per block to use when the user does not request one.
  uint32_t DefaultNumThreads = 0;
  /// Number of blocks to use when the user does not request one.
  uint32_t DefaultNumBlocks = 0;
  /// Maximum number of blocks that can be launched.
  uint32_t MaxNumBlocks = 0;
  /// Maximum number of threads (blocks * threads per block) a launch can have.
  uint64_t MaxGridSize = 0;
  /// Maximum amount of shared memory, in bytes, a block can use in a kernel.
  uint64_t MaxBlockSharedMemSize = 0;
  /// Number of compute units, or zero if the device does not report it.
  uint32_t NumComputeUnits = 0;
  /// Backend of the platform of the device.
  ol_platform_backend_t Backend = OL_PLATFORM_BACKEND_UNKNOWN;
};

struct PluginManager;

struct DeviceTy {
  int32_t DeviceID;
  GenericPluginTy *RTL = nullptr;
  int32_t RTLDeviceID = -1;
  ol_platform_handle_t Platform;
  ol_device_handle_t DeviceHandle;
  PluginManager *PM;
  ol_context_handle_t Context = nullptr;

  DeviceTy(int32_t DeviceID, ol_platform_handle_t Platform,
           ol_device_handle_t DeviceHandle, PluginManager *PM);
  // DeviceTy is not copyable
  DeviceTy(const DeviceTy &D) = delete;
  DeviceTy &operator=(const DeviceTy &D) = delete;

  ~DeviceTy();

  /// Try to initialize the device and return any failure.
  llvm::Error init();

  /// Deinitialize the OpenMP device.
  llvm::Error deinit();

  /// Provide access to the mapping handler.
  MappingInfoTy &getMappingInfo() { return MappingInfo; }

  /// Whether this device executes kernels on the host CPU.
  bool isHostDevice() const { return IsHostDevice; }

  /// Get this device's backend.
  ol_platform_backend_t getBackend() const { return Backend; }

  /// Load \p Img onto the device and return the resulting program.
  llvm::Expected<ProgramTy> loadBinary(__tgt_device_image *Img);

  /// Resolve the device address of the global variable \p Name in \p Program,
  /// recording it for kernel record/replay if recording is currently active.
  llvm::Expected<void *> getAndRecordGlobalAddress(const ProgramTy &Program,
                                                   const char *Name,
                                                   size_t ExpectedSize);

  // device memory allocation/deallocation routines
  /// Allocates \p Size bytes on the device, host or shared memory space
  /// (depending on \p Kind) and returns the address/nullptr when
  /// succeeds/fails. \p HstPtr is an address of the host data which the
  /// allocated target data will be associated with. If it is unknown, the
  /// default value of \p HstPtr is nullptr. Note: this function doesn't do
  /// pointer association. Actually, all the __tgt_rtl_data_alloc
  /// implementations ignore \p HstPtr. \p Kind dictates what allocator should
  /// be used (host, shared, device).
  void *allocData(int64_t Size, void *HstPtr = nullptr,
                  int32_t Kind = TARGET_ALLOC_DEFAULT);

  /// Deallocates memory which \p TgtPtrBegin points at and returns
  /// OFFLOAD_SUCCESS/OFFLOAD_FAIL when succeeds/fails. p Kind dictates what
  /// allocator should be used (host, shared, device).
  int32_t deleteData(void *TgtPtrBegin, int32_t Kind = TARGET_ALLOC_DEFAULT);

  /// Allocate \p Size bytes of \p Type memory in the device's context, which,
  /// unlike allocData, caches the allocation for reuse. \p Type must be
  /// OL_ALLOC_TYPE_DEVICE or OL_ALLOC_TYPE_MANAGED. Free it with freeCtxMem.
  llvm::Expected<void *> allocCtxMem(int64_t Size, ol_alloc_type_t Type);

  /// Free memory allocated with allocCtxMem.
  llvm::Error freeCtxMem(void *Ptr);

  // Data transfer. The transfer is enqueued on the queue associated with
  // \p AsyncInfo; the *Sync variants below do not need a queue and always
  // transfer the data synchronously.
  // Copy data from host to device
  int32_t submitData(void *TgtPtrBegin, void *HstPtrBegin, int64_t Size,
                     AsyncInfoTy &AsyncInfo,
                     HostDataToTargetTy *Entry = nullptr,
                     MappingInfoTy::HDTTMapAccessorTy *HDTTMapPtr = nullptr,
                     const ident_t *Loc = nullptr, const char *Name = nullptr);

  // Copy data from device back to host
  int32_t retrieveData(void *HstPtrBegin, void *TgtPtrBegin, int64_t Size,
                       AsyncInfoTy &AsyncInfo,
                       HostDataToTargetTy *Entry = nullptr,
                       MappingInfoTy::HDTTMapAccessorTy *HDTTMapPtr = nullptr,
                       const ident_t *Loc = nullptr);

  // Copy data from host to device synchronously
  int32_t submitDataSync(void *TgtPtrBegin, void *HstPtrBegin, int64_t Size,
                         HostDataToTargetTy *Entry = nullptr,
                         MappingInfoTy::HDTTMapAccessorTy *HDTTMapPtr = nullptr,
                         const ident_t *Loc = nullptr,
                         const char *Name = nullptr);

  // Copy data from device back to host synchronously
  int32_t
  retrieveDataSync(void *HstPtrBegin, void *TgtPtrBegin, int64_t Size,
                   HostDataToTargetTy *Entry = nullptr,
                   MappingInfoTy::HDTTMapAccessorTy *HDTTMapPtr = nullptr,
                   const ident_t *Loc = nullptr);

  // Copy data from current device to destination device directly
  int32_t dataExchange(void *SrcPtr, DeviceTy &DstDev, void *DstPtr,
                       int64_t Size, AsyncInfoTy &AsyncInfo);

  /// Register (and, if \p LockMemory, page-lock) the host buffer \p HstPtr
  /// with \p Size bytes, returning the device-accessible pointer.
  llvm::Expected<void *> registerMemory(void *HstPtr, int64_t Size,
                                        bool LockMemory = true);

  /// Unregister (and, if \p UnlockMemory, page-unlock) a host buffer
  /// previously registered via registerMemory.
  llvm::Error unregisterMemory(void *HstPtr, bool UnlockMemory = true);

  /// Notify the plugin about a new mapping starting at the host address
  /// \p HstPtr and \p Size bytes.
  int32_t notifyDataMapped(void *HstPtr, int64_t Size);

  /// Notify the plugin about an existing mapping being unmapped starting at
  /// the host address \p HstPtr.
  int32_t notifyDataUnmapped(void *HstPtr);

  // Launch the kernel identified by \p TgtEntryPtr with the given arguments.
  int32_t launchKernel(void *TgtEntryPtr, void **TgtVarsPtr,
                       ptrdiff_t *TgtOffsets, KernelArgsTy &KernelArgs,
                       KernelReplayOutcomeTy *ReplayOutcome,
                       AsyncInfoTy &AsyncInfo);

  /// Synchronize device/queue/event based on \p AsyncInfo and return
  /// OFFLOAD_SUCCESS/OFFLOAD_FAIL when succeeds/fails.
  int32_t synchronize(AsyncInfoTy &AsyncInfo);

  /// Query for device/queue/event based completion on \p AsyncInfo in a
  /// non-blocking manner and return OFFLOAD_SUCCESS/OFFLOAD_FAIL when
  /// succeeds/fails. Must be called multiple times until AsyncInfo is
  /// completed and AsyncInfo.isDone() returns true.
  int32_t queryAsync(AsyncInfoTy &AsyncInfo);

  /// Event related interfaces.
  /// {
  /// Record the event based on status in AsyncInfo->Queue at the moment the
  /// function is called. If \p Event already holds an event, it is replaced
  /// with a new one representing the current queue state and the old event
  /// is destroyed.
  int32_t recordEvent(void **Event, AsyncInfoTy &AsyncInfo);

  /// Wait for an event. This function can be blocking or non-blocking,
  /// depending on the implementation. It is expected to set a dependence on the
  /// event such that corresponding operations shall only start once the event
  /// is fulfilled.
  int32_t waitEvent(void *Event, AsyncInfoTy &AsyncInfo);

  /// Synchronize the event. It is expected to block the thread.
  int32_t syncEvent(void *Event);

  /// Destroy the event.
  int32_t destroyEvent(void *Event);
  /// }

  /// Print all offload entries to stderr.
  void dumpOffloadEntries();

  /// Ask the device whether the runtime should use auto zero-copy.
  bool useAutoZeroCopy();

  /// Ask the device whether the storage is accessible.
  bool isAccessiblePtr(const void *Ptr, size_t Size);

  /// Check if there are pending images for this device.
  bool hasPendingImages() const { return HasPendingImages; }

  /// Indicate that there are pending images for this device or not.
  void setHasPendingImages(bool V) { HasPendingImages = V; }

  /// Return the unique identifier of the device.
  llvm::StringRef getUid() const { return Uid; }

  /// Get information from the device.
  template <typename T> T getInfo(ol_device_info_t Info) const {
    T Value{};
    if (olGetDeviceInfo(DeviceHandle, Info, sizeof(Value), &Value))
      return T{};
    return Value;
  }

  /// Get string information from the device into \p Out, which is left empty
  /// if the information cannot be obtained. \p Out is always null terminated.
  template <unsigned N>
  void getInfoString(ol_device_info_t Info, llvm::SmallString<N> &Out) const {
    size_t Size = 0;
    if (!olGetDeviceInfoSize(DeviceHandle, Info, &Size) && Size > 0) {
      Out.resize(Size);
      if (olGetDeviceInfo(DeviceHandle, Info, Size, Out.data()))
        Out.clear();
      else
        // Drop the null terminator from the length, c_str() puts it back.
        Out.pop_back();
    } else {
      Out.clear();
    }
    // Make sure the data can be handed out as a C string.
    Out.c_str();
  }

  /// Return the maximum number of threads per team this device supports,
  /// further limited by OMP_TEAMS_THREAD_LIMIT if the user set it.
  uint32_t getThreadLimit() const;

  /// Return the number of threads per team to use when the user does not
  /// request one.
  uint32_t getDefaultNumThreads() const {
    return GeometryProps.DefaultNumThreads;
  }

  /// Return the maximum amount of shared memory, in bytes, a team can use in
  /// any kernel.
  uint64_t getMaxBlockSharedMemSize() const {
    return GeometryProps.MaxBlockSharedMemSize;
  }

  /// Record the launch-geometry properties for the kernel at \p KernelPtr,
  /// read once at registration time from its "<name>_kernel_environment"
  /// device global.
  void setKernelLaunchInfo(void *KernelPtr, KernelLaunchInfoTy Info) {
    (*KernelLaunchInfoMap.getExclusiveAccessor())[KernelPtr] = Info;
  }

  /// Return the launch-geometry properties recorded for the kernel at
  /// \p KernelPtr, or a default-constructed KernelLaunchInfoTy if none were
  /// recorded.
  KernelLaunchInfoTy getKernelLaunchInfo(void *KernelPtr) {
    return (*KernelLaunchInfoMap.getExclusiveAccessor())[KernelPtr];
  }

private:
  /// Query the properties used to compute the launch geometry of a kernel.
  llvm::Error initGeometryProps();

  /// Copy data from host to device. A null \p Queue means the transfer is
  /// performed synchronously.
  int32_t submitDataImpl(void *TgtPtrBegin, void *HstPtrBegin, int64_t Size,
                         ol_queue_handle_t Queue, HostDataToTargetTy *Entry,
                         MappingInfoTy::HDTTMapAccessorTy *HDTTMapPtr,
                         const ident_t *Loc, const char *Name);

  /// Copy data from device back to host. A null \p Queue means the transfer is
  /// performed synchronously.
  int32_t retrieveDataImpl(void *HstPtrBegin, void *TgtPtrBegin, int64_t Size,
                           ol_queue_handle_t Queue, HostDataToTargetTy *Entry,
                           MappingInfoTy::HDTTMapAccessorTy *HDTTMapPtr,
                           const ident_t *Loc);

  /// Unique identifier of the device.
  llvm::SmallString<32> Uid;

  /// Properties used to compute the launch geometry of a kernel.
  DeviceGeometryPropsTy GeometryProps;

  /// All offload entries available on this device.
  using DeviceOffloadEntriesMapTy =
      llvm::DenseMap<llvm::StringRef, OffloadEntryTy>;
  ProtectedObj<DeviceOffloadEntriesMapTy> DeviceOffloadEntries;

  /// Launch-geometry properties for each kernel registered on this device.
  using KernelLaunchInfoMapTy = llvm::DenseMap<void *, KernelLaunchInfoTy>;
  ProtectedObj<KernelLaunchInfoMapTy> KernelLaunchInfoMap;

  /// Whether this is a host device.
  bool IsHostDevice = false;

  /// The backend of this device.
  ol_platform_backend_t Backend = OL_PLATFORM_BACKEND_UNKNOWN;

  /// Handler to collect and organize host-2-device mapping information.
  MappingInfoTy MappingInfo;

  /// Flag to indicate pending images (true after construction).
  bool HasPendingImages = true;

  /// Indicate whether mapped host buffers should be locked automatically.
  bool LockMappedBuffers = false;

  /// Indicate whether failures when locking mapped buffers should be ignored.
  bool IgnoreLockMappedFailures = true;
};

#endif
