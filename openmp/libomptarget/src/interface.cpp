//===-------- interface.cpp - Target independent OpenMP target RTL --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implementation of the interface to be used by Clang during the codegen of a
// target region.
//
//===----------------------------------------------------------------------===//

#include "OmptCallback.h"
#include "OmptInterface.h"
#include "device.h"
#include "omptarget.h"
#include "private.h"
#include "rtl.h"

#include "Utilities.h"

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <type_traits>

#ifdef OMPT_SUPPORT
using namespace llvm::omp::target::ompt;
#endif

////////////////////////////////////////////////////////////////////////////////
/// adds requires flags
EXTERN void __tgt_register_requires(int64_t Flags) {
  TIMESCOPE();
  PM->RTLs.registerRequires(Flags);
}

////////////////////////////////////////////////////////////////////////////////
/// adds a target shared library to the target execution image
EXTERN void __tgt_register_lib(__tgt_bin_desc *Desc) {
  TIMESCOPE();
  if (PM->maybeDelayRegisterLib(Desc))
    return;

  for (auto &RTL : PM->RTLs.AllRTLs) {
    if (RTL.register_lib) {
      if ((*RTL.register_lib)(Desc) != OFFLOAD_SUCCESS) {
        DP("Could not register library with %s", RTL.RTLName.c_str());
      }
    }
  }
  PM->RTLs.registerLib(Desc);
}

////////////////////////////////////////////////////////////////////////////////
/// Initialize all available devices without registering any image
EXTERN void __tgt_init_all_rtls() { PM->RTLs.initAllRTLs(); }

////////////////////////////////////////////////////////////////////////////////
/// unloads a target shared library
EXTERN void __tgt_unregister_lib(__tgt_bin_desc *Desc) {
  TIMESCOPE();
  PM->RTLs.unregisterLib(Desc);
  for (auto &RTL : PM->RTLs.UsedRTLs) {
    if (RTL->unregister_lib) {
      if ((*RTL->unregister_lib)(Desc) != OFFLOAD_SUCCESS) {
        DP("Could not register library with %s", RTL->RTLName.c_str());
      }
    }
  }
}

template <typename TargetAsyncInfoTy>
static inline void
targetDataMapper(ident_t *Loc, int64_t DeviceId, int32_t ArgNum,
                 void **ArgsBase, void **Args, int64_t *ArgSizes,
                 int64_t *ArgTypes, map_var_info_t *ArgNames, void **ArgMappers,
                 TargetDataFuncPtrTy TargetDataFunction,
                 const char *RegionTypeMsg, const char *RegionName) {
  static_assert(std::is_convertible_v<TargetAsyncInfoTy, AsyncInfoTy>,
                "TargetAsyncInfoTy must be convertible to AsyncInfoTy.");

  TIMESCOPE_WITH_RTM_AND_IDENT(RegionTypeMsg, Loc);

  DP("Entering data %s region for device %" PRId64 " with %d mappings\n",
     RegionName, DeviceId, ArgNum);

  if (checkDeviceAndCtors(DeviceId, Loc)) {
    DP("Not offloading to device %" PRId64 "\n", DeviceId);
    return;
  }

  if (getInfoLevel() & OMP_INFOTYPE_KERNEL_ARGS)
    printKernelArguments(Loc, DeviceId, ArgNum, ArgSizes, ArgTypes, ArgNames,
                         RegionTypeMsg);
#ifdef OMPTARGET_DEBUG
  for (int I = 0; I < ArgNum; ++I) {
    DP("Entry %2d: Base=" DPxMOD ", Begin=" DPxMOD ", Size=%" PRId64
       ", Type=0x%" PRIx64 ", Name=%s\n",
       I, DPxPTR(ArgsBase[I]), DPxPTR(Args[I]), ArgSizes[I], ArgTypes[I],
       (ArgNames) ? getNameFromMapping(ArgNames[I]).c_str() : "unknown");
  }
#endif

  DeviceTy &Device = *PM->Devices[DeviceId];
  TargetAsyncInfoTy TargetAsyncInfo(Device);
  AsyncInfoTy &AsyncInfo = TargetAsyncInfo;

  /// RAII to establish tool anchors before and after data begin / end / update
  OMPT_IF_BUILT(assert((TargetDataFunction == targetDataBegin ||
                        TargetDataFunction == targetDataEnd ||
                        TargetDataFunction == targetDataUpdate) &&
                       "Encountered unexpected TargetDataFunction during "
                       "execution of targetDataMapper");
                auto CallbackFunctions =
                    (TargetDataFunction == targetDataBegin)
                        ? RegionInterface.getCallbacks<ompt_target_enter_data>()
                    : (TargetDataFunction == targetDataEnd)
                        ? RegionInterface.getCallbacks<ompt_target_exit_data>()
                        : RegionInterface.getCallbacks<ompt_target_update>();
                InterfaceRAII TargetDataRAII(CallbackFunctions, DeviceId,
                                             OMPT_GET_RETURN_ADDRESS(0));)

  int Rc = OFFLOAD_SUCCESS;
  Rc = TargetDataFunction(Loc, Device, ArgNum, ArgsBase, Args, ArgSizes,
                          ArgTypes, ArgNames, ArgMappers, AsyncInfo,
                          false /* FromMapper */);

  if (Rc == OFFLOAD_SUCCESS)
    Rc = AsyncInfo.synchronize();

  handleTargetOutcome(Rc == OFFLOAD_SUCCESS, Loc);
}

/// creates host-to-target data mapping, stores it in the
/// libomptarget.so internal structure (an entry in a stack of data maps)
/// and passes the data to the device.
EXTERN void __tgt_target_data_begin_mapper(ident_t *Loc, int64_t DeviceId,
                                           int32_t ArgNum, void **ArgsBase,
                                           void **Args, int64_t *ArgSizes,
                                           int64_t *ArgTypes,
                                           map_var_info_t *ArgNames,
                                           void **ArgMappers) {

  targetDataMapper<AsyncInfoTy>(Loc, DeviceId, ArgNum, ArgsBase, Args, ArgSizes,
                                ArgTypes, ArgNames, ArgMappers, targetDataBegin,
                                "Entering OpenMP data region with being_mapper",
                                "begin");
}

EXTERN void __tgt_target_data_begin_nowait_mapper(
    ident_t *Loc, int64_t DeviceId, int32_t ArgNum, void **ArgsBase,
    void **Args, int64_t *ArgSizes, int64_t *ArgTypes, map_var_info_t *ArgNames,
    void **ArgMappers, int32_t DepNum, void *DepList, int32_t NoAliasDepNum,
    void *NoAliasDepList) {

  targetDataMapper<TaskAsyncInfoWrapperTy>(
      Loc, DeviceId, ArgNum, ArgsBase, Args, ArgSizes, ArgTypes, ArgNames,
      ArgMappers, targetDataBegin,
      "Entering OpenMP data region with being_nowait_mapper", "begin");
}

/// passes data from the target, releases target memory and destroys
/// the host-target mapping (top entry from the stack of data maps)
/// created by the last __tgt_target_data_begin.
EXTERN void __tgt_target_data_end_mapper(ident_t *Loc, int64_t DeviceId,
                                         int32_t ArgNum, void **ArgsBase,
                                         void **Args, int64_t *ArgSizes,
                                         int64_t *ArgTypes,
                                         map_var_info_t *ArgNames,
                                         void **ArgMappers) {

  targetDataMapper<AsyncInfoTy>(Loc, DeviceId, ArgNum, ArgsBase, Args, ArgSizes,
                                ArgTypes, ArgNames, ArgMappers, targetDataEnd,
                                "Exiting OpenMP data region with end_mapper",
                                "end");
}

EXTERN void __tgt_target_data_end_nowait_mapper(
    ident_t *Loc, int64_t DeviceId, int32_t ArgNum, void **ArgsBase,
    void **Args, int64_t *ArgSizes, int64_t *ArgTypes, map_var_info_t *ArgNames,
    void **ArgMappers, int32_t DepNum, void *DepList, int32_t NoAliasDepNum,
    void *NoAliasDepList) {

  targetDataMapper<TaskAsyncInfoWrapperTy>(
      Loc, DeviceId, ArgNum, ArgsBase, Args, ArgSizes, ArgTypes, ArgNames,
      ArgMappers, targetDataEnd,
      "Exiting OpenMP data region with end_nowait_mapper", "end");
}

EXTERN void __tgt_target_data_update_mapper(ident_t *Loc, int64_t DeviceId,
                                            int32_t ArgNum, void **ArgsBase,
                                            void **Args, int64_t *ArgSizes,
                                            int64_t *ArgTypes,
                                            map_var_info_t *ArgNames,
                                            void **ArgMappers) {

  targetDataMapper<AsyncInfoTy>(
      Loc, DeviceId, ArgNum, ArgsBase, Args, ArgSizes, ArgTypes, ArgNames,
      ArgMappers, targetDataUpdate,
      "Updating data within the OpenMP data region with update_mapper",
      "update");
}

EXTERN void __tgt_target_data_update_nowait_mapper(
    ident_t *Loc, int64_t DeviceId, int32_t ArgNum, void **ArgsBase,
    void **Args, int64_t *ArgSizes, int64_t *ArgTypes, map_var_info_t *ArgNames,
    void **ArgMappers, int32_t DepNum, void *DepList, int32_t NoAliasDepNum,
    void *NoAliasDepList) {
  targetDataMapper<TaskAsyncInfoWrapperTy>(
      Loc, DeviceId, ArgNum, ArgsBase, Args, ArgSizes, ArgTypes, ArgNames,
      ArgMappers, targetDataUpdate,
      "Updating data within the OpenMP data region with update_nowait_mapper",
      "update");
}

static KernelArgsTy *upgradeKernelArgs(KernelArgsTy *KernelArgs,
                                       KernelArgsTy &LocalKernelArgs,
                                       int32_t NumTeams, int32_t ThreadLimit) {
  if (KernelArgs->Version > 2)
    DP("Unexpected ABI version: %u\n", KernelArgs->Version);

  if (KernelArgs->Version == 1) {
    LocalKernelArgs.Version = 2;
    LocalKernelArgs.NumArgs = KernelArgs->NumArgs;
    LocalKernelArgs.ArgBasePtrs = KernelArgs->ArgBasePtrs;
    LocalKernelArgs.ArgPtrs = KernelArgs->ArgPtrs;
    LocalKernelArgs.ArgSizes = KernelArgs->ArgSizes;
    LocalKernelArgs.ArgTypes = KernelArgs->ArgTypes;
    LocalKernelArgs.ArgNames = KernelArgs->ArgNames;
    LocalKernelArgs.ArgMappers = KernelArgs->ArgMappers;
    LocalKernelArgs.Tripcount = KernelArgs->Tripcount;
    LocalKernelArgs.Flags = KernelArgs->Flags;
    LocalKernelArgs.DynCGroupMem = 0;
    LocalKernelArgs.NumTeams[0] = NumTeams;
    LocalKernelArgs.NumTeams[1] = 0;
    LocalKernelArgs.NumTeams[2] = 0;
    LocalKernelArgs.ThreadLimit[0] = ThreadLimit;
    LocalKernelArgs.ThreadLimit[1] = 0;
    LocalKernelArgs.ThreadLimit[2] = 0;
    return &LocalKernelArgs;
  }

  return KernelArgs;
}

template <typename TargetAsyncInfoTy>
static inline int targetKernel(ident_t *Loc, int64_t DeviceId, int32_t NumTeams,
                               int32_t ThreadLimit, void *HostPtr,
                               KernelArgsTy *KernelArgs) {
  static_assert(std::is_convertible_v<TargetAsyncInfoTy, AsyncInfoTy>,
                "Target AsyncInfoTy must be convertible to AsyncInfoTy.");

  TIMESCOPE_WITH_IDENT(Loc);

  DP("Entering target region for device %" PRId64 " with entry point " DPxMOD
     "\n",
     DeviceId, DPxPTR(HostPtr));

  if (checkDeviceAndCtors(DeviceId, Loc)) {
    DP("Not offloading to device %" PRId64 "\n", DeviceId);
    return OMP_TGT_FAIL;
  }

  bool IsTeams = NumTeams != -1;
  if (!IsTeams)
    KernelArgs->NumTeams[0] = NumTeams = 1;

  // Auto-upgrade kernel args version 1 to 2.
  KernelArgsTy LocalKernelArgs;
  KernelArgs =
      upgradeKernelArgs(KernelArgs, LocalKernelArgs, NumTeams, ThreadLimit);

  assert(KernelArgs->NumTeams[0] == static_cast<uint32_t>(NumTeams) &&
         !KernelArgs->NumTeams[1] && !KernelArgs->NumTeams[2] &&
         "OpenMP interface should not use multiple dimensions");
  assert(KernelArgs->ThreadLimit[0] == static_cast<uint32_t>(ThreadLimit) &&
         !KernelArgs->ThreadLimit[1] && !KernelArgs->ThreadLimit[2] &&
         "OpenMP interface should not use multiple dimensions");

  if (getInfoLevel() & OMP_INFOTYPE_KERNEL_ARGS)
    printKernelArguments(Loc, DeviceId, KernelArgs->NumArgs,
                         KernelArgs->ArgSizes, KernelArgs->ArgTypes,
                         KernelArgs->ArgNames, "Entering OpenMP kernel");
#ifdef OMPTARGET_DEBUG
  for (uint32_t I = 0; I < KernelArgs->NumArgs; ++I) {
    DP("Entry %2d: Base=" DPxMOD ", Begin=" DPxMOD ", Size=%" PRId64
       ", Type=0x%" PRIx64 ", Name=%s\n",
       I, DPxPTR(KernelArgs->ArgBasePtrs[I]), DPxPTR(KernelArgs->ArgPtrs[I]),
       KernelArgs->ArgSizes[I], KernelArgs->ArgTypes[I],
       (KernelArgs->ArgNames)
           ? getNameFromMapping(KernelArgs->ArgNames[I]).c_str()
           : "unknown");
  }
#endif

  DeviceTy &Device = *PM->Devices[DeviceId];
  TargetAsyncInfoTy TargetAsyncInfo(Device);
  AsyncInfoTy &AsyncInfo = TargetAsyncInfo;
  /// RAII to establish tool anchors before and after target region
  OMPT_IF_BUILT(InterfaceRAII TargetRAII(
                    RegionInterface.getCallbacks<ompt_target>(), DeviceId,
                    /* CodePtr */ OMPT_GET_RETURN_ADDRESS(0));)

  int Rc = OFFLOAD_SUCCESS;
  Rc = target(Loc, Device, HostPtr, *KernelArgs, AsyncInfo);

  if (Rc == OFFLOAD_SUCCESS)
    Rc = AsyncInfo.synchronize();

  handleTargetOutcome(Rc == OFFLOAD_SUCCESS, Loc);
  assert(Rc == OFFLOAD_SUCCESS && "__tgt_target_kernel unexpected failure!");

  return OMP_TGT_SUCCESS;
}

/// Implements a kernel entry that executes the target region on the specified
/// device.
///
/// \param Loc Source location associated with this target region.
/// \param DeviceId The device to execute this region, -1 indicated the default.
/// \param NumTeams Number of teams to launch the region with, -1 indicates a
///                 non-teams region and 0 indicates it was unspecified.
/// \param ThreadLimit Limit to the number of threads to use in the kernel
///                    launch, 0 indicates it was unspecified.
/// \param HostPtr  The pointer to the host function registered with the kernel.
/// \param Args     All arguments to this kernel launch (see struct definition).
EXTERN int __tgt_target_kernel(ident_t *Loc, int64_t DeviceId, int32_t NumTeams,
                               int32_t ThreadLimit, void *HostPtr,
                               KernelArgsTy *KernelArgs) {
  if (KernelArgs->Flags.NoWait)
    return targetKernel<TaskAsyncInfoWrapperTy>(
        Loc, DeviceId, NumTeams, ThreadLimit, HostPtr, KernelArgs);
  else
    return targetKernel<AsyncInfoTy>(Loc, DeviceId, NumTeams, ThreadLimit,
                                     HostPtr, KernelArgs);
}

/// Activates the record replay mechanism.
/// \param DeviceId The device identifier to execute the target region.
/// \param MemorySize The number of bytes to be (pre-)allocated
///                   by the bump allocator
/// /param IsRecord Activates the record replay mechanism in
///                 'record' mode or 'replay' mode.
/// /param SaveOutput Store the device memory after kernel
///                   execution on persistent storage
EXTERN int __tgt_activate_record_replay(int64_t DeviceId, uint64_t MemorySize,
                                        bool IsRecord, bool SaveOutput) {
  if (!deviceIsReady(DeviceId)) {
    DP("Device %" PRId64 " is not ready\n", DeviceId);
    return OMP_TGT_FAIL;
  }

  DeviceTy &Device = *PM->Devices[DeviceId];
  [[maybe_unused]] int Rc =
      target_activate_rr(Device, MemorySize, IsRecord, SaveOutput);
  assert(Rc == OFFLOAD_SUCCESS &&
         "__tgt_activate_record_replay unexpected failure!");
  return OMP_TGT_SUCCESS;
};

/// Implements a target kernel entry that replays a pre-recorded kernel.
/// \param Loc Source location associated with this target region (unused).
/// \param DeviceId The device identifier to execute the target region.
/// \param HostPtr A pointer to an address that uniquely identifies the kernel.
/// \param DeviceMemory A pointer to an array storing device memory data to move
///                     prior to kernel execution.
/// \param DeviceMemorySize The size of the above device memory data in bytes.
/// \param TgtArgs An array of pointers of the pre-recorded target kernel
///                arguments.
/// \param TgtOffsets An array of pointers of the pre-recorded target kernel
///                   argument offsets.
/// \param NumArgs The number of kernel arguments.
/// \param NumTeams Number of teams to launch the target region with.
/// \param ThreadLimit Limit to the number of threads to use in kernel
///                    execution.
/// \param LoopTripCount The pre-recorded value of the loop tripcount, if any.
/// \return OMP_TGT_SUCCESS on success, OMP_TGT_FAIL on failure.
EXTERN int __tgt_target_kernel_replay(ident_t *Loc, int64_t DeviceId,
                                      void *HostPtr, void *DeviceMemory,
                                      int64_t DeviceMemorySize, void **TgtArgs,
                                      ptrdiff_t *TgtOffsets, int32_t NumArgs,
                                      int32_t NumTeams, int32_t ThreadLimit,
                                      uint64_t LoopTripCount) {

  if (checkDeviceAndCtors(DeviceId, Loc)) {
    DP("Not offloading to device %" PRId64 "\n", DeviceId);
    return OMP_TGT_FAIL;
  }
  DeviceTy &Device = *PM->Devices[DeviceId];
  /// RAII to establish tool anchors before and after target region
  OMPT_IF_BUILT(InterfaceRAII TargetRAII(
                    RegionInterface.getCallbacks<ompt_target>(), DeviceId,
                    /* CodePtr */ OMPT_GET_RETURN_ADDRESS(0));)

  AsyncInfoTy AsyncInfo(Device);
  int Rc = target_replay(Loc, Device, HostPtr, DeviceMemory, DeviceMemorySize,
                         TgtArgs, TgtOffsets, NumArgs, NumTeams, ThreadLimit,
                         LoopTripCount, AsyncInfo);
  if (Rc == OFFLOAD_SUCCESS)
    Rc = AsyncInfo.synchronize();
  handleTargetOutcome(Rc == OFFLOAD_SUCCESS, Loc);
  assert(Rc == OFFLOAD_SUCCESS &&
         "__tgt_target_kernel_replay unexpected failure!");
  return OMP_TGT_SUCCESS;
}

// Get the current number of components for a user-defined mapper.
EXTERN int64_t __tgt_mapper_num_components(void *RtMapperHandle) {
  TIMESCOPE();
  auto *MapperComponentsPtr = (struct MapperComponentsTy *)RtMapperHandle;
  int64_t Size = MapperComponentsPtr->Components.size();
  DP("__tgt_mapper_num_components(Handle=" DPxMOD ") returns %" PRId64 "\n",
     DPxPTR(RtMapperHandle), Size);
  return Size;
}

// Push back one component for a user-defined mapper.
EXTERN void __tgt_push_mapper_component(void *RtMapperHandle, void *Base,
                                        void *Begin, int64_t Size, int64_t Type,
                                        void *Name) {
  TIMESCOPE();
  DP("__tgt_push_mapper_component(Handle=" DPxMOD
     ") adds an entry (Base=" DPxMOD ", Begin=" DPxMOD ", Size=%" PRId64
     ", Type=0x%" PRIx64 ", Name=%s).\n",
     DPxPTR(RtMapperHandle), DPxPTR(Base), DPxPTR(Begin), Size, Type,
     (Name) ? getNameFromMapping(Name).c_str() : "unknown");
  auto *MapperComponentsPtr = (struct MapperComponentsTy *)RtMapperHandle;
  MapperComponentsPtr->Components.push_back(
      MapComponentInfoTy(Base, Begin, Size, Type, Name));
}

EXTERN void __tgt_set_info_flag(uint32_t NewInfoLevel) {
  std::atomic<uint32_t> &InfoLevel = getInfoLevelInternal();
  InfoLevel.store(NewInfoLevel);
  for (auto &R : PM->RTLs.AllRTLs) {
    if (R.set_info_flag)
      R.set_info_flag(NewInfoLevel);
  }
}

EXTERN int __tgt_print_device_info(int64_t DeviceId) {
  // Make sure the device is ready.
  if (!deviceIsReady(DeviceId)) {
    DP("Device %" PRId64 " is not ready\n", DeviceId);
    return OMP_TGT_FAIL;
  }

  return PM->Devices[DeviceId]->printDeviceInfo(
      PM->Devices[DeviceId]->RTLDeviceID);
}

EXTERN void __tgt_target_nowait_query(void **AsyncHandle) {
  if (!AsyncHandle || !*AsyncHandle) {
    FATAL_MESSAGE0(
        1, "Receive an invalid async handle from the current OpenMP task. Is "
           "this a target nowait region?\n");
  }

  // Exponential backoff tries to optimally decide if a thread should just query
  // for the device operations (work/spin wait on them) or block until they are
  // completed (use device side blocking mechanism). This allows the runtime to
  // adapt itself when there are a lot of long-running target regions in-flight.
  using namespace llvm::omp::target;
  static thread_local ExponentialBackoff QueryCounter(
      Int64Envar("OMPTARGET_QUERY_COUNT_MAX", 10),
      Int64Envar("OMPTARGET_QUERY_COUNT_THRESHOLD", 5),
      Envar<float>("OMPTARGET_QUERY_COUNT_BACKOFF_FACTOR", 0.5f));

  auto *AsyncInfo = (AsyncInfoTy *)*AsyncHandle;

  // If the thread is actively waiting on too many target nowait regions, we
  // should use the blocking sync type.
  if (QueryCounter.isAboveThreshold())
    AsyncInfo->SyncType = AsyncInfoTy::SyncTy::BLOCKING;

  if (const int Rc = AsyncInfo->synchronize())
    FATAL_MESSAGE0(1, "Error while querying the async queue for completion.\n");
  // If there are device operations still pending, return immediately without
  // deallocating the handle and increase the current thread query count.
  if (!AsyncInfo->isDone()) {
    QueryCounter.increment();
    return;
  }

  // When a thread successfully completes a target nowait region, we
  // exponentially backoff its query counter by the query factor.
  QueryCounter.decrement();

  // Delete the handle and unset it from the OpenMP task data.
  delete AsyncInfo;
  *AsyncHandle = nullptr;
}
