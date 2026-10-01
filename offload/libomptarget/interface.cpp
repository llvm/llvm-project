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

#include "OpenMP/OMPT/Interface.h"
#include "OffloadPolicy.h"
#include "OpenMP/Mapping.h"
#include "OpenMP/OMPT/Callback.h"
#include "OpenMP/OffloadRTL.h"
#include "OpenMP/TaskGraphPlan.h"
#include "OpenMP/omp.h"
#include "PluginManager.h"
#include "device.h"
#include "omptarget.h"
#include "private.h"

#include "Shared/EnvironmentVar.h"
#include "Shared/Profile.h"
#include "Shared/TaskGraph.h"
#include "Shared/TaskGraphBuilder.h"
#include "Shared/TaskGraphDebugDump.h"

#include "Utils/ExponentialBackoff.h"

#include "llvm/Frontend/OpenMP/OMPConstants.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/MathExtras.h"

#include <atomic>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <new>
#include <vector>

#ifdef OMPT_SUPPORT
using namespace llvm::omp::target::ompt;
#endif
using namespace llvm::omp::target::debug;
using namespace llvm::omp::target::helpers;

// If offload is enabled, ensure that device DeviceID has been initialized.
//
// The return bool indicates if the offload is to the host device
// There are three possible results:
// - Return false if the target device is ready for offload
// - Return true without reporting a runtime error if offload is
//   disabled, perhaps because the initial device was specified.
// - Report a runtime error and return true.
//
// If DeviceID == OFFLOAD_DEVICE_DEFAULT, set DeviceID to the default device.
// This step might be skipped if offload is disabled.
bool checkDevice(int64_t &DeviceID, ident_t *Loc) {
  if (OffloadPolicy::get(*PM).Kind == OffloadPolicy::DISABLED) {
    ODBG(ODT_Device) << "Offload is disabled";
    return true;
  }

  if (DeviceID == OFFLOAD_DEVICE_DEFAULT) {
    DeviceID = omp_get_default_device();
    ODBG(ODT_Device) << "Use default device id " << DeviceID;
  }

  // Proposed behavior for OpenMP 5.2 in OpenMP spec github issue 2669.
  if (omp_get_num_devices() == 0) {
    ODBG(ODT_Device) << "omp_get_num_devices() == 0 but offload is manadatory";
    handleTargetOutcome(false, Loc);
    return true;
  }

  if (isInitialDevice(static_cast<int>(DeviceID))) {
    ODBG(ODT_Device) << "Device is host (" << DeviceID
                     << "), returning as if offload is disabled";
    return true;
  }
  return false;
}

////////////////////////////////////////////////////////////////////////////////
/// adds requires flags
EXTERN void __tgt_register_requires(int64_t Flags) {
  MESSAGE("The %s function has been removed. Old OpenMP requirements will not "
          "be handled",
          __PRETTY_FUNCTION__);
}

EXTERN void __tgt_rtl_init() { initRuntime(); }
EXTERN void __tgt_rtl_deinit() { deinitRuntime(); }

////////////////////////////////////////////////////////////////////////////////
/// adds a target shared library to the target execution image
EXTERN void __tgt_register_lib(__tgt_bin_desc *Desc) {
  initRuntime();
  if (PM->delayRegisterLib(Desc))
    return;

  PM->registerLib(Desc);
}

////////////////////////////////////////////////////////////////////////////////
/// Initialize all available devices without registering any image
EXTERN void __tgt_init_all_rtls() {
  assert(PM && "Runtime not initialized");
  PM->initializeAllDevices();
}

////////////////////////////////////////////////////////////////////////////////
/// unloads a target shared library
EXTERN void __tgt_unregister_lib(__tgt_bin_desc *Desc) {
  PM->unregisterLib(Desc);

  deinitRuntime();
}

template <typename TargetAsyncInfoTy>
static inline void
targetData(ident_t *Loc, int64_t DeviceId, int32_t ArgNum, void **ArgsBase,
           void **Args, int64_t *ArgSizes, int64_t *ArgTypes,
           map_var_info_t *ArgNames, void **ArgMappers,
           TargetDataFuncPtrTy TargetDataFunction, const char *RegionTypeMsg,
           const char *RegionName) {
  assert(PM && "Runtime not initialized");
  static_assert(std::is_convertible_v<TargetAsyncInfoTy &, AsyncInfoTy &>,
                "TargetAsyncInfoTy must be convertible to AsyncInfoTy.");

  TIMESCOPE_WITH_DETAILS_AND_IDENT("Runtime: Data Copy",
                                   "NumArgs=" + std::to_string(ArgNum), Loc);

  ODBG(ODT_Interface) << "Entering data " << RegionName << " region for device "
                      << DeviceId << " with " << ArgNum << " mappings";

  if (checkDevice(DeviceId, Loc)) {
    ODBG(ODT_Interface) << "Not offloading to device " << DeviceId;
    return;
  }

  if (getInfoLevel() & OMP_INFOTYPE_KERNEL_ARGS)
    printKernelArguments(Loc, DeviceId, ArgNum, ArgSizes, ArgTypes, ArgNames,
                         RegionTypeMsg);
  ODBG_OS(ODT_Kernel, [&](llvm::raw_ostream &Os) {
    for (int I = 0; I < ArgNum; ++I) {
      Os << "Entry " << llvm::format("%2d", I) << ": Base=" << ArgsBase[I]
         << ", Begin=" << Args[I] << ", Size=" << ArgSizes[I]
         << ", Type=" << llvm::format("0x%" PRIx64, ArgTypes[I]) << ", Name="
         << ((ArgNames) ? getNameFromMapping(ArgNames[I]) : "unknown") << "\n";
    }
  });

  auto DeviceOrErr = PM->getDevice(DeviceId);
  if (!DeviceOrErr)
    FATAL_MESSAGE(DeviceId, "%s", toString(DeviceOrErr.takeError()).c_str());

  TargetAsyncInfoTy TargetAsyncInfo(*DeviceOrErr);
  AsyncInfoTy &AsyncInfo = TargetAsyncInfo;

  /// RAII to establish tool anchors before and after data begin / end / update
  OMPT_IF_BUILT(assert((TargetDataFunction == targetDataBegin ||
                        TargetDataFunction == targetDataEnd ||
                        TargetDataFunction == targetDataUpdate) &&
                       "Encountered unexpected TargetDataFunction during "
                       "execution of targetData");
                auto CallbackFunctions =
                    (TargetDataFunction == targetDataBegin)
                        ? RegionInterface.getCallbacks<ompt_target_enter_data>()
                    : (TargetDataFunction == targetDataEnd)
                        ? RegionInterface.getCallbacks<ompt_target_exit_data>()
                        : RegionInterface.getCallbacks<ompt_target_update>();
                InterfaceRAII TargetDataRAII(CallbackFunctions, DeviceId,
                                             OMPT_GET_RETURN_ADDRESS);)

  int Rc = OFFLOAD_SUCCESS;

  // Allocate StateInfo for targetDataBegin and targetDataEnd to track
  // allocations, pointer attachments and deferred transfers.
  // This is not needed for targetDataUpdate.
  std::unique_ptr<StateInfoTy> StateInfo;
  if (TargetDataFunction == targetDataBegin ||
      TargetDataFunction == targetDataEnd)
    StateInfo = std::make_unique<StateInfoTy>();

  Rc = TargetDataFunction(Loc, *DeviceOrErr, ArgNum, ArgsBase, Args, ArgSizes,
                          ArgTypes, ArgNames, ArgMappers, AsyncInfo,
                          StateInfo.get(), /*FromMapper=*/false);

  if (Rc == OFFLOAD_SUCCESS) {
    // Process deferred ATTACH entries BEFORE synchronization
    if (StateInfo && !StateInfo->AttachEntries.empty())
      Rc = processAttachEntries(*DeviceOrErr, *StateInfo, AsyncInfo);

    if (Rc == OFFLOAD_SUCCESS) {
      printSyncInfo(Loc, DeviceId, AsyncInfo);
      Rc = AsyncInfo.synchronize();
    }
  }

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
  OMPT_IF_BUILT(ReturnAddressSetterRAII RA(__builtin_return_address(0)));
  targetData<AsyncInfoTy>(Loc, DeviceId, ArgNum, ArgsBase, Args, ArgSizes,
                          ArgTypes, ArgNames, ArgMappers, targetDataBegin,
                          "Entering OpenMP data region with being_mapper",
                          "begin");
}

EXTERN void __tgt_target_data_begin_nowait_mapper(
    ident_t *Loc, int64_t DeviceId, int32_t ArgNum, void **ArgsBase,
    void **Args, int64_t *ArgSizes, int64_t *ArgTypes, map_var_info_t *ArgNames,
    void **ArgMappers, int32_t DepNum, void *DepList, int32_t NoAliasDepNum,
    void *NoAliasDepList) {
  OMPT_IF_BUILT(ReturnAddressSetterRAII RA(__builtin_return_address(0)));
  targetData<TaskAsyncInfoWrapperTy>(
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
  OMPT_IF_BUILT(ReturnAddressSetterRAII RA(__builtin_return_address(0)));
  targetData<AsyncInfoTy>(Loc, DeviceId, ArgNum, ArgsBase, Args, ArgSizes,
                          ArgTypes, ArgNames, ArgMappers, targetDataEnd,
                          "Exiting OpenMP data region with end_mapper", "end");
}

EXTERN void __tgt_target_data_end_nowait_mapper(
    ident_t *Loc, int64_t DeviceId, int32_t ArgNum, void **ArgsBase,
    void **Args, int64_t *ArgSizes, int64_t *ArgTypes, map_var_info_t *ArgNames,
    void **ArgMappers, int32_t DepNum, void *DepList, int32_t NoAliasDepNum,
    void *NoAliasDepList) {
  OMPT_IF_BUILT(ReturnAddressSetterRAII RA(__builtin_return_address(0)));
  targetData<TaskAsyncInfoWrapperTy>(
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
  OMPT_IF_BUILT(ReturnAddressSetterRAII RA(__builtin_return_address(0)));
  targetData<AsyncInfoTy>(
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
  OMPT_IF_BUILT(ReturnAddressSetterRAII RA(__builtin_return_address(0)));
  targetData<TaskAsyncInfoWrapperTy>(
      Loc, DeviceId, ArgNum, ArgsBase, Args, ArgSizes, ArgTypes, ArgNames,
      ArgMappers, targetDataUpdate,
      "Updating data within the OpenMP data region with update_nowait_mapper",
      "update");
}

/// Holds dynamically allocated argument arrays when upgrading old-format
/// kernel arguments to include the dyn_ptr slot.
struct UpgradedArgBuffersTy {
  llvm::SmallVector<void *, 0> BasePtrs;
  llvm::SmallVector<void *, 0> Ptrs;
  llvm::SmallVector<int64_t, 0> Sizes;
  llvm::SmallVector<int64_t, 0> Types;
  llvm::SmallVector<map_var_info_t, 0> Names;
  llvm::SmallVector<void *, 0> Mappers;
};

/// Give the unused dimensions of \p KernelArgs' launch a size of one.
// FIXME: This is a WA to "calibrate" the bad work done in the front end.
// Delete this ugly code after the front end emits proper values.
static void correctMultiDim(KernelArgsTy &KernelArgs) {
  auto Correct = [](uint32_t (&Val)[3]) {
    if (Val[1] == 0)
      Val[1] = 1;
    if (Val[2] == 0)
      Val[2] = 1;
  };
  Correct(KernelArgs.UserThreadLimit);
  Correct(KernelArgs.UserNumBlocks);
}

static KernelArgsTy *upgradeKernelArgs(KernelArgsTy *KernelArgs,
                                       KernelArgsTy &LocalKernelArgs,
                                       UpgradedArgBuffersTy &Bufs,
                                       int32_t NumTeams, int32_t ThreadLimit) {
  if (KernelArgs->Version > OMP_KERNEL_ARG_VERSION)
    ODBG(ODT_Interface) << "Unexpected ABI version: " << KernelArgs->Version;

  // Versions before OMP_KERNEL_ARG_MIN_VERSION_WITH_DYN_PTR used an older
  // struct layout missing several fields. Reconstruct a complete struct.
  if (KernelArgs->Version < OMP_KERNEL_ARG_MIN_VERSION_WITH_DYN_PTR) {
    // Maintain the version so the runtime can match the device ABI.
    LocalKernelArgs.Version = KernelArgs->Version;
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
    LocalKernelArgs.UserNumBlocks[0] = NumTeams;
    LocalKernelArgs.UserNumBlocks[1] = 1;
    LocalKernelArgs.UserNumBlocks[2] = 1;
    LocalKernelArgs.UserThreadLimit[0] = ThreadLimit;
    LocalKernelArgs.UserThreadLimit[1] = 1;
    LocalKernelArgs.UserThreadLimit[2] = 1;
    return &LocalKernelArgs;
  }

  correctMultiDim(*KernelArgs);

  // Version 3 put the implicit argument at the front with no storage.
  if (KernelArgs->Version == OMP_KERNEL_ARG_MIN_VERSION_WITH_DYN_PTR) {
    uint32_t NewSize = KernelArgs->NumArgs + 1;

    Bufs.BasePtrs.resize(NewSize, nullptr);
    Bufs.Ptrs.resize(NewSize, nullptr);
    Bufs.Sizes.resize(NewSize, 0);
    Bufs.Types.resize(NewSize, 0);
    Bufs.Names.resize(NewSize, nullptr);
    Bufs.Mappers.resize(NewSize, nullptr);

    for (uint32_t I = 0; I < KernelArgs->NumArgs; ++I) {
      Bufs.BasePtrs[I] = KernelArgs->ArgBasePtrs[I];
      Bufs.Ptrs[I] = KernelArgs->ArgPtrs[I];
      Bufs.Sizes[I] = KernelArgs->ArgSizes[I];
      Bufs.Types[I] = KernelArgs->ArgTypes[I];
      if (KernelArgs->ArgNames)
        Bufs.Names[I] = KernelArgs->ArgNames[I];
      if (KernelArgs->ArgMappers)
        Bufs.Mappers[I] = KernelArgs->ArgMappers[I];
    }

    Bufs.Types[KernelArgs->NumArgs] =
        OMP_TGT_MAPTYPE_TARGET_PARAM | OMP_TGT_MAPTYPE_LITERAL;

    LocalKernelArgs = *KernelArgs;
    LocalKernelArgs.NumArgs = NewSize;
    LocalKernelArgs.ArgBasePtrs = Bufs.BasePtrs.data();
    LocalKernelArgs.ArgPtrs = Bufs.Ptrs.data();
    LocalKernelArgs.ArgSizes = Bufs.Sizes.data();
    LocalKernelArgs.ArgTypes = Bufs.Types.data();
    LocalKernelArgs.ArgNames = Bufs.Names.data();
    LocalKernelArgs.ArgMappers = Bufs.Mappers.data();
    return &LocalKernelArgs;
  }

  return KernelArgs;
}

template <typename TargetAsyncInfoTy>
static inline int targetKernel(ident_t *Loc, int64_t DeviceId, int32_t NumTeams,
                               int32_t ThreadLimit, void *HostPtr,
                               KernelArgsTy *KernelArgs) {
  assert(PM && "Runtime not initialized");
  static_assert(std::is_convertible_v<TargetAsyncInfoTy &, AsyncInfoTy &>,
                "Target AsyncInfoTy must be convertible to AsyncInfoTy.");
  ODBG(ODT_Interface) << "Entering target region for device " << DeviceId
                      << " with entry point " << HostPtr;

  if (checkDevice(DeviceId, Loc)) {
    ODBG(ODT_Interface) << "Not offloading to device " << DeviceId;
    return OMP_TGT_FAIL;
  }

  bool IsTeams = NumTeams != -1;
  if (!IsTeams)
    KernelArgs->UserNumBlocks[0] = NumTeams = 1;

  KernelArgsTy LocalKernelArgs;
  UpgradedArgBuffersTy UpgradedBufs;
  KernelArgs = upgradeKernelArgs(KernelArgs, LocalKernelArgs, UpgradedBufs,
                                 NumTeams, ThreadLimit);

  TIMESCOPE_WITH_DETAILS_AND_IDENT(
      "Runtime: target exe",
      "NumTeams=" + std::to_string(NumTeams) +
          ";NumArgs=" + std::to_string(KernelArgs->NumArgs),
      Loc);

  // The implicit dyn_ptr slot is always the last entry for versions that
  // support it.  Exclude it from user-facing info output.
  uint32_t UserArgCount = KernelArgs->NumArgs;
  if (KernelArgs->Version >= OMP_KERNEL_ARG_MIN_VERSION_WITH_DYN_PTR &&
      UserArgCount > 0)
    --UserArgCount;

  if (getInfoLevel() & OMP_INFOTYPE_KERNEL_ARGS)
    printKernelArguments(Loc, DeviceId, UserArgCount, KernelArgs->ArgSizes,
                         KernelArgs->ArgTypes, KernelArgs->ArgNames,
                         "Entering OpenMP kernel");

  ODBG_OS(ODT_Kernel, [&](llvm::raw_ostream &Os) {
    for (uint32_t I = 0; I < KernelArgs->NumArgs; ++I) {
      Os << "Entry" << llvm::format("%2d", I)
         << ": Base=" << KernelArgs->ArgBasePtrs[I]
         << ", Begin=" << KernelArgs->ArgPtrs[I]
         << ", Size=" << KernelArgs->ArgSizes[I]
         << ", Type=" << llvm::format("0x%" PRIx64, KernelArgs->ArgTypes[I])
         << ", Name="
         << (KernelArgs->ArgNames
                 ? getNameFromMapping(KernelArgs->ArgNames[I]).c_str()
                 : "unknown")
         << "\n";
    }
  });

  auto DeviceOrErr = PM->getDevice(DeviceId);
  if (!DeviceOrErr)
    FATAL_MESSAGE(DeviceId, "%s", toString(DeviceOrErr.takeError()).c_str());

  TargetAsyncInfoTy TargetAsyncInfo(*DeviceOrErr);
  AsyncInfoTy &AsyncInfo = TargetAsyncInfo;
  /// RAII to establish tool anchors before and after target region
  OMPT_IF_BUILT(InterfaceRAII TargetRAII(
                    RegionInterface.getCallbacks<ompt_target>(), DeviceId,
                    /*CodePtr=*/OMPT_GET_RETURN_ADDRESS);)

  int Rc = OFFLOAD_SUCCESS;
  Rc = target(Loc, *DeviceOrErr, HostPtr, *KernelArgs, AsyncInfo);
  { // required to show synchronization
    TIMESCOPE_WITH_DETAILS_AND_IDENT("Runtime: synchronize", "", Loc);
    if (Rc == OFFLOAD_SUCCESS) {
      printSyncInfo(Loc, DeviceId, AsyncInfo);
      Rc = AsyncInfo.synchronize();
    }

    handleTargetOutcome(Rc == OFFLOAD_SUCCESS, Loc);
    assert(Rc == OFFLOAD_SUCCESS && "__tgt_target_kernel unexpected failure!");
  }
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
  OMPT_IF_BUILT(ReturnAddressSetterRAII RA(__builtin_return_address(0)));
  if (KernelArgs->Flags.NoWait)
    return targetKernel<TaskAsyncInfoWrapperTy>(
        Loc, DeviceId, NumTeams, ThreadLimit, HostPtr, KernelArgs);
  return targetKernel<AsyncInfoTy>(Loc, DeviceId, NumTeams, ThreadLimit,
                                   HostPtr, KernelArgs);
}

/// Activates the record replay mechanism.
/// \param DeviceId The device identifier to execute the target region.
/// \param MemorySize The number of bytes to be (pre-)allocated
///                   by the record replay allocator.
/// /param IsRecord Activates the record replay mechanism in
///                 'record' or 'replay' mode.
/// /param SaveOutput Store the device memory after kernel
///                   execution on persistent storage.
/// /param EmitReport Emit a summary report after the recording.
/// /param OutputDirPath The output directory where the record replay files
/// should be stored. An empty string or nullptr indicates the current working
/// directory should be used.
EXTERN int __tgt_activate_record_replay(int64_t DeviceId, uint64_t MemorySize,
                                        void *VAddr, bool IsRecord,
                                        bool SaveOutput, bool EmitReport,
                                        const char *OutputDirPath) {
  assert(PM && "Runtime not initialized");
  OMPT_IF_BUILT(ReturnAddressSetterRAII RA(__builtin_return_address(0)));
  auto DeviceOrErr = PM->getDevice(DeviceId);
  if (!DeviceOrErr)
    FATAL_MESSAGE(DeviceId, "%s", toString(DeviceOrErr.takeError()).c_str());

  int Rc = target_activate_rr(*DeviceOrErr, MemorySize, VAddr, IsRecord,
                              SaveOutput, EmitReport, OutputDirPath);
  if (Rc != OFFLOAD_SUCCESS) {
    ODBG(ODT_Interface) << "Record replay failed to activate in device "
                        << DeviceId;
    return OMP_TGT_FAIL;
  }
  return OMP_TGT_SUCCESS;
}

/// Implements a target kernel entry that replays a pre-recorded kernel.
/// \param Loc Source location associated with this target region (unused).
/// \param DeviceId The device identifier to execute the target region.
/// \param HostPtr A pointer to an address that uniquely identifies the kernel.
/// \param DeviceMemory A pointer to an array storing device memory data to move
///                     prior to kernel execution.
/// \param DeviceMemorySize The size of the above device memory data in bytes.
/// \param ReuseDeviceAlloc Pointer to a device memory allocation that should be
///                         reused for the replay. If null, the replay will
///                         allocate the necessary device buffer.
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
EXTERN int __tgt_target_kernel_replay(
    ident_t *Loc, int64_t DeviceId, void *HostPtr, void *DeviceMemory,
    void *ReuseDeviceAlloc, int64_t DeviceMemorySize,
    const llvm::offloading::EntryTy *Globals, int32_t NumGlobals,
    void **TgtArgs, ptrdiff_t *TgtOffsets, int32_t NumArgs, int32_t NumTeams,
    int32_t ThreadLimit, uint32_t SharedMemorySize, uint64_t LoopTripCount,
    KernelReplayOutcomeTy *ReplayOutcome) {
  assert(PM && "Runtime not initialized");
  OMPT_IF_BUILT(ReturnAddressSetterRAII RA(__builtin_return_address(0)));
  if (checkDevice(DeviceId, Loc)) {
    ODBG(ODT_Interface) << "Not offloading to device " << DeviceId;
    return OMP_TGT_FAIL;
  }
  auto DeviceOrErr = PM->getDevice(DeviceId);
  if (!DeviceOrErr)
    FATAL_MESSAGE(DeviceId, "%s", toString(DeviceOrErr.takeError()).c_str());

  /// RAII to establish tool anchors before and after target region
  OMPT_IF_BUILT(InterfaceRAII TargetRAII(
                    RegionInterface.getCallbacks<ompt_target>(), DeviceId,
                    /*CodePtr=*/OMPT_GET_RETURN_ADDRESS);)

  AsyncInfoTy AsyncInfo(*DeviceOrErr);
  int Rc =
      target_replay(Loc, *DeviceOrErr, HostPtr, DeviceMemory, DeviceMemorySize,
                    ReuseDeviceAlloc, Globals, NumGlobals, TgtArgs, TgtOffsets,
                    NumArgs, NumTeams, ThreadLimit, SharedMemorySize,
                    LoopTripCount, AsyncInfo, ReplayOutcome);

  if (Rc == OFFLOAD_SUCCESS)
    Rc = AsyncInfo.synchronize();

  if (Rc != OFFLOAD_SUCCESS) {
    ODBG(ODT_Interface) << "Kernel replay failed in device " << DeviceId;
    return OMP_TGT_FAIL;
  }
  return OMP_TGT_SUCCESS;
}

// Get the current number of components for a user-defined mapper.
EXTERN int64_t __tgt_mapper_num_components(void *RtMapperHandle) {
  auto *MapperComponentsPtr = (struct MapperComponentsTy *)RtMapperHandle;
  int64_t Size = MapperComponentsPtr->Components.size();
  ODBG(ODT_Interface) << "__tgt_mapper_num_components(Handle=" << RtMapperHandle
                      << ") returns " << Size;
  return Size;
}

// Push back one component for a user-defined mapper.
EXTERN void __tgt_push_mapper_component(void *RtMapperHandle, void *Base,
                                        void *Begin, int64_t Size, int64_t Type,
                                        void *Name) {
  ODBG(ODT_Interface) << "__tgt_push_mapper_component(Handle=" << RtMapperHandle
                      << ") adds an entry (Base=" << Base << ", Begin=" << Begin
                      << ", Size=" << Size
                      << ", Type=" << llvm::format("0x%" PRIx64, Type)
                      << ", Name="
                      << ((Name) ? getNameFromMapping(Name) : "unknown") << ")";
  auto *MapperComponentsPtr = (struct MapperComponentsTy *)RtMapperHandle;
  MapperComponentsPtr->Components.push_back(
      MapComponentInfoTy(Base, Begin, Size, Type, Name));
}

EXTERN void __tgt_set_info_flag(uint32_t NewInfoLevel) {
  assert(PM && "Runtime not initialized");
  std::atomic<uint32_t> &InfoLevel = getInfoLevelInternal();
  InfoLevel.store(NewInfoLevel);
}

EXTERN int __tgt_print_device_info(int64_t DeviceId) {
  MESSAGE("The %s function is deprecated and no longer prints any "
          "information. Use olGetDeviceInfo instead",
          __PRETTY_FUNCTION__);
  return false;
}

EXTERN void __tgt_target_nowait_query(void **AsyncHandle) {
  assert(PM && "Runtime not initialized");
  OMPT_IF_BUILT(ReturnAddressSetterRAII RA(__builtin_return_address(0)));

  if (!AsyncHandle || !*AsyncHandle) {
    FATAL_MESSAGE0(
        1, "Receive an invalid async handle from the current OpenMP task. Is "
           "this a target nowait region?\n");
  }

  // Exponential backoff tries to optimally decide if a thread should just query
  // for the device operations (work/spin wait on them) or block until they are
  // completed (use device side blocking mechanism). This allows the runtime to
  // adapt itself when there are a lot of long-running target regions in-flight.
  static thread_local utils::ExponentialBackoff QueryCounter(
      Int64Envar("OMPTARGET_QUERY_COUNT_MAX", 10),
      Int64Envar("OMPTARGET_QUERY_COUNT_THRESHOLD", 5),
      Envar<float>("OMPTARGET_QUERY_COUNT_BACKOFF_FACTOR", 0.5f));

  auto *AsyncInfo = (AsyncInfoTy *)*AsyncHandle;

  // If the thread is actively waiting on too many target nowait regions, we
  // should use the blocking sync type.
  if (QueryCounter.isAboveThreshold())
    AsyncInfo->SyncType = AsyncInfoTy::SyncTy::BLOCKING;

  if (AsyncInfo->synchronize())
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

EXTERN void __tgt_register_rpc_callback(unsigned (*Callback)(void *,
                                                             unsigned)) {
  if (!PM || !PM->isLibOffloadInitialized())
    return;

  if (auto Err = iteratePlatforms(
          [](ol_platform_handle_t Platform, void *Data) {
            bool Active = false;
            if (olGetPlatformInfo(Platform, OL_PLATFORM_INFO_ACTIVE,
                                  sizeof(Active), &Active) == OL_SUCCESS &&
                Active)
              olPlatformRegisterRPCCallback(
                  Platform, reinterpret_cast<ol_platform_rpc_cb_t>(Data));
            return true;
          },
          reinterpret_cast<void *>(Callback)))
    REPORT() << "Failed to iterate platforms: " << toString(std::move(Err));
}

EXTERN void *__tgt_get_mapped_ptr(int64_t DeviceId, const void *HostPtr) {
  void *TargetPtr = omp_get_mapped_ptr(HostPtr, DeviceId);
  if (!TargetPtr)
    return const_cast<void *>(HostPtr);
  return TargetPtr;
}

//===----------------------------------------------------------------------===//
// Taskgraph transmission (libomp -> libomptarget)
//
// The taskgraph is transmitted twice by libomp: the first time to measure the
// total needed size, the second to do the transfer into a single allocated
// block of that size.
//===----------------------------------------------------------------------===//

using llvm::omp::target::acquireTaskGraphMappings;
using llvm::omp::target::buildTaskGraphPlan;
using llvm::omp::target::commitTaskGraphUpdates;
using llvm::omp::target::releaseTaskGraphPlan;
using llvm::omp::target::TaskGraphArgUpdateTy;
using llvm::omp::target::TaskGraphBuilder;
using llvm::omp::target::TaskGraphDebugDump;
using llvm::omp::target::TaskGraphEdge;
using llvm::omp::target::TaskGraphElement;
using llvm::omp::target::TaskGraphExclRegion;
using llvm::omp::target::TaskGraphIrreducibleRegion;
using llvm::omp::target::TaskGraphMappingsGuard;
using llvm::omp::target::TaskGraphNode;
using llvm::omp::target::TaskGraphParRegion;
using llvm::omp::target::TaskGraphRegion;
using llvm::omp::target::TaskGraphSeqRegion;
using llvm::omp::target::TaskGraphTargetDataNode;
using llvm::omp::target::TaskGraphTargetEnterDataNode;
using llvm::omp::target::TaskGraphTargetExitDataNode;
using llvm::omp::target::TaskGraphTargetNode;
using llvm::omp::target::TaskGraphTargetUpdateDataNode;
using llvm::omp::target::TaskGraphTy;
using llvm::omp::target::TGKind;

namespace {

/// Resolve the (possibly default) device id used to host a graph.
static int64_t resolveGraphDevice(int64_t DeviceId) {
  return DeviceId == OFFLOAD_DEVICE_DEFAULT ? omp_get_default_device()
                                            : DeviceId;
}

/// Append a target enter/exit/update data leaf to the graph.
template <class NodeT>
static void emitTaskGraphData(void *Graph, int64_t DeviceId, int32_t ArgNum,
                              void **ArgsBase, void **Args, int64_t *ArgSizes,
                              int64_t *ArgTypes, void **ArgNames,
                              void **ArgMappers) {
  TaskGraphBuilder B(*static_cast<TaskGraphTy *>(Graph));
  NodeT *N = B.makeLeaf<NodeT>();
  if (!N)
    return;
  N->DeviceId = DeviceId;
  N->ArgNum = ArgNum;
  N->ArgsBase = ArgsBase;
  N->Args = Args;
  N->ArgSizes = ArgSizes;
  N->ArgTypes = ArgTypes;
  N->ArgNames = ArgNames;
  N->ArgMappers = ArgMappers;
  B.linkChild(N);
}

/// Open a region of \p NodeCount children: the measuring pass sizes it and its
/// child array, the build pass constructs it and fills it with the children
/// that follow.
template <class RegionT>
static void startRegion(void *Graph, int32_t NodeCount) {
  TaskGraphBuilder B(*static_cast<TaskGraphTy *>(Graph));
  if (auto *R = B.makeRegion<RegionT>(NodeCount))
    B.pushRegion(R);
}

/// Close the region being filled.
static void endRegion(void *Graph) {
  TaskGraphBuilder B(*static_cast<TaskGraphTy *>(Graph));
  if (B.building())
    B.popRegion();
}

} // namespace

EXTERN void *__tgt_taskgraph_start(int64_t DeviceId, uintptr_t GraphId,
                                   size_t ByteSize) {
  assert(PM && "Runtime not initialized");
  // ByteSize == 0 is the measuring pass (accumulate size only); ByteSize > 0 is
  // the build pass.
  return TaskGraphBuilder::start(DeviceId, GraphId, ByteSize);
}

EXTERN void __tgt_taskgraph_start_parallel(void *Graph, int32_t NodeCount) {
  startRegion<TaskGraphParRegion>(Graph, NodeCount);
}
EXTERN void __tgt_taskgraph_end_parallel(void *Graph) { endRegion(Graph); }
EXTERN void __tgt_taskgraph_start_sequential(void *Graph, int32_t NodeCount) {
  startRegion<TaskGraphSeqRegion>(Graph, NodeCount);
}
EXTERN void __tgt_taskgraph_end_sequential(void *Graph) { endRegion(Graph); }
EXTERN void __tgt_taskgraph_start_exclusive(void *Graph, int32_t NodeCount) {
  startRegion<TaskGraphExclRegion>(Graph, NodeCount);
}
EXTERN void __tgt_taskgraph_end_exclusive(void *Graph) { endRegion(Graph); }
EXTERN void __tgt_taskgraph_start_irreducible(void *Graph, int32_t NodeCount,
                                              int32_t EdgeCount) {
  TaskGraphBuilder B(*static_cast<TaskGraphTy *>(Graph));
  // The edges, filled by the emit_edge calls that follow the children.
  if (auto *R = B.makeIrreducibleRegion(NodeCount, EdgeCount))
    B.pushRegion(R);
}
EXTERN void __tgt_taskgraph_end_irreducible(void *Graph) { endRegion(Graph); }
EXTERN void __tgt_taskgraph_emit_edge(void *Graph, int32_t SrcChild,
                                      int32_t DstChild) {
  // Edges are sized up front with the region's edge array (start_irreducible),
  // so the measuring pass has nothing to do here.  On the build pass they land
  // in the enclosing irreducible region, which the cursor is currently filling.
  TaskGraphBuilder B(*static_cast<TaskGraphTy *>(Graph));
  if (B.building())
    B.addEdge(SrcChild, DstChild);
}

EXTERN void __tgt_taskgraph_emit_target(void *Graph, int64_t DeviceId,
                                        int32_t NumTeams, int32_t ThreadLimit,
                                        void *HostPtr, void *KernelArgs) {
  TaskGraphBuilder B(*static_cast<TaskGraphTy *>(Graph));
  auto *N = B.makeLeaf<TaskGraphTargetNode>();
  if (!N)
    return;
  N->DeviceId = DeviceId;
  N->NumTeams = NumTeams;
  N->ThreadLimit = ThreadLimit;
  N->HostPtr = HostPtr;
  N->KernelArgs = KernelArgs;
  B.linkChild(N);
}

EXTERN void __tgt_taskgraph_emit_target_enter_data(
    void *Graph, int64_t DeviceId, int32_t ArgNum, void **ArgsBase, void **Args,
    int64_t *ArgSizes, int64_t *ArgTypes, void **ArgNames, void **ArgMappers) {
  emitTaskGraphData<TaskGraphTargetEnterDataNode>(
      Graph, DeviceId, ArgNum, ArgsBase, Args, ArgSizes, ArgTypes, ArgNames,
      ArgMappers);
}

EXTERN void __tgt_taskgraph_emit_target_exit_data(
    void *Graph, int64_t DeviceId, int32_t ArgNum, void **ArgsBase, void **Args,
    int64_t *ArgSizes, int64_t *ArgTypes, void **ArgNames, void **ArgMappers) {
  emitTaskGraphData<TaskGraphTargetExitDataNode>(
      Graph, DeviceId, ArgNum, ArgsBase, Args, ArgSizes, ArgTypes, ArgNames,
      ArgMappers);
}

EXTERN void __tgt_taskgraph_emit_target_update(
    void *Graph, int64_t DeviceId, int32_t ArgNum, void **ArgsBase, void **Args,
    int64_t *ArgSizes, int64_t *ArgTypes, void **ArgNames, void **ArgMappers) {
  emitTaskGraphData<TaskGraphTargetUpdateDataNode>(
      Graph, DeviceId, ArgNum, ArgsBase, Args, ArgSizes, ArgTypes, ArgNames,
      ArgMappers);
}

/// Answer a recorded `target data` region's use_device_ptr / use_device_addr
/// list items, as the begin mapper would, when the region's body starts on a
/// replay.  \p Out receives one device base per entry.
EXTERN void __tgt_taskgraph_lookup_use_device(int64_t DeviceId, int32_t ArgNum,
                                              void **ArgsBase, void **Args,
                                              int64_t *ArgTypes, void **Out) {
  assert(PM && "Runtime not initialized");
  auto DeviceOrErr = PM->getDevice(resolveGraphDevice(DeviceId));
  if (!DeviceOrErr)
    FATAL_MESSAGE(DeviceId, "%s", toString(DeviceOrErr.takeError()).c_str());
  MappingInfoTy &MappingInfo = DeviceOrErr->getMappingInfo();

  for (int32_t I = 0; I < ArgNum; ++I) {
    if (!(ArgTypes[I] & OMP_TGT_MAPTYPE_RETURN_PARAM))
      continue;
    const uintptr_t Delta = reinterpret_cast<uintptr_t>(Args[I]) -
                            reinterpret_cast<uintptr_t>(ArgsBase[I]);
    // Deliberately no reference-count movement and no hold: this is the
    // question `where is it`, asked of a mapping some map is already
    // responsible for.  A graph asks it once per replay, so answering it with
    // a reference would leak one per replay.
    TargetPointerResultTy TPR = MappingInfo.getTgtPtrBegin(
        Args[I], /*Size=*/0, /*UpdateRefCount=*/false,
        /*UseHoldRefCount=*/false);
    uintptr_t Begin = reinterpret_cast<uintptr_t>(TPR.TargetPointer);
    // An absent entry comes back as the host address, so it is told apart by
    // the flags rather than by a null.  It takes the same two fallbacks as in
    // the mapper: nullify when the clause asked for it, and otherwise hand back
    // the host address, which OpenMP makes the program's responsibility to
    // have made device-accessible.  Host-resident memory under unified shared
    // memory is found, not absent.
    if (!TPR.isPresent() && !TPR.isHostPointer() &&
        (ArgTypes[I] & OMP_TGT_MAPTYPE_FB_NULLIFY))
      Begin = 0;
    Out[I] = reinterpret_cast<void *>(Begin - Delta);
  }
}

EXTERN size_t __tgt_taskgraph_end(void *Graph) {
  auto *G = static_cast<TaskGraphTy *>(Graph);
  TaskGraphBuilder B(*G);
  const bool Measuring = !B.building();
  size_t Bytes = B.finish();
  // The measuring pass exists only to compute the block size: libomp feeds it
  // back to __tgt_taskgraph_start and re-streams the graph on a build pass, so
  // this throwaway handle is discarded here.  The build pass keeps the handle:
  // its block is fully built, ready to be handed to a plugin.
  if (Measuring)
    delete G;
  return Bytes;
}

//===----------------------------------------------------------------------===//
// Services a plugin needs while lowering a graph
//
// The libomptarget-only operations a plugin cannot perform for itself, handed
// over as function pointers in TaskGraphServicesTy because a plugin must not
// name a libomptarget symbol: the plugin archives are linked into LLVMOffload
// as well as into libomptarget.so, and LLVMOffload contains none of this.
//===----------------------------------------------------------------------===//

/// TaskGraphServicesTy::resolveKernel.
static void *taskGraphResolveKernel(int64_t DeviceId, void *HostPtr) {
  return getDeviceKernelEntry(static_cast<int32_t>(DeviceId), HostPtr);
}

/// Why \p G cannot be offloaded to \p DeviceId, or null if it can, settling
/// what every backend would otherwise check for itself.  Also resolves the
/// default device across the tree, and settles each recorded launch's
/// configuration as targetKernel does.
static const char *whyNotOffloadable(TaskGraphTy &G, int64_t DeviceId) {
  if (const char *Why = TaskGraphBuilder::whyMalformed(G))
    return Why;

  // Safe to do in place: finalize runs once, on the build-pass handle, before
  // anything reads the leaves.
  G.DeviceId = DeviceId;
  for (TaskGraphNode *N : G.leaves()) {
    if (N->DeviceId == OFFLOAD_DEVICE_DEFAULT)
      N->DeviceId = DeviceId;
    // A backend lowers the graph for the one device it is offered on.
    if (N->DeviceId != DeviceId)
      return "its leaves target more than one device";

    auto *K = llvm::dyn_cast<TaskGraphTargetNode>(N);
    if (!K)
      continue;
    // The recording is libomp's own copy, and host replay goes through
    // targetKernel, which makes the same adjustments again.
    auto *KA = static_cast<KernelArgsTy *>(K->KernelArgs);
    if (!KA)
      return "a target leaf with no recorded kernel arguments";
    // Version 3 put the launch environment first, with no slot recorded for
    // it, which only upgradeKernelArgs's per-launch copy straightens out.
    if (KA->Version <= OMP_KERNEL_ARG_MIN_VERSION_WITH_DYN_PTR)
      return "a target leaf built for an older kernel-argument ABI";
    if (K->NumTeams == -1)
      KA->UserNumBlocks[0] = 1;
    correctMultiDim(*KA);
  }
  return nullptr;
}

EXTERN int __tgt_taskgraph_finalize(void *Graph) {
  auto *G = static_cast<TaskGraphTy *>(Graph);
  int64_t ResolvedDevice = resolveGraphDevice(G->DeviceId);
  auto DeviceOrErr = PM->getDevice(ResolvedDevice);
  if (!DeviceOrErr) {
    consumeError(DeviceOrErr.takeError());
    return OFFLOAD_FAIL;
  }
  DeviceTy &Device = *DeviceOrErr;

  // Building the plan allocates device memory, which is wasted on a device
  // with no graph backend.
  if (!Device.RTL->supports_taskgraphs(Device.RTLDeviceID))
    return OFFLOAD_FAIL;

  if (const char *Why = whyNotOffloadable(*G, ResolvedDevice)) {
    INFO(OMP_INFOTYPE_ALL, Device.DeviceID,
         "taskgraph %" PRIu64 " replays on the host: %s\n",
         static_cast<uint64_t>(G->GraphId), Why);
    return OFFLOAD_FAIL;
  }

  G->Services.resolveKernel = taskGraphResolveKernel;

  // Work out where the graph's mapped data lives before offering it to the
  // plugin.  A graph the plan cannot express is replayed on the host.
  if (!buildTaskGraphPlan(Device, *G)) {
    releaseTaskGraphPlan(Device, *G);
    return OFFLOAD_FAIL;
  }

  // Offer the tree to the plugin, which lowers it however suits its backend.
  // The base plugin implementation declines, as does a backend that meets
  // something in the graph it cannot express; either way libomp keeps the graph
  // and replays it on the host.
  void *PluginGraph = nullptr;
  if (Device.RTL->finalize_taskgraph(Device.RTLDeviceID, G, &PluginGraph) !=
      OFFLOAD_SUCCESS) {
    releaseTaskGraphPlan(Device, *G);
    return OFFLOAD_FAIL;
  }
  assert(PluginGraph && "plugin claimed a taskgraph without a handle");
  G->PluginGraph = PluginGraph;
  return OFFLOAD_SUCCESS;
}

EXTERN void __tgt_taskgraph_begin_replay(void *Graph) {
  auto *G = static_cast<TaskGraphTy *>(Graph);
  assert(G->PluginGraph && "replaying a graph no plugin claimed");
  beginTaskGraphReplay(*G);
}

static bool dumpTaskGraphPlans() {
  static BoolEnvar Dump("LIBOMPTARGET_TASKGRAPH_TRACE", false);
  return Dump;
}

EXTERN int __tgt_taskgraph_replay(void *Graph) {
  auto *G = static_cast<TaskGraphTy *>(Graph);
  assert(G->PluginGraph && "replaying a graph no plugin claimed");

  auto DeviceOrErr = PM->getDevice(G->DeviceId);
  if (!DeviceOrErr)
    FATAL_MESSAGE(G->DeviceId, "%s", toString(DeviceOrErr.takeError()).c_str());
  DeviceTy &Device = *DeviceOrErr;

  // Resolve every range against the mapping table as it stands now, mapping
  // the ones the graph is responsible for and holding all of them until the
  // graph has run, then hand the plugin whatever changed.
  //
  // Until the handover nothing has been issued, so a graph that cannot be run
  // is declined: libomp runs its target constructs itself instead.  The guard
  // gives back whatever was acquired.
  auto decline = [&](const char *Why) {
    if (Why)
      MESSAGE("taskgraph %" PRIu64 " replays on the host this time: %s",
              static_cast<uint64_t>(G->GraphId), Why);
    return OFFLOAD_FAIL;
  };
  llvm::SmallVector<TaskGraphArgUpdateTy, 8> Updates;
  bool Resync = false;
  TaskGraphMappingsGuard Mappings(Device, *G);
  if (!acquireTaskGraphMappings(Device, *G, Updates, Resync))
    return decline(G->Plan.declineReported()
                       ? nullptr
                       : "it cannot be replayed against the current mapping "
                         "state");

  if (dumpTaskGraphPlans()) {
    llvm::errs() << "Target graph pre-replay:\n";
    TaskGraphDebugDump TGD(*G, llvm::errs());
    TGD.dumpDataPlan();
    TGD.dumpUpdates(Updates, Resync);
  }

  // Pinned every replay, since the ranges' host memory is only known to be the
  // program's while the graph runs.  A failed pin holds nothing.
  if (Device.RTL->pin_taskgraph(Device.RTLDeviceID, G) != OFFLOAD_SUCCESS)
    return decline("the device could not pin the graph's host memory");
  if (llvm::any_of(Updates, [&](const TaskGraphArgUpdateTy &U) {
        return U.has(TaskGraphArgUpdateTy::DataChanged) &&
               llvm::isa<llvm::omp::target::TaskGraphLiteralArgTy>(
                   G->Plan.getArg(U.Arg).get());
      }))
    ODBG(ODT_Interface) << "taskgraph " << G->GraphId
                        << " reparametrizes for a changed literal argument";
  if ((!Updates.empty() || Resync) &&
      Device.RTL->reparametrize_taskgraph(Device.RTLDeviceID, G, Updates,
                                          Resync) != OFFLOAD_SUCCESS) {
    Device.RTL->unpin_taskgraph(Device.RTLDeviceID, G);
    return decline("the device could not adopt the current mapping state");
  }
  commitTaskGraphUpdates(*G);
  Mappings.release();

  // Past this point the backend may have issued part of the graph, so a
  // failure cannot fall back to the host.
  int32_t Rc = Device.RTL->replay_taskgraph(Device.RTLDeviceID, G);
  if (Device.RTL->unpin_taskgraph(Device.RTLDeviceID, G) != OFFLOAD_SUCCESS)
    Rc = OFFLOAD_FAIL;
  if (Rc != OFFLOAD_SUCCESS)
    FATAL_MESSAGE(G->DeviceId, "%s", "taskgraph replay failed");

  // The graph has run, so its hold on the mappings can go.
  releaseTaskGraphMappings(Device, *G);
  return OFFLOAD_SUCCESS;
}

EXTERN void __tgt_taskgraph_destroy(void *Graph) {
  auto *G = static_cast<TaskGraphTy *>(Graph);
  if (G->PluginGraph) {
    // DeviceId was resolved to a concrete device by finalize, which is also the
    // only thing that can set PluginGraph.
    auto DeviceOrErr = PM->getDevice(G->DeviceId);
    if (DeviceOrErr) {
      DeviceOrErr->RTL->destroy_taskgraph(DeviceOrErr->RTLDeviceID, G);
      releaseTaskGraphPlan(*DeviceOrErr, *G);
    } else {
      consumeError(DeviceOrErr.takeError());
    }
  }
  delete G;
}

EXTERN void __tgt_taskgraph_dup_kernel_args(void *Dst, void *Src,
                                            size_t *AllocSize) {
  // Deep-copy a kernel-arguments blob into a single caller-provided block, so
  // that libomp can capture a target launch without knowing the layout of
  // KernelArgsTy or of the six argument arrays hanging off it.  Called twice:
  // once with a null Dst to compute the size of the block (returned in
  // AllocSize), then again to fill a block of that size.
  bool SizeOnly = (Dst == nullptr);
  KernelArgsTy *KernArgs = static_cast<KernelArgsTy *>(Src);
  size_t N = KernArgs->NumArgs;
  namespace mt = omptarget::maptype;
  // A firstprivate aggregate whose clause carried the "saved" modifier: its
  // data has to be duplicated here.  We only need to do this for aggregates,
  // because scalars or pointers are passed by value.
  auto FpAggregateP = [&](size_t Idx) {
    int64_t ArgType = KernArgs->ArgTypes[Idx];
    return mt::isPrivate(ArgType) && !mt::isLiteral(ArgType) &&
           mt::isSaved(ArgType) && KernArgs->ArgPtrs[Idx] &&
           KernArgs->ArgSizes[Idx] > 0;
  };
  // ArgNames and ArgMappers are optional; a launch with neither debug names nor
  // user-defined mappers leaves them null.  The copy has to preserve that, or
  // every argument would appear to carry a mapper (pointing at uninitialized
  // bytes of the block).  Both passes read the same blob, so both agree on
  // whether the arrays are there.
  const bool HasNames = KernArgs->ArgNames != nullptr;
  const bool HasMappers = KernArgs->ArgMappers != nullptr;
  if (SizeOnly) {
    size_t Total = 0;
    auto CountAligned = [&](size_t Size, size_t Align) {
      Total = llvm::alignTo(Total, Align) + Size;
    };
    CountAligned(sizeof(KernelArgsTy), alignof(KernelArgsTy));
    // ArgBasePtrs
    CountAligned(N * sizeof(void *), alignof(void *));
    // ArgPtrs
    CountAligned(N * sizeof(void *), alignof(void *));
    // ArgSizes
    CountAligned(N * sizeof(uint64_t), alignof(uint64_t));
    // ArgTypes
    CountAligned(N * sizeof(uint64_t), alignof(uint64_t));
    // ArgNames
    if (HasNames)
      CountAligned(N * sizeof(void *), alignof(void *));
    // ArgMappers
    if (HasMappers)
      CountAligned(N * sizeof(void *), alignof(void *));
    // One snapshot per saved firstprivate aggregate.
    for (size_t I = 0; I < N; I++)
      if (FpAggregateP(I))
        CountAligned(KernArgs->ArgSizes[I], alignof(std::max_align_t));
    *AllocSize = Total;
  } else {
    size_t Allocated = 0;
    auto BumpAllocDup = [&](size_t Size, size_t Align,
                            void *Src = nullptr) -> void * {
      size_t ThisBlockIdx = llvm::alignTo(Allocated, Align);
      char *Block = &(static_cast<char *>(Dst))[ThisBlockIdx];
      Allocated = ThisBlockIdx + Size;
      if (Src)
        std::memcpy(Block, Src, Size);
      return Block;
    };
    auto *DupArgs = static_cast<KernelArgsTy *>(
        BumpAllocDup(sizeof(KernelArgsTy), alignof(KernelArgsTy), Src));
    DupArgs->ArgBasePtrs = static_cast<void **>(BumpAllocDup(
        N * sizeof(void *), alignof(void *), KernArgs->ArgBasePtrs));
    DupArgs->ArgPtrs = static_cast<void **>(
        BumpAllocDup(N * sizeof(void *), alignof(void *), KernArgs->ArgPtrs));
    DupArgs->ArgSizes = static_cast<int64_t *>(BumpAllocDup(
        N * sizeof(int64_t), alignof(uint64_t), KernArgs->ArgSizes));
    DupArgs->ArgTypes = static_cast<int64_t *>(BumpAllocDup(
        N * sizeof(int64_t), alignof(uint64_t), KernArgs->ArgTypes));
    DupArgs->ArgNames =
        HasNames ? static_cast<void **>(BumpAllocDup(
                       N * sizeof(void *), alignof(void *), KernArgs->ArgNames))
                 : nullptr;
    DupArgs->ArgMappers =
        HasMappers
            ? static_cast<void **>(BumpAllocDup(
                  N * sizeof(void *), alignof(void *), KernArgs->ArgMappers))
            : nullptr;
    // Snapshot each saved firstprivate aggregate and redirect the argument at
    // the copy.
    for (size_t I = 0; I < N; I++) {
      if (!FpAggregateP(I))
        continue;
      void *Saved =
          BumpAllocDup(KernArgs->ArgSizes[I], alignof(std::max_align_t),
                       KernArgs->ArgPtrs[I]);
      DupArgs->ArgPtrs[I] = Saved;
      DupArgs->ArgBasePtrs[I] = Saved;
    }
    assert(Allocated == *AllocSize);
  }
}

EXTERN void __tgt_taskgraph_dup_data_args(void *Dst, int32_t ArgNum,
                                          void ***ArgsBase, void ***Args,
                                          int64_t **ArgSizes,
                                          int64_t **ArgTypes, void ***ArgNames,
                                          void ***ArgMappers,
                                          size_t *AllocSize) {
  // Deep-copy the map arrays of a target data construct into a single
  // caller-provided block, so that libomp can capture one without knowing what
  // an element of them means.  Called twice, as
  // __tgt_taskgraph_dup_kernel_args is: once with a null Dst to compute the
  // size of the block (returned in AllocSize), then again to fill a block of
  // that size, which is when the in-out array parameters are repointed into it.
  bool SizeOnly = (Dst == nullptr);
  size_t N = ArgNum;
  namespace mt = omptarget::maptype;
  // Copying the arrays is not enough on its own for a non-contiguous update.
  // Its args[i] is a pointer to a stride descriptor rather than to mapped
  // data, and clang emits that descriptor into the encountering function's
  // frame, so a copy of the pointer dangles as soon as the recording encounter
  // returns.  The descriptors are copied into the same block and the copied
  // args[i] repointed at them, which also puts them beyond the reach of
  // relocation for good: they are runtime-owned from here on and, being pure
  // offsets, counts and strides, have nothing frame-dependent left in them to
  // rewrite.  arg_sizes[i] is the dimension count for such an entry, not a
  // byte size.
  auto NonContigDims = [&](size_t Idx) -> size_t {
    return mt::isNonContig((*ArgTypes)[Idx]) ? (size_t)(*ArgSizes)[Idx] : 0;
  };
  // ArgNames and ArgMappers are optional; a construct with neither debug names
  // nor user-defined mappers leaves them null, and the copy has to preserve
  // that rather than hand back pointers to uninitialized bytes of the block.
  // Both calls are given the same arrays, so both agree on whether they are
  // there.
  const bool HasNames = *ArgNames != nullptr;
  const bool HasMappers = *ArgMappers != nullptr;
  if (SizeOnly) {
    size_t Total = 0;
    auto CountAligned = [&](size_t Size, size_t Align) {
      Total = llvm::alignTo(Total, Align) + Size;
    };
    // ArgsBase
    CountAligned(N * sizeof(void *), alignof(void *));
    // Args
    CountAligned(N * sizeof(void *), alignof(void *));
    // ArgSizes
    CountAligned(N * sizeof(int64_t), alignof(int64_t));
    // ArgTypes
    CountAligned(N * sizeof(int64_t), alignof(int64_t));
    if (HasNames)
      CountAligned(N * sizeof(void *), alignof(void *));
    if (HasMappers)
      CountAligned(N * sizeof(void *), alignof(void *));
    // One descriptor array per non-contiguous entry.
    for (size_t I = 0; I < N; I++)
      if (size_t Dims = NonContigDims(I))
        CountAligned(Dims * sizeof(__tgt_target_non_contig),
                     alignof(__tgt_target_non_contig));
    *AllocSize = Total;
    return;
  }

  size_t Allocated = 0;
  auto BumpAllocDup = [&](size_t Size, size_t Align, void *Src) -> void * {
    size_t ThisBlockIdx = llvm::alignTo(Allocated, Align);
    char *Block = &(static_cast<char *>(Dst))[ThisBlockIdx];
    Allocated = ThisBlockIdx + Size;
    if (Src)
      std::memcpy(Block, Src, Size);
    return Block;
  };
  auto *DupArgsBase = static_cast<void **>(
      BumpAllocDup(N * sizeof(void *), alignof(void *), *ArgsBase));
  auto *DupArgs = static_cast<void **>(
      BumpAllocDup(N * sizeof(void *), alignof(void *), *Args));
  auto *DupArgSizes = static_cast<int64_t *>(
      BumpAllocDup(N * sizeof(int64_t), alignof(int64_t), *ArgSizes));
  auto *DupArgTypes = static_cast<int64_t *>(
      BumpAllocDup(N * sizeof(int64_t), alignof(int64_t), *ArgTypes));
  auto *DupArgNames =
      HasNames ? static_cast<void **>(BumpAllocDup(N * sizeof(void *),
                                                   alignof(void *), *ArgNames))
               : nullptr;
  auto *DupArgMappers =
      HasMappers ? static_cast<void **>(BumpAllocDup(
                       N * sizeof(void *), alignof(void *), *ArgMappers))
                 : nullptr;
  for (size_t I = 0; I < N; I++)
    if (size_t Dims = NonContigDims(I))
      DupArgs[I] = BumpAllocDup(Dims * sizeof(__tgt_target_non_contig),
                                alignof(__tgt_target_non_contig), (*Args)[I]);
  assert(Allocated == *AllocSize);

  *ArgsBase = DupArgsBase;
  *Args = DupArgs;
  *ArgSizes = DupArgSizes;
  *ArgTypes = DupArgTypes;
  *ArgNames = DupArgNames;
  *ArgMappers = DupArgMappers;
}
