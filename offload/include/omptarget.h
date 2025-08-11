//===-------- omptarget.h - Target independent OpenMP target RTL -- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Interface to be used by Clang during the codegen of a
// target region.
//
//===----------------------------------------------------------------------===//

#ifndef _OMPTARGET_H_
#define _OMPTARGET_H_

#include "Shared/APITypes.h"
#include "Shared/Environment.h"
#include "Shared/SourceInfo.h"

#include "OpenMP/InternalTypes.h"

#include <cstddef>
#include <cstdint>
#include <deque>
#include <functional>
#include <type_traits>

#include "llvm/ADT/SmallVector.h"

#define OFFLOAD_SUCCESS (0)
#define OFFLOAD_FAIL (~0)

#define OFFLOAD_DEVICE_DEFAULT -1

/// return flags of __tgt_target_XXX public APIs
enum __tgt_target_return_t : int {
  /// successful offload executed on a target device
  OMP_TGT_SUCCESS = 0,
  /// offload may not execute on the requested target device
  /// this scenario can be caused by the device not available or unsupported
  /// as described in the Execution Model in the specification
  /// this status may not be used for target device execution failure
  /// which should be handled internally in libomptarget
  OMP_TGT_FAIL = ~0
};

/// Data attributes for each data reference used in an OpenMP target region.
enum tgt_map_type {
  // No flags
  OMP_TGT_MAPTYPE_NONE = 0x000,
  // copy data from host to device
  OMP_TGT_MAPTYPE_TO = 0x001,
  // copy data from device to host
  OMP_TGT_MAPTYPE_FROM = 0x002,
  // copy regardless of the reference count
  OMP_TGT_MAPTYPE_ALWAYS = 0x004,
  // force unmapping of data
  OMP_TGT_MAPTYPE_DELETE = 0x008,
  // map the pointer as well as the pointee
  OMP_TGT_MAPTYPE_PTR_AND_OBJ = 0x010,
  // pass device base address to kernel
  OMP_TGT_MAPTYPE_TARGET_PARAM = 0x020,
  // return base device address of mapped data
  OMP_TGT_MAPTYPE_RETURN_PARAM = 0x040,
  // private variable - not mapped
  OMP_TGT_MAPTYPE_PRIVATE = 0x080,
  // copy by value - not mapped
  OMP_TGT_MAPTYPE_LITERAL = 0x100,
  // mapping is implicit
  OMP_TGT_MAPTYPE_IMPLICIT = 0x200,
  // copy data to device
  OMP_TGT_MAPTYPE_CLOSE = 0x400,
  // runtime error if not already allocated
  OMP_TGT_MAPTYPE_PRESENT = 0x1000,
  // use a separate reference counter so that the data cannot be unmapped within
  // the structured region
  // This is an OpenMP extension for the sake of OpenACC support.
  OMP_TGT_MAPTYPE_OMPX_HOLD = 0x2000,
  // descriptor for non-contiguous target-update
  OMP_TGT_MAPTYPE_NON_CONTIG = 0x100000000000,
  // member of struct, member given by [16 MSBs] - 1
  OMP_TGT_MAPTYPE_MEMBER_OF = 0xffff000000000000
};

/// Flags for offload entries.
enum OpenMPOffloadingDeclareTargetFlags {
  /// Mark the entry global as having a 'link' attribute.
  OMP_DECLARE_TARGET_LINK = 0x01,
  /// Mark the entry global as being an indirectly callable function.
  OMP_DECLARE_TARGET_INDIRECT = 0x08,
  /// This is an entry corresponding to a requirement to be registered.
  OMP_REGISTER_REQUIRES = 0x10,
};

enum TargetAllocTy : int32_t {
  TARGET_ALLOC_DEVICE = 0,
  TARGET_ALLOC_HOST,
  TARGET_ALLOC_SHARED,
  TARGET_ALLOC_DEFAULT,
  /// The allocation will not block on other streams.
  TARGET_ALLOC_DEVICE_NON_BLOCKING,
};

inline KernelArgsTy CTorDTorKernelArgs = {
    1,       0, nullptr,   nullptr,   nullptr,   nullptr, nullptr,
    nullptr, 0, {0, 0, 0}, {1, 0, 0}, {1, 0, 0}, 0};

struct DeviceTy;

/// The libomptarget wrapper around a __tgt_async_info object directly
/// associated with a libomptarget layer device. RAII semantics to avoid
/// mistakes.
class AsyncInfoTy {
public:
  enum class SyncTy { BLOCKING, NON_BLOCKING };

private:
  /// Locations we used in (potentially) asynchronous calls which should live
  /// as long as this AsyncInfoTy object.
  std::deque<void *> BufferLocations;

  /// Post-processing operations executed after a successful synchronization.
  /// \note the post-processing function should return OFFLOAD_SUCCESS or
  /// OFFLOAD_FAIL appropriately.
  using PostProcFuncTy = std::function<int()>;
  llvm::SmallVector<PostProcFuncTy> PostProcessingFunctions;

  __tgt_async_info AsyncInfo;
  DeviceTy &Device;

public:
  /// Synchronization method to be used.
  SyncTy SyncType;

  AsyncInfoTy(DeviceTy &Device, SyncTy SyncType = SyncTy::BLOCKING)
      : Device(Device), SyncType(SyncType) {}
  ~AsyncInfoTy() { synchronize(); }

  /// Implicit conversion to the __tgt_async_info which is used in the
  /// plugin interface.
  operator __tgt_async_info *() { return &AsyncInfo; }

  /// Synchronize all pending actions.
  ///
  /// \note synchronization will be performance in a blocking or non-blocking
  /// manner, depending on the SyncType.
  ///
  /// \note if the operations are completed, the registered post-processing
  /// functions will be executed once and unregistered afterwards.
  ///
  /// \returns OFFLOAD_FAIL or OFFLOAD_SUCCESS appropriately.
  int synchronize();

  /// Return a void* reference with a lifetime that is at least as long as this
  /// AsyncInfoTy object. The location can be used as intermediate buffer.
  void *&getVoidPtrLocation();

  /// Check if all asynchronous operations are completed.
  ///
  /// \note only a lightweight check. If needed, use synchronize() to query the
  /// status of AsyncInfo before checking.
  ///
  /// \returns true if there is no pending asynchronous operations, false
  /// otherwise.
  bool isDone() const;

  /// Add a new post-processing function to be executed after synchronization.
  ///
  /// \param[in] Function is a templated function (e.g., function pointers,
  /// lambdas, std::function) that can be convertible to a PostProcFuncTy (i.e.,
  /// it must have int() as its function signature).
  template <typename FuncTy> void addPostProcessingFunction(FuncTy &&Function) {
    static_assert(std::is_convertible_v<FuncTy, PostProcFuncTy>,
                  "Invalid post-processing function type. Please check "
                  "function signature!");
    PostProcessingFunctions.emplace_back(Function);
  }

private:
  /// Run all the post-processing functions sequentially.
  ///
  /// \note after a successful execution, all previously registered functions
  /// are unregistered.
  ///
  /// \returns OFFLOAD_FAIL if any post-processing function failed,
  /// OFFLOAD_SUCCESS otherwise.
  int32_t runPostProcessing();

  /// Check if the internal asynchronous info queue is empty or not.
  ///
  /// \returns true if empty, false otherwise.
  bool isQueueEmpty() const;
};

// Wrapper for task stored async info objects.
class TaskAsyncInfoWrapperTy {
  // Invalid GTID as defined by libomp; keep in sync
  static constexpr int KMP_GTID_DNE = -2;

  const int ExecThreadID = KMP_GTID_DNE;
  AsyncInfoTy LocalAsyncInfo;
  AsyncInfoTy *AsyncInfo = &LocalAsyncInfo;
  void **TaskAsyncInfoPtr = nullptr;

public:
  TaskAsyncInfoWrapperTy(DeviceTy &Device)
      : ExecThreadID(__kmpc_global_thread_num(NULL)), LocalAsyncInfo(Device) {
    // If we failed to acquired the current global thread id, we cannot
    // re-enqueue the current task. Thus we should use the local blocking async
    // info.
    if (ExecThreadID == KMP_GTID_DNE)
      return;

    // Only tasks with an assigned task team can be re-enqueue and thus can
    // use the non-blocking synchronization scheme. Thus we should use the local
    // blocking async info, if we don´t have one.
    if (!__kmpc_omp_has_task_team(ExecThreadID))
      return;

    // Acquire a pointer to the AsyncInfo stored inside the current task being
    // executed.
    TaskAsyncInfoPtr = __kmpc_omp_get_target_async_handle_ptr(ExecThreadID);

    // If we cannot acquire such pointer, fallback to using the local blocking
    // async info.
    if (!TaskAsyncInfoPtr)
      return;

    // When creating a new task async info, the task handle must always be
    // invalid. We must never overwrite any task async handle and there should
    // never be any valid handle store inside the task at this point.
    assert((*TaskAsyncInfoPtr) == nullptr &&
           "Task async handle is not empty when dispatching new device "
           "operations. The handle was not cleared properly or "
           "__tgt_target_nowait_query should have been called!");

    // If no valid async handle is present, a new AsyncInfo will be allocated
    // and stored in the current task.
    AsyncInfo = new AsyncInfoTy(Device, AsyncInfoTy::SyncTy::NON_BLOCKING);
    *TaskAsyncInfoPtr = (void *)AsyncInfo;
  }

  ~TaskAsyncInfoWrapperTy() {
    // Local async info destruction is automatically handled by ~AsyncInfoTy.
    if (AsyncInfo == &LocalAsyncInfo)
      return;

    // If the are device operations still pending, return immediately without
    // deallocating the handle.
    if (!AsyncInfo->isDone())
      return;

    // Delete the handle and unset it from the OpenMP task data.
    delete AsyncInfo;
    *TaskAsyncInfoPtr = nullptr;
  }

  operator AsyncInfoTy &() { return *AsyncInfo; }
};

/// This struct is a record of non-contiguous information
struct __tgt_target_non_contig {
  uint64_t Offset;
  uint64_t Count;
  uint64_t Stride;
};

#ifdef __cplusplus
extern "C" {
#endif

void ompx_dump_mapping_tables(void);
int omp_get_num_devices(void);
int omp_get_device_num(void);
int omp_get_initial_device(void);
void *omp_target_alloc(size_t Size, int DeviceNum);
void omp_target_free(void *DevicePtr, int DeviceNum);
int omp_target_is_present(const void *Ptr, int DeviceNum);
int omp_target_memcpy(void *Dst, const void *Src, size_t Length,
                      size_t DstOffset, size_t SrcOffset, int DstDevice,
                      int SrcDevice);
int omp_target_memcpy_rect(void *Dst, const void *Src, size_t ElementSize,
                           int NumDims, const size_t *Volume,
                           const size_t *DstOffsets, const size_t *SrcOffsets,
                           const size_t *DstDimensions,
                           const size_t *SrcDimensions, int DstDevice,
                           int SrcDevice);
void *omp_target_memset(void *Ptr, int C, size_t N, int DeviceNum);
int omp_target_associate_ptr(const void *HostPtr, const void *DevicePtr,
                             size_t Size, size_t DeviceOffset, int DeviceNum);
int omp_target_disassociate_ptr(const void *HostPtr, int DeviceNum);

/// Explicit target memory allocators
/// Using the llvm_ prefix until they become part of the OpenMP standard.
void *llvm_omp_target_alloc_device(size_t Size, int DeviceNum);
void *llvm_omp_target_alloc_host(size_t Size, int DeviceNum);
void *llvm_omp_target_alloc_shared(size_t Size, int DeviceNum);

/// Explicit target memory deallocators
/// Using the llvm_ prefix until they become part of the OpenMP standard.
void llvm_omp_target_free_device(void *DevicePtr, int DeviceNum);
void llvm_omp_target_free_host(void *DevicePtr, int DeviceNum);
void llvm_omp_target_free_shared(void *DevicePtr, int DeviceNum);

/// Dummy target so we have a symbol for generating host fallback.
void *llvm_omp_target_dynamic_shared_alloc();

/// add the clauses of the requires directives in a given file
void __tgt_register_requires(int64_t Flags);

/// Initializes the runtime library.
void __tgt_rtl_init();

/// Deinitializes the runtime library.
void __tgt_rtl_deinit();

/// adds a target shared library to the target execution image
void __tgt_register_lib(__tgt_bin_desc *Desc);

/// Initialize all RTLs at once
void __tgt_init_all_rtls();

/// removes a target shared library from the target execution image
void __tgt_unregister_lib(__tgt_bin_desc *Desc);

// creates the host to target data mapping, stores it in the
// libomptarget.so internal structure (an entry in a stack of data maps) and
// passes the data to the device;
void __tgt_target_data_begin(int64_t DeviceId, int32_t ArgNum, void **ArgsBase,
                             void **Args, int64_t *ArgSizes, int64_t *ArgTypes);
void __tgt_target_data_begin_nowait(int64_t DeviceId, int32_t ArgNum,
                                    void **ArgsBase, void **Args,
                                    int64_t *ArgSizes, int64_t *ArgTypes,
                                    int32_t DepNum, void *DepList,
                                    int32_t NoAliasDepNum,
                                    void *NoAliasDepList);
void __tgt_target_data_begin_mapper(ident_t *Loc, int64_t DeviceId,
                                    int32_t ArgNum, void **ArgsBase,
                                    void **Args, int64_t *ArgSizes,
                                    int64_t *ArgTypes, map_var_info_t *ArgNames,
                                    void **ArgMappers);
void __tgt_target_data_begin_nowait_mapper(
    ident_t *Loc, int64_t DeviceId, int32_t ArgNum, void **ArgsBase,
    void **Args, int64_t *ArgSizes, int64_t *ArgTypes, map_var_info_t *ArgNames,
    void **ArgMappers, int32_t DepNum, void *DepList, int32_t NoAliasDepNum,
    void *NoAliasDepList);

// passes data from the target, release target memory and destroys the
// host-target mapping (top entry from the stack of data maps) created by
// the last __tgt_target_data_begin
void __tgt_target_data_end(int64_t DeviceId, int32_t ArgNum, void **ArgsBase,
                           void **Args, int64_t *ArgSizes, int64_t *ArgTypes);
void __tgt_target_data_end_nowait(int64_t DeviceId, int32_t ArgNum,
                                  void **ArgsBase, void **Args,
                                  int64_t *ArgSizes, int64_t *ArgTypes,
                                  int32_t DepNum, void *DepList,
                                  int32_t NoAliasDepNum, void *NoAliasDepList);
void __tgt_target_data_end_mapper(ident_t *Loc, int64_t DeviceId,
                                  int32_t ArgNum, void **ArgsBase, void **Args,
                                  int64_t *ArgSizes, int64_t *ArgTypes,
                                  map_var_info_t *ArgNames, void **ArgMappers);
void __tgt_target_data_end_nowait_mapper(
    ident_t *Loc, int64_t DeviceId, int32_t ArgNum, void **ArgsBase,
    void **Args, int64_t *ArgSizes, int64_t *ArgTypes, map_var_info_t *ArgNames,
    void **ArgMappers, int32_t depNum, void *depList, int32_t NoAliasDepNum,
    void *NoAliasDepList);

/// passes data to/from the target
void __tgt_target_data_update(int64_t DeviceId, int32_t ArgNum, void **ArgsBase,
                              void **Args, int64_t *ArgSizes,
                              int64_t *ArgTypes);
void __tgt_target_data_update_nowait(int64_t DeviceId, int32_t ArgNum,
                                     void **ArgsBase, void **Args,
                                     int64_t *ArgSizes, int64_t *ArgTypes,
                                     int32_t DepNum, void *DepList,
                                     int32_t NoAliasDepNum,
                                     void *NoAliasDepList);
void __tgt_target_data_update_mapper(ident_t *Loc, int64_t DeviceId,
                                     int32_t ArgNum, void **ArgsBase,
                                     void **Args, int64_t *ArgSizes,
                                     int64_t *ArgTypes,
                                     map_var_info_t *ArgNames,
                                     void **ArgMappers);
void __tgt_target_data_update_nowait_mapper(
    ident_t *Loc, int64_t DeviceId, int32_t ArgNum, void **ArgsBase,
    void **Args, int64_t *ArgSizes, int64_t *ArgTypes, map_var_info_t *ArgNames,
    void **ArgMappers, int32_t DepNum, void *DepList, int32_t NoAliasDepNum,
    void *NoAliasDepList);

// Performs the same actions as data_begin in case ArgNum is non-zero
// and initiates run of offloaded region on target platform; if ArgNum
// is non-zero after the region execution is done it also performs the
// same action as data_end above. The following types are used; this
// function returns 0 if it was able to transfer the execution to a
// target and an int different from zero otherwise.
int __tgt_target_kernel(ident_t *Loc, int64_t DeviceId, int32_t NumTeams,
                        int32_t ThreadLimit, void *HostPtr, KernelArgsTy *Args);

// Non-blocking synchronization for target nowait regions. This function
// acquires the asynchronous context from task data of the current task being
// executed and tries to query for the completion of its operations. If the
// operations are still pending, the function returns immediately. If the
// operations are completed, all the post-processing procedures stored in the
// asynchronous context are executed and the context is removed from the task
// data.
void __tgt_target_nowait_query(void **AsyncHandle);

/// Executes a target kernel by replaying recorded kernel arguments and
/// device memory.
int __tgt_target_kernel_replay(ident_t *Loc, int64_t DeviceId, void *HostPtr,
                               void *DeviceMemory, int64_t DeviceMemorySize,
                               void **TgtArgs, ptrdiff_t *TgtOffsets,
                               int32_t NumArgs, int32_t NumTeams,
                               int32_t ThreadLimit, uint64_t LoopTripCount);

void __tgt_set_info_flag(uint32_t);

int __tgt_print_device_info(int64_t DeviceId);

int __tgt_activate_record_replay(int64_t DeviceId, uint64_t MemorySize,
                                 void *VAddr, bool IsRecord, bool SaveOutput,
                                 uint64_t &ReqPtrArgOffset);

#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
#define EXTERN extern "C"
#else
#define EXTERN extern
#endif

#endif // _OMPTARGET_H_
