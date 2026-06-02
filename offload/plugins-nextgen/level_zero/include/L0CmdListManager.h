//===--- Level Zero Target RTL Implementation -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Command List Manager for Level Zero.
//
//===----------------------------------------------------------------------===//

#ifndef OPENMP_LIBOMPTARGET_PLUGINS_NEXTGEN_LEVEL_ZERO_L0CMDLISTMANAGER_H
#define OPENMP_LIBOMPTARGET_PLUGINS_NEXTGEN_LEVEL_ZERO_L0CMDLISTMANAGER_H

#include "L0Defs.h"
#include "L0Trace.h"
#include <mutex>

namespace llvm::omp::target::plugin {

/// Manager for Level Zero command list operations.
/// Encapsulates all zeCommandListAppend* calls with thread safety.
class L0CmdListManagerTy {
  /// Underlying immediate command list.
  ze_command_list_handle_t CmdList;
  /// Mutex to protect L0 operations that are not thread safe.
  std::mutex Mtx;

public:
  L0CmdListManagerTy(ze_command_list_handle_t CmdList) : CmdList(CmdList) {}

  ze_command_list_handle_t getCmdList() const { return CmdList; }

  Error hostSynchronize(uint64_t TimeoutNs = L0DefaultTimeout) {
    CALL_ZE_RET_ERROR(zeCommandListHostSynchronize, CmdList, TimeoutNs);
    return Plugin::success();
  }

  Expected<bool> queryPendingWork() {
    ze_result_t RC;
    CALL_ZE(RC, zeCommandListHostSynchronize, CmdList, 0);
    if (RC == ZE_RESULT_SUCCESS)
      return false;
    if (RC == ZE_RESULT_NOT_READY)
      return true;
    return Plugin::error(ErrorCode::UNKNOWN,
                         "zeCommandListHostSynchronize query failed: %s",
                         getZeErrorName(RC));
  }

  Error eventHostSynchronize(ze_event_handle_t Event,
                             uint64_t TimeoutNs = L0DefaultTimeout) {
    CALL_ZE_RET_ERROR(zeEventHostSynchronize, Event, TimeoutNs);
    return Plugin::success();
  }

  Error appendMemoryCopy(void *Dst, const void *Src, size_t Size,
                         ze_event_handle_t SignalEvent = nullptr,
                         uint32_t NumWaitEvents = 0,
                         ze_event_handle_t *WaitEvents = nullptr) {
    std::lock_guard<std::mutex> Lock(Mtx);
    CALL_ZE_RET_ERROR(zeCommandListAppendMemoryCopy, CmdList, Dst, Src, Size,
                      SignalEvent, NumWaitEvents, WaitEvents);
    return Plugin::success();
  }

  Error appendMemoryCopyRegion(void *Dst, const ze_copy_region_t *DstRegion,
                               uint32_t DstPitch, uint32_t DstSlicePitch,
                               const void *Src,
                               const ze_copy_region_t *SrcRegion,
                               uint32_t SrcPitch, uint32_t SrcSlicePitch,
                               ze_event_handle_t SignalEvent = nullptr,
                               uint32_t NumWaitEvents = 0,
                               ze_event_handle_t *WaitEvents = nullptr) {
    std::lock_guard<std::mutex> Lock(Mtx);
    CALL_ZE_RET_ERROR(zeCommandListAppendMemoryCopyRegion, CmdList, Dst,
                      DstRegion, DstPitch, DstSlicePitch, Src, SrcRegion,
                      SrcPitch, SrcSlicePitch, SignalEvent, NumWaitEvents,
                      WaitEvents);
    return Plugin::success();
  }

  Error appendMemoryFill(void *Ptr, const void *Pattern, size_t PatternSize,
                         size_t Size, ze_event_handle_t SignalEvent = nullptr,
                         uint32_t NumWaitEvents = 0,
                         ze_event_handle_t *WaitEvents = nullptr) {
    std::lock_guard<std::mutex> Lock(Mtx);
    CALL_ZE_RET_ERROR(zeCommandListAppendMemoryFill, CmdList, Ptr, Pattern,
                      PatternSize, Size, SignalEvent, NumWaitEvents,
                      WaitEvents);
    return Plugin::success();
  }

  Error appendLaunchKernel(ze_kernel_handle_t Kernel,
                           const ze_group_count_t *pLaunchFuncArgs,
                           ze_event_handle_t SignalEvent = nullptr,
                           uint32_t NumWaitEvents = 0,
                           ze_event_handle_t *WaitEvents = nullptr,
                           bool Cooperative = false) {
    std::lock_guard<std::mutex> Lock(Mtx);
    if (Cooperative) {
      CALL_ZE_RET_ERROR(zeCommandListAppendLaunchCooperativeKernel, CmdList,
                        Kernel, pLaunchFuncArgs, SignalEvent, NumWaitEvents,
                        WaitEvents);
    } else {
      CALL_ZE_RET_ERROR(zeCommandListAppendLaunchKernel, CmdList, Kernel,
                        pLaunchFuncArgs, SignalEvent, NumWaitEvents,
                        WaitEvents);
    }
    return Plugin::success();
  }

  Error appendLaunchKernelWithArgs(
      ze_kernel_handle_t Kernel, const ze_group_count_t *GroupCounts,
      const ze_group_size_t *GroupSizes, void **ArgPtrs,
      ze_event_handle_t SignalEvent = nullptr, uint32_t NumWaitEvents = 0,
      ze_event_handle_t *WaitEvents = nullptr, bool IsCooperative = false) {
    ze_command_list_append_launch_kernel_param_cooperative_desc_t CoopDesc = {
        ZE_STRUCTURE_TYPE_COMMAND_LIST_APPEND_PARAM_COOPERATIVE_DESC, nullptr,
        static_cast<ze_bool_t>(IsCooperative)};
    std::lock_guard<std::mutex> Lock(Mtx);
    CALL_ZE_RET_ERROR(zeCommandListAppendLaunchKernelWithArguments, CmdList,
                      Kernel, *GroupCounts, *GroupSizes, ArgPtrs,
                      IsCooperative ? &CoopDesc : nullptr, SignalEvent,
                      NumWaitEvents, WaitEvents);
    return Plugin::success();
  }
};

} // namespace llvm::omp::target::plugin
#endif // OPENMP_LIBOMPTARGET_PLUGINS_NEXTGEN_LEVEL_ZERO_L0CMDLISTMANAGER_H
