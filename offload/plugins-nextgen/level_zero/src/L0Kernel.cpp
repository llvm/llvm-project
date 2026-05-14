//===--- Level Zero Target RTL Implementation -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// GenericKernel implementation for SPIR-V/Xe machine.
//
//===----------------------------------------------------------------------===//

#include "L0Kernel.h"
#include "L0Device.h"
#include "L0Plugin.h"
#include "L0Program.h"

#include "llvm/ADT/ScopeExit.h"

namespace llvm::omp::target::plugin {

Error L0KernelTy::readKernelProperties(L0ProgramTy &Program) {
  const auto &l0Device = L0DeviceTy::makeL0Device(Program.getDevice());
  auto &KernelPR = getProperties();
  ze_kernel_properties_t KP = {};
  KP.stype = ZE_STRUCTURE_TYPE_KERNEL_PROPERTIES;
  KP.pNext = nullptr;
  ze_kernel_preferred_group_size_properties_t KPrefGRPSize = {};
  KPrefGRPSize.stype = ZE_STRUCTURE_TYPE_KERNEL_PREFERRED_GROUP_SIZE_PROPERTIES;
  KPrefGRPSize.pNext = nullptr;
  if (l0Device.getDriverAPIVersion() >= ZE_API_VERSION_1_2)
    KP.pNext = &KPrefGRPSize;

  CALL_ZE_RET_ERROR(zeKernelGetProperties, zeKernel, &KP);
  KernelPR.SIMDWidth = KP.maxSubgroupSize;
  KernelPR.Width = KP.maxSubgroupSize;
  KernelPR.NumKernelArgs = KP.numKernelArgs;

  if (KP.pNext)
    KernelPR.Width = KPrefGRPSize.preferredMultiple;

  if (!l0Device.isDeviceArch(DeviceArchTy::DeviceArch_Gen)) {
    KernelPR.Width = (std::max)(KernelPR.Width, 2 * KernelPR.SIMDWidth);
  }
  KernelPR.MaxThreadGroupSize = KP.maxSubgroupSize * KP.maxNumSubgroups;

  // Query and cache argument sizes if extension is available.
  auto &Context = l0Device.getL0Context();
  if (KernelPR.NumKernelArgs > 0 && Context.zexKernelGetArgumentSize) {
    KernelPR.ArgSizes = std::make_unique<uint32_t[]>(KernelPR.NumKernelArgs);
    for (uint32_t I = 0; I < KernelPR.NumKernelArgs; I++) {
      CALL_ZE_RET_ERROR(Context.zexKernelGetArgumentSize, zeKernel, I,
                        &KernelPR.ArgSizes[I]);
    }
  }

  return Plugin::success();
}

Error L0KernelTy::buildKernel(L0ProgramTy &Program) {
  const auto *KernelName = getName();

  auto Module = Program.findModuleFromKernelName(KernelName);
  if (!Module)
    return Plugin::error(ErrorCode::NOT_FOUND,
                         "kernel '%s' not found in the program", KernelName);

  ze_kernel_desc_t KernelDesc = {ZE_STRUCTURE_TYPE_KERNEL_DESC, nullptr, 0,
                                 KernelName};
  CALL_ZE_RET_ERROR(zeKernelCreate, Module, &KernelDesc, &zeKernel);
  if (auto Err = readKernelProperties(Program))
    return Err;

  return Plugin::success();
}

Error L0KernelTy::initImpl(GenericDeviceTy &GenericDevice,
                           DeviceImageTy &Image) {
  auto &Program = L0ProgramTy::makeL0Program(Image);

  if (auto Err = buildKernel(Program))
    return Err;
  Program.addKernel(this);

  return Plugin::success();
}

static Error launchKernelWithImmCmdList(L0DeviceTy &l0Device,
                                        ze_kernel_handle_t zeKernel,
                                        L0LaunchEnvTy &KEnv,
                                        CommandModeTy CommandMode) {
  const auto DeviceId = l0Device.getDeviceId();
  auto *IdStr = l0Device.getZeIdCStr();
  auto CmdListOrErr = l0Device.getImmCmdList();
  if (!CmdListOrErr)
    return CmdListOrErr.takeError();
  const ze_command_list_handle_t CmdList = *CmdListOrErr;
  // Command queue is not used with immediate command list.

  INFO(OMP_INFOTYPE_PLUGIN_KERNEL, DeviceId,
       "Using immediate command list for kernel submission.\n");
  auto EventOrError = l0Device.getEvent();
  if (!EventOrError)
    return EventOrError.takeError();
  ze_event_handle_t Event = *EventOrError;
  size_t NumWaitEvents = 0;
  ze_event_handle_t *WaitEvents = nullptr;
  auto *AsyncQueue = KEnv.AsyncQueue;
  if (KEnv.IsAsync && !AsyncQueue->WaitEvents.empty()) {
    if (CommandMode == CommandModeTy::AsyncOrdered) {
      NumWaitEvents = 1;
      WaitEvents = &AsyncQueue->WaitEvents.back();
    } else {
      NumWaitEvents = AsyncQueue->WaitEvents.size();
      WaitEvents = AsyncQueue->WaitEvents.data();
    }
  }
  INFO(OMP_INFOTYPE_PLUGIN_KERNEL, DeviceId,
       "Kernel depends on %zu data copying events.\n", NumWaitEvents);
  Error AllErrors = Error::success();

  CALL_ZE_ACCUM_ERROR(AllErrors, zeCommandListAppendLaunchKernel, CmdList,
                      zeKernel, &KEnv.GroupCounts, Event, NumWaitEvents,
                      WaitEvents);
  KEnv.Lock.unlock();
  if (AllErrors) {
    if (auto Err = l0Device.releaseEvent(Event))
      AllErrors = joinErrors(std::move(AllErrors), std::move(Err));
    return AllErrors;
  }
  INFO(OMP_INFOTYPE_PLUGIN_KERNEL, DeviceId,
       "Submitted kernel " DPxMOD " to device %s\n", DPxPTR(zeKernel), IdStr);

  if (KEnv.IsAsync) {
    AsyncQueue->WaitEvents.push_back(Event);
    AsyncQueue->KernelEvent = Event;
  } else {
    CALL_ZE_ACCUM_ERROR(AllErrors, zeEventHostSynchronize, Event,
                        L0DefaultTimeout);
    if (auto Err = l0Device.releaseEvent(Event))
      AllErrors = joinErrors(std::move(AllErrors), std::move(Err));
    if (AllErrors)
      return AllErrors;
  }
  INFO(OMP_INFOTYPE_PLUGIN_KERNEL, DeviceId,
       "Executed kernel entry " DPxMOD " on device %s\n", DPxPTR(zeKernel),
       IdStr);

  return Plugin::success();
}

static Error launchKernelWithCmdQueue(L0DeviceTy &l0Device,
                                      ze_kernel_handle_t zeKernel,
                                      L0LaunchEnvTy &KEnv) {
  const auto DeviceId = l0Device.getDeviceId();
  const auto *IdStr = l0Device.getZeIdCStr();

  auto CmdListOrErr = l0Device.getCmdList();
  if (!CmdListOrErr)
    return CmdListOrErr.takeError();
  ze_command_list_handle_t CmdList = *CmdListOrErr;
  auto CmdQueueOrErr = l0Device.getCmdQueue();
  if (!CmdQueueOrErr)
    return CmdQueueOrErr.takeError();
  const ze_command_queue_handle_t CmdQueue = *CmdQueueOrErr;

  INFO(OMP_INFOTYPE_PLUGIN_KERNEL, DeviceId,
       "Using regular command list for kernel submission.\n");

  ze_event_handle_t Event = nullptr;
  CALL_ZE_RET_ERROR(zeCommandListAppendLaunchKernel, CmdList, zeKernel,
                    &KEnv.GroupCounts, Event, 0, nullptr);
  KEnv.Lock.unlock();
  CALL_ZE_RET_ERROR(zeCommandListClose, CmdList);

  // Ensure command list is reset even on errors after this point.
  llvm::scope_exit ResetOnExit(
      [&]() { CALL_ZE_SILENT(zeCommandListReset, CmdList); });

  CALL_ZE_RET_ERROR_MTX(zeCommandQueueExecuteCommandLists, l0Device.getMutex(),
                        CmdQueue, 1, &CmdList, nullptr);
  INFO(OMP_INFOTYPE_PLUGIN_KERNEL, DeviceId,
       "Submitted kernel " DPxMOD " to device %s\n", DPxPTR(zeKernel), IdStr);
  CALL_ZE_RET_ERROR(zeCommandQueueSynchronize, CmdQueue, L0DefaultTimeout);
  if (Event) {
    if (auto Err = l0Device.releaseEvent(Event))
      return Err;
  }
  INFO(OMP_INFOTYPE_PLUGIN_KERNEL, DeviceId,
       "Executed kernel entry " DPxMOD " on device %s\n", DPxPTR(zeKernel),
       IdStr);

  return Plugin::success();
}

Error L0KernelTy::setKernelGroups(L0DeviceTy &l0Device, L0LaunchEnvTy &KEnv,
                                  uint32_t NumThreads[3],
                                  uint32_t NumBlocks[3]) const {
  assert(NumThreads[0] > 0 && NumThreads[1] > 0 && NumThreads[2] > 0 &&
         "Pre-computed ThreadLimit values must be non-zero");
  assert(NumBlocks[0] > 0 && NumBlocks[1] > 0 && NumBlocks[2] > 0 &&
         "Pre-computed NumTeams values must be non-zero");

  uint32_t GroupSizes[3];
  KEnv.GroupCounts = {NumBlocks[0], NumBlocks[1], NumBlocks[2]};
  // Respect max group size attribute in the kernel.
  uint32_t MaxGroupSize = KEnv.KernelPR.MaxThreadGroupSize;
  GroupSizes[0] = std::min<uint32_t>(MaxGroupSize, NumThreads[0]);
  GroupSizes[1] = std::min<uint32_t>(MaxGroupSize, NumThreads[1]);
  GroupSizes[2] = std::min<uint32_t>(MaxGroupSize, NumThreads[2]);

  auto DeviceId = l0Device.getDeviceId();
  INFO(OMP_INFOTYPE_PLUGIN_KERNEL, DeviceId,
       "Team sizes = {%" PRIu32 ", %" PRIu32 ", %" PRIu32 "}\n", GroupSizes[0],
       GroupSizes[1], GroupSizes[2]);
  INFO(OMP_INFOTYPE_PLUGIN_KERNEL, DeviceId,
       "Number of teams = {%" PRIu32 ", %" PRIu32 ", %" PRIu32 "}\n",
       KEnv.GroupCounts.groupCountX, KEnv.GroupCounts.groupCountY,
       KEnv.GroupCounts.groupCountZ);

  CALL_ZE_RET_ERROR(zeKernelSetGroupSize, getZeKernel(), GroupSizes[0],
                    GroupSizes[1], GroupSizes[2]);

  return Plugin::success();
}

Error L0KernelTy::setIndirectFlags(L0DeviceTy &l0Device,
                                   L0LaunchEnvTy &KEnv) const {
  // Set Kernel Indirect flags.
  ze_kernel_indirect_access_flags_t Flags = 0;
  Flags |= l0Device.getMemAllocator(TARGET_ALLOC_HOST).getIndirectFlags();
  Flags |= l0Device.getMemAllocator(TARGET_ALLOC_DEVICE).getIndirectFlags();

  if (KEnv.KernelPR.IndirectAccessFlags != Flags) {
    // Combine with common access flags.
    const auto FinalFlags = l0Device.getIndirectFlags() | Flags;
    CALL_ZE_RET_ERROR(zeKernelSetIndirectAccess, zeKernel, FinalFlags);
    ODBG(OLDT_Kernel) << "Setting indirect access flags "
                      << reinterpret_cast<void *>(FinalFlags);
    KEnv.KernelPR.IndirectAccessFlags = Flags;
  }

  return Plugin::success();
}

Error L0KernelTy::launchImpl(GenericDeviceTy &GenericDevice,
                             uint32_t NumThreads[3], uint32_t NumBlocks[3],
                             uint32_t DynBlockMemSize, KernelArgsTy &KernelArgs,
                             KernelLaunchParamsTy LaunchParams,
                             AsyncInfoWrapperTy &AsyncInfoWrapper) const {
  if (DynBlockMemSize > 0)
    return Plugin::error(ErrorCode::UNSUPPORTED,
                         "dynamic shared memory is unsupported in L0 plugin");

  auto &l0Device = L0DeviceTy::makeL0Device(GenericDevice);
  __tgt_async_info *AsyncInfo = AsyncInfoWrapper;

  auto zeKernel = getZeKernel();
  auto DeviceId = l0Device.getDeviceId();
  INFO(OMP_INFOTYPE_PLUGIN_KERNEL, DeviceId, "Launching kernel " DPxMOD "...\n",
       DPxPTR(zeKernel));

  auto &Plugin = l0Device.getPlugin();
  auto *IdStr = l0Device.getZeIdCStr();
  auto &Options = Plugin.getOptions();
  bool IsAsync = AsyncInfo && l0Device.asyncEnabled();
  if (IsAsync && !AsyncInfo->Queue) {
    AsyncInfo->Queue = reinterpret_cast<void *>(Plugin.getAsyncQueue());
    if (!AsyncInfo->Queue)
      IsAsync = false; // Couldn't get a queue, revert to sync.
  }
  auto *AsyncQueue =
      IsAsync ? static_cast<AsyncQueueTy *>(AsyncInfo->Queue) : nullptr;
  auto &KernelPR = getProperties();

  L0LaunchEnvTy KEnv(IsAsync, AsyncQueue, KernelPR);

  // Protect from kernel preparation to submission as kernels are shared.
  KEnv.Lock.lock();

  if (auto Err = setKernelGroups(l0Device, KEnv, NumThreads, NumBlocks))
    return Err;

  // Set kernel arguments.
  uint32_t NumKernelArgs = KernelPR.NumKernelArgs;
  if (NumKernelArgs > 0) {
    if (!KernelPR.ArgSizes)
      return Plugin::error(ErrorCode::INVALID_ARGUMENT,
                           "level zero plugin requires kernel argument sizes.");
    // Use sizes from kernel properties.
    // TODO: This is temporary workaround it will not work if there is
    // padding/alignment between arguments.
    char *Arg = static_cast<char *>(LaunchParams.Data);
    for (uint32_t I = 0; I < NumKernelArgs; I++) {
      uint32_t ArgSize = KernelPR.ArgSizes[I];
      CALL_ZE_RET_ERROR(zeKernelSetArgumentValue, zeKernel, I, ArgSize, Arg);

      INFO(OMP_INFOTYPE_PLUGIN_KERNEL, DeviceId,
           "Kernel Pointer argument %" PRIu32 " (value: " DPxMOD
           ") was set successfully for device %s.\n",
           I, DPxPTR(Arg), IdStr);
      Arg += ArgSize;
    }
  }

  if (auto Err = setIndirectFlags(l0Device, KEnv))
    return Err;

  // The next calls should unlock the KernelLock internally.
  const bool UseImmCmdList = l0Device.useImmForCompute();
  if (UseImmCmdList)
    return launchKernelWithImmCmdList(l0Device, zeKernel, KEnv,
                                      Options.CommandMode);

  return launchKernelWithCmdQueue(l0Device, zeKernel, KEnv);
}

} // namespace llvm::omp::target::plugin
