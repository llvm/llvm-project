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

namespace llvm::omp::target::plugin {

bool KernelPropertiesTy::reuseGroupParams(const int32_t NumTeamsIn,
                                          const int32_t ThreadLimitIn,
                                          uint32_t *GroupSizesOut,
                                          L0LaunchEnvTy &KEnv) const {
  if (NumTeamsIn != NumTeams || ThreadLimitIn != ThreadLimit)
    return false;
  // Found matching input parameters.
  std::copy_n(GroupSizes, 3, GroupSizesOut);
  KEnv.GroupCounts = GroupCounts;
  return true;
}

void KernelPropertiesTy::cacheGroupParams(const int32_t NumTeamsIn,
                                          const int32_t ThreadLimitIn,
                                          const uint32_t *GroupSizesIn,
                                          L0LaunchEnvTy &KEnv) {
  NumTeams = NumTeamsIn;
  ThreadLimit = ThreadLimitIn;
  std::copy_n(GroupSizesIn, 3, GroupSizes);
  GroupCounts = KEnv.GroupCounts;
}

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

  if (KP.pNext)
    KernelPR.Width = KPrefGRPSize.preferredMultiple;

  if (!l0Device.isDeviceArch(DeviceArchTy::DeviceArch_Gen)) {
    KernelPR.Width = (std::max)(KernelPR.Width, 2 * KernelPR.SIMDWidth);
  }
  KernelPR.MaxThreadGroupSize = KP.maxSubgroupSize * KP.maxNumSubgroups;
  return Plugin::success();
}

Error L0KernelTy::buildKernel(L0ProgramTy &Program) {
  const auto *KernelName = getName();

  auto Module = Program.findModuleFromKernelName(KernelName);
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

void L0KernelTy::decideKernelGroupArguments(L0DeviceTy &Device,
                                            uint32_t NumTeams,
                                            uint32_t ThreadLimit,
                                            uint32_t *GroupSizes,
                                            L0LaunchEnvTy &KEnv) const {

  const KernelPropertiesTy &KernelPR = getProperties();

  const auto DeviceId = Device.getDeviceId();
  bool MaxGroupSizeForced = false;
  bool MaxGroupCountForced = false;
  uint32_t MaxGroupSize = Device.getMaxGroupSize();
  const auto &Option = LevelZeroPluginTy::getOptions();
  const auto OptSubscRate = Option.SubscriptionRate;
  auto &GroupCounts = KEnv.GroupCounts;

  uint32_t SIMDWidth = KernelPR.SIMDWidth;
  uint32_t KernelWidth = KernelPR.Width;
  uint32_t KernelMaxThreadGroupSize = KernelPR.MaxThreadGroupSize;

  if (KernelMaxThreadGroupSize < MaxGroupSize) {
    MaxGroupSize = KernelMaxThreadGroupSize;
    INFO(OMP_INFOTYPE_PLUGIN_KERNEL, DeviceId,
         "Capping maximum team size to %" PRIu32
         " due to kernel constraints.\n",
         MaxGroupSize);
  }

  if (ThreadLimit > 0) {
    MaxGroupSizeForced = true;
    MaxGroupSize = ThreadLimit;
  }

  uint32_t MaxGroupCount = 0;
  if (NumTeams > 0) {
    MaxGroupCount = NumTeams;
    MaxGroupCountForced = true;
  }

  if (MaxGroupCountForced) {
    // If number of teams is specified by the user, then use KernelWidth.
    // WIs per WG by default, so that it matches
    // decideLoopKernelGroupArguments() behavior.
    if (!MaxGroupSizeForced) {
      MaxGroupSize = KernelWidth;
    }
  } else {
    const uint32_t NumSubslices = Device.getNumSubslices();
    uint32_t NumThreadsPerSubslice = Device.getNumThreadsPerSubslice();
    if (KEnv.HalfNumThreads)
      NumThreadsPerSubslice /= 2;

    MaxGroupCount = NumSubslices * NumThreadsPerSubslice;
    if (MaxGroupSizeForced) {
      // Set group size for the HW capacity.
      uint32_t NumThreadsPerGroup = (MaxGroupSize + SIMDWidth - 1) / SIMDWidth;
      uint32_t NumGroupsPerSubslice =
          (NumThreadsPerSubslice + NumThreadsPerGroup - 1) / NumThreadsPerGroup;
      MaxGroupCount = NumGroupsPerSubslice * NumSubslices;
    } else {
      assert(!MaxGroupSizeForced && !MaxGroupCountForced);
      assert((MaxGroupSize <= KernelWidth || MaxGroupSize % KernelWidth == 0) &&
             "Invalid maxGroupSize");
      // Maximize group size.
      while (MaxGroupSize >= KernelWidth) {
        uint32_t NumThreadsPerGroup =
            (MaxGroupSize + SIMDWidth - 1) / SIMDWidth;

        if (NumThreadsPerSubslice % NumThreadsPerGroup == 0) {
          uint32_t NumGroupsPerSubslice =
              NumThreadsPerSubslice / NumThreadsPerGroup;
          MaxGroupCount = NumGroupsPerSubslice * NumSubslices;
          break;
        }
        MaxGroupSize -= KernelWidth;
      }
    }
  }

  uint32_t GRPCounts[3] = {MaxGroupCount, 1, 1};
  uint32_t GRPSizes[3] = {MaxGroupSize, 1, 1};
  if (!MaxGroupCountForced) {
    GRPCounts[0] *= OptSubscRate;
  }
  GroupCounts.groupCountX = GRPCounts[0];
  GroupCounts.groupCountY = GRPCounts[1];
  GroupCounts.groupCountZ = GRPCounts[2];
  std::copy(GRPSizes, GRPSizes + 3, GroupSizes);
}

Error L0KernelTy::getGroupsShape(L0DeviceTy &Device, int32_t NumTeams,
                                 int32_t ThreadLimit, uint32_t *GroupSizes,
                                 L0LaunchEnvTy &KEnv) const {

  const auto DeviceId = Device.getDeviceId();
  const auto &KernelPR = getProperties();

  // Read the most recent global thread limit and max teams.
  const int32_t NumTeamsICV = 0;
  const int32_t ThreadLimitICV = 0;

  bool IsXeHPG = Device.isDeviceArch(DeviceArchTy::DeviceArch_XeHPG);
  KEnv.HalfNumThreads =
      LevelZeroPluginTy::getOptions().ZeDebugEnabled && IsXeHPG;
  uint32_t KernelWidth = KernelPR.Width;
  uint32_t SIMDWidth = KernelPR.SIMDWidth;
  INFO(OMP_INFOTYPE_PLUGIN_KERNEL, DeviceId,
       "Assumed kernel SIMD width is %" PRIu32 "\n", SIMDWidth);
  INFO(OMP_INFOTYPE_PLUGIN_KERNEL, DeviceId,
       "Preferred team size is multiple of %" PRIu32 "\n", KernelWidth);
  assert(SIMDWidth <= KernelWidth && "Invalid SIMD width.");

  if (ThreadLimit > 0) {
    // use thread_limit clause value default.
    ODBG(OLDT_Kernel) << "Max team size is set to " << ThreadLimit
                      << " (thread_limit clause)";
  } else if (ThreadLimitICV > 0) {
    // else use thread-limit-var ICV.
    ThreadLimit = ThreadLimitICV;
    ODBG(OLDT_Kernel) << "Max team size is set to " << ThreadLimit
                      << " (thread-limit-icv)";
  }

  size_t MaxThreadLimit = Device.getMaxGroupSize();
  // Set correct max group size if the kernel was compiled with explicit SIMD.
  if (SIMDWidth == 1)
    MaxThreadLimit = Device.getNumThreadsPerSubslice();

  if (KernelPR.MaxThreadGroupSize < MaxThreadLimit) {
    MaxThreadLimit = KernelPR.MaxThreadGroupSize;
    ODBG(OLDT_Kernel) << "Capping maximum team size to " << MaxThreadLimit
                      << " due to kernel constraints.";
  }

  if (ThreadLimit > static_cast<int32_t>(MaxThreadLimit)) {
    ThreadLimit = MaxThreadLimit;
    ODBG(OLDT_Kernel) << "Max team size exceeds current maximum "
                      << MaxThreadLimit << ". Adjusted";
  }
  // scope code to ease integration with downstream custom code.
  {
    if (NumTeams > 0) {
      ODBG(OLDT_Kernel) << "Number of teams is set to " << NumTeams
                        << " (num_teams clause or no teams construct)";
    } else if (NumTeamsICV > 0) {
      // OMP_NUM_TEAMS only matters, if num_teams() clause is absent.
      INFO(OMP_INFOTYPE_PLUGIN_KERNEL, DeviceId,
           "OMP_NUM_TEAMS(%" PRId32 ") is ignored\n", NumTeamsICV);

      NumTeams = NumTeamsICV;
      ODBG(OLDT_Kernel) << "Max number of teams is set to " << NumTeams
                        << " (OMP_NUM_TEAMS)";
    }

    decideKernelGroupArguments(Device, (uint32_t)NumTeams,
                               (uint32_t)ThreadLimit, GroupSizes, KEnv);
  }

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
  CALL_ZE_RET_ERROR(zeCommandListAppendLaunchKernel, CmdList, zeKernel,
                    &KEnv.GroupCounts, Event, NumWaitEvents, WaitEvents);
  KEnv.KernelPR.Mtx.unlock();
  INFO(OMP_INFOTYPE_PLUGIN_KERNEL, DeviceId,
       "Submitted kernel " DPxMOD " to device %s\n", DPxPTR(zeKernel), IdStr);

  if (KEnv.IsAsync) {
    AsyncQueue->WaitEvents.push_back(Event);
    AsyncQueue->KernelEvent = Event;
  } else {
    CALL_ZE_RET_ERROR(zeEventHostSynchronize, Event, L0DefaultTimeout);
    if (auto Err = l0Device.releaseEvent(Event))
      return Err;
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
  KEnv.KernelPR.Mtx.unlock();
  CALL_ZE_RET_ERROR(zeCommandListClose, CmdList);
  CALL_ZE_RET_ERROR_MTX(zeCommandQueueExecuteCommandLists, l0Device.getMutex(),
                        CmdQueue, 1, &CmdList, nullptr);
  INFO(OMP_INFOTYPE_PLUGIN_KERNEL, DeviceId,
       "Submitted kernel " DPxMOD " to device %s\n", DPxPTR(zeKernel), IdStr);
  CALL_ZE_RET_ERROR(zeCommandQueueSynchronize, CmdQueue, L0DefaultTimeout);
  CALL_ZE_RET_ERROR(zeCommandListReset, CmdList);
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

  if (KernelEnvironment.Configuration.ExecMode != OMP_TGT_EXEC_MODE_BARE) {
    // For non-bare mode, the groups are already set in the launch.
    KEnv.GroupCounts = {NumBlocks[0], NumBlocks[1], NumBlocks[2]};
    CALL_ZE_RET_ERROR(zeKernelSetGroupSize, getZeKernel(), NumThreads[0],
                      NumThreads[1], NumThreads[2]);
    return Plugin::success();
  }

  int32_t NumTeams = NumBlocks[0];
  int32_t ThreadLimit = NumThreads[0];
  if (NumTeams < 0)
    NumTeams = 0;
  if (ThreadLimit < 0)
    ThreadLimit = 0;

  uint32_t GroupSizes[3];
  auto DeviceId = l0Device.getDeviceId();
  auto &KernelPR = KEnv.KernelPR;
  // Check if we can reuse previous group parameters.
  bool GroupParamsReused =
      KernelPR.reuseGroupParams(NumTeams, ThreadLimit, GroupSizes, KEnv);

  if (!GroupParamsReused) {
    if (auto Err =
            getGroupsShape(l0Device, NumTeams, ThreadLimit, GroupSizes, KEnv))
      return Err;
    KernelPR.cacheGroupParams(NumTeams, ThreadLimit, GroupSizes, KEnv);
  }

  INFO(OMP_INFOTYPE_PLUGIN_KERNEL, DeviceId,
       "Team sizes = {%" PRIu32 ", %" PRIu32 ", %" PRIu32 "}\n", GroupSizes[0],
       GroupSizes[1], GroupSizes[2]);
  INFO(OMP_INFOTYPE_PLUGIN_KERNEL, DeviceId,
       "Number of teams = {%" PRIu32 ", %" PRIu32 ", %" PRIu32 "}\n",
       KEnv.GroupCounts.groupCountX, KEnv.GroupCounts.groupCountY,
       KEnv.GroupCounts.groupCountZ);

  if (!GroupParamsReused) {
    CALL_ZE_RET_ERROR(zeKernelSetGroupSize, getZeKernel(), GroupSizes[0],
                      GroupSizes[1], GroupSizes[2]);
  }

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
                             KernelArgsTy &KernelArgs,
                             KernelLaunchParamsTy LaunchParams,
                             AsyncInfoWrapperTy &AsyncInfoWrapper) const {
  auto &l0Device = L0DeviceTy::makeL0Device(GenericDevice);
  __tgt_async_info *AsyncInfo = AsyncInfoWrapper;

  auto zeKernel = getZeKernel();
  auto DeviceId = l0Device.getDeviceId();
  int32_t NumArgs = KernelArgs.NumArgs;
  INFO(OMP_INFOTYPE_PLUGIN_KERNEL, DeviceId, "Launching kernel " DPxMOD "...\n",
       DPxPTR(zeKernel));

  auto &Plugin = l0Device.getPlugin();
  auto *IdStr = l0Device.getZeIdCStr();
  auto &Options = LevelZeroPluginTy::getOptions();
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
  KernelPR.Mtx.lock();

  if (auto Err = setKernelGroups(l0Device, KEnv, NumThreads, NumBlocks))
    return Err;

  // Set kernel arguments.
  for (int32_t I = 0; I < NumArgs; I++) {
    // Scope code to ease integration with downstream custom code.
    {
      void *Arg = (static_cast<void **>(LaunchParams.Data))[I];
      CALL_ZE_RET_ERROR(zeKernelSetArgumentValue, zeKernel, I, sizeof(Arg),
                        Arg == nullptr ? nullptr : &Arg);
      INFO(OMP_INFOTYPE_PLUGIN_KERNEL, DeviceId,
           "Kernel Pointer argument %" PRId32 " (value: " DPxMOD
           ") was set successfully for device %s.\n",
           I, DPxPTR(Arg), IdStr);
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
