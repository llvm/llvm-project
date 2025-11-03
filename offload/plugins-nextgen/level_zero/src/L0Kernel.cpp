//===--- Level Zero Target RTL Implementation -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// GenericKernel implementation for SPIR-V/Xe machine
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
  if (!KEnv.LoopDesc && LoopDescInit != LoopDesc)
    return false;
  if (KEnv.LoopDesc && *KEnv.LoopDesc != LoopDesc)
    return false;
  if (NumTeamsIn != NumTeams || ThreadLimitIn != ThreadLimit)
    return false;
  // Found matching input parameters.
  std::copy_n(GroupSizes, 3, GroupSizesOut);
  KEnv.GroupCounts = GroupCounts;
  KEnv.AllowCooperative = AllowCooperative;
  return true;
}

void KernelPropertiesTy::cacheGroupParams(const int32_t NumTeamsIn,
                                          const int32_t ThreadLimitIn,
                                          const uint32_t *GroupSizesIn,
                                          L0LaunchEnvTy &KEnv) {
  LoopDesc = KEnv.LoopDesc ? *KEnv.LoopDesc : LoopDescInit;
  NumTeams = NumTeamsIn;
  ThreadLimit = ThreadLimitIn;
  std::copy_n(GroupSizesIn, 3, GroupSizes);
  GroupCounts = KEnv.GroupCounts;
  AllowCooperative = KEnv.AllowCooperative;
}

Error L0KernelTy::buildKernel(L0ProgramTy &Program) {
  const auto *KernelName = getName();

  auto Module = Program.findModuleFromKernelName(KernelName);
  ze_kernel_desc_t KernelDesc = {ZE_STRUCTURE_TYPE_KERNEL_DESC, nullptr, 0,
                                 KernelName};
  CALL_ZE_RET_ERROR(zeKernelCreate, Module, &KernelDesc, &zeKernel);
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
    // If number of teams is specified by the user, then use KernelWidth
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
      // Set group size for the HW capacity
      uint32_t NumThreadsPerGroup = (MaxGroupSize + SIMDWidth - 1) / SIMDWidth;
      uint32_t NumGroupsPerSubslice =
          (NumThreadsPerSubslice + NumThreadsPerGroup - 1) / NumThreadsPerGroup;
      MaxGroupCount = NumGroupsPerSubslice * NumSubslices;
    } else {
      assert(!MaxGroupSizeForced && !MaxGroupCountForced);
      assert((MaxGroupSize <= KernelWidth || MaxGroupSize % KernelWidth == 0) &&
             "Invalid maxGroupSize");
      // Maximize group size
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
  bool UsedReductionSubscriptionRate = false;
  if (!MaxGroupCountForced) {
    {
      GRPCounts[0] *= OptSubscRate;
    }

    size_t LoopTripcount = 0;
    TgtNDRangeDescTy *LoopLevels = KEnv.LoopDesc;
    if (LoopLevels) {
      // TODO: consider other possible LoopDesc uses
      INFO(OMP_INFOTYPE_PLUGIN_KERNEL, DeviceId,
           "Loop desciptor provided but specific ND-range is disabled\n");
      // TODO: get rid of this constraint
      if (LoopLevels->NumLoops > 1) {
        INFO(OMP_INFOTYPE_PLUGIN_KERNEL, DeviceId,
             "More than 1 loop found (%" PRIu32 "), ignoring loop info\n",
             LoopLevels->NumLoops);
      } else if (LoopLevels->Levels[0].Ub >= LoopLevels->Levels[0].Lb) {
        LoopTripcount = (LoopLevels->Levels[0].Ub - LoopLevels->Levels[0].Lb +
                         LoopLevels->Levels[0].Stride) /
                        LoopLevels->Levels[0].Stride;
        INFO(OMP_INFOTYPE_PLUGIN_KERNEL, DeviceId,
             "Loop TC = (%" PRId64 " - %" PRId64 " + %" PRId64 ") / %" PRId64
             " = %zu\n",
             LoopLevels->Levels[0].Ub, LoopLevels->Levels[0].Lb,
             LoopLevels->Levels[0].Stride, LoopLevels->Levels[0].Stride,
             LoopTripcount);
      }
    }

    if (LoopTripcount && !UsedReductionSubscriptionRate) {
      const size_t MaxTotalThreads = Device.getNumThreadsPerSubslice() *
                                     Device.getNumSubslices() * SIMDWidth;
      size_t AdjustedGroupCount =
          KEnv.IsTeamsNDRange
              ? (std::min)(((LoopTripcount + 7) & ~7),
                           MaxTotalThreads / GRPSizes[0])
              : ((LoopTripcount + GRPSizes[0] - 1) / GRPSizes[0]);
      AdjustedGroupCount = std::max(AdjustedGroupCount, size_t{1});
      AdjustedGroupCount *= OptSubscRate;
      INFO(OMP_INFOTYPE_PLUGIN_KERNEL, DeviceId,
           "Adjusting number of teams using the loop tripcount\n");
      if (AdjustedGroupCount < GRPCounts[0])
        GRPCounts[0] = AdjustedGroupCount;
    }
  }
  GroupCounts.groupCountX = GRPCounts[0];
  GroupCounts.groupCountY = GRPCounts[1];
  GroupCounts.groupCountZ = GRPCounts[2];
  std::copy(GRPSizes, GRPSizes + 3, GroupSizes);
}

// Return the number of total HW threads required to execute
// a loop kernel compiled with the given SIMDWidth, and the given
// loop(s) trip counts and group sizes.
// Returns UINT64_MAX, if computations overflow.
static uint64_t computeThreadsNeeded(const llvm::ArrayRef<size_t> TripCounts,
                                     const llvm::ArrayRef<uint32_t> GroupSizes,
                                     uint32_t SIMDWidth) {
  assert(TripCounts.size() == 3 && "Invalid trip counts array size");
  assert(GroupSizes.size() == 3 && "Invalid group sizes array size");
  // Compute the number of groups in each dimension.
  std::array<uint64_t, 3> GroupCount;

  for (int I = 0; I < 3; ++I) {
    if (TripCounts[I] == 0 || GroupSizes[I] == 0)
      return (std::numeric_limits<uint64_t>::max)();
    GroupCount[I] =
        (uint64_t(TripCounts[I]) + GroupSizes[I] - 1) / GroupSizes[I];
    if (GroupCount[I] > (std::numeric_limits<uint32_t>::max)())
      return (std::numeric_limits<uint64_t>::max)();
  }
  for (int I = 1; I < 3; ++I) {
    if ((std::numeric_limits<uint64_t>::max)() / GroupCount[0] < GroupCount[I])
      return (std::numeric_limits<uint64_t>::max)();
    GroupCount[0] *= GroupCount[I];
  }
  // Multiplication of the group sizes must never overflow uint64_t
  // for any existing device.
  uint64_t LocalWorkSize =
      uint64_t(GroupSizes[0]) * GroupSizes[1] * GroupSizes[2];
  uint64_t ThreadsPerWG = ((LocalWorkSize + SIMDWidth - 1) / SIMDWidth);

  // Check that the total number of threads fits uint64_t.
  if ((std::numeric_limits<uint64_t>::max)() / GroupCount[0] < ThreadsPerWG)
    return (std::numeric_limits<uint64_t>::max)();

  return GroupCount[0] * ThreadsPerWG;
}

Error L0KernelTy::decideLoopKernelGroupArguments(L0DeviceTy &Device,
                                                 uint32_t ThreadLimit,
                                                 uint32_t *GroupSizes,
                                                 L0LaunchEnvTy &KEnv) const {

  const auto DeviceId = Device.getDeviceId();
  const auto &Options = LevelZeroPluginTy::getOptions();
  const auto &KernelPR = getProperties();
  uint32_t MaxGroupSize = Device.getMaxGroupSize();
  TgtNDRangeDescTy *LoopLevels = KEnv.LoopDesc;
  auto &GroupCounts = KEnv.GroupCounts;

  bool MaxGroupSizeForced = false;
  if (ThreadLimit > 0) {
    MaxGroupSizeForced = true;
    MaxGroupSize = ThreadLimit;
  }

  uint32_t GRPCounts[3] = {1, 1, 1};
  uint32_t GRPSizes[3] = {MaxGroupSize, 1, 1};
  TgtLoopDescTy *Levels = LoopLevels->Levels;
  int32_t DistributeDim = LoopLevels->DistributeDim;
  assert(DistributeDim >= 0 && DistributeDim <= 2 &&
         "Invalid distribute dimension.");
  int32_t NumLoops = LoopLevels->NumLoops;
  assert((NumLoops > 0 && NumLoops <= 3) &&
         "Invalid loop nest description for ND partitioning");

  // Compute global widths for X/Y/Z dimensions.
  size_t TripCounts[3] = {1, 1, 1};

  for (int32_t I = 0; I < NumLoops; I++) {
    assert(Levels[I].Stride > 0 && "Invalid loop stride for ND partitioning");
    INFO(OMP_INFOTYPE_PLUGIN_KERNEL, DeviceId,
         "Loop %" PRIu32 ": lower bound = %" PRId64 ", upper bound = %" PRId64
         ", Stride = %" PRId64 "\n",
         I, Levels[I].Lb, Levels[I].Ub, Levels[I].Stride);
    if (Levels[I].Ub < Levels[I].Lb)
      TripCounts[I] = 0;
    else
      TripCounts[I] =
          (Levels[I].Ub - Levels[I].Lb + Levels[I].Stride) / Levels[I].Stride;
  }

  // Check if any of the loop has zero iterations.
  if (TripCounts[0] == 0 || TripCounts[1] == 0 || TripCounts[2] == 0) {
    std::fill(GroupSizes, GroupSizes + 3, 1);
    std::fill(GRPCounts, GRPCounts + 3, 1);
    if (DistributeDim > 0 && TripCounts[DistributeDim] != 0) {
      // There is a distribute dimension, and the distribute loop
      // has non-zero iterations, but some inner parallel loop
      // has zero iterations. We still want to split the distribute
      // loop's iterations between many WGs (of size 1), but the inner/lower
      // dimensions should be 1x1.
      // Note that this code is currently dead, because we are not
      // hoisting the inner loops' bounds outside of the target regions.
      // The code is here just for completeness.
      size_t DistributeTripCount = TripCounts[DistributeDim];
      if (DistributeTripCount > UINT32_MAX) {
        INFO(OMP_INFOTYPE_PLUGIN_KERNEL, DeviceId,
             "Invalid number of teams %zu due to large loop trip count\n",
             DistributeTripCount);
        return Plugin::success();
      }
      GRPCounts[DistributeDim] = DistributeTripCount;
    }
    KEnv.AllowCooperative = false;
    GroupCounts.groupCountX = GRPCounts[0];
    GroupCounts.groupCountY = GRPCounts[1];
    GroupCounts.groupCountZ = GRPCounts[2];
    return Plugin::success();
  }

  if (!MaxGroupSizeForced) {
    // Use zeKernelSuggestGroupSize to compute group sizes,
    // or fallback to setting dimension 0 width to SIMDWidth.
    // Note that in case of user-specified LWS GRPSizes[0]
    // is already set according to the specified value.
    size_t GlobalSizes[3] = {TripCounts[0], TripCounts[1], TripCounts[2]};
    if (DistributeDim > 0) {
      // There is a distribute dimension.
      GlobalSizes[DistributeDim - 1] *= GlobalSizes[DistributeDim];
      GlobalSizes[DistributeDim] = 1;
    }

    {
      if (MaxGroupSize > KernelPR.Width) {
        GRPSizes[0] = KernelPR.Width;
      }
      if (DistributeDim == 0) {
        // If there is a distribute dimension, then we do not use
        // thin HW threads, since we do not know anything about
        // the iteration space of the inner parallel loop regions.
        //
        // If there is no distribute dimension, then try to use thiner
        // HW threads to get more independent HW threads executing
        // the kernel - this may allow more parallelism due to
        // the stalls being distributed across multiple HW threads rather
        // than across SIMD lanes within one HW thread.
        assert(GRPSizes[1] == 1 && GRPSizes[2] == 1 &&
               "Unexpected team sizes for dimensions 1 or/and 2.");
        uint32_t SimdWidth = KernelPR.SIMDWidth;
        uint64_t TotalThreads = Device.getTotalThreads();
        TotalThreads *= Options.ThinThreadsThreshold;

        uint32_t GRPSizePrev = GRPSizes[0];
        uint64_t ThreadsNeeded =
            computeThreadsNeeded(TripCounts, GRPSizes, SimdWidth);
        while (ThreadsNeeded < TotalThreads) {
          GRPSizePrev = GRPSizes[0];
          // Try to half the local work size (if possible) and see
          // how many HW threads the kernel will require with this
          // new local work size.
          // In most implementations the initial GRPSizes[0]
          // will be a power-of-two.
          if (GRPSizes[0] <= 1)
            break;
          GRPSizes[0] >>= 1;
          ThreadsNeeded = computeThreadsNeeded(TripCounts, GRPSizes, SimdWidth);
        }
        GRPSizes[0] = GRPSizePrev;
      }
    }
  }

  for (int32_t I = 0; I < NumLoops; I++) {
    if (I < DistributeDim) {
      GRPCounts[I] = 1;
      continue;
    }
    size_t Trip = TripCounts[I];
    if (GRPSizes[I] >= Trip)
      GRPSizes[I] = Trip;
    size_t Count = (Trip + GRPSizes[I] - 1) / GRPSizes[I];
    if (Count > UINT32_MAX) {
      return Plugin::error(ErrorCode::INVALID_ARGUMENT,
                           "Invalid number of teams %zu due to large loop "
                           "trip count\n",
                           Count);
    }
    GRPCounts[I] = (uint32_t)Count;
  }
  KEnv.AllowCooperative = false;
  GroupCounts.groupCountX = GRPCounts[0];
  GroupCounts.groupCountY = GRPCounts[1];
  GroupCounts.groupCountZ = GRPCounts[2];
  std::copy(GRPSizes, GRPSizes + 3, GroupSizes);

  return Plugin::success();
}

Error L0KernelTy::getGroupsShape(L0DeviceTy &Device, int32_t NumTeams,
                                 int32_t ThreadLimit, uint32_t *GroupSizes,
                                 L0LaunchEnvTy &KEnv) const {

  const auto DeviceId = Device.getDeviceId();
  const auto &KernelPR = getProperties();

  // Read the most recent global thread limit and max teams.
  const auto [NumTeamsICV, ThreadLimitICV] = std::make_tuple(0, 0);

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
    // use thread_limit clause value default
    DP("Max team size is set to %" PRId32 " (thread_limit clause)\n",
       ThreadLimit);
  } else if (ThreadLimitICV > 0) {
    // else use thread-limit-var ICV
    ThreadLimit = ThreadLimitICV;
    DP("Max team size is set to %" PRId32 " (thread-limit-icv)\n", ThreadLimit);
  }

  size_t MaxThreadLimit = Device.getMaxGroupSize();
  // Set correct max group size if the kernel was compiled with explicit SIMD
  if (SIMDWidth == 1) {
    MaxThreadLimit = Device.getNumThreadsPerSubslice();
  }

  if (KernelPR.MaxThreadGroupSize < MaxThreadLimit) {
    MaxThreadLimit = KernelPR.MaxThreadGroupSize;
    DP("Capping maximum team size to %zu due to kernel constraints.\n",
       MaxThreadLimit);
  }

  if (ThreadLimit > static_cast<int32_t>(MaxThreadLimit)) {
    ThreadLimit = MaxThreadLimit;
    DP("Max team size execceds current maximum %zu. Adjusted\n",
       MaxThreadLimit);
  }
  {
    if (NumTeams > 0) {
      DP("Number of teams is set to %" PRId32
         "(num_teams clause or no teams construct)\n",
         NumTeams);
    } else if (NumTeamsICV > 0) {
      // OMP_NUM_TEAMS only matters, if num_teams() clause is absent.
      INFO(OMP_INFOTYPE_PLUGIN_KERNEL, DeviceId,
           "OMP_NUM_TEAMS(%" PRId32 ") is ignored\n", NumTeamsICV);

      NumTeams = NumTeamsICV;
      DP("Max number of teams is set to %" PRId32 " (OMP_NUM_TEAMS)\n",
         NumTeams);
    }

    decideKernelGroupArguments(Device, (uint32_t)NumTeams,
                               (uint32_t)ThreadLimit, GroupSizes, KEnv);
    KEnv.AllowCooperative = false;
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
  // Command queue is not used with immediate command list

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
  if (KEnv.AllowCooperative)
    CALL_ZE_RET_ERROR(zeCommandListAppendLaunchCooperativeKernel, CmdList,
                      zeKernel, &KEnv.GroupCounts, Event, NumWaitEvents,
                      WaitEvents);
  else
    CALL_ZE_RET_ERROR(zeCommandListAppendLaunchKernel, CmdList, zeKernel,
                      &KEnv.GroupCounts, Event, NumWaitEvents, WaitEvents);
  KEnv.KernelPR.Mtx.unlock();
  INFO(OMP_INFOTYPE_PLUGIN_KERNEL, DeviceId,
       "Submitted kernel " DPxMOD " to device %s\n", DPxPTR(zeKernel), IdStr);

  if (KEnv.IsAsync) {
    AsyncQueue->WaitEvents.push_back(Event);
    AsyncQueue->KernelEvent = Event;
  } else {
    CALL_ZE_RET_ERROR(zeEventHostSynchronize, Event, UINT64_MAX);
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
  if (KEnv.AllowCooperative)
    CALL_ZE_RET_ERROR(zeCommandListAppendLaunchCooperativeKernel, CmdList,
                      zeKernel, &KEnv.GroupCounts, Event, 0, nullptr);
  else
    CALL_ZE_RET_ERROR(zeCommandListAppendLaunchKernel, CmdList, zeKernel,
                      &KEnv.GroupCounts, Event, 0, nullptr);
  KEnv.KernelPR.Mtx.unlock();
  CALL_ZE_RET_ERROR(zeCommandListClose, CmdList);
  CALL_ZE_RET_ERROR_MTX(zeCommandQueueExecuteCommandLists, l0Device.getMutex(),
                        CmdQueue, 1, &CmdList, nullptr);
  INFO(OMP_INFOTYPE_PLUGIN_KERNEL, DeviceId,
       "Submitted kernel " DPxMOD " to device %s\n", DPxPTR(zeKernel), IdStr);
  CALL_ZE_RET_ERROR(zeCommandQueueSynchronize, CmdQueue, UINT64_MAX);
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
                                  int32_t NumTeams, int32_t ThreadLimit) const {
  uint32_t GroupSizes[3];
  auto DeviceId = l0Device.getDeviceId();
  auto &KernelPR = KEnv.KernelPR;
  // Check if we can reuse previous group parameters
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
  // Set Kernel Indirect flags
  ze_kernel_indirect_access_flags_t Flags = 0;
  Flags |= l0Device.getMemAllocator(TARGET_ALLOC_HOST).getIndirectFlags();
  Flags |= l0Device.getMemAllocator(TARGET_ALLOC_DEVICE).getIndirectFlags();

  if (KEnv.KernelPR.IndirectAccessFlags != Flags) {
    // Combine with common access flags
    const auto FinalFlags = l0Device.getIndirectFlags() | Flags;
    CALL_ZE_RET_ERROR(zeKernelSetIndirectAccess, zeKernel, FinalFlags);
    DP("Setting indirect access flags " DPxMOD "\n", DPxPTR(FinalFlags));
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
  int32_t NumTeams = KernelArgs.NumTeams[0];
  int32_t ThreadLimit = KernelArgs.ThreadLimit[0];
  if (NumTeams < 0)
    NumTeams = 0;
  if (ThreadLimit < 0)
    ThreadLimit = 0;
  INFO(OMP_INFOTYPE_PLUGIN_KERNEL, DeviceId, "Launching kernel " DPxMOD "...\n",
       DPxPTR(zeKernel));

  auto &Plugin = l0Device.getPlugin();
  auto *IdStr = l0Device.getZeIdCStr();
  auto &Options = LevelZeroPluginTy::getOptions();
  bool IsAsync = AsyncInfo && l0Device.asyncEnabled();
  if (IsAsync && !AsyncInfo->Queue) {
    AsyncInfo->Queue = reinterpret_cast<void *>(Plugin.getAsyncQueue());
    if (!AsyncInfo->Queue)
      IsAsync = false; // Couldn't get a queue, revert to sync
  }
  auto *AsyncQueue =
      IsAsync ? static_cast<AsyncQueueTy *>(AsyncInfo->Queue) : NULL;

  // We need to get a non-const version of the Properties structure in order to
  // use its lock and be able to cache the group params and indirect flags
  auto &KernelPR = const_cast<KernelPropertiesTy &>(getProperties());

  L0LaunchEnvTy KEnv(IsAsync, AsyncQueue, KernelPR);

  // Protect from kernel preparation to submission as kernels are shared.
  KernelPR.Mtx.lock();

  if (auto Err = setKernelGroups(l0Device, KEnv, NumTeams, ThreadLimit))
    return Err;

  // Set kernel arguments
  for (int32_t I = 0; I < NumArgs; I++) {
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

  // The next calls should unlock the KernelLock internally
  const bool UseImmCmdList = l0Device.useImmForCompute();
  if (UseImmCmdList)
    return launchKernelWithImmCmdList(l0Device, zeKernel, KEnv,
                                      Options.CommandMode);

  return launchKernelWithCmdQueue(l0Device, zeKernel, KEnv);
}

} // namespace llvm::omp::target::plugin
