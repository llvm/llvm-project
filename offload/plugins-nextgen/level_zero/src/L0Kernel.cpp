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

Error L0KernelTy::launchImpl(GenericDeviceTy &GenericDevice,
                             uint32_t NumThreads[3], uint32_t NumBlocks[3],
                             KernelArgsTy &KernelArgs,
                             KernelLaunchParamsTy LaunchParams,
                             AsyncInfoWrapperTy &AsyncInfoWrapper) const {

  auto &l0Device = L0DeviceTy::makeL0Device(GenericDevice);
  int32_t RC = runTargetTeamRegion(l0Device, KernelArgs,
                                   std::move(LaunchParams), AsyncInfoWrapper);
  if (RC == OFFLOAD_SUCCESS)
    return Plugin::success();
  return Plugin::error(error::ErrorCode::UNKNOWN,
                       "Error in launch Kernel %s: %d", getName(), RC);
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

  Error Err = buildKernel(Program);
  if (Err)
    return Err;
  Program.addKernel(this);

  return Plugin::success();
}

/// Read global thread limit and max teams from the host runtime. These values
/// are subject to change at any program point, so every kernel execution
/// needs to read the most recent values.
static std::tuple<int32_t, int32_t> readTeamsThreadLimit() {
  int32_t ThreadLimit;
  ThreadLimit = omp_get_teams_thread_limit();
  DP("omp_get_teams_thread_limit() returned %" PRId32 "\n", ThreadLimit);
  // omp_get_thread_limit() would return INT_MAX by default.
  // NOTE: Windows.h defines max() macro, so we have to guard
  //       the call with parentheses.
  ThreadLimit =
      (ThreadLimit > 0 && ThreadLimit != (std::numeric_limits<int32_t>::max)())
          ? ThreadLimit
          : 0;

  int NTeams = omp_get_max_teams();
  DP("omp_get_max_teams() returned %" PRId32 "\n", NTeams);
  // omp_get_max_teams() would return INT_MAX by default.
  // NOTE: Windows.h defines max() macro, so we have to guard
  //       the call with parentheses.
  int32_t NumTeams =
      (NTeams > 0 && NTeams != (std::numeric_limits<int32_t>::max)()) ? NTeams
                                                                      : 0;

  return {NumTeams, ThreadLimit};
}

void L0KernelTy::decideKernelGroupArguments(
    L0DeviceTy &Device, uint32_t NumTeams, uint32_t ThreadLimit,
    TgtNDRangeDescTy *LoopLevels, uint32_t *GroupSizes,
    ze_group_count_t &GroupCounts, bool HalfNumThreads,
    bool IsTeamsNDRange) const {

  const KernelPropertiesTy &KernelPR = getProperties();

  const auto DeviceId = Device.getDeviceId();
  bool MaxGroupSizeForced = false;
  bool MaxGroupCountForced = false;
  uint32_t MaxGroupSize = Device.getMaxGroupSize();
  const auto &Option = LevelZeroPluginTy::getOptions();
  const auto OptSubscRate = Option.SubscriptionRate;

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
    if (HalfNumThreads)
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
          IsTeamsNDRange ? (std::min)(((LoopTripcount + 7) & ~7),
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

int32_t L0KernelTy::decideLoopKernelGroupArguments(
    L0DeviceTy &Device, uint32_t ThreadLimit, TgtNDRangeDescTy *LoopLevels,
    uint32_t *GroupSizes, ze_group_count_t &GroupCounts, bool HalfNumThreads,
    bool &AllowCooperative) const {

  const auto DeviceId = Device.getDeviceId();
  const auto &Options = LevelZeroPluginTy::getOptions();
  const auto &KernelPR = getProperties();
  uint32_t MaxGroupSize = Device.getMaxGroupSize();

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
        return OFFLOAD_FAIL;
      }
      GRPCounts[DistributeDim] = DistributeTripCount;
    }
    AllowCooperative = false;
    GroupCounts.groupCountX = GRPCounts[0];
    GroupCounts.groupCountY = GRPCounts[1];
    GroupCounts.groupCountZ = GRPCounts[2];
    return OFFLOAD_SUCCESS;
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
      INFO(OMP_INFOTYPE_PLUGIN_KERNEL, DeviceId,
           "Invalid number of teams %zu due to large loop trip count\n", Count);
      return OFFLOAD_FAIL;
    }
    GRPCounts[I] = (uint32_t)Count;
  }
  AllowCooperative = false;
  GroupCounts.groupCountX = GRPCounts[0];
  GroupCounts.groupCountY = GRPCounts[1];
  GroupCounts.groupCountZ = GRPCounts[2];
  std::copy(GRPSizes, GRPSizes + 3, GroupSizes);

  return OFFLOAD_SUCCESS;
}

int32_t L0KernelTy::getGroupsShape(L0DeviceTy &SubDevice, int32_t NumTeams,
                                   int32_t ThreadLimit, uint32_t *GroupSizes,
                                   ze_group_count_t &GroupCounts,
                                   void *LoopDesc,
                                   bool &AllowCooperative) const {

  const auto SubId = SubDevice.getDeviceId();
  const auto &KernelPR = getProperties();

  // Detect if we need to reduce available HW threads. We need this adjustment
  // on XeHPG when L0 debug is enabled (ZET_ENABLE_PROGRAM_DEBUGGING=1).
  static std::once_flag OnceFlag;
  static bool ZeDebugEnabled = false;
  std::call_once(OnceFlag, []() {
    const char *EnvVal = std::getenv("ZET_ENABLE_PROGRAM_DEBUGGING");
    if (EnvVal && std::atoi(EnvVal) == 1)
      ZeDebugEnabled = true;
  });

  // Read the most recent global thread limit and max teams.
  auto [NumTeamsICV, ThreadLimitICV] = readTeamsThreadLimit();

  bool IsXeHPG = SubDevice.isDeviceArch(DeviceArchTy::DeviceArch_XeHPG);
  bool HalfNumThreads = ZeDebugEnabled && IsXeHPG;
  uint32_t KernelWidth = KernelPR.Width;
  uint32_t SIMDWidth = KernelPR.SIMDWidth;
  INFO(OMP_INFOTYPE_PLUGIN_KERNEL, SubId,
       "Assumed kernel SIMD width is %" PRIu32 "\n", SIMDWidth);
  INFO(OMP_INFOTYPE_PLUGIN_KERNEL, SubId,
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

  size_t MaxThreadLimit = SubDevice.getMaxGroupSize();
  // Set correct max group size if the kernel was compiled with explicit SIMD
  if (SIMDWidth == 1) {
    MaxThreadLimit = SubDevice.getNumThreadsPerSubslice();
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
      INFO(OMP_INFOTYPE_PLUGIN_KERNEL, SubId,
           "OMP_NUM_TEAMS(%" PRId32 ") is ignored\n", NumTeamsICV);

      NumTeams = NumTeamsICV;
      DP("Max number of teams is set to %" PRId32 " (OMP_NUM_TEAMS)\n",
         NumTeams);
    }

    bool UseLoopTC = LoopDesc;
    decideKernelGroupArguments(
        SubDevice, (uint32_t)NumTeams, (uint32_t)ThreadLimit,
        UseLoopTC ? (TgtNDRangeDescTy *)LoopDesc : nullptr, GroupSizes,
        GroupCounts, HalfNumThreads, false);
    AllowCooperative = false;
  }

  return OFFLOAD_SUCCESS;
}

int32_t L0KernelTy::runTargetTeamRegion(L0DeviceTy &l0Device,
                                        KernelArgsTy &KernelArgs,
                                        KernelLaunchParamsTy LaunchParams,
                                        __tgt_async_info *AsyncInfo) const {
  // Libomptarget can pass negative NumTeams and ThreadLimit now after
  // introducing __tgt_target_kernel. This happens only when we have valid
  // LoopDesc and the region is not a teams region.

  auto zeKernel = getZeKernel();
  auto DeviceId = l0Device.getDeviceId();
  int32_t NumArgs = KernelArgs.NumArgs;
  int32_t NumTeams = KernelArgs.NumTeams[0];
  int32_t ThreadLimit = KernelArgs.ThreadLimit[0];
  void *LoopDesc = nullptr;

  if (NumTeams < 0)
    NumTeams = 0;
  if (ThreadLimit < 0)
    ThreadLimit = 0;
  INFO(OMP_INFOTYPE_PLUGIN_KERNEL, DeviceId,
       "Executing a kernel " DPxMOD "...\n", DPxPTR(zeKernel));

  auto &Plugin = l0Device.getPlugin();
  auto &Device = Plugin.getDeviceFromId(DeviceId);

  auto *IdStr = Device.getZeIdCStr();
  auto &Options = LevelZeroPluginTy::getOptions();
  bool IsAsync = AsyncInfo && Device.asyncEnabled();
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
  // Protect from kernel preparation to submission as kernels are shared.
  std::unique_lock<std::mutex> KernelLock(KernelPR.Mtx);

  // Decide group sizes and counts
  uint32_t GroupSizes[3];
  ze_group_count_t GroupCounts;

  bool AllowCooperative = false;

  // Check if we can reuse previous group parameters
  bool GroupParamsReused = KernelPR.reuseGroupParams(
      static_cast<TgtNDRangeDescTy *>(LoopDesc), NumTeams, ThreadLimit,
      GroupSizes, GroupCounts, AllowCooperative);

  if (!GroupParamsReused) {
    auto RC = getGroupsShape(Device, NumTeams, ThreadLimit, GroupSizes,
                             GroupCounts, LoopDesc, AllowCooperative);

    if (RC != OFFLOAD_SUCCESS) {
      return RC;
    }

    KernelPR.cacheGroupParams(static_cast<TgtNDRangeDescTy *>(LoopDesc),
                              NumTeams, ThreadLimit, GroupSizes, GroupCounts,
                              AllowCooperative);
  }

  INFO(OMP_INFOTYPE_PLUGIN_KERNEL, DeviceId,
       "Team sizes = {%" PRIu32 ", %" PRIu32 ", %" PRIu32 "}\n", GroupSizes[0],
       GroupSizes[1], GroupSizes[2]);
  INFO(OMP_INFOTYPE_PLUGIN_KERNEL, DeviceId,
       "Number of teams = {%" PRIu32 ", %" PRIu32 ", %" PRIu32 "}\n",
       GroupCounts.groupCountX, GroupCounts.groupCountY,
       GroupCounts.groupCountZ);
  for (int32_t I = 0; I < NumArgs; I++) {
    {
      void *Arg = (static_cast<void **>(LaunchParams.Data))[I];
      CALL_ZE_RET_FAIL(zeKernelSetArgumentValue, zeKernel, I, sizeof(Arg),
                       Arg == nullptr ? nullptr : &Arg);
      INFO(OMP_INFOTYPE_PLUGIN_KERNEL, DeviceId,
           "Kernel Pointer argument %" PRId32 " (value: " DPxMOD
           ") was set successfully for device %s.\n",
           I, DPxPTR(Arg), IdStr);
    }
  }

  // Set Kernel Indirect flags
  auto &PrevFlags = KernelPR.IndirectAccessFlags;
  ze_kernel_indirect_access_flags_t Flags = 0;
  Flags |= Device.getMemAllocator(TARGET_ALLOC_HOST).getIndirectFlags();
  Flags |= Device.getMemAllocator(TARGET_ALLOC_DEVICE).getIndirectFlags();

  if (PrevFlags != Flags) {
    // Combine with common access flags
    const auto FinalFlags = Device.getIndirectFlags() | Flags;
    CALL_ZE_RET_FAIL(zeKernelSetIndirectAccess, getZeKernel(), FinalFlags);
    DP("Setting indirect access flags " DPxMOD "\n", DPxPTR(FinalFlags));
    PrevFlags = Flags;
  }

  if (!GroupParamsReused) {
    CALL_ZE_RET_FAIL(zeKernelSetGroupSize, zeKernel, GroupSizes[0],
                     GroupSizes[1], GroupSizes[2]);
  }

  ze_command_list_handle_t CmdList = nullptr;
  ze_command_queue_handle_t CmdQueue = nullptr;
  const bool UseImmCmdList = Device.useImmForCompute();

  if (UseImmCmdList) {
    CmdList = Device.getImmCmdList();
    // Command queue is not used with immediate command list
  } else {
    CmdList = Device.getCmdList();
    CmdQueue = Device.getCmdQueue();
  }

  if (UseImmCmdList) {
    INFO(OMP_INFOTYPE_PLUGIN_KERNEL, DeviceId,
         "Using immediate command list for kernel submission.\n");
    auto Event = Device.getEvent();
    size_t NumWaitEvents = 0;
    ze_event_handle_t *WaitEvents = nullptr;
    if (IsAsync && !AsyncQueue->WaitEvents.empty()) {
      if (Options.CommandMode == CommandModeTy::AsyncOrdered) {
        NumWaitEvents = 1;
        WaitEvents = &AsyncQueue->WaitEvents.back();
      } else {
        NumWaitEvents = AsyncQueue->WaitEvents.size();
        WaitEvents = AsyncQueue->WaitEvents.data();
      }
    }
    INFO(OMP_INFOTYPE_PLUGIN_KERNEL, DeviceId,
         "Kernel depends on %zu data copying events.\n", NumWaitEvents);
    if (AllowCooperative)
      CALL_ZE_RET_FAIL(zeCommandListAppendLaunchCooperativeKernel, CmdList,
                       zeKernel, &GroupCounts, Event, NumWaitEvents,
                       WaitEvents);
    else
      CALL_ZE_RET_FAIL(zeCommandListAppendLaunchKernel, CmdList, zeKernel,
                       &GroupCounts, Event, NumWaitEvents, WaitEvents);
    KernelLock.unlock();
    if (IsAsync) {
      AsyncQueue->WaitEvents.push_back(Event);
      AsyncQueue->KernelEvent = Event;
    } else {
      CALL_ZE_RET_FAIL(zeEventHostSynchronize, Event, UINT64_MAX);
      Device.releaseEvent(Event);
    }
  } else {
    ze_event_handle_t Event = nullptr;
    if (AllowCooperative)
      CALL_ZE_RET_FAIL(zeCommandListAppendLaunchCooperativeKernel, CmdList,
                       zeKernel, &GroupCounts, Event, 0, nullptr);
    else
      CALL_ZE_RET_FAIL(zeCommandListAppendLaunchKernel, CmdList, zeKernel,
                       &GroupCounts, Event, 0, nullptr);
    KernelLock.unlock();
    CALL_ZE_RET_FAIL(zeCommandListClose, CmdList);
    CALL_ZE_RET_FAIL_MTX(zeCommandQueueExecuteCommandLists, Device.getMutex(),
                         CmdQueue, 1, &CmdList, nullptr);
    INFO(OMP_INFOTYPE_PLUGIN_KERNEL, DeviceId,
         "Submitted kernel " DPxMOD " to device %s\n", DPxPTR(zeKernel), IdStr);
    CALL_ZE_RET_FAIL(zeCommandQueueSynchronize, CmdQueue, UINT64_MAX);
    CALL_ZE_RET_FAIL(zeCommandListReset, CmdList);
    if (Event) {
      Device.releaseEvent(Event);
    }
  }

  INFO(OMP_INFOTYPE_PLUGIN_KERNEL, DeviceId,
       "Executed kernel entry " DPxMOD " on device %s\n", DPxPTR(zeKernel),
       IdStr);

  return OFFLOAD_SUCCESS;
}

} // namespace llvm::omp::target::plugin
