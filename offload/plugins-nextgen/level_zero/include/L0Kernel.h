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

#ifndef OPENMP_LIBOMPTARGET_PLUGINS_NEXTGEN_LEVEL_ZERO_L0KERNEL_H
#define OPENMP_LIBOMPTARGET_PLUGINS_NEXTGEN_LEVEL_ZERO_L0KERNEL_H

#include "AsyncQueue.h"
#include "L0Defs.h"
#include "L0Trace.h"
#include "PluginInterface.h"

namespace llvm::omp::target::plugin {

class L0DeviceTy;
class L0ProgramTy;

/// Loop descriptor.
struct TgtLoopDescTy {
  int64_t Lb = 0;     // The lower bound of the i-th loop.
  int64_t Ub = 0;     // The upper bound of the i-th loop.
  int64_t Stride = 0; // The stride of the i-th loop.

  bool operator==(const TgtLoopDescTy &other) const {
    return Lb == other.Lb && Ub == other.Ub && Stride == other.Stride;
  }
};

struct TgtNDRangeDescTy {
  int32_t NumLoops = 0;      // Number of loops/dimensions.
  int32_t DistributeDim = 0; // Dimensions lower than this one
                             // must end up in one WG.
  TgtLoopDescTy Levels[3];   // Up to 3 loops.

  bool operator==(const TgtNDRangeDescTy &other) const {
    return NumLoops == other.NumLoops && DistributeDim == other.DistributeDim &&
           std::equal(Levels, Levels + 3, other.Levels);
  }
  bool operator!=(const TgtNDRangeDescTy &other) const {
    return !(*this == other);
  }
};

/// Forward declaration.
struct L0LaunchEnvTy;

/// Kernel properties.
struct KernelPropertiesTy {
  uint32_t Width = 0;
  uint32_t SIMDWidth = 0;
  uint32_t MaxThreadGroupSize = 0;

  /// Cached input parameters used in the previous launch.
  int32_t NumTeams = -1;
  int32_t ThreadLimit = -1;

  /// Cached parameters used in the previous launch.
  ze_kernel_indirect_access_flags_t IndirectAccessFlags =
      std::numeric_limits<decltype(IndirectAccessFlags)>::max();
  uint32_t GroupSizes[3] = {0, 0, 0};
  ze_group_count_t GroupCounts{0, 0, 0};

  std::mutex Mtx;

  /// Check if we can reuse group parameters.
  bool reuseGroupParams(const int32_t NumTeamsIn, const int32_t ThreadLimitIn,
                        uint32_t *GroupSizesOut, L0LaunchEnvTy &KEnv) const;

  /// Update cached group parameters.
  void cacheGroupParams(const int32_t NumTeamsIn, const int32_t ThreadLimitIn,
                        const uint32_t *GroupSizesIn, L0LaunchEnvTy &KEnv);
};

struct L0LaunchEnvTy {
  bool IsAsync;
  AsyncQueueTy *AsyncQueue;
  ze_group_count_t GroupCounts = {0, 0, 0};
  KernelPropertiesTy &KernelPR;
  bool HalfNumThreads = false;
  bool IsTeamsNDRange = false;

  L0LaunchEnvTy(bool IsAsync, AsyncQueueTy *AsyncQueue,
                KernelPropertiesTy &KernelPR)
      : IsAsync(IsAsync), AsyncQueue(AsyncQueue), KernelPR(KernelPR) {}
};

class L0KernelTy : public GenericKernelTy {
  // L0 Kernel Handle.
  ze_kernel_handle_t zeKernel;
  // Kernel Properties.
  mutable KernelPropertiesTy Properties;

  void decideKernelGroupArguments(L0DeviceTy &Device, uint32_t NumTeams,
                                  uint32_t ThreadLimit, uint32_t *GroupSizes,
                                  L0LaunchEnvTy &KEnv) const;

  Error buildKernel(L0ProgramTy &Program);
  Error readKernelProperties(L0ProgramTy &Program);

  Error setKernelGroups(L0DeviceTy &l0Device, L0LaunchEnvTy &KEnv,
                        uint32_t NumThreads[3], uint32_t NumBlocks[3]) const;
  Error setIndirectFlags(L0DeviceTy &l0Device, L0LaunchEnvTy &KEnv) const;

public:
  /// Create a L0 kernel with a name and an execution mode.
  L0KernelTy(const char *Name) : GenericKernelTy(Name), zeKernel(nullptr) {}
  ~L0KernelTy() = default;
  L0KernelTy(const L0KernelTy &) = delete;
  L0KernelTy(L0KernelTy &&) = delete;
  L0KernelTy &operator=(const L0KernelTy &) = delete;
  L0KernelTy &operator=(const L0KernelTy &&) = delete;

  KernelPropertiesTy &getProperties() const { return Properties; }

  /// Initialize the L0 kernel.
  Error initImpl(GenericDeviceTy &GenericDevice, DeviceImageTy &Image) override;
  /// Launch the L0 kernel function.
  Error launchImpl(GenericDeviceTy &GenericDevice, uint32_t NumThreads[3],
                   uint32_t NumBlocks[3], KernelArgsTy &KernelArgs,
                   KernelLaunchParamsTy LaunchParams,
                   AsyncInfoWrapperTy &AsyncInfoWrapper) const override;
  Error deinit() {
    CALL_ZE_RET_ERROR(zeKernelDestroy, zeKernel);
    return Plugin::success();
  }

  Expected<uint64_t> maxGroupSize(GenericDeviceTy &GenericDevice,
                                  uint64_t DynamicMemSize) const override {
    return Plugin::error(ErrorCode::UNIMPLEMENTED,
                         "maxGroupSize not implemented yet");
  }

  ze_kernel_handle_t getZeKernel() const { return zeKernel; }

  Error getGroupsShape(L0DeviceTy &Device, int32_t NumTeams,
                       int32_t ThreadLimit, uint32_t *GroupSizes,
                       L0LaunchEnvTy &KEnv) const;
};

} // namespace llvm::omp::target::plugin

#endif // OPENMP_LIBOMPTARGET_PLUGINS_NEXTGEN_LEVEL_ZERO_L0KERNEL_H
