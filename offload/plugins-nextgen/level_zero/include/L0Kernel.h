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

#include "L0Defs.h"
#include "L0Trace.h"
#include "PluginInterface.h"

namespace llvm::omp::target::plugin {

// Forward declarations.
class L0DeviceTy;
class L0ProgramTy;
class L0QueueTy;
struct L0LaunchEnvTy;

/// Kernel properties.
struct KernelPropertiesTy {
  uint32_t Width = 0;
  uint32_t SIMDWidth = 0;
  uint32_t MaxThreadGroupSize = 0;
  uint32_t NumKernelArgs = 0;
  std::unique_ptr<uint32_t[]> ArgSizes;
  ze_kernel_indirect_access_flags_t IndirectAccessFlags =
      std::numeric_limits<decltype(IndirectAccessFlags)>::max();
  std::mutex Mtx;
};

struct L0LaunchEnvTy {
  ze_group_count_t GroupCounts = {0, 0, 0};
  ze_group_size_t GroupSizes = {0, 0, 0};
  KernelPropertiesTy &KernelPR;
  bool HalfNumThreads = false;
  bool IsTeamsNDRange = false;
  bool IsCooperative = false;
  bool IsPtrArg = false;
  void **ArgPtrs = nullptr;
  std::unique_lock<std::mutex> Lock;

  L0LaunchEnvTy(KernelPropertiesTy &KernelPR, KernelArgsTy &KernelArgs,
                KernelLaunchParamsTy LaunchParams)
      : KernelPR(KernelPR), IsCooperative(KernelArgs.Flags.Cooperative),
        IsPtrArg(LaunchParams.Args != nullptr), ArgPtrs(LaunchParams.Args),
        Lock(KernelPR.Mtx, std::defer_lock) {}
};

class L0KernelTy : public GenericKernelTy {
  // L0 Kernel Handle.
  ze_kernel_handle_t zeKernel;
  // Kernel Properties.
  mutable KernelPropertiesTy Properties;

  Error buildKernel(L0ProgramTy &Program);
  Error readKernelProperties(L0ProgramTy &Program);

  ze_group_size_t createKernelGroups(L0DeviceTy &l0Device, L0LaunchEnvTy &KEnv,
                                     uint32_t NumThreads[3],
                                     uint32_t NumBlocks[3]) const;
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
                   uint32_t NumBlocks[3], uint32_t DynBlockMemSize,
                   KernelArgsTy &KernelArgs, KernelLaunchParamsTy LaunchParams,
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

  Expected<uint32_t>
  getMaxCooperativeGroupCount(GenericDeviceTy &GenericDevice,
                              const uint32_t NumThreads[3],
                              uint32_t DynBlockMemSize) const override;

  ze_kernel_handle_t getZeKernel() const { return zeKernel; }
};

} // namespace llvm::omp::target::plugin

#endif // OPENMP_LIBOMPTARGET_PLUGINS_NEXTGEN_LEVEL_ZERO_L0KERNEL_H
