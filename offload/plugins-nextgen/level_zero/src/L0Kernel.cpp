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

ze_group_size_t L0KernelTy::createKernelGroups(L0DeviceTy &l0Device,
                                               L0LaunchEnvTy &KEnv,
                                               uint32_t NumThreads[3],
                                               uint32_t NumBlocks[3]) const {
  assert(NumThreads[0] > 0 && NumThreads[1] > 0 && NumThreads[2] > 0 &&
         "Pre-computed ThreadLimit values must be non-zero");
  assert(NumBlocks[0] > 0 && NumBlocks[1] > 0 && NumBlocks[2] > 0 &&
         "Pre-computed NumTeams values must be non-zero");

  KEnv.GroupCounts = {NumBlocks[0], NumBlocks[1], NumBlocks[2]};
  // Respect max group size attribute in the kernel.
  uint32_t MaxGroupSize = KEnv.KernelPR.MaxThreadGroupSize;
  KEnv.GroupSizes.groupSizeX = std::min<uint32_t>(MaxGroupSize, NumThreads[0]);
  KEnv.GroupSizes.groupSizeY = std::min<uint32_t>(MaxGroupSize, NumThreads[1]);
  KEnv.GroupSizes.groupSizeZ = std::min<uint32_t>(MaxGroupSize, NumThreads[2]);

  auto DeviceId = l0Device.getDeviceId();
  INFO(OMP_INFOTYPE_PLUGIN_KERNEL, DeviceId,
       "Team sizes = {%" PRIu32 ", %" PRIu32 ", %" PRIu32 "}\n",
       KEnv.GroupSizes.groupSizeX, KEnv.GroupSizes.groupSizeY,
       KEnv.GroupSizes.groupSizeZ);
  INFO(OMP_INFOTYPE_PLUGIN_KERNEL, DeviceId,
       "Number of teams = {%" PRIu32 ", %" PRIu32 ", %" PRIu32 "}\n",
       KEnv.GroupCounts.groupCountX, KEnv.GroupCounts.groupCountY,
       KEnv.GroupCounts.groupCountZ);

  return KEnv.GroupSizes;
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
  assert(AsyncInfo && "AsyncInfo must be provided for L0 kernel launch");

  auto zeKernel = getZeKernel();
  auto DeviceId = l0Device.getDeviceId();
  INFO(OMP_INFOTYPE_PLUGIN_KERNEL, DeviceId, "Launching kernel " DPxMOD "...\n",
       DPxPTR(zeKernel));

  auto *IdStr = l0Device.getZeIdCStr();
  bool IsCooperative = KernelArgs.Flags.Cooperative;

  if (IsCooperative && !l0Device.supportsCooperativeKernels()) {
    return Plugin::error(
        ErrorCode::UNSUPPORTED,
        "cooperative kernel launch is not supported by the device");
  }
  auto QueueOrErr = l0Device.getOrCreateQueue(AsyncInfo);
  if (!QueueOrErr)
    return QueueOrErr.takeError();
  auto *Queue = *QueueOrErr;
  auto &KernelPR = getProperties();

  L0LaunchEnvTy KEnv(KernelPR, KernelArgs, LaunchParams);

  // Protect from kernel preparation to submission as kernels are shared.
  KEnv.Lock.lock();

  createKernelGroups(l0Device, KEnv, NumThreads, NumBlocks);

  // Validate cooperative kernel launch constraints
  if (IsCooperative) {
    uint32_t MaxCooperativeGroupCount = 0;
    CALL_ZE_RET_ERROR(zeKernelSuggestMaxCooperativeGroupCount, zeKernel,
                      &MaxCooperativeGroupCount);

    uint32_t TotalGroupCount = KEnv.GroupCounts.groupCountX *
                               KEnv.GroupCounts.groupCountY *
                               KEnv.GroupCounts.groupCountZ;

    if (TotalGroupCount > MaxCooperativeGroupCount) {
      KernelPR.Mtx.unlock();
      return Plugin::error(
          ErrorCode::INVALID_ARGUMENT,
          "cooperative kernel launch failed: requested %u groups exceeds "
          "maximum %u cooperative groups supported by device",
          TotalGroupCount, MaxCooperativeGroupCount);
    }

    INFO(OMP_INFOTYPE_PLUGIN_KERNEL, DeviceId,
         "Cooperative kernel validated: using %u groups (max: %u)\n",
         TotalGroupCount, MaxCooperativeGroupCount);
  }

  // With pointer-array arguments, zeCommandListAppendLaunchKernelWithArguments
  // folds group-size, per-argument set, and launch into a single call.
  if (LaunchParams.NumArgs != KernelPR.NumKernelArgs)
    return Plugin::error(
        ErrorCode::INVALID_ARGUMENT,
        "Number of arguments (%u) does not match the number of arguments "
        "expected by the kernel (%u)",
        LaunchParams.NumArgs, KernelPR.NumKernelArgs);

  if (auto Err = setIndirectFlags(l0Device, KEnv))
    return Err;

  // The next call should unlock the KernelLock internally.
  if (auto Err = Queue->launchKernel(zeKernel, KEnv))
    return Err;
  INFO(OMP_INFOTYPE_PLUGIN_KERNEL, DeviceId,
       "Submitted kernel " DPxMOD " to device %s\n", DPxPTR(zeKernel), IdStr);

  return Plugin::success();
}

Expected<uint32_t>
L0KernelTy::getMaxCooperativeGroupCount(GenericDeviceTy &GenericDevice,
                                        const uint32_t NumThreads[3],
                                        uint32_t DynBlockMemSize) const {
  ze_result_t Res = zeKernelSetGroupSize(zeKernel, NumThreads[0], NumThreads[1],
                                         NumThreads[2]);
  if (Res != ZE_RESULT_SUCCESS)
    return Plugin::error(ErrorCode::UNSUPPORTED,
                         "failed to set group size for cooperative launch");

  uint32_t MaxCooperativeGroupCount = 0;
  Res = zeKernelSuggestMaxCooperativeGroupCount(zeKernel,
                                                &MaxCooperativeGroupCount);

  if (Res != ZE_RESULT_SUCCESS)
    return Plugin::error(ErrorCode::UNSUPPORTED,
                         "failed to query max cooperative group count");

  return MaxCooperativeGroupCount;
}

} // namespace llvm::omp::target::plugin
