//===- AMDGPUDeviceOptions.h - AMDGPU per-device options --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef OFFLOAD_PLUGINS_NEXTGEN_AMDGPU_AMDGPUDEVICEOPTIONS_H
#define OFFLOAD_PLUGINS_NEXTGEN_AMDGPU_AMDGPUDEVICEOPTIONS_H

#include "DeviceOptions.h"

#include "llvm/ADT/StringRef.h"

namespace llvm::omp::target::plugin {

/// Runtime options of an AMDGPU device.
struct AMDGPUDeviceOptionsTy {
  /// Use ROCm 5.7 interface for multiple SDMA engines.
  DeviceOptionTy<bool> UseMultipleSdmaEngines{
      "LIBOMPTARGET_AMDGPU_USE_MULTIPLE_SDMA_ENGINES", false};

  /// Get the defaults for architectures.
  static AMDGPUDeviceOptionsTy getDeviceDefaults(StringRef Arch, bool IsAPU) {
    AMDGPUDeviceOptionsTy Options;
    // Adjust the defaults here for specific architectures.
    return Options;
  }

  void readFromEnv() { UseMultipleSdmaEngines.readFromEnv(); }
};

} // namespace llvm::omp::target::plugin

#endif // OFFLOAD_PLUGINS_NEXTGEN_AMDGPU_AMDGPUDEVICEOPTIONS_H
