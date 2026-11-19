//===- AMDGPUDeviceOptions.h - AMDGPU per-device options --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef OFFLOAD_PLUGIN_AMDGPU_AMDGPUDEVICEOPTIONS_H
#define OFFLOAD_PLUGIN_AMDGPU_AMDGPUDEVICEOPTIONS_H

#include "DeviceOptions.h"

#include "llvm/ADT/StringRef.h"

namespace llvm::offload::plugin {

/// Runtime options of an AMDGPU device.
struct AMDGPUDeviceOptionsTy {
  /// Use ROCm 5.7 interface for multiple SDMA engines.
  DeviceOptionTy<bool> UseMultipleSdmaEngines{
      "LIBOMPTARGET_AMDGPU_USE_MULTIPLE_SDMA_ENGINES", false};

  AMDGPUDeviceOptionsTy() = default;

  /// Initialize the options with the defaults for the architecture. Values set
  /// through environment variables override the architecture defaults.
  AMDGPUDeviceOptionsTy(StringRef Arch, bool IsAPU) {
    // Adjust the defaults here for specific architectures.

    // Read the environment variables after the architecture defaults are set so
    // that values set by the user take precedence.
    UseMultipleSdmaEngines.readFromEnv();
  }
};

} // namespace llvm::offload::plugin

#endif // OFFLOAD_PLUGIN_AMDGPU_AMDGPUDEVICEOPTIONS_H
