//===-- OpenMP/BackendConfig/AMDGPUConfig.h - AMDGPU constants --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef OMPTARGET_BACKENDCONFIG_AMDGPUCONFIG_H
#define OMPTARGET_BACKENDCONFIG_AMDGPUCONFIG_H

#include "OpenMP/BackendConfig.h"

namespace llvm::omp::target {

/// Constants of the AMDGPU backend.
struct AMDGPUConfig final : BackendConfig {
  AMDGPUConfig() { DefaultGenericSPMDTeamsPerCU = 6; }

  interop_spec_t
  selectInteropPreference(int32_t InteropType, int32_t NumPrefers,
                          interop_spec_t *Prefers) const override {
    // TODO: update once targetsync is supported.
    if (InteropType == kmp_interop_type_targetsync)
      return NoInteropSpec;
    return interop_spec_t{/*fr_id=*/omp_ifr_hsa,
                          /*attrs=*/{/*inorder=*/true, /*reserved=*/0},
                          /*impl_attrs=*/0};
  }
};

} // namespace llvm::omp::target

#endif // OMPTARGET_BACKENDCONFIG_AMDGPUCONFIG_H
