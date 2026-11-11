//===-- OpenMP/BackendConfig/CUDAConfig.h - CUDA constants ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef OMPTARGET_BACKENDCONFIG_CUDACONFIG_H
#define OMPTARGET_BACKENDCONFIG_CUDACONFIG_H

#include "OpenMP/BackendConfig.h"

namespace llvm::omp::target {

/// Constants of the CUDA backend.
struct CUDAConfig final : BackendConfig {
  interop_spec_t
  selectInteropPreference(int32_t InteropType, int32_t NumPrefers,
                          interop_spec_t *Prefers) const override {
    return interop_spec_t{/*fr_id=*/omp_ifr_cuda,
                          /*attrs=*/{/*inorder=*/true, /*reserved=*/0},
                          /*impl_attrs=*/0};
  }
};

} // namespace llvm::omp::target

#endif // OMPTARGET_BACKENDCONFIG_CUDACONFIG_H
