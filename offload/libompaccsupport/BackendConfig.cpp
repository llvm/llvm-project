//===-- BackendConfig.cpp - Per-backend constants for libomptarget --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "OpenMP/BackendConfig.h"
#include "OpenMP/BackendConfig/AMDGPU.h"
#include "OpenMP/BackendConfig/CUDA.h"
#include "OpenMP/BackendConfig/Host.h"
#include "OpenMP/BackendConfig/LevelZero.h"

using namespace llvm::omp::target;

const BackendConfig &BackendConfig::get(ol_platform_backend_t Backend) {
  static const BackendConfig Default;
  static const LevelZeroConfig LevelZero;
  static const AMDGPUConfig AMDGPU;
  static const CUDAConfig CUDA;
  static const HostConfig Host;

  switch (Backend) {
  case OL_PLATFORM_BACKEND_LEVEL_ZERO:
    return LevelZero;
  case OL_PLATFORM_BACKEND_AMDGPU:
    return AMDGPU;
  case OL_PLATFORM_BACKEND_HOST:
    return Host;
  case OL_PLATFORM_BACKEND_CUDA:
    return CUDA;
  default:
    return Default;
  }
}
