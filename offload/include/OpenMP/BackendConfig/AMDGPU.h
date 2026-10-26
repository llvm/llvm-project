//===-- OpenMP/BackendConfig/AMDGPU.h - AMDGPU constants --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef OMPTARGET_BACKENDCONFIG_AMDGPU_H
#define OMPTARGET_BACKENDCONFIG_AMDGPU_H

#include "OpenMP/BackendConfig.h"

namespace llvm::omp::target {

/// Constants of the AMDGPU backend.
struct AMDGPUConfig final : BackendConfig {};

} // namespace llvm::omp::target

#endif // OMPTARGET_BACKENDCONFIG_AMDGPU_H
