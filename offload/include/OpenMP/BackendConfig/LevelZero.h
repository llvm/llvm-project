//===-- OpenMP/BackendConfig/LevelZero.h - Level Zero ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef OMPTARGET_BACKENDCONFIG_LEVELZERO_H
#define OMPTARGET_BACKENDCONFIG_LEVELZERO_H

#include "OpenMP/BackendConfig.h"

namespace llvm::omp::target {

/// Constants of the Level Zero backend.
struct LevelZeroConfig final : BackendConfig {};

} // namespace llvm::omp::target

#endif // OMPTARGET_BACKENDCONFIG_LEVELZERO_H
