//===-- OpenMP/BackendConfig/Host.h - Host constants --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef OMPTARGET_BACKENDCONFIG_HOST_H
#define OMPTARGET_BACKENDCONFIG_HOST_H

#include "OpenMP/BackendConfig.h"

namespace llvm::omp::target {

/// Constants of the host backend.
struct HostConfig final : BackendConfig {
  HostConfig() { RequiresLatestKernelArgVersion = true; }
};

} // namespace llvm::omp::target

#endif // OMPTARGET_BACKENDCONFIG_HOST_H
