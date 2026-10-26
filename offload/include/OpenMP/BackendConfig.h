//===-- OpenMP/BackendConfig.h - Per-backend constants ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Static properties of each offloading backend that libomptarget needs to
// know about but that are not worth querying from the plugin at runtime.
//
//===----------------------------------------------------------------------===//

#ifndef OMPTARGET_BACKENDCONFIG_H
#define OMPTARGET_BACKENDCONFIG_H

#include "OffloadAPI.h"

namespace llvm::omp::target {

/// Base class for backend specific configurations.
struct BackendConfig {
  /// Whether kernels can only be launched with the latest kernel arguments ABI
  /// version (OMP_KERNEL_ARG_VERSION).
  bool RequiresLatestKernelArgVersion = false;

  /// Get the configuration of \p Backend.
  static const BackendConfig &get(ol_platform_backend_t Backend);
};

} // namespace llvm::omp::target

#endif // OMPTARGET_BACKENDCONFIG_H
