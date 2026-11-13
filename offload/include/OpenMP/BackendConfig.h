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
#include "OpenMP/InteropAPI.h"

namespace llvm::omp::target {

/// Base class for backend specific configurations.
struct BackendConfig {
  /// Whether kernels can only be launched with the latest kernel arguments ABI
  /// version (OMP_KERNEL_ARG_VERSION).
  bool RequiresLatestKernelArgVersion = false;

  /// Default number of teams (blocks) per compute unit for kernels in
  /// Generic-SPMD mode. If zero, Generic-SPMD kernels use the same default
  /// number of teams as kernels in any other execution mode. Can be overridden
  /// by the user with LIBOMPTARGET_GENERIC_SPMD_TEAMS_PER_CU.
  uint32_t DefaultGenericSPMDTeamsPerCU = 0;

  BackendConfig() = default;
  BackendConfig(const BackendConfig &) = default;
  BackendConfig(BackendConfig &&) = default;
  BackendConfig &operator=(const BackendConfig &) = default;
  BackendConfig &operator=(BackendConfig &&) = default;
  virtual ~BackendConfig() = default;

  /// Return the interop specification supported by the backend for the given
  /// interop type, or NoInteropSpec if the backend supports none.
  /// TODO: the user-provided preference list is not honored yet.
  virtual interop_spec_t
  selectInteropPreference(int32_t InteropType, int32_t NumPrefers,
                          interop_spec_t *Prefers) const {
    return NoInteropSpec;
  }

  /// Get the configuration of \p Backend.
  static const BackendConfig &get(ol_platform_backend_t Backend);
};

} // namespace llvm::omp::target

#endif // OMPTARGET_BACKENDCONFIG_H
