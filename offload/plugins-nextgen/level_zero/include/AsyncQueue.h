//===--- Level Zero Target RTL Implementation -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Async Queue wrapper for Level Zero.
//
//===----------------------------------------------------------------------===//

#ifndef OPENMP_LIBOMPTARGET_PLUGINS_NEXTGEN_LEVEL_ZERO_ASYNCQUEUE_H
#define OPENMP_LIBOMPTARGET_PLUGINS_NEXTGEN_LEVEL_ZERO_ASYNCQUEUE_H

#include <tuple>

#include "L0Memory.h"

namespace llvm::omp::target::plugin {

/// Abstract queue that supports asynchronous command submission.
struct AsyncQueueTy {
  /// List of events attached to submitted commands.
  llvm::SmallVector<ze_event_handle_t> WaitEvents;
  /// Pending staging buffer to host copies.
  llvm::SmallVector<std::tuple<void *, void *, size_t>> H2MList;
  /// Pending USM memory copy commands that must wait for kernel completion.
  llvm::SmallVector<std::tuple<const void *, void *, size_t>> USM2MList;
  /// Kernel event not signaled.
  ze_event_handle_t KernelEvent = nullptr;
  /// Clear data.
  void reset() {
    WaitEvents.clear();
    H2MList.clear();
    USM2MList.clear();
    KernelEvent = nullptr;
  }
};

using AsyncQueuePoolTy = ObjPool<AsyncQueueTy>;

} // namespace llvm::omp::target::plugin
#endif // OPENMP_LIBOMPTARGET_PLUGINS_NEXTGEN_LEVEL_ZERO_ASYNCQUEUE_H
