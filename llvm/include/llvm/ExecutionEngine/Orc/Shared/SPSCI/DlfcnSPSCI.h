//===----- DlfcnSPSCI.h - SPS CI for dlfcn-style calls ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// SPS controller-interface descriptors for the runtime's dlfcn-style calls
// (dlopen / dlupdate / dlclose). See CallSPSCI.h for the descriptor scheme.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_EXECUTIONENGINE_ORC_SHARED_SPSCI_DLFCNSPSCI_H
#define LLVM_EXECUTIONENGINE_ORC_SHARED_SPSCI_DLFCNSPSCI_H

#include "llvm/ExecutionEngine/Orc/Shared/ExecutorAddress.h"
#include "llvm/ExecutionEngine/Orc/Shared/SimplePackedSerialization.h"
#include "llvm/ExecutionEngine/Orc/Shared/SymbolNameSpec.h"

#include <cstdint>

namespace llvm::orc::rt::sps_ci {

/// Open a JITDylib by name with the given dlopen-style mode flags; returns its
/// dso handle.
struct DlfcnOpen {
  static constexpr SymbolNameSpec Name =
      SymbolNameSpec::c("__orc_rt_jit_dlopen_wrapper");
  using SPSSig = shared::SPSExecutorAddr(shared::SPSString, int32_t);
};

/// Re-run initializers for an already-open dso handle; returns nonzero on
/// failure.
struct DlfcnUpdate {
  static constexpr SymbolNameSpec Name =
      SymbolNameSpec::c("__orc_rt_jit_dlupdate_wrapper");
  using SPSSig = int32_t(shared::SPSExecutorAddr);
};

/// Close the given dso handle, running its deinitializers; returns nonzero on
/// failure.
struct DlfcnClose {
  static constexpr SymbolNameSpec Name =
      SymbolNameSpec::c("__orc_rt_jit_dlclose_wrapper");
  using SPSSig = int32_t(shared::SPSExecutorAddr);
};

} // namespace llvm::orc::rt::sps_ci

#endif // LLVM_EXECUTIONENGINE_ORC_SHARED_SPSCI_DLFCNSPSCI_H
