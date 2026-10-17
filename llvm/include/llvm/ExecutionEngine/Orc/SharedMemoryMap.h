//===- SharedMemoryMap.h - Shared-memory map bindings -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A controller-side handle to an executor-side shared-memory mapper.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_EXECUTIONENGINE_ORC_SHAREDMEMORYMAP_H
#define LLVM_EXECUTIONENGINE_ORC_SHAREDMEMORYMAP_H

#include "llvm/ExecutionEngine/Orc/Proxy.h"
#include "llvm/ExecutionEngine/Orc/Shared/TargetProcessControlTypes.h"

#include <cstdint>
#include <string>
#include <utility>

namespace llvm::orc {

/// The resolved controller-side handle to an executor-side shared-memory
/// mapper: the address of the mapper instance, which is passed as the first
/// argument to each call, plus the proxies for its operations.
///
/// This is the shared-memory analogue of SimpleMemoryMapBindings: reserve
/// additionally returns the name of the shared-memory object backing the range
/// (which the controller maps into its own address space), and initialize
/// describes its segments by address and size, since their content is written
/// directly into that shared region rather than shipped inline.
///
/// These are protocol-agnostic: sps::createSharedMemoryMapBindings populates
/// them over the runtime's SPS controller interface, but a client targeting a
/// different protocol -- or a different executor-side implementation of these
/// operations -- can build its own and pass them to the utility that will use
/// them.
struct SharedMemoryMapBindings {
  /// Reserve an address range of the given size; returns its base together with
  /// the name of the shared-memory object backing it, which the controller maps
  /// into its own address space.
  using ReserveProxy = Proxy<Expected<std::pair<ExecutorAddr, std::string>>(
      ExecutorAddr, uint64_t)>;

  /// Apply a finalize request to the reservation with the given base; returns a
  /// key for the initialized allocation.
  using InitializeProxy = Proxy<Expected<ExecutorAddr>(
      ExecutorAddr, ExecutorAddr, tpctypes::SharedMemoryFinalizeRequest)>;

  /// Deinitialize the allocations with the given keys (running their
  /// deallocation actions) without releasing their memory.
  using DeinitializeProxy = Proxy<Error(ExecutorAddr, ArrayRef<ExecutorAddr>)>;

  /// Release the reservations with the given base addresses.
  using ReleaseProxy = Proxy<Error(ExecutorAddr, ArrayRef<ExecutorAddr>)>;

  ExecutorAddr Instance;
  ReserveProxy Reserve;
  InitializeProxy Initialize;
  DeinitializeProxy Deinitialize;
  ReleaseProxy Release;
};

} // namespace llvm::orc

#endif // LLVM_EXECUTIONENGINE_ORC_SHAREDMEMORYMAP_H
