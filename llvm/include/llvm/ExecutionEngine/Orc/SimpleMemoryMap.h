//===- SimpleMemoryMap.h - Memory-map bindings ------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A controller-side handle to an executor-side memory manager implementing the
// simple memory-map interface -- reserve, initialize, deinitialize, release --
// plus the address of the manager instance those calls operate on.
//
// Utilities that drive such a manager share this handle:
// EPCGenericJITLinkMemoryManager and SimpleRemoteMemoryMapper both hold one, so
// the operations are described once rather than per client.
//
// This header is protocol-agnostic, and says nothing about which executor-side
// implementation is on the other end. See SimpleMemoryMapSPS.h to bind a handle
// over the ORC runtime's SPS controller interface; that resolves the runtime's
// SimpleNativeMemoryMap by default, but any implementation exporting the same
// operations can be bound by resolving them under its own names.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_EXECUTIONENGINE_ORC_SIMPLEMEMORYMAP_H
#define LLVM_EXECUTIONENGINE_ORC_SIMPLEMEMORYMAP_H

#include "llvm/ExecutionEngine/Orc/Proxy.h"
#include "llvm/ExecutionEngine/Orc/Shared/TargetProcessControlTypes.h"

#include <cstdint>

namespace llvm::orc {

/// The resolved controller-side handle to an executor-side memory manager: the
/// address of the manager instance, which is passed as the first argument to
/// each call, plus the proxies for its operations.
///
/// These are protocol-agnostic: sps::createSimpleMemoryMapBindings populates
/// them over the runtime's SPS controller interface, but a client targeting a
/// different protocol -- or a different executor-side implementation of these
/// operations -- can build its own and pass them to the utility that will use
/// them.
struct SimpleMemoryMapBindings {
  /// Reserve an address range of the given size; returns its base.
  using ReserveProxy = Proxy<Expected<ExecutorAddr>(ExecutorAddr, uint64_t)>;

  /// Apply a finalize request; returns a key for the initialized allocation.
  using InitializeProxy =
      Proxy<Expected<ExecutorAddr>(ExecutorAddr, tpctypes::FinalizeRequest)>;

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

#endif // LLVM_EXECUTIONENGINE_ORC_SIMPLEMEMORYMAP_H
