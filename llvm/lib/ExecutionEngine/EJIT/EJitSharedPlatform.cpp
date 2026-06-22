//===-- EJitSharedPlatform.cpp - Cross-core shared taskpool platform seam -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  Implementation of the cross-core platform seam. This file deliberately uses
//  no std::thread / std::mutex / std::condition_variable: the host backing is a
//  single thread_local scalar used by the deterministic multi-core simulation.
//
//===----------------------------------------------------------------------===//

#include "llvm/ExecutionEngine/EJIT/EJitSharedPlatform.h"

namespace llvm {
namespace ejit {

#ifdef EJIT_SRE_SHARED_TASKPOOL_PLATFORM

// Declared-only platform symbol: a real cross-core build MUST link a strong
// definition (e.g. the SRE core-id primitive). No weak/local fallback is
// provided on purpose, so a missing implementation is a clean link error rather
// than a silently-wrong constant.
extern "C" uint32_t ejit_sre_current_core_id();

uint32_t EJitCoreId::current() { return ejit_sre_current_core_id(); }

#else

namespace {
// Per-thread simulated core id. A host test switches this between calls to
// model cross-core interleaving inside one process and one thread. thread_local
// (not a plain static) so the optional single-real-worker host test keeps its
// own identity without disturbing the producer thread.
thread_local uint32_t gSimulatedCoreId = 0u;
} // namespace

uint32_t EJitCoreId::current() { return gSimulatedCoreId; }

void EJitCoreId::setCurrentForTest(uint32_t coreId) {
  gSimulatedCoreId = coreId;
}

void EJitCoreId::resetForTest() { gSimulatedCoreId = 0u; }

#endif // EJIT_SRE_SHARED_TASKPOOL_PLATFORM

} // namespace ejit
} // namespace llvm
