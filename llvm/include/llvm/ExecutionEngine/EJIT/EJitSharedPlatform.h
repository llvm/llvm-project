//===-- EJitSharedPlatform.h - Cross-core shared taskpool platform seam ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  Platform seam for the cross-core SHARED taskpool (EJIT_SRE_SHARED_TASKPOOL).
//
//  Two host-overridable / platform-injected primitives live here so the rest of
//  the shared taskpool never names a platform symbol directly:
//
//   * EJIT_SHARED_SECTION: the section attribute that places the single shared
//     state blob into inter-core shared memory. The build overrides it via
//     -D'EJIT_SHARED_SECTION=__attribute__((section(".xxxxx")))'. The default
//     is empty so host / unit-test builds use ordinary static storage (a single
//     process already shares one address space, which is exactly what the
//     deterministic multi-core simulation needs).
//
//   * EJitCoreId::current(): the identity of the core running the call. Real
//     cross-core builds (EJIT_SRE_SHARED_TASKPOOL_PLATFORM) bind it to a
//     declared-only platform symbol with NO weak fallback, so a missing
//     platform implementation is a link error rather than a silently-wrong
//     constant. Host builds use a per-thread settable value so a single test
//     process can simulate many cores deterministically, without any real
//     thread.
//
//  This header pulls in no STL and no <atomic>; it is safe in freestanding
//  builds.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_EXECUTIONENGINE_EJIT_EJITSHAREDPLATFORM_H
#define LLVM_EXECUTIONENGINE_EJIT_EJITSHAREDPLATFORM_H

#include <cstdint>

//===----------------------------------------------------------------------===//
// Shared-section placement attribute (overridable by the build).
//===----------------------------------------------------------------------===//
#ifndef EJIT_SHARED_SECTION
#define EJIT_SHARED_SECTION
#endif

namespace llvm {
namespace ejit {

//===----------------------------------------------------------------------===//
// Shared-state ABI identity. Stored as separate fixed-width scalar fields and
// always compared by value (never byte-parsed), so the same definition is valid
// on little- and big-endian targets.
//===----------------------------------------------------------------------===//

/// Magic word stamped into the shared state header. Distinct, non-symmetric
/// value so a partially-zeroed or foreign blob is rejected.
constexpr uint32_t kEJitSharedAbiMagic = 0x456A5370u; // "EjSp"

/// ABI version of the shared state layout. Bump on any field/layout change.
/// v2: EJitCompileRequest carries a generation field and the flat dedup slot
/// stores the owning generation (0 = free) instead of a 1-bit flag.
/// v3: the shared state carries an owner registration fingerprint (peers
/// validate their funcIndex/dimType mapping against the owner before use).
constexpr uint32_t kEJitSharedAbiVersion = 3u;

/// Sentinel "no core" id. Out of any plausible core-id range.
constexpr uint32_t kEJitInvalidCoreId = 0xFFFFFFFFu;

//===----------------------------------------------------------------------===//
// EJitCoreId: injectable current-core identity.
//===----------------------------------------------------------------------===//
class EJitCoreId {
public:
  /// Identity of the core executing this call.
  static uint32_t current();

#ifndef EJIT_SRE_SHARED_TASKPOOL_PLATFORM
  /// Host/unit-test only: set the simulated current-core id for the calling
  /// thread. Lets one process deterministically model many cores with no real
  /// thread. Unavailable in a real platform build (which uses the declared-only
  /// platform symbol).
  static void setCurrentForTest(uint32_t coreId);

  /// Host/unit-test only: restore the default simulated core id (0).
  static void resetForTest();
#endif
};

} // namespace ejit
} // namespace llvm

#endif // LLVM_EXECUTIONENGINE_EJIT_EJITSHAREDPLATFORM_H
