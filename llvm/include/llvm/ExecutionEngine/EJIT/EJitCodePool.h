//===-- EJitCodePool.h - EmbeddedJIT SRE machine-code memory pool ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  Dedicated machine-code memory pool for EmbeddedJIT on SRE-style targets.
//
//  Background: the target's execute-permission primitive (enable_ex) can only
//  flip permissions at 2MiB granularity, and once a 2MiB region is made RX it
//  can no longer be written. Wrapping the global mprotect breaks ORC/JITLink's
//  later writes, so instead EmbeddedJIT owns the JIT code memory directly:
//
//    * Each pool is a 2MiB-aligned region carved from a raw SRE allocation.
//    * While JITLink writes machine code the pool stays RW.
//    * Before a JIT function pointer is handed back to the caller, the pool
//      containing it is sealed: enable_ex flips it to RX and it is marked
//      executable. A sealed pool is never written or allocated from again.
//    * New JIT code always lands in a fresh (or current unsealed) pool.
//
//  This class is intentionally free of any SRE header dependency: the raw
//  allocator and the seal (enable_ex) primitive are injected as callbacks so
//  that the manager is fully unit-testable on a host with mocks. The thin SRE
//  platform adapter lives in EJitSrePlatform.h and is only pulled in when
//  EJIT_SRE_CODE_POOL is enabled.
//
//  Memory boundary: the *machine code bytes* always come from the injected raw
//  allocator (SRE on the target), never from malloc/new/business heap. The
//  small pool-descriptor bookkeeping below is ordinary host memory and never
//  holds executable code.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_EXECUTIONENGINE_EJIT_EJITCODEPOOL_H
#define LLVM_EXECUTIONENGINE_EJIT_EJITCODEPOOL_H

#include "llvm/Support/Error.h"
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

#ifndef EJIT_FREESTANDING
#include <mutex>
#endif

namespace llvm {
namespace ejit {

/// A single 2MiB-aligned JIT code pool.
struct CodePool {
  /// Raw pointer returned by the injected allocator (SRE_MemDbgAlloc on the
  /// target). Retained for debugging and possible future reclamation.
  uint8_t *raw = nullptr;
  /// 2MiB-aligned usable base inside [raw, raw + rawSize).
  uint8_t *base = nullptr;
  /// Usable size of the pool (the configured pool size, default 2MiB).
  size_t size = 0;
  /// Bump-allocation offset within [base, base + size).
  size_t used = 0;
  /// false = RW and still accepting allocations; true = sealed RX, frozen.
  bool executable = false;
};

/// Manages a set of 2MiB JIT code pools with bump allocation and RW->RX
/// sealing. Not coupled to LLVM containers for the code bytes themselves; the
/// code storage comes exclusively from the injected raw allocator.
class EJitCodePoolManager {
public:
  /// Allocate at least `bytes` of writable memory. Returns nullptr on failure.
  /// On the target this is backed by SRE_MemDbgAlloc; in tests, by a mock.
  using RawAllocFn = std::function<void *(size_t bytes)>;

  /// Seal the 2MiB region whose base is `base2m` (must be 2MiB aligned),
  /// switching it to RX. Returns 0 on success, non-zero on failure. On the
  /// target this calls enable_ex(1, base2m); in tests, a mock.
  using SealFn = std::function<unsigned(void *base2m)>;

  struct Options {
    /// Usable bytes per pool (EJIT_SRE_CODE_POOL_SIZE). Default 2MiB.
    size_t poolSize = static_cast<size_t>(2) * 1024 * 1024;
    /// Alignment of each pool base — the enable_ex granularity. Default 2MiB.
    size_t poolAlign = static_cast<size_t>(2) * 1024 * 1024;
    /// Minimum alignment applied to every code allocation. Default 64.
    size_t minCodeAlign = 64;
  };

  struct Stats {
    size_t poolCount = 0;       ///< total pools created
    size_t sealedCount = 0;     ///< pools currently sealed (RX)
    size_t activeCount = 0;     ///< pools still RW
    size_t usedBytes = 0;       ///< sum of bump offsets across all pools
    size_t reservedBytes = 0;   ///< sum of pool sizes across all pools
    size_t wastedBytes = 0;     ///< unused tail bytes inside sealed pools
    size_t sealInvocations = 0; ///< number of successful seal (enable_ex) calls
  };

  EJitCodePoolManager(Options Opts, RawAllocFn Alloc, SealFn Seal);
  ~EJitCodePoolManager();

  EJitCodePoolManager(const EJitCodePoolManager &) = delete;
  EJitCodePoolManager &operator=(const EJitCodePoolManager &) = delete;

  /// Bump-allocate `Size` bytes of RW code memory aligned to
  /// max(Align, minCodeAlign). Allocation strategy:
  ///   1. no active pool          -> new 2MiB pool
  ///   2. active pool sealed       -> new 2MiB pool
  ///   3. active pool out of room  -> seal active pool, then new 2MiB pool
  ///   4. otherwise                -> bump-allocate inside the active pool
  /// A request larger than the pool size is a clean error (no silent fallback).
  /// If sealing the full active pool (case 3) fails, returns that Error.
  Expected<void *> allocateCode(size_t Size, size_t Align);

  /// Seal the pool that contains `Ptr` (RW -> RX) if it is not already sealed.
  /// Idempotent: a second call for an address in an already-sealed pool is a
  /// no-op success and does not re-invoke enable_ex. Returns an Error if the
  /// pointer is not owned by any pool, or if enable_ex fails.
  Error sealPoolContaining(const void *Ptr);

  /// Seal every pool that is still writable. Used at shutdown/quiesce points.
  Error sealAllWritablePools();

  /// True if `Ptr` falls inside the usable range of any owned pool.
  bool contains(const void *Ptr) const;

  /// Snapshot of pool statistics (thread-safe).
  Stats getStats() const;

private:
  CodePool *findPoolLocked(const void *Ptr);
  Error newActivePoolLocked();
  Error sealPoolLocked(CodePool &P);
  bool poolHasRoomLocked(const CodePool &P, size_t Size, size_t Align) const;

  Options Opts_;
  RawAllocFn Alloc_;
  SealFn Seal_;

  // Pool descriptors (ordinary host bookkeeping; never holds code bytes).
  std::vector<std::unique_ptr<CodePool>> Pools_;
  CodePool *Active_ = nullptr;
  size_t SealInvocations_ = 0;

#ifndef EJIT_FREESTANDING
  mutable std::mutex Mutex_;
#endif
};

} // namespace ejit
} // namespace llvm

#endif // LLVM_EXECUTIONENGINE_EJIT_EJITCODEPOOL_H
