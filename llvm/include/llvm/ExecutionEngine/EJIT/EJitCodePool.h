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
//  Background: the target's execute-permission primitive (enable_ex) flips
//  permissions on the 4KiB page containing the supplied VA, but the underlying
//  large page is still 2MiB: a 2MiB-aligned region must first be split into 4K
//  mappings via split_2m_to_4k before any per-page enable_ex is legal. Wrapping
//  the global mprotect breaks ORC/JITLink's later writes, so instead EmbeddedJIT
//  owns the JIT code memory directly. There are two sealing modes:
//
//    * Legacy whole-pool seal (fourKSeal = false): each 2MiB-aligned pool is
//      sealed as a unit (one enable_ex on the pool base); a sealed pool is
//      never written or allocated from again.
//    * 4K page seal (fourKSeal = true): the pool is still a 2MiB-aligned region
//      (split into 4K mappings at creation), but only the 4KiB pages that a
//      finalized allocation actually covers are sealed (one enable_ex per
//      page). Each allocation starts on a fresh 4K page and is rounded up to a
//      4K multiple, so subsequent allocations never touch an already-RX page
//      and the rest of the pool stays RW. This is far cheaper than burning a
//      whole 2MiB pool per function.
//
//  In both modes, while JITLink writes machine code the affected memory stays
//  RW; the RW->RX seal happens only after finalize, before a function pointer
//  is handed back to the caller. New JIT code always lands in writable memory.
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
  /// Whole-pool seal flag (legacy mode only): false = RW and still accepting
  /// allocations; true = sealed RX, frozen. In 4K seal mode the pool is sealed
  /// per page and this stays false (the pool keeps serving fresh-page allocs).
  bool executable = false;
};

/// Manages a set of 2MiB-aligned JIT code pools with bump allocation and
/// RW->RX sealing (whole-pool or per-4K-page). Not coupled to LLVM containers
/// for the code bytes themselves; the code storage comes exclusively from the
/// injected raw allocator.
class EJitCodePoolManager {
public:
  /// Allocate at least `bytes` of writable memory. Returns nullptr on failure.
  /// On the target this is backed by SRE_MemDbgAlloc; in tests, by a mock.
  using RawAllocFn = std::function<void *(size_t bytes)>;

  /// Seal one execute-permission unit, switching it to RX. Returns 0 on
  /// success, non-zero on failure. On the target this calls enable_ex(1, va);
  /// in tests, a mock. In legacy whole-pool mode `va` is the 2MiB pool base; in
  /// 4K page-seal mode it is the base VA of a single 4KiB page.
  using SealFn = std::function<unsigned(void *va)>;

  /// Split a 2MiB-aligned region [base, base + size) into 4KiB mappings so the
  /// platform can later flip execute permission per 4K page. Returns 0 on
  /// success, non-zero on failure. On the target this calls
  /// split_2m_to_4k(base, size); in tests, a mock. Only used when
  /// Options::fourKSeal is set.
  using SplitFn = std::function<unsigned(void *base, size_t size)>;

  struct Options {
    /// Usable bytes per pool (EJIT_SRE_CODE_POOL_SIZE). Default 2MiB. In 4K
    /// seal mode this is rounded up to a multiple of poolAlign.
    size_t poolSize = static_cast<size_t>(2) * 1024 * 1024;
    /// Alignment of each pool base — the large-page / split granularity.
    /// Default 2MiB.
    size_t poolAlign = static_cast<size_t>(2) * 1024 * 1024;
    /// Minimum alignment applied to every code allocation. Default 64.
    size_t minCodeAlign = 64;
    /// When true, seal execute permission per 4KiB page (split_2m_to_4k at pool
    /// creation + enable_ex per covered page at finalize) instead of sealing
    /// the whole 2MiB pool. Default false (legacy whole-pool seal).
    bool fourKSeal = false;
    /// Execute-permission seal granularity in 4K mode. Platform constant 4KiB.
    size_t sealPageSize = 4096;
  };

  struct Stats {
    size_t poolCount = 0;       ///< total pools created
    size_t sealedCount = 0;     ///< pools currently sealed (RX)
    size_t activeCount = 0;     ///< pools still RW
    size_t usedBytes = 0;       ///< sum of bump offsets across all pools
    size_t reservedBytes = 0;   ///< sum of pool sizes across all pools
    size_t wastedBytes = 0;     ///< unused tail bytes inside sealed pools
    size_t sealInvocations = 0; ///< number of successful seal (enable_ex) calls
                                ///< (per 4K page in 4K seal mode)
    size_t splitInvocations = 0; ///< number of successful split_2m_to_4k calls
                                 ///< (one per pool in 4K seal mode)
  };

  EJitCodePoolManager(Options Opts, RawAllocFn Alloc, SealFn Seal,
                      SplitFn Split = nullptr);
  ~EJitCodePoolManager();

  EJitCodePoolManager(const EJitCodePoolManager &) = delete;
  EJitCodePoolManager &operator=(const EJitCodePoolManager &) = delete;

  /// Bump-allocate `Size` bytes of RW code memory aligned to
  /// max(Align, minCodeAlign) (and to sealPageSize in 4K seal mode). Allocation
  /// strategy:
  ///   1. no active pool          -> new 2MiB-aligned pool
  ///   2. active pool out of room  -> new 2MiB-aligned pool
  ///   3. (legacy mode) active pool sealed -> new pool; a full active pool is
  ///      sealed before rolling over so it is never written again
  ///   4. otherwise                -> bump-allocate inside the active pool
  /// In 4K seal mode every allocation starts on a fresh 4KiB page and its used
  /// extent is rounded up to a 4K multiple, so a later allocation never lands
  /// on an already-sealed (RX) page; pools are not whole-sealed on rollover.
  /// A request larger than the pool size is a clean error (no silent fallback).
  /// If sealing the full active pool (legacy case 3) fails, returns that Error.
  Expected<void *> allocateCode(size_t Size, size_t Align);

  /// Seal the pool that contains `Ptr` (RW -> RX) if it is not already sealed.
  /// Idempotent: a second call for an address in an already-sealed pool is a
  /// no-op success and does not re-invoke enable_ex. Returns an Error if the
  /// pointer is not owned by any pool, or if enable_ex fails.
  Error sealPoolContaining(const void *Ptr);

  /// Seal every pool that is still writable. Used at shutdown/quiesce points.
  Error sealAllWritablePools();

  /// 4K seal mode: seal the 4KiB pages covering [Start, Start + Size), i.e.
  /// [alignDown(Start, sealPageSize), alignUp(Start + Size, sealPageSize)),
  /// invoking enable_ex(1, pageVA) once per page. Used after a JIT allocation
  /// has been fully written/finalized, before its function pointer is returned.
  /// Returns an Error if `Start` is not owned by any pool or if any page seal
  /// fails (in which case no callable pointer must be handed back).
  Error sealCodeRange(const void *Start, size_t Size);

  /// True if this manager seals execute permission per 4KiB page (rather than
  /// per whole 2MiB pool).
  bool usesPageSeal() const { return Opts_.fourKSeal; }

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
  SplitFn Split_;

  // Pool descriptors (ordinary host bookkeeping; never holds code bytes).
  std::vector<std::unique_ptr<CodePool>> Pools_;
  CodePool *Active_ = nullptr;
  size_t SealInvocations_ = 0;
  size_t SplitInvocations_ = 0;

#ifndef EJIT_FREESTANDING
  mutable std::mutex Mutex_;
#endif
};

} // namespace ejit
} // namespace llvm

#endif // LLVM_EXECUTIONENGINE_EJIT_EJITCODEPOOL_H
