//===-- EJitCodePool.cpp - EmbeddedJIT SRE machine-code memory pool -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ExecutionEngine/EJIT/EJitCodePool.h"
#include "llvm/Support/MathExtras.h"

using namespace llvm;
using namespace llvm::ejit;

namespace {

inline size_t alignUp(size_t V, size_t A) { return (V + (A - 1)) & ~(A - 1); }

inline uintptr_t alignDownAddr(uintptr_t V, size_t A) {
  return V & ~(static_cast<uintptr_t>(A) - 1);
}

inline uintptr_t addr(const void *P) {
  return reinterpret_cast<uintptr_t>(P);
}

} // namespace

EJitCodePoolManager::EJitCodePoolManager(Options Opts, RawAllocFn Alloc,
                                         SealFn Seal, SplitFn Split)
    : Opts_(Opts), Alloc_(std::move(Alloc)), Seal_(std::move(Seal)),
      Split_(std::move(Split)) {
  // minCodeAlign and poolAlign must be powers of two for the masking math.
  if (Opts_.minCodeAlign == 0 || !isPowerOf2_64(Opts_.minCodeAlign))
    Opts_.minCodeAlign = 64;
  if (Opts_.poolAlign == 0 || !isPowerOf2_64(Opts_.poolAlign))
    Opts_.poolAlign = static_cast<size_t>(2) * 1024 * 1024;
  if (Opts_.fourKSeal) {
    // 4K seal granularity must be a power of two.
    if (Opts_.sealPageSize == 0 || !isPowerOf2_64(Opts_.sealPageSize))
      Opts_.sealPageSize = 4096;
    // The pool size must be a whole multiple of the large-page / split
    // granularity (poolAlign); round up if a configured size is not.
    Opts_.poolSize = alignUp(Opts_.poolSize, Opts_.poolAlign);
    if (Opts_.poolSize == 0)
      Opts_.poolSize = Opts_.poolAlign;
  }
}

EJitCodePoolManager::~EJitCodePoolManager() = default;

bool EJitCodePoolManager::poolHasRoomLocked(const CodePool &P, size_t Size,
                                            size_t Align) const {
  size_t Off = alignUp(P.used, Align);
  // Guard against overflow before comparing against the pool size.
  if (Off < P.used)
    return false;
  return Off <= P.size && (P.size - Off) >= Size;
}

Error EJitCodePoolManager::newActivePoolLocked() {
  // Reserve alignment slack so the base can be rounded up to poolAlign while
  // still leaving poolSize usable bytes. In 4K seal mode the platform contract
  // wants a full large-page (poolAlign) of slack; legacy mode needs only
  // poolAlign - 1.
  size_t RawSize = Opts_.fourKSeal ? (Opts_.poolSize + Opts_.poolAlign)
                                   : (Opts_.poolSize + Opts_.poolAlign - 1);
  void *Raw = Alloc_ ? Alloc_(RawSize) : nullptr;
  if (!Raw)
    return make_error<StringError>(
        "EJitCodePool: raw allocation of " + Twine(RawSize) + " bytes failed",
        inconvertibleErrorCode());

  auto *RawBytes = static_cast<uint8_t *>(Raw);
  auto *Base = reinterpret_cast<uint8_t *>(
      alignUp(addr(RawBytes), Opts_.poolAlign));

  if (Opts_.fourKSeal) {
    // The 2MiB-aligned usable window must stay inside the raw allocation.
    if (addr(Base) + Opts_.poolSize > addr(RawBytes) + RawSize)
      return make_error<StringError>(
          "EJitCodePool: aligned pool window exceeds raw allocation",
          inconvertibleErrorCode());
    // Split the 2MiB-aligned region into 4K mappings before any enable_ex. One
    // split per pool; per-page enable_ex happens later at seal time.
    unsigned Rc = Split_ ? Split_(Base, Opts_.poolSize) : 1;
    if (Rc != 0)
      return make_error<StringError>(
          "EJitCodePool: split_2m_to_4k failed (rc=" + Twine(Rc) +
              ") for pool base " + Twine(addr(Base)),
          inconvertibleErrorCode());
    ++SplitInvocations_;
  }

  auto P = std::make_unique<CodePool>();
  P->raw = RawBytes;
  P->base = Base;
  P->size = Opts_.poolSize;
  P->used = 0;
  P->executable = false;
  Active_ = P.get();
  Pools_.push_back(std::move(P));
  return Error::success();
}

Error EJitCodePoolManager::sealPoolLocked(CodePool &P) {
  if (P.executable)
    return Error::success(); // already sealed — never re-invoke enable_ex
  unsigned Rc = Seal_ ? Seal_(P.base) : 1;
  if (Rc != 0)
    return make_error<StringError>(
        "EJitCodePool: enable_ex failed (rc=" + Twine(Rc) + ") for pool base " +
            Twine(addr(P.base)),
        inconvertibleErrorCode());
  P.executable = true;
  ++SealInvocations_;
  // Per platform guidance, enable_ex performs its own permission/cache
  // synchronization, so we deliberately do NOT call __builtin___clear_cache.
  return Error::success();
}

Expected<void *> EJitCodePoolManager::allocateCode(size_t Size, size_t Align) {
  if (Size == 0)
    return make_error<StringError>("EJitCodePool: zero-size allocation",
                                   inconvertibleErrorCode());

  size_t EffAlign = std::max(Align, Opts_.minCodeAlign);
  if (!isPowerOf2_64(EffAlign))
    EffAlign = alignUp(EffAlign, Opts_.minCodeAlign);
  // In 4K seal mode every allocation must start on a fresh seal page so a later
  // allocation never lands on an already-RX page.
  if (Opts_.fourKSeal)
    EffAlign = std::max(EffAlign, Opts_.sealPageSize);

  // A single allocation can never span pools; reject oversize requests cleanly.
  if (Size > Opts_.poolSize)
    return make_error<StringError>(
        "EJitCodePool: request of " + Twine(Size) +
            " bytes exceeds pool size " + Twine(Opts_.poolSize),
        inconvertibleErrorCode());

#ifndef EJIT_FREESTANDING
  std::lock_guard<std::mutex> Lock(Mutex_);
#endif

  if (Opts_.fourKSeal) {
    // 4K seal mode: never whole-seal a pool on rollover. Individual pages are
    // sealed at finalize; when the active pool runs out of room we simply move
    // to a fresh pool (the old pool's already-sealed pages stay RX, any unused
    // tail is just abandoned RW memory).
    if (!Active_ || !poolHasRoomLocked(*Active_, Size, EffAlign)) {
      if (auto Err = newActivePoolLocked())
        return std::move(Err);
    }
    size_t Off = alignUp(Active_->used, EffAlign);
    uint8_t *Ptr = Active_->base + Off;
    // Round the bump cursor up to a whole seal page so the next allocation
    // starts on a page this one does not share.
    Active_->used = alignUp(Off + Size, Opts_.sealPageSize);
    return static_cast<void *>(Ptr);
  }

  // Legacy whole-pool seal mode. Decide whether we can use the active pool.
  if (!Active_ || Active_->executable ||
      !poolHasRoomLocked(*Active_, Size, EffAlign)) {
    // Case 3: an unsealed-but-full active pool is sealed before rolling over so
    // it is frozen and never written again. (Safe under EmbeddedJIT's
    // synchronous compilation: the previous module in this pool has already
    // finalized by the time a new allocation arrives.)
    if (Active_ && !Active_->executable) {
      if (auto Err = sealPoolLocked(*Active_))
        return std::move(Err);
    }
    if (auto Err = newActivePoolLocked())
      return std::move(Err);
  }

  size_t Off = alignUp(Active_->used, EffAlign);
  uint8_t *Ptr = Active_->base + Off;
  Active_->used = Off + Size;
  return static_cast<void *>(Ptr);
}

CodePool *EJitCodePoolManager::findPoolLocked(const void *Ptr) {
  uintptr_t A = addr(Ptr);
  for (auto &P : Pools_) {
    uintptr_t B = addr(P->base);
    if (A >= B && A < B + P->size)
      return P.get();
  }
  return nullptr;
}

Error EJitCodePoolManager::sealPoolContaining(const void *Ptr) {
#ifndef EJIT_FREESTANDING
  std::lock_guard<std::mutex> Lock(Mutex_);
#endif
  // Whole-pool sealing is meaningless in 4K page-seal mode: a bare pointer
  // carries no size, so we cannot know which 4K pages to flip. Callers must use
  // sealCodeRange(start, size) instead. Returning an Error (rather than sealing
  // the whole pool) prevents the easy misuse of marking a 2MiB pool executable
  // off a single address.
  if (Opts_.fourKSeal) {
    return make_error<StringError>(
        "EJitCodePool: sealPoolContaining is not supported in 4K page-seal "
        "mode; use sealCodeRange to seal the written code range",
        inconvertibleErrorCode());
  }
  CodePool *P = findPoolLocked(Ptr);
  if (!P)
    return make_error<StringError>(
        "EJitCodePool: address " + Twine(addr(Ptr)) +
            " is not owned by any code pool",
        inconvertibleErrorCode());
  return sealPoolLocked(*P);
}

Error EJitCodePoolManager::sealAllWritablePools() {
#ifndef EJIT_FREESTANDING
  std::lock_guard<std::mutex> Lock(Mutex_);
#endif
  // Whole-pool sealing is not supported in 4K page-seal mode: in that mode a
  // pool intentionally stays partially writable (only the 4K pages of finalized
  // code are RX), so "seal every writable pool" has no correct meaning. Return
  // an Error rather than silently no-op so a caller cannot mistakenly believe
  // all pools are now fully sealed. Use sealCodeRange per finalized allocation.
  if (Opts_.fourKSeal)
    return make_error<StringError>(
        "EJitCodePool: sealAllWritablePools (whole-pool sealing) is not "
        "supported in 4K page-seal mode; use sealCodeRange",
        inconvertibleErrorCode());
  Error Err = Error::success();
  for (auto &P : Pools_)
    if (!P->executable)
      Err = joinErrors(std::move(Err), sealPoolLocked(*P));
  return Err;
}

Error EJitCodePoolManager::sealCodeRange(const void *Start, size_t Size) {
#ifndef EJIT_FREESTANDING
  std::lock_guard<std::mutex> Lock(Mutex_);
#endif
  if (Size == 0)
    return Error::success();
  CodePool *P = findPoolLocked(Start);
  if (!P)
    return make_error<StringError>(
        "EJitCodePool: address " + Twine(addr(Start)) +
            " is not owned by any code pool",
        inconvertibleErrorCode());

  // Seal every 4KiB page the written code overlaps: page-align the start down
  // and the end up, then enable_ex(1, pageVA) per page.
  size_t Page = Opts_.sealPageSize;
  uintptr_t PageStart = alignDownAddr(addr(Start), Page);
  uintptr_t PageEnd = alignUp(addr(Start) + Size, Page);
  for (uintptr_t VA = PageStart; VA < PageEnd; VA += Page) {
    unsigned Rc = Seal_ ? Seal_(reinterpret_cast<void *>(VA)) : 1;
    if (Rc != 0)
      return make_error<StringError>(
          "EJitCodePool: enable_ex failed (rc=" + Twine(Rc) + ") for page " +
              Twine(VA),
          inconvertibleErrorCode());
    ++SealInvocations_;
  }
  // Per platform guidance, enable_ex performs its own permission/cache
  // synchronization, so we deliberately do NOT call __builtin___clear_cache.
  return Error::success();
}

bool EJitCodePoolManager::contains(const void *Ptr) const {
#ifndef EJIT_FREESTANDING
  std::lock_guard<std::mutex> Lock(Mutex_);
#endif
  uintptr_t A = addr(Ptr);
  for (auto &P : Pools_) {
    uintptr_t B = addr(P->base);
    if (A >= B && A < B + P->size)
      return true;
  }
  return false;
}

EJitCodePoolManager::Stats EJitCodePoolManager::getStats() const {
#ifndef EJIT_FREESTANDING
  std::lock_guard<std::mutex> Lock(Mutex_);
#endif
  Stats S;
  S.poolCount = Pools_.size();
  S.sealInvocations = SealInvocations_;
  S.splitInvocations = SplitInvocations_;
  for (auto &P : Pools_) {
    S.reservedBytes += P->size;
    S.usedBytes += P->used;
    if (P->executable) {
      ++S.sealedCount;
      S.wastedBytes += (P->size - P->used);
    } else {
      ++S.activeCount;
    }
  }
  return S;
}
