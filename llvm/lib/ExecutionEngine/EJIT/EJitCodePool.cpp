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

inline uintptr_t addr(const void *P) {
  return reinterpret_cast<uintptr_t>(P);
}

} // namespace

EJitCodePoolManager::EJitCodePoolManager(Options Opts, RawAllocFn Alloc,
                                         SealFn Seal)
    : Opts_(Opts), Alloc_(std::move(Alloc)), Seal_(std::move(Seal)) {
  // minCodeAlign and poolAlign must be powers of two for the masking math.
  if (Opts_.minCodeAlign == 0 || !isPowerOf2_64(Opts_.minCodeAlign))
    Opts_.minCodeAlign = 64;
  if (Opts_.poolAlign == 0 || !isPowerOf2_64(Opts_.poolAlign))
    Opts_.poolAlign = static_cast<size_t>(2) * 1024 * 1024;
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
  // raw size must allow rounding the base up to poolAlign while still leaving
  // poolSize usable bytes: rawSize >= poolSize + poolAlign - 1.
  size_t RawSize = Opts_.poolSize + Opts_.poolAlign - 1;
  void *Raw = Alloc_ ? Alloc_(RawSize) : nullptr;
  if (!Raw)
    return make_error<StringError>(
        "EJitCodePool: raw allocation of " + Twine(RawSize) + " bytes failed",
        inconvertibleErrorCode());

  auto *RawBytes = static_cast<uint8_t *>(Raw);
  auto *Base = reinterpret_cast<uint8_t *>(
      alignUp(addr(RawBytes), Opts_.poolAlign));

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

  // A single allocation can never span pools; reject oversize requests cleanly.
  if (Size > Opts_.poolSize)
    return make_error<StringError>(
        "EJitCodePool: request of " + Twine(Size) +
            " bytes exceeds pool size " + Twine(Opts_.poolSize),
        inconvertibleErrorCode());

#ifndef EJIT_FREESTANDING
  std::lock_guard<std::mutex> Lock(Mutex_);
#endif

  // Decide whether we can use the current active pool.
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
  Error Err = Error::success();
  for (auto &P : Pools_)
    if (!P->executable)
      Err = joinErrors(std::move(Err), sealPoolLocked(*P));
  return Err;
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
