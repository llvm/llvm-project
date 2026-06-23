//===-- EJitSrePlatform.cpp - SRE platform adapter for the code pool ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  Wires EJitCodePoolManager to the real SRE platform primitives. Compiled
//  only when EJIT_SRE_CODE_POOL is enabled. The platform symbols (enable_ex,
//  split_2m_to_4k, SRE_MemDbgAlloc) are ONLY declared here — never defined and
//  never given weak fallbacks. The real platform / business link environment
//  must supply their strong definitions; if a symbol is missing it must surface
//  as a link-time error rather than be silently satisfied by a no-op. Host unit
//  tests do not reference makeSreCodePoolManager (they inject mock callbacks
//  into EJitCodePoolManager directly), so this translation unit's external
//  references are never pulled into a host test link.
//
//===----------------------------------------------------------------------===//

#ifdef EJIT_SRE_CODE_POOL

#include "llvm/ExecutionEngine/EJIT/EJitSrePlatform.h"

#ifndef EJIT_SRE_CODE_POOL_SIZE
#define EJIT_SRE_CODE_POOL_SIZE                                                \
  (static_cast<unsigned long long>(2) * 1024 * 1024)
#endif

#ifndef EJIT_SRE_CODE_POOL_PTNO
#define EJIT_SRE_CODE_POOL_PTNO 8
#endif

// Memory module id passed to SRE_MemDbgAlloc. Not architecturally significant
// for the pool; overridable if a deployment needs a specific id.
#ifndef EJIT_SRE_CODE_POOL_MID
#define EJIT_SRE_CODE_POOL_MID 0
#endif

namespace {
constexpr unsigned long long kSrePoolSize = EJIT_SRE_CODE_POOL_SIZE;
constexpr unsigned char kSrePtNo =
    static_cast<unsigned char>(EJIT_SRE_CODE_POOL_PTNO);
constexpr unsigned kSreMid = static_cast<unsigned>(EJIT_SRE_CODE_POOL_MID);
constexpr size_t k2MiB = static_cast<size_t>(2) * 1024 * 1024;
constexpr size_t k4KiB = static_cast<size_t>(4) * 1024;
} // namespace

//===----------------------------------------------------------------------===//
// Platform primitives (declaration only — defined by the platform/business)
//
// enable_ex / split_2m_to_4k are renamed via asm labels so the generic
// identifiers (ejit_sre_enable_ex / ejit_sre_split_2m_to_4k) are used in C++
// while the linker sees the real platform symbol names. These are intentionally
// NOT given weak fallbacks: in static-pack / partial-link / platform-SDK
// scenarios a weak local definition could shadow or collide with the real
// symbol or bind incorrectly. EmbeddedJIT only declares and calls them; the
// platform must provide the strong definitions.
//===----------------------------------------------------------------------===//
extern "C" unsigned
ejit_sre_enable_ex(unsigned startLevel,
                   unsigned long long va) __asm__("enable_ex");

// Split a 2MiB-aligned [va, va + size) window into 4KiB mappings. Must be
// called before any per-page enable_ex on that window. Returns 0 on success.
extern "C" unsigned
ejit_sre_split_2m_to_4k(unsigned long long va,
                        unsigned long long size) __asm__("split_2m_to_4k");

extern "C" void *SRE_MemDbgAlloc(unsigned int mid, unsigned char ptNo,
                                 unsigned long size, const char *func,
                                 unsigned int line);

std::unique_ptr<llvm::ejit::EJitCodePoolManager>
llvm::ejit::makeSreCodePoolManager() {
  EJitCodePoolManager::Options Opts;
  Opts.poolSize = static_cast<size_t>(kSrePoolSize);
  Opts.poolAlign = k2MiB; // large-page / split granularity
  Opts.minCodeAlign = 64;
#ifdef EJIT_CODE_POOL_4K_SEAL
  // Adapt to the platform's 4K execute-permission interface: the 2MiB pool is
  // split into 4K mappings at creation and sealed one 4KiB page at a time.
  Opts.fourKSeal = true;
  Opts.sealPageSize = k4KiB;
#endif

  auto RawAlloc = [](size_t Bytes) -> void * {
    return SRE_MemDbgAlloc(kSreMid, kSrePtNo, static_cast<unsigned long>(Bytes),
                           __func__, __LINE__);
  };

  auto Seal = [](void *Va) -> unsigned {
#ifdef EJIT_SRE_ENABLE_EX
    // In 4K seal mode Va is a single 4KiB page; in legacy mode it is the 2MiB
    // pool base. enable_ex flips the page containing Va to RX either way.
    return ejit_sre_enable_ex(1, reinterpret_cast<unsigned long long>(Va));
#else
    // Code-pool routing without permission flips (bring-up / measurement).
    (void)Va;
    return 0;
#endif
  };

  auto Split = [](void *Base, size_t Size) -> unsigned {
#ifdef EJIT_CODE_POOL_4K_SEAL
    return ejit_sre_split_2m_to_4k(reinterpret_cast<unsigned long long>(Base),
                                   static_cast<unsigned long long>(Size));
#else
    (void)Base;
    (void)Size;
    return 0;
#endif
  };

  return std::make_unique<EJitCodePoolManager>(Opts, RawAlloc, Seal, Split);
}

bool llvm::ejit::prepareSreCodeForCurrentCore(const void *FnPtr) {
#if !defined(EJIT_SRE_ENABLE_EX) || defined(EJIT_CODE_POOL_4K_SEAL)
  (void)FnPtr;
  return false;
#else
  if (!FnPtr)
    return false;
  const auto Address = reinterpret_cast<uintptr_t>(FnPtr);
  const auto PoolBase = Address & ~(static_cast<uintptr_t>(k2MiB) - 1);
  return ejit_sre_enable_ex(1, static_cast<unsigned long long>(PoolBase)) == 0;
#endif
}

#endif // EJIT_SRE_CODE_POOL
