//===-- EJitSrePlatform.cpp - SRE platform adapter for the code pool ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  Wires EJitCodePoolManager to the real SRE platform primitives. Compiled
//  only when EJIT_SRE_CODE_POOL is enabled. Provides weak host fallbacks so a
//  host build links and runs without the real SRE symbols (the real platform
//  supplies strong overrides).
//
//===----------------------------------------------------------------------===//

#ifdef EJIT_SRE_CODE_POOL

#include "llvm/ExecutionEngine/EJIT/EJitSrePlatform.h"

#include <cstdlib>

#ifndef EJIT_SRE_CODE_POOL_SIZE
#define EJIT_SRE_CODE_POOL_SIZE (static_cast<unsigned long long>(2) * 1024 * 1024)
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
constexpr unsigned char kSrePtNo = static_cast<unsigned char>(EJIT_SRE_CODE_POOL_PTNO);
constexpr unsigned kSreMid = static_cast<unsigned>(EJIT_SRE_CODE_POOL_MID);
constexpr size_t k2MiB = static_cast<size_t>(2) * 1024 * 1024;
} // namespace

//===----------------------------------------------------------------------===//
// Platform primitives
//
// enable_ex is renamed via an asm label so the generic identifier
// ejit_sre_enable_ex is used in C++ while the linker sees "enable_ex".
//===----------------------------------------------------------------------===//
extern "C" unsigned ejit_sre_enable_ex(unsigned startLevel,
                                       unsigned long long va)
    __asm__("enable_ex");

extern "C" void *SRE_MemDbgAlloc(unsigned int mid, unsigned char ptNo,
                                 unsigned long size, const char *func,
                                 unsigned int line);

//===----------------------------------------------------------------------===//
// Weak host fallbacks
//
// On the real target the platform provides strong definitions that override
// these. On a host without SRE, the weak versions let an EJIT_SRE_CODE_POOL
// build link and run: enable_ex becomes a no-op success (host pages are already
// writable+executable for smoke testing) and allocation uses aligned host
// memory. They are intentionally NOT used by the unit tests, which inject their
// own mocks.
//===----------------------------------------------------------------------===//
__attribute__((weak)) unsigned ejit_sre_enable_ex(unsigned /*startLevel*/,
                                                  unsigned long long /*va*/) {
  return 0; // success no-op on host
}

__attribute__((weak)) void *SRE_MemDbgAlloc(unsigned int /*mid*/,
                                            unsigned char /*ptNo*/,
                                            unsigned long size,
                                            const char * /*func*/,
                                            unsigned int /*line*/) {
  void *P = nullptr;
  // 2MiB alignment so the pool base is already aligned on the host path.
  if (posix_memalign(&P, k2MiB, static_cast<size_t>(size)) != 0)
    return nullptr;
  return P;
}

std::unique_ptr<llvm::ejit::EJitCodePoolManager>
llvm::ejit::makeSreCodePoolManager() {
  EJitCodePoolManager::Options Opts;
  Opts.poolSize = static_cast<size_t>(kSrePoolSize);
  Opts.poolAlign = k2MiB; // enable_ex granularity
  Opts.minCodeAlign = 64;

  auto RawAlloc = [](size_t Bytes) -> void * {
    return SRE_MemDbgAlloc(kSreMid, kSrePtNo,
                           static_cast<unsigned long>(Bytes), __func__,
                           __LINE__);
  };

  auto Seal = [](void *Base2M) -> unsigned {
#ifdef EJIT_SRE_ENABLE_EX
    return ejit_sre_enable_ex(1, reinterpret_cast<unsigned long long>(Base2M));
#else
    // Code-pool routing without permission flips (bring-up / measurement).
    (void)Base2M;
    return 0;
#endif
  };

  return std::make_unique<EJitCodePoolManager>(Opts, RawAlloc, Seal);
}

#endif // EJIT_SRE_CODE_POOL
