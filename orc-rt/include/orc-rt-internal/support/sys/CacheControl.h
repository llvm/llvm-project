//===- CacheControl.h - Instruction cache maintenance -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Host cache management APIs.
//
//===----------------------------------------------------------------------===//

#ifndef ORC_RT_INTERNAL_SUPPORT_SYS_CACHECONTROL_H
#define ORC_RT_INTERNAL_SUPPORT_SYS_CACHECONTROL_H

#include "orc-rt/support/Compiler.h"

#include <cstddef>

namespace orc_rt::sys {

/// Make writes to [Addr, Addr + Size) visible to instruction fetch.
///
/// Call this after writing instructions to memory and before executing them.
/// SimpleNativeMemoryMap does so for any segment made executable, so clients
/// finalizing memory through it do not need to call this themselves.
inline void clear_icache(void *Addr, size_t Size);

} // namespace orc_rt::sys

// Definition of the above.

#if defined(__APPLE__)

// Use libSystem's implementation, which knows the cache geometry of the running
// CPU. Preferred over __builtin___clear_cache on Darwin.
extern "C" void sys_icache_invalidate(const void *Addr, size_t Size);
inline void orc_rt::sys::clear_icache(void *Addr, size_t Size) {
  sys_icache_invalidate(Addr, Size);
}

#elif ORC_RT_HAS_BUILTIN(__builtin___clear_cache) || defined(__GNUC__)

// For systems supporting __builtin___clear_cache, use that.
inline void orc_rt::sys::clear_icache(void *Addr, size_t Size) {
  char *Start = static_cast<char *>(Addr);
  __builtin___clear_cache(Start, Start + Size);
}

#else

#error "No clear_icache implementation for this target"

#endif

#endif // ORC_RT_INTERNAL_SUPPORT_SYS_CACHECONTROL_H
