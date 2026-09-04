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

// Definition of the above: the default is here, and a system that can do better
// than the compiler builtin gets a header of its own, selected below.
#if defined(__APPLE__)

#include "orc-rt-internal/support/sys/darwin/CacheControl.h"

#elif ORC_RT_HAS_BUILTIN(__builtin___clear_cache) || defined(__GNUC__)

namespace orc_rt::sys {

inline void clear_icache(void *Addr, size_t Size) {
  char *Start = static_cast<char *>(Addr);
  __builtin___clear_cache(Start, Start + Size);
}

} // namespace orc_rt::sys

#else

#error "No clear_icache implementation for this target"

#endif

#endif // ORC_RT_INTERNAL_SUPPORT_SYS_CACHECONTROL_H
