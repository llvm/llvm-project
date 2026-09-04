//===- CacheControl.h - Instruction cache maintenance on Darwin -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Darwin definitions for the operations declared in sys/CacheControl.h, which
// is the header callers should include.
//
//===----------------------------------------------------------------------===//

#ifndef ORC_RT_INTERNAL_SUPPORT_SYS_DARWIN_CACHECONTROL_H
#define ORC_RT_INTERNAL_SUPPORT_SYS_DARWIN_CACHECONTROL_H

#include <cstddef>

extern "C" void sys_icache_invalidate(const void *Addr, size_t Size);

namespace orc_rt::sys {

inline void clear_icache(void *Addr, size_t Size) {
  // libSystem's implementation, which knows the cache geometry of the running
  // CPU. Preferred over __builtin___clear_cache on Darwin.
  sys_icache_invalidate(Addr, Size);
}

} // namespace orc_rt::sys

#endif // ORC_RT_INTERNAL_SUPPORT_SYS_DARWIN_CACHECONTROL_H
