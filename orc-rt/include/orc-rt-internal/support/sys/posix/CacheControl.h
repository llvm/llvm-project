//===- CacheControl.h - Instruction cache maintenance, generic --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Generic definitions for the operations declared in sys/CacheControl.h, which
// is the header callers should include.
//
//===----------------------------------------------------------------------===//

#ifndef ORC_RT_INTERNAL_SUPPORT_SYS_POSIX_CACHECONTROL_H
#define ORC_RT_INTERNAL_SUPPORT_SYS_POSIX_CACHECONTROL_H

#include <cstddef>

namespace orc_rt::sys {

inline void clear_icache(void *Addr, size_t Size) {
  // A builtin rather than a declaration of __clear_cache, so that the compiler
  // can emit the cache-maintenance instructions inline where the target has
  // them, instead of an opaque call.
  char *Start = static_cast<char *>(Addr);
  __builtin___clear_cache(Start, Start + Size);
}

} // namespace orc_rt::sys

#endif // ORC_RT_INTERNAL_SUPPORT_SYS_POSIX_CACHECONTROL_H
