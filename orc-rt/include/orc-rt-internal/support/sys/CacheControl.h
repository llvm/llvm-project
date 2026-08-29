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
// Unlike the rest of sys/, where the build compiles one implementation .cpp per
// system, these operations want to inline: on targets where they lower to
// cache-maintenance instructions rather than a call, a cross-TU call would cost
// more than the operation itself. So the declarations live here and the
// definitions come from a per-system header selected below. The conditionals
// are confined to this file; each implementation header is unconditional.
//
//===----------------------------------------------------------------------===//

#ifndef ORC_RT_INTERNAL_SUPPORT_SYS_CACHECONTROL_H
#define ORC_RT_INTERNAL_SUPPORT_SYS_CACHECONTROL_H

#include <cstddef>

namespace orc_rt::sys {

/// Make writes to [Addr, Addr + Size) visible to instruction fetch.
///
/// Call this after writing instructions to memory and before executing them.
/// SimpleNativeMemoryMap does so for any segment made executable, so clients
/// finalizing memory through it do not need to call this themselves.
inline void clear_icache(void *Addr, size_t Size);

} // namespace orc_rt::sys

// Definition of the above. Selected here rather than by the build system so
// that a reader of this header can see which implementation applies.
#if defined(__APPLE__)
#include "orc-rt-internal/support/sys/darwin/CacheControl.h"
#else
#include "orc-rt-internal/support/sys/posix/CacheControl.h"
#endif

#endif // ORC_RT_INTERNAL_SUPPORT_SYS_CACHECONTROL_H
