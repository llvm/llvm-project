//===- Memory.h - System memory operations ----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The host memory operations that SimpleNativeMemoryMap is built on.
//
// Exactly one implementation is compiled into the runtime, chosen by the build:
// see lib/bedrock/sys/posix/Memory.cpp and its siblings. A target that
// cannot provide these has no native memory map.
//
//===----------------------------------------------------------------------===//

#ifndef ORC_RT_INTERNAL_SUPPORT_SYS_MEMORY_H
#define ORC_RT_INTERNAL_SUPPORT_SYS_MEMORY_H

#include "orc-rt/support/Error.h"
#include "orc-rt/support/MemoryFlags.h"

#include <cstddef>

namespace orc_rt::sys {

/// Reserve Size bytes of read/write memory. Returns null if Size is zero.
///
/// The returned range is not yet executable: use protectMemory to change
/// its permissions.
Expected<void *> reserveMemory(size_t Size);

/// Release a range previously returned by reserveMemory.
Error releaseMemory(void *Base, size_t Size);

/// Set the permissions of a range previously returned by reserveMemory.
///
/// If MP includes Exec then the instruction cache is invalidated for the range,
/// so callers do not need to do so themselves.
Error protectMemory(void *Base, size_t Size, MemProt MP);

} // namespace orc_rt::sys

#endif // ORC_RT_INTERNAL_SUPPORT_SYS_MEMORY_H
