//===- Logging_oslog.cpp - os_log logging backend ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Runtime support for the os_log logging backend declared in
// orc-rt-c/Logging.h.
//
// The log records themselves are emitted at the call site by os_log_with_type
// (see the ORC_RT_LOG_* macros). The only runtime support needed here is a
// per-category os_log_t handle, which orc_rt_log_osLogHandle reads inline; this
// file homes the cache and provides the cold path that fills it.
//
//===----------------------------------------------------------------------===//

#include "orc-rt-c/Logging.h"

#include <os/log.h>

// Cache for the inline accessor (declared in Logging.h). Zero-initialized, so
// no static constructor runs; slots are filled lazily by the cold path below.
os_log_t orc_rt_log_OSLogHandles[orc_rt_log_Category_Count];

extern "C" os_log_t
orc_rt_log_osLogHandleSlow(orc_rt_log_Category Category) noexcept {
  // Category is already range-checked by the inline caller.
  //
  // ORC rt logs to the "org.llvm.orc-rt" subsystem.
  //
  // Combined with the per-category name it lets records be filtered in Console
  // or `log` (e.g. `log stream --predicate 'subsystem == "org.llvm.orc-rt"'`).
  //
  // Note: os_log_create deduplicates by (subsystem, category), so if two
  // threads race here they get the same handle back; storing either one is
  // correct and no lock is needed. The store publishes it for future warm-path
  // loads.
  os_log_t Handle =
      os_log_create("org.llvm.orc-rt", orc_rt_log_Category_getName(Category));
  __atomic_store_n(&orc_rt_log_OSLogHandles[Category], Handle,
                   __ATOMIC_RELEASE);
  return Handle;
}
