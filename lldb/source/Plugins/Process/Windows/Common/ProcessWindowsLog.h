//===-- ProcessWindowsLog.h -------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_ProcessWindowsLog_h_
#define liblldb_ProcessWindowsLog_h_

#include "lldb/Utility/Log.h"

#define WINDOWS_LOG_PROCESS (1u << 1)     // Log process operations
#define WINDOWS_LOG_EXCEPTION (1u << 1)   // Log exceptions
#define WINDOWS_LOG_THREAD (1u << 2)      // Log thread operations
#define WINDOWS_LOG_MEMORY (1u << 3)      // Log memory reads/writes calls
#define WINDOWS_LOG_BREAKPOINTS (1u << 4) // Log breakpoint operations
#define WINDOWS_LOG_STEP (1u << 5)        // Log step operations
#define WINDOWS_LOG_REGISTERS (1u << 6)   // Log register operations
#define WINDOWS_LOG_EVENT (1u << 7)       // Low level debug events

namespace lldb_private {
class ProcessWindowsLog {
  static Log::Channel g_channel;

public:
  static void Initialize();
  static void Terminate();

  static Log *GetLogIfAny(uint32_t mask) { return g_channel.GetLogIfAny(mask); }
};
}

#endif // liblldb_ProcessWindowsLog_h_
