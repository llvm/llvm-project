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

namespace lldb_private {

enum class WindowsLog : Log::MaskType {
  Breakpoints = Log::ChannelFlag<0>, // Log breakpoint operations
  Event = Log::ChannelFlag<1>,       // Low level debug events
  Exception = Log::ChannelFlag<2>,   // Log exceptions
  Memory = Log::ChannelFlag<3>,      // Log memory reads/writes calls
  Process = Log::ChannelFlag<4>,     // Log process operations
  Registers = Log::ChannelFlag<5>,   // Log register operations
  Step = Log::ChannelFlag<6>,        // Log step operations
  Thread = Log::ChannelFlag<7>,      // Log thread operations
  LLVM_MARK_AS_BITMASK_ENUM(Thread)
};

#define WINDOWS_LOG_PROCESS ::lldb_private::WindowsLog::Process
#define WINDOWS_LOG_EXCEPTION ::lldb_private::WindowsLog::Exception
#define WINDOWS_LOG_THREAD ::lldb_private::WindowsLog::Thread
#define WINDOWS_LOG_MEMORY ::lldb_private::WindowsLog::Memory
#define WINDOWS_LOG_BREAKPOINTS ::lldb_private::WindowsLog::Breakpoints
#define WINDOWS_LOG_STEP ::lldb_private::WindowsLog::Step
#define WINDOWS_LOG_REGISTERS ::lldb_private::WindowsLog::Registers
#define WINDOWS_LOG_EVENT ::lldb_private::WindowsLog::Event

class ProcessWindowsLog {
public:
  static void Initialize();
  static void Terminate();

  static Log *GetLogIfAny(WindowsLog mask) { return GetLog(mask); }
};

template <> Log::Channel &LogChannelFor<WindowsLog>();
}

#endif // liblldb_ProcessWindowsLog_h_
