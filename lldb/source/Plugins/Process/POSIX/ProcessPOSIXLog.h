//===-- ProcessPOSIXLog.h -----------------------------------------*- C++
//-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_ProcessPOSIXLog_h_
#define liblldb_ProcessPOSIXLog_h_


#include "lldb/Utility/Log.h"

namespace lldb_private {

enum class POSIXLog : Log::MaskType {
  Breakpoints = Log::ChannelFlag<0>,
  Memory = Log::ChannelFlag<1>,
  Process = Log::ChannelFlag<2>,
  Ptrace = Log::ChannelFlag<3>,
  Registers = Log::ChannelFlag<4>,
  Thread = Log::ChannelFlag<5>,
  Watchpoints = Log::ChannelFlag<6>,
  LLVM_MARK_AS_BITMASK_ENUM(Watchpoints)
};

#define POSIX_LOG_PROCESS ::lldb_private::POSIXLog::Process
#define POSIX_LOG_THREAD ::lldb_private::POSIXLog::Thread
#define POSIX_LOG_MEMORY ::lldb_private::POSIXLog::Memory
#define POSIX_LOG_PTRACE ::lldb_private::POSIXLog::Ptrace
#define POSIX_LOG_REGISTERS ::lldb_private::POSIXLog::Registers
#define POSIX_LOG_BREAKPOINTS ::lldb_private::POSIXLog::Breakpoints
#define POSIX_LOG_WATCHPOINTS ::lldb_private::POSIXLog::Watchpoints

class ProcessPOSIXLog {
public:
  static void Initialize();

  static Log *GetLogIfAllCategoriesSet(POSIXLog mask) { return GetLog(mask); }
};

template <> Log::Channel &LogChannelFor<POSIXLog>();
} // namespace lldb_private

#endif // liblldb_ProcessPOSIXLog_h_
