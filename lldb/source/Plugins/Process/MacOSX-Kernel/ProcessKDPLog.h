//===-- ProcessKDPLog.h -----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SOURCE_PLUGINS_PROCESS_MACOSX_KERNEL_PROCESSKDPLOG_H
#define LLDB_SOURCE_PLUGINS_PROCESS_MACOSX_KERNEL_PROCESSKDPLOG_H

#include "lldb/Utility/Log.h"


namespace lldb_private {

enum class KDPLog : Log::MaskType {
  Async = Log::ChannelFlag<0>,
  Breakpoints = Log::ChannelFlag<1>,
  Comm = Log::ChannelFlag<2>,
  MemoryDataLong = Log::ChannelFlag<3>,  // Log all memory reads/writes bytes
  MemoryDataShort = Log::ChannelFlag<4>, // Log short memory reads/writes bytes
  Memory = Log::ChannelFlag<5>,          // Log memory reads/writes calls
  Packets = Log::ChannelFlag<6>,
  Process = Log::ChannelFlag<7>,
  Step = Log::ChannelFlag<8>,
  Thread = Log::ChannelFlag<9>,
  Watchpoints = Log::ChannelFlag<10>,
  LLVM_MARK_AS_BITMASK_ENUM(Watchpoints)
};
#define KDP_LOG_PROCESS ::lldb_private::KDPLog::Process
#define KDP_LOG_THREAD ::lldb_private::KDPLog::Thread
#define KDP_LOG_PACKETS ::lldb_private::KDPLog::Packets
#define KDP_LOG_MEMORY ::lldb_private::KDPLog::Memory
#define KDP_LOG_MEMORY_DATA_SHORT ::lldb_private::KDPLog::MemoryDataShort
#define KDP_LOG_MEMORY_DATA_LONG ::lldb_private::KDPLog::MemoryDataLong
#define KDP_LOG_BREAKPOINTS ::lldb_private::KDPLog::Breakpoints
#define KDP_LOG_WATCHPOINTS ::lldb_private::KDPLog::Watchpoints
#define KDP_LOG_STEP ::lldb_private::KDPLog::Step
#define KDP_LOG_COMM ::lldb_private::KDPLog::Comm
#define KDP_LOG_ASYNC ::lldb_private::KDPLog::Async
#define KDP_LOG_DEFAULT KDP_LOG_PACKETS

class ProcessKDPLog {
public:
  static void Initialize();

  static Log *GetLogIfAllCategoriesSet(KDPLog mask) { return GetLog(mask); }
};

template <> Log::Channel &LogChannelFor<KDPLog>();
}

#endif // LLDB_SOURCE_PLUGINS_PROCESS_MACOSX_KERNEL_PROCESSKDPLOG_H
