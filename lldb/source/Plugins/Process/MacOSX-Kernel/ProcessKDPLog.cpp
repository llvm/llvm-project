//===-- ProcessKDPLog.cpp ---------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "ProcessKDPLog.h"

using namespace lldb_private;

static constexpr Log::Category g_categories[] = {
    {{"async"}, {"log asynchronous activity"}, KDP_LOG_ASYNC},
    {{"break"}, {"log breakpoints"}, KDP_LOG_BREAKPOINTS},
    {{"comm"}, {"log communication activity"}, KDP_LOG_COMM},
    {{"data-long"},
     {"log memory bytes for memory reads and writes for all transactions"},
     KDP_LOG_MEMORY_DATA_LONG},
    {{"data-short"},
     {"log memory bytes for memory reads and writes for short transactions "
      "only"},
     KDP_LOG_MEMORY_DATA_SHORT},
    {{"memory"}, {"log memory reads and writes"}, KDP_LOG_MEMORY},
    {{"packets"}, {"log gdb remote packets"}, KDP_LOG_PACKETS},
    {{"process"}, {"log process events and activities"}, KDP_LOG_PROCESS},
    {{"step"}, {"log step related activities"}, KDP_LOG_STEP},
    {{"thread"}, {"log thread events and activities"}, KDP_LOG_THREAD},
    {{"watch"}, {"log watchpoint related activities"}, KDP_LOG_WATCHPOINTS},
};

Log::Channel ProcessKDPLog::g_channel(g_categories, KDP_LOG_DEFAULT);

void ProcessKDPLog::Initialize() { Log::Register("kdp-remote", g_channel); }
