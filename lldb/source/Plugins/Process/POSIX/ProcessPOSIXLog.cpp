//===-- ProcessPOSIXLog.cpp -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ProcessPOSIXLog.h"

#include "llvm/Support/Threading.h"

using namespace lldb_private;

static constexpr Log::Category g_categories[] = {
  {{"break"}, {"log breakpoints"}, POSIX_LOG_BREAKPOINTS},
  {{"memory"}, {"log memory reads and writes"}, POSIX_LOG_MEMORY},
  {{"process"}, {"log process events and activities"}, POSIX_LOG_PROCESS},
  {{"ptrace"}, {"log all calls to ptrace"}, POSIX_LOG_PTRACE},
  {{"registers"}, {"log register read/writes"}, POSIX_LOG_REGISTERS},
  {{"thread"}, {"log thread events and activities"}, POSIX_LOG_THREAD},
  {{"watch"}, {"log watchpoint related activities"}, POSIX_LOG_WATCHPOINTS},
};

Log::Channel ProcessPOSIXLog::g_channel(g_categories, POSIX_LOG_DEFAULT);

void ProcessPOSIXLog::Initialize() {
  static llvm::once_flag g_once_flag;
  llvm::call_once(g_once_flag, []() { Log::Register("posix", g_channel); });
}
