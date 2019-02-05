//===-- ProcessWindowsLog.cpp -----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ProcessWindowsLog.h"

using namespace lldb_private;

static constexpr Log::Category g_categories[] = {
    {{"break"}, {"log breakpoints"}, WINDOWS_LOG_BREAKPOINTS},
    {{"event"}, {"log low level debugger events"}, WINDOWS_LOG_EVENT},
    {{"exception"}, {"log exception information"}, WINDOWS_LOG_EXCEPTION},
    {{"memory"}, {"log memory reads and writes"}, WINDOWS_LOG_MEMORY},
    {{"process"}, {"log process events and activities"}, WINDOWS_LOG_PROCESS},
    {{"registers"}, {"log register read/writes"}, WINDOWS_LOG_REGISTERS},
    {{"step"}, {"log step related activities"}, WINDOWS_LOG_STEP},
    {{"thread"}, {"log thread events and activities"}, WINDOWS_LOG_THREAD},
};

Log::Channel ProcessWindowsLog::g_channel(g_categories, WINDOWS_LOG_PROCESS);

void ProcessWindowsLog::Initialize() {
  static llvm::once_flag g_once_flag;
  llvm::call_once(g_once_flag, []() { Log::Register("windows", g_channel); });
}

void ProcessWindowsLog::Terminate() {}









