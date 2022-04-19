//===-- LogChannelSwift.cpp -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "LogChannelSwift.h"

using namespace lldb_private;

static constexpr Log::Category g_categories[] = {
    {{"health"},
     {"log all messages related to lldb Swift operational health"},
     SwiftLog::Health},
};

static Log::Channel g_channel(g_categories, SwiftLog::Health);

static std::string g_swift_log_buffer;

template <> Log::Channel &lldb_private::LogChannelFor<SwiftLog>() {
  return g_channel;
}

void LogChannelSwift::Initialize() {
  Log::Register("swift", g_channel);

  llvm::raw_null_ostream error_stream;
  Log::EnableLogChannel(
      std::make_shared<llvm::raw_string_ostream>(g_swift_log_buffer),
      LLDB_LOG_OPTION_THREADSAFE, "swift", {"health"}, error_stream);
}

void LogChannelSwift::Terminate() { Log::Unregister("swift"); }

Log *lldb_private::GetSwiftHealthLog() { return GetLog(SwiftLog::Health); }

llvm::StringRef lldb_private::GetSwiftHealthLogData() {
  return g_swift_log_buffer;
}
