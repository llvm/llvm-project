//===-- LogChannelSwift.cpp -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "LogChannelSwift.h"
#include "lldb/Utility/Log.h"
#include "lldb/Version/Version.h"

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

class StringLogHandler : public LogHandler {
public:
  StringLogHandler(std::string& str) : m_string(str) {}

  void Emit(llvm::StringRef message) override {
    std::lock_guard<std::mutex> guard(m_mutex);
    m_string += std::string(message);
  }

  bool isA(const void *ClassID) const override { return ClassID == &ID; }
  static bool classof(const LogHandler *obj) { return obj->isA(&ID); }

private:
  std::mutex m_mutex;
  std::string& m_string;
  static char ID;
};

char StringLogHandler::ID;

void LogChannelSwift::Initialize() {
  Log::Register("swift", g_channel);

  auto log_handler_sp = std::make_shared<StringLogHandler>(g_swift_log_buffer);
  Log::EnableLogChannel(log_handler_sp, 0, "swift", {"health"}, llvm::nulls());
  if (Log *log = GetSwiftHealthLog())
    log->Printf(
        "==== LLDB swift-healthcheck log. ===\n"
        "This file contains the configuration of LLDB's embedded Swift "
        "compiler to help diagnosing module import and search path issues. "
        "The swift-healthcheck command is meant to be run *after* an error "
        "has occurred.\n%s",
        lldb_private::GetVersion());
}

void LogChannelSwift::Terminate() { Log::Unregister("swift"); }

Log *lldb_private::GetSwiftHealthLog() { return GetLog(SwiftLog::Health); }

llvm::StringRef lldb_private::GetSwiftHealthLogData() {
  return g_swift_log_buffer;
}
