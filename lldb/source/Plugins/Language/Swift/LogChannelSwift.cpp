//===-- LogChannelSwift.cpp -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "LogChannelSwift.h"
#include "lldb/Core/Diagnostics.h"
#include "lldb/Host/Host.h"
#include "lldb/Utility/Log.h"
#include "lldb/Version/Version.h"
#include "llvm/Support/raw_ostream.h"

using namespace lldb_private;

static constexpr Log::Category g_categories[] = {
    {{"health"},
     {"log all messages related to lldb Swift operational health"},
     SwiftLog::Health},
};

static Log::Channel g_channel(g_categories, SwiftLog::Health);

static constexpr size_t g_health_log_size = 5000;
static std::shared_ptr<RotatingLogHandler> g_health_log_handler;
static std::optional<Diagnostics::ArtifactProviderID> g_diagnostics_artifact_id;

template <> Log::Channel &lldb_private::LogChannelFor<SwiftLog>() {
  return g_channel;
}

void LogChannelSwift::Initialize() {
  Log::Register("swift", g_channel);

  g_health_log_handler =
      std::make_shared<RotatingLogHandler>(g_health_log_size);
  auto system_log_handler_sp = std::make_shared<SystemLogHandler>();
  auto log_handler_sp = std::make_shared<TeeLogHandler>(g_health_log_handler,
                                                        system_log_handler_sp);

  llvm::consumeError(
      Log::EnableLogChannel(log_handler_sp, 0, "swift", {"health"}));
  if (Log *log = GetSwiftHealthLog())
    log->Printf(
        "==== LLDB swift-healthcheck log. ===\n"
        "This file contains the configuration of LLDB's embedded Swift "
        "compiler to help diagnosing module import and search path issues. "
        "The swift-healthcheck command is meant to be run *after* an error "
        "has occurred.\n%s",
        lldb_private::GetVersion());

  if (Diagnostics::Enabled()) {
    g_diagnostics_artifact_id = Diagnostics::Instance().AddArtifactProvider(
        "swift-healthcheck.log", []() -> std::string {
          std::string content;
          llvm::raw_string_ostream stream(content);
          if (g_health_log_handler)
            g_health_log_handler->Dump(stream);
          return content;
        });
  }
}

void LogChannelSwift::Terminate() {
  if (g_diagnostics_artifact_id && Diagnostics::Enabled())
    Diagnostics::Instance().RemoveArtifactProvider(*g_diagnostics_artifact_id);
  g_diagnostics_artifact_id.reset();
  g_health_log_handler.reset();
  Log::Unregister("swift");
}

Log *lldb_private::GetSwiftHealthLog() { return GetLog(SwiftLog::Health); }

void lldb_private::DumpSwiftHealthLog(llvm::raw_ostream &stream) {
  if (g_health_log_handler)
    g_health_log_handler->Dump(stream);
}
