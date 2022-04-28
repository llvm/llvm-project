//===-- LLDBLog.cpp -------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Utility/LLDBLog.h"
#include "lldb/Utility/Log.h"
#include "llvm/ADT/ArrayRef.h"
#include <cstdarg>

using namespace lldb_private;

static constexpr Log::Category g_categories[] = {
    {{"api"}, {"log API calls and return values"}, LLDBLog::API},
    {{"ast"}, {"log AST"}, LLDBLog::AST},
    {{"break"}, {"log breakpoints"}, LLDBLog::Breakpoints},
    {{"commands"}, {"log command argument parsing"}, LLDBLog::Commands},
    {{"comm"}, {"log communication activities"}, LLDBLog::Communication},
    {{"conn"}, {"log connection details"}, LLDBLog::Connection},
    {{"demangle"},
     {"log mangled names to catch demangler crashes"},
     LLDBLog::Demangle},
    {{"dyld"},
     {"log shared library related activities"},
     LLDBLog::DynamicLoader},
    {{"event"},
     {"log broadcaster, listener and event queue activities"},
     LLDBLog::Events},
    {{"expr"}, {"log expressions"}, LLDBLog::Expressions},
    {{"formatters"},
     {"log data formatters related activities"},
     LLDBLog::DataFormatters},
    {{"host"}, {"log host activities"}, LLDBLog::Host},
    {{"jit"}, {"log JIT events in the target"}, LLDBLog::JITLoader},
    {{"language"}, {"log language runtime events"}, LLDBLog::Language},
    {{"mmap"}, {"log mmap related activities"}, LLDBLog::MMap},
    {{"module"},
     {"log module activities such as when modules are created, destroyed, "
      "replaced, and more"},
     LLDBLog::Modules},
    {{"object"},
     {"log object construction/destruction for important objects"},
     LLDBLog::Object},
    {{"os"}, {"log OperatingSystem plugin related activities"}, LLDBLog::OS},
    {{"platform"}, {"log platform events and activities"}, LLDBLog::Platform},
    {{"process"}, {"log process events and activities"}, LLDBLog::Process},
    {{"script"}, {"log events about the script interpreter"}, LLDBLog::Script},
    {{"state"},
     {"log private and public process state changes"},
     LLDBLog::State},
    {{"step"}, {"log step related activities"}, LLDBLog::Step},
    {{"symbol"}, {"log symbol related issues and warnings"}, LLDBLog::Symbols},
    {{"system-runtime"}, {"log system runtime events"}, LLDBLog::SystemRuntime},
    {{"target"}, {"log target events and activities"}, LLDBLog::Target},
    {{"temp"}, {"log internal temporary debug messages"}, LLDBLog::Temporary},
    {{"thread"}, {"log thread events and activities"}, LLDBLog::Thread},
    {{"types"}, {"log type system related activities"}, LLDBLog::Types},
    {{"unwind"}, {"log stack unwind activities"}, LLDBLog::Unwind},
    {{"watch"}, {"log watchpoint related activities"}, LLDBLog::Watchpoints},
    {{"on-demand"},
     {"log symbol on-demand related activities"},
     LLDBLog::OnDemand},
};

static Log::Channel g_log_channel(g_categories,
                                  LLDBLog::Process | LLDBLog::Thread |
                                      LLDBLog::DynamicLoader |
                                      LLDBLog::Breakpoints |
                                      LLDBLog::Watchpoints | LLDBLog::Step |
                                      LLDBLog::State | LLDBLog::Symbols |
                                      LLDBLog::Target | LLDBLog::Commands);

template <> Log::Channel &lldb_private::LogChannelFor<LLDBLog>() {
  return g_log_channel;
}

#ifdef LLDB_ENABLE_SWIFT

static constexpr Log::Category g_swift_categories[] = {
  {{"health"}, {"log all messages related to lldb Swift operational health"}, SwiftLog::Health},
};

static Log::Channel g_swift_log_channel(g_swift_categories, SwiftLog::Health);

template <> Log::Channel &lldb_private::LogChannelFor<SwiftLog>() {
  return g_swift_log_channel;
}

static std::string g_swift_log_buffer;

#endif

void lldb_private::InitializeLldbChannel() {
  Log::Register("lldb", g_log_channel);
#ifdef LLDB_ENABLE_SWIFT
  Log::Register("swift", g_swift_log_channel);

  llvm::raw_null_ostream error_stream;
  Log::EnableLogChannel(
      std::make_shared<llvm::raw_string_ostream>(g_swift_log_buffer),
      LLDB_LOG_OPTION_THREADSAFE, "swift", {"health"}, error_stream);
#endif
}

Log *lldb_private::GetLogIfAllCategoriesSet(LLDBLog mask) {
  return GetLog(mask);
}

Log *lldb_private::GetLogIfAnyCategoriesSet(LLDBLog mask) {
  return GetLog(mask);
}

#ifdef LLDB_ENABLE_SWIFT

Log *lldb_private::GetSwiftHealthLog() {
  return GetLog(SwiftLog::Health);
}

llvm::StringRef lldb_private::GetSwiftHealthLogData() {
  return g_swift_log_buffer;
}

#endif
