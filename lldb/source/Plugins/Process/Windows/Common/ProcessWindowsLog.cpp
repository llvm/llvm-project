//===-- ProcessWindowsLog.cpp -----------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "ProcessWindowsLog.h"

#include <mutex>

#include "lldb/Core/StreamFile.h"
#include "lldb/Interpreter/Args.h"
#include "llvm/Support/ManagedStatic.h"

using namespace lldb;
using namespace lldb_private;

// We want to avoid global constructors where code needs to be run so here we
// control access to our static g_log_sp by hiding it in a singleton function
// that will construct the static g_log_sp the first time this function is
// called.
static bool g_log_enabled = false;
static Log *g_log = nullptr;

static llvm::ManagedStatic<std::once_flag> g_once_flag;

void ProcessWindowsLog::Initialize() {
  static ConstString g_name("windows");

  std::call_once(*g_once_flag, []() {
    Log::Callbacks log_callbacks = {DisableLog, EnableLog, ListLogCategories};

    Log::RegisterLogChannel(g_name, log_callbacks);
    RegisterPluginName(g_name);
  });
}

void ProcessWindowsLog::Terminate() {}

Log *ProcessWindowsLog::GetLog() { return (g_log_enabled) ? g_log : nullptr; }

bool ProcessWindowsLog::TestLogFlags(uint32_t mask, LogMaskReq req) {
  Log *log = GetLog();
  if (!log)
    return false;

  uint32_t log_mask = log->GetMask().Get();
  if (req == LogMaskReq::All)
    return ((log_mask & mask) == mask);
  else
    return (log_mask & mask);
}

static uint32_t GetFlagBits(const char *arg) {
  if (::strcasecmp(arg, "all") == 0)
    return WINDOWS_LOG_ALL;
  else if (::strcasecmp(arg, "break") == 0)
    return WINDOWS_LOG_BREAKPOINTS;
  else if (::strcasecmp(arg, "event") == 0)
    return WINDOWS_LOG_EVENT;
  else if (::strcasecmp(arg, "exception") == 0)
    return WINDOWS_LOG_EXCEPTION;
  else if (::strcasecmp(arg, "memory") == 0)
    return WINDOWS_LOG_MEMORY;
  else if (::strcasecmp(arg, "process") == 0)
    return WINDOWS_LOG_PROCESS;
  else if (::strcasecmp(arg, "registers") == 0)
    return WINDOWS_LOG_REGISTERS;
  else if (::strcasecmp(arg, "step") == 0)
    return WINDOWS_LOG_STEP;
  else if (::strcasecmp(arg, "thread") == 0)
    return WINDOWS_LOG_THREAD;
  else if (::strcasecmp(arg, "verbose") == 0)
    return WINDOWS_LOG_VERBOSE;
  return 0;
}

void ProcessWindowsLog::DisableLog(const char **args, Stream *feedback_strm) {
  Log *log(GetLog());
  if (log) {
    uint32_t flag_bits = 0;

    if (args[0] != nullptr) {
      flag_bits = log->GetMask().Get();
      for (; args[0]; args++) {
        const char *arg = args[0];
        uint32_t bits = GetFlagBits(arg);

        if (bits) {
          flag_bits &= ~bits;
        } else {
          feedback_strm->Printf("error: unrecognized log category '%s'\n", arg);
          ListLogCategories(feedback_strm);
        }
      }
    }

    log->GetMask().Reset(flag_bits);
    if (flag_bits == 0) {
      g_log_enabled = false;
      log->SetStream(lldb::StreamSP());
    }
  }

  return;
}

Log *ProcessWindowsLog::EnableLog(StreamSP &log_stream_sp, uint32_t log_options,
                                  const char **args, Stream *feedback_strm) {
  // Try see if there already is a log - that way we can reuse its settings.
  // We could reuse the log in toto, but we don't know that the stream is the
  // same.
  uint32_t flag_bits = 0;
  if (g_log)
    flag_bits = g_log->GetMask().Get();

  // Now make a new log with this stream if one was provided
  if (log_stream_sp) {
    if (g_log)
      g_log->SetStream(log_stream_sp);
    else
      g_log = new Log(log_stream_sp);
  }

  if (g_log) {
    bool got_unknown_category = false;
    for (; args[0]; args++) {
      const char *arg = args[0];
      uint32_t bits = GetFlagBits(arg);

      if (bits) {
        flag_bits |= bits;
      } else {
        feedback_strm->Printf("error: unrecognized log category '%s'\n", arg);
        if (got_unknown_category == false) {
          got_unknown_category = true;
          ListLogCategories(feedback_strm);
        }
      }
    }
    if (flag_bits == 0)
      flag_bits = WINDOWS_LOG_ALL;
    g_log->GetMask().Reset(flag_bits);
    g_log->GetOptions().Reset(log_options);
    g_log_enabled = true;
  }
  return g_log;
}

void ProcessWindowsLog::ListLogCategories(Stream *strm) {
  strm->Printf("Logging categories for '%s':\n"
               "  all - turn on all available logging categories\n"
               "  break - log breakpoints\n"
               "  event - log low level debugger events\n"
               "  exception - log exception information\n"
               "  memory - log memory reads and writes\n"
               "  process - log process events and activities\n"
               "  registers - log register read/writes\n"
               "  thread - log thread events and activities\n"
               "  step - log step related activities\n"
               "  verbose - enable verbose logging\n",
               ProcessWindowsLog::m_pluginname);
}

const char *ProcessWindowsLog::m_pluginname = "";
