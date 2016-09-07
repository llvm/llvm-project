//===-- ProcessWindowsLog.h -------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_ProcessWindowsLog_h_
#define liblldb_ProcessWindowsLog_h_

#include "lldb/Core/Log.h"

#define WINDOWS_LOG_VERBOSE (1u << 0)
#define WINDOWS_LOG_PROCESS (1u << 1)     // Log process operations
#define WINDOWS_LOG_EXCEPTION (1u << 1)   // Log exceptions
#define WINDOWS_LOG_THREAD (1u << 2)      // Log thread operations
#define WINDOWS_LOG_MEMORY (1u << 3)      // Log memory reads/writes calls
#define WINDOWS_LOG_BREAKPOINTS (1u << 4) // Log breakpoint operations
#define WINDOWS_LOG_STEP (1u << 5)        // Log step operations
#define WINDOWS_LOG_REGISTERS (1u << 6)   // Log register operations
#define WINDOWS_LOG_EVENT (1u << 7)       // Low level debug events
#define WINDOWS_LOG_ALL (UINT32_MAX)

enum class LogMaskReq { All, Any };

class ProcessWindowsLog {
  static const char *m_pluginname;

public:
  // ---------------------------------------------------------------------
  // Public Static Methods
  // ---------------------------------------------------------------------
  static void Initialize();

  static void Terminate();

  static void RegisterPluginName(const char *pluginName) {
    m_pluginname = pluginName;
  }

  static void RegisterPluginName(lldb_private::ConstString pluginName) {
    m_pluginname = pluginName.GetCString();
  }

  static bool TestLogFlags(uint32_t mask, LogMaskReq req);

  static lldb_private::Log *GetLog();

  static void DisableLog(const char **args,
                         lldb_private::Stream *feedback_strm);

  static lldb_private::Log *EnableLog(lldb::StreamSP &log_stream_sp,
                                      uint32_t log_options, const char **args,
                                      lldb_private::Stream *feedback_strm);

  static void ListLogCategories(lldb_private::Stream *strm);
};

#define WINLOGF_IF(Flags, Req, Method, ...)                                    \
  {                                                                            \
    if (ProcessWindowsLog::TestLogFlags(Flags, Req)) {                         \
      Log *log = ProcessWindowsLog::GetLog();                                  \
      if (log)                                                                 \
        log->Method(__VA_ARGS__);                                              \
    }                                                                          \
  }

#define WINLOG_IFANY(Flags, ...)                                               \
  WINLOGF_IF(Flags, LogMaskReq::Any, Printf, __VA_ARGS__)
#define WINLOG_IFALL(Flags, ...)                                               \
  WINLOGF_IF(Flags, LogMaskReq::All, Printf, __VA_ARGS__)
#define WINLOGV_IFANY(Flags, ...)                                              \
  WINLOGF_IF(Flags, LogMaskReq::Any, Verbose, __VA_ARGS__)
#define WINLOGV_IFALL(Flags, ...)                                              \
  WINLOGF_IF(Flags, LogMaskReq::All, Verbose, __VA_ARGS__)
#define WINLOGD_IFANY(Flags, ...)                                              \
  WINLOGF_IF(Flags, LogMaskReq::Any, Debug, __VA_ARGS__)
#define WINLOGD_IFALL(Flags, ...)                                              \
  WINLOGF_IF(Flags, LogMaskReq::All, Debug, __VA_ARGS__)
#define WINERR_IFANY(Flags, ...)                                               \
  WINLOGF_IF(Flags, LogMaskReq::Any, Error, __VA_ARGS__)
#define WINERR_IFALL(Flags, ...)                                               \
  WINLOGF_IF(Flags, LogMaskReq::All, Error, __VA_ARGS__)
#define WINWARN_IFANY(Flags, ...)                                              \
  WINLOGF_IF(Flags, LogMaskReq::Any, Warning, __VA_ARGS__)
#define WINWARN_IFALL(Flags, ...)                                              \
  WINLOGF_IF(Flags, LogMaskReq::All, Warning, __VA_ARGS__)

#endif // liblldb_ProcessWindowsLog_h_
