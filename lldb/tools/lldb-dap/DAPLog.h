//===-- DAPLog.h ----------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_TOOLS_LLDB_DAP_DAPLOG_H
#define LLDB_TOOLS_LLDB_DAP_DAPLOG_H

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FormatAdapters.h"
#include "llvm/Support/FormatVariadic.h"
#include <chrono>
#include <fstream>
#include <mutex>
#include <string>

// Write a message to log, if logging is enabled.
#define DAP_LOG(log, ...)                                                      \
  do {                                                                         \
    ::lldb_dap::Log *log_private = (log);                                      \
    if (log_private) {                                                         \
      ::std::chrono::duration<double> now{                                     \
          ::std::chrono::system_clock::now().time_since_epoch()};              \
      ::std::string out;                                                       \
      ::llvm::raw_string_ostream os(out);                                      \
      os << ::llvm::formatv("{0:f9} ", now.count()).str()                      \
         << ::llvm::formatv(__VA_ARGS__).str() << "\n";                        \
      log_private->WriteMessage(out);                                          \
    }                                                                          \
  } while (0)

// Write message to log, if error is set. In the log message refer to the error
// with {0}. Error is cleared regardless of whether logging is enabled.
#define DAP_LOG_ERROR(log, error, ...)                                         \
  do {                                                                         \
    ::lldb_dap::Log *log_private = (log);                                      \
    ::llvm::Error error_private = (error);                                     \
    if (log_private && error_private) {                                        \
      ::std::chrono::duration<double> now{                                     \
          std::chrono::system_clock::now().time_since_epoch()};                \
      ::std::string out;                                                       \
      ::llvm::raw_string_ostream os(out);                                      \
      os << ::llvm::formatv("{0:f9} ", now.count()).str()                      \
         << ::lldb_dap::FormatError(::std::move(error_private), __VA_ARGS__)   \
         << "\n";                                                              \
      log_private->WriteMessage(out);                                          \
    } else                                                                     \
      ::llvm::consumeError(::std::move(error_private));                        \
  } while (0)

namespace lldb_dap {

class Log {
public:
  Log(std::ofstream stream) : m_stream(std::move(stream)) {}

  void WriteMessage(llvm::StringRef message) {
    std::scoped_lock<std::mutex> lock(m_mutex);
    m_stream << message.str();
    m_stream.flush();
  }

private:
  std::mutex m_mutex;
  std::ofstream m_stream;
};

template <typename... Args>
inline auto FormatError(llvm::Error error, const char *format, Args &&...args) {
  return llvm::formatv(format, llvm::toString(std::move(error)),
                       std::forward<Args>(args)...)
      .str();
}
} // namespace lldb_dap

#endif
