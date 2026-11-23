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
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"
#include <mutex>
#include <string>

// Write a message to log, if logging is enabled.
#define DAP_LOG(log, ...)                                                      \
  do {                                                                         \
    ::lldb_dap::Log &log_private = (log);                                      \
    log_private.Emit(::llvm::formatv(__VA_ARGS__).str(), __FILE__, __LINE__);  \
  } while (0)

// Write message to log, if error is set. In the log message refer to the error
// with {0}. Error is cleared regardless of whether logging is enabled.
#define DAP_LOG_ERROR(log, error, ...)                                         \
  do {                                                                         \
    ::lldb_dap::Log &log_private = (log);                                      \
    ::llvm::Error error_private = (error);                                     \
    if (error_private)                                                         \
      log_private.Emit(                                                        \
          ::lldb_dap::FormatError(::std::move(error_private), __VA_ARGS__),    \
          __FILE__, __LINE__);                                                 \
  } while (0)

namespace lldb_dap {

/// Log manages the lldb-dap log file, used with the corresponding `DAP_LOG` and
/// `DAP_LOG_ERROR` helpers.
class Log final {
public:
  using Mutex = std::mutex;

  Log(llvm::raw_ostream &stream, Mutex &mutex)
      : m_stream(stream), m_mutex(mutex) {}
  Log(llvm::StringRef prefix, const Log &log)
      : m_prefix(prefix), m_stream(log.m_stream), m_mutex(log.m_mutex) {}

  /// Retuns a new Log instance with the associated prefix for all messages.
  inline Log WithPrefix(llvm::StringRef prefix) const {
    std::string full_prefix =
        m_prefix.empty() ? prefix.str() : m_prefix + prefix.str();
    full_prefix += " ";
    return Log(full_prefix, *this);
  }

  /// Emit writes a message to the underlying stream.
  void Emit(llvm::StringRef message);

  /// Emit writes a message to the underlying stream, including the file and
  /// line the message originated from.
  void Emit(llvm::StringRef message, llvm::StringRef file, size_t line);

private:
  std::string m_prefix;
  llvm::raw_ostream &m_stream;
  Mutex &m_mutex;
};

template <typename... Args>
inline auto FormatError(llvm::Error error, const char *format, Args &&...args) {
  return llvm::formatv(format, llvm::toString(std::move(error)),
                       std::forward<Args>(args)...)
      .str();
}
} // namespace lldb_dap

#endif
