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
#include "llvm/Support/raw_ostream.h"
#include <mutex>
#include <string>
#include <system_error>

// Write a message to log, if logging is enabled.
#define DAP_LOG(log, ...)                                                      \
  do {                                                                         \
    ::lldb_dap::Log *log_private = (log);                                      \
    if (log_private) {                                                         \
      log_private->WriteMessage(::llvm::formatv(__VA_ARGS__).str());           \
    }                                                                          \
  } while (0)

// Write message to log, if error is set. In the log message refer to the error
// with {0}. Error is cleared regardless of whether logging is enabled.
#define DAP_LOG_ERROR(log, error, ...)                                         \
  do {                                                                         \
    ::lldb_dap::Log *log_private = (log);                                      \
    ::llvm::Error error_private = (error);                                     \
    if (log_private && error_private) {                                        \
      log_private->WriteMessage(                                               \
          ::lldb_dap::FormatError(::std::move(error_private), __VA_ARGS__));   \
    } else                                                                     \
      ::llvm::consumeError(::std::move(error_private));                        \
  } while (0)

namespace lldb_dap {

/// Log manages the lldb-dap log file, used with the corresponding `DAP_LOG` and
/// `DAP_LOG_ERROR` helpers.
class Log final {
public:
  /// Creates a log file with the given filename.
  Log(llvm::StringRef filename, std::error_code &EC);

  void WriteMessage(llvm::StringRef message);

private:
  std::mutex m_mutex;
  llvm::raw_fd_ostream m_stream;
};

template <typename... Args>
inline auto FormatError(llvm::Error error, const char *format, Args &&...args) {
  return llvm::formatv(format, llvm::toString(std::move(error)),
                       std::forward<Args>(args)...)
      .str();
}
} // namespace lldb_dap

#endif
