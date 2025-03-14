//===-- DAPLog.h ----------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_TOOLS_LLDB_DAP_DAPLOG_H
#define LLDB_TOOLS_LLDB_DAP_DAPLOG_H

#include "llvm/Support/Error.h"
#include "llvm/Support/FormatAdapters.h"
#include "llvm/Support/FormatVariadic.h"
#include <chrono>
#include <fstream>
#include <string>

// Write a message to log, if logging is enabled.
#define DAP_LOG(log, ...)                                                      \
  do {                                                                         \
    ::std::ofstream *log_private = (log);                                      \
    if (log_private) {                                                         \
      ::std::chrono::duration<double> now{                                     \
          ::std::chrono::system_clock::now().time_since_epoch()};              \
      *log_private << ::llvm::formatv("{0:f9} ", now.count()).str()            \
                   << ::llvm::formatv(__VA_ARGS__).str() << std::endl;         \
    }                                                                          \
  } while (0)

// Write message to log, if error is set. In the log message refer to the error
// with {0}. Error is cleared regardless of whether logging is enabled.
#define DAP_LOG_ERROR(log, error, ...)                                         \
  do {                                                                         \
    ::std::ofstream *log_private = (log);                                      \
    ::llvm::Error error_private = (error);                                     \
    if (log_private && error_private) {                                        \
      ::std::chrono::duration<double> now{                                     \
          std::chrono::system_clock::now().time_since_epoch()};                \
      *log_private << ::llvm::formatv("{0:f9} ", now.count()).str()            \
                   << ::lldb_dap::FormatError(::std::move(error_private),      \
                                              __VA_ARGS__)                     \
                   << std::endl;                                               \
    } else                                                                     \
      ::llvm::consumeError(::std::move(error_private));                        \
  } while (0)

namespace lldb_dap {

template <typename... Args>
inline auto FormatError(llvm::Error error, const char *format, Args &&...args) {
  return llvm::formatv(format, llvm::toString(std::move(error)),
                       std::forward<Args>(args)...)
      .str();
}
} // namespace lldb_dap

#endif
