//===- Logging.h - LSP Server Logging ----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_LSP_LOGGING_H
#define LLVM_SUPPORT_LSP_LOGGING_H

#include "llvm/Support/Debug.h"
#include "llvm/Support/FormatVariadic.h"
#include <memory>
#include <mutex>

namespace llvm {
namespace lsp {

/// This class represents the main interface for logging, and allows for
/// filtering logging based on different levels of severity or significance.
class Logger {
public:
  /// The level of significance for a log message.
  enum class Level { Debug, Info, Error };

  /// Set the severity level of the logger.
  static void setLogLevel(Level LogLevel);

  /// Initiate a log message at various severity levels. These should be called
  /// after a call to `initialize`.
  template <typename... Ts> static void debug(const char *Fmt, Ts &&...Vals) {
    log(Level::Debug, Fmt, llvm::formatv(Fmt, std::forward<Ts>(Vals)...));
  }
  template <typename... Ts> static void info(const char *Fmt, Ts &&...Vals) {
    log(Level::Info, Fmt, llvm::formatv(Fmt, std::forward<Ts>(Vals)...));
  }
  template <typename... Ts> static void error(const char *Fmt, Ts &&...Vals) {
    log(Level::Error, Fmt, llvm::formatv(Fmt, std::forward<Ts>(Vals)...));
  }

private:
  Logger() = default;

  /// Return the main logger instance.
  static Logger &get();

  /// Start a log message with the given severity level.
  static void log(Level LogLevel, const char *Fmt,
                  const llvm::formatv_object_base &Message);

  /// The minimum logging level. Messages with lower level are ignored.
  Level LogLevel = Level::Error;

  /// A mutex used to guard logging.
  std::mutex Mutex;
};
} // namespace lsp
} // namespace llvm

#endif // LLVM_SUPPORT_LSP_LOGGING_H
