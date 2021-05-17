//===- Logging.h - MLIR LSP Server Logging ----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LIB_MLIR_TOOLS_MLIRLSPSERVER_LSP_LOGGING_H
#define LIB_MLIR_TOOLS_MLIRLSPSERVER_LSP_LOGGING_H

#include "mlir/Support/LLVM.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FormatVariadic.h"
#include <memory>
#include <mutex>

namespace mlir {
namespace lsp {

/// This class represents the main interface for logging, and allows for
/// filtering logging based on different levels of severity or significance.
class Logger {
public:
  /// The level of significance for a log message.
  enum class Level { Debug, Info, Error };

  /// Set the severity level of the logger.
  static void setLogLevel(Level logLevel);

  /// Initiate a log message at various severity levels. These should be called
  /// after a call to `initialize`.
  template <typename... Ts> static void debug(const char *fmt, Ts &&... vals) {
    log(Level::Debug, fmt, llvm::formatv(fmt, std::forward<Ts>(vals)...));
  }
  template <typename... Ts> static void info(const char *fmt, Ts &&... vals) {
    log(Level::Info, fmt, llvm::formatv(fmt, std::forward<Ts>(vals)...));
  }
  template <typename... Ts> static void error(const char *fmt, Ts &&... vals) {
    log(Level::Error, fmt, llvm::formatv(fmt, std::forward<Ts>(vals)...));
  }

private:
  Logger() = default;

  /// Return the main logger instance.
  static Logger &get();

  /// Start a log message with the given severity level.
  static void log(Level logLevel, const char *fmt,
                  const llvm::formatv_object_base &message);

  /// The minimum logging level. Messages with lower level are ignored.
  Level logLevel = Level::Error;

  /// A mutex used to guard logging.
  std::mutex mutex;
};
} // namespace lsp
} // namespace mlir

#endif // LIB_MLIR_TOOLS_MLIRLSPSERVER_LSP_LOGGING_H
