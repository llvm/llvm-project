//===- Logging.cpp --------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Tools/lsp-server-support/Logging.h"
#include "llvm/Support/Chrono.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace mlir::lsp;

void Logger::setLogLevel(Level logLevel) { get().logLevel = logLevel; }

Logger &Logger::get() {
  static Logger logger;
  return logger;
}

void Logger::log(Level logLevel, const char *fmt,
                 const llvm::formatv_object_base &message) {
  Logger &logger = get();

  // Ignore messages with log levels below the current setting in the logger.
  if (logLevel < logger.logLevel)
    return;

  // An indicator character for each log level.
  const char *logLevelIndicators = "DIE";

  // Format the message and print to errs.
  llvm::sys::TimePoint<> timestamp = std::chrono::system_clock::now();
  std::lock_guard<std::mutex> logGuard(logger.mutex);
  llvm::errs() << llvm::formatv(
      "{0}[{1:%H:%M:%S.%L}] {2}\n",
      logLevelIndicators[static_cast<unsigned>(logLevel)], timestamp, message);
  llvm::errs().flush();
}
