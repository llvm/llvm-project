//===------------------- Logging.h - LLVM Advisor -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Structured logging with multiple severity levels.
// Provides logging infrastructure for the entire advisor.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "AdvisorCommon.h"

namespace llvm::advisor {

enum class LogLevel { Error, Warning, Info, Debug };

/// Simple logger that writes messages to a stream with severity labels.
class Logger {
public:
  explicit Logger(raw_ostream &OS) : OS(OS) {}

  /// Toggle verbose (debug) output.
  void setVerbose(bool Value) { Verbose = Value; }

  /// Write a message at the given severity level.
  void log(LogLevel Level, StringRef Message);

private:
  raw_ostream &OS;
  bool Verbose = false;
};

} // namespace llvm::advisor
