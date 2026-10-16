//===------------------- Logging.cpp - LLVM Advisor -------------------===//
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
#include "Utils/Logging.h"

using namespace llvm;
using namespace llvm::advisor;

static StringRef levelLabel(LogLevel Level) {
  switch (Level) {
  case LogLevel::Error:
    return "error";
  case LogLevel::Warning:
    return "warning";
  case LogLevel::Info:
    return "info";
  case LogLevel::Debug:
    return "debug";
  }
  llvm_unreachable("Unknown log level");
}

void Logger::log(LogLevel Level, StringRef Message) {
  if (Level == LogLevel::Debug && !Verbose)
    return;
  OS << "llvm-advisor: " << levelLabel(Level) << ": " << Message << '\n';
}
