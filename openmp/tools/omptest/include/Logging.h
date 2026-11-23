//===- Logging.h - General logging class ------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Provides ompTest-tailored logging, with log-levels and formatting/coloring.
///
//===----------------------------------------------------------------------===//

#ifndef OPENMP_TOOLS_OMPTEST_INCLUDE_LOGGING_H
#define OPENMP_TOOLS_OMPTEST_INCLUDE_LOGGING_H

#include "OmptAssertEvent.h"

#include <iostream>
#include <map>
#include <mutex>
#include <set>
#include <sstream>
#include <string>

namespace omptest {
namespace logging {

enum class Level : uint32_t {
  // Levels (Note: DEBUG may already be reserved)
  Diagnostic = 10,
  Info = 20,
  Warning = 30,
  Error = 40,
  Critical = 50,

  // Types used for formatting options
  Default,
  ExpectedEvent,
  ObservedEvent,
  OffendingEvent,

  // Suppress all prints
  Silent = 0xFFFFFFFF
};

enum class FormatOption : uint32_t {
  // General options
  // Note: Bold is actually "BRIGHT" -- But it will be perceived as 'bold' font
  //       It is implicitly switching colors to the 'Light' variant
  //       Thus, it has -NO EFFECT- when already using a Light* color
  None = 0,
  Bold = 1,
  Dim = 2,
  Underlined = 4,
  Blink = 5,
  Inverted = 7,
  Hidden = 8,
  // Foreground colors
  ColorDefault = 39,
  ColorBlack = 30,
  ColorRed = 31,
  ColorGreen = 32,
  ColorYellow = 33,
  ColorBlue = 34,
  ColorMagenta = 35,
  ColorCyan = 36,
  ColorLightGray = 37,
  ColorDarkGray = 90,
  ColorLightRed = 91,
  ColorLightGreen = 92,
  ColorLightYellow = 93,
  ColorLightBlue = 94,
  ColorLightMagenta = 95,
  ColorLightCyan = 96,
  ColorWhite = 97,
  // Background colors
  ColorBackgroundDefault = 49,
  ColorBackgroundBlack = 40,
  ColorBackgroundRed = 41,
  ColorBackgroundGreen = 42,
  ColorBackgroundYellow = 43,
  ColorBackgroundBlue = 44,
  ColorBackgroundMagenta = 45,
  ColorBackgroundCyan = 46,
  ColorBackgroundLightGray = 47,
  ColorBackgroundDarkGray = 100,
  ColorBackgroundLightRed = 101,
  ColorBackgroundLightGreen = 102,
  ColorBackgroundLightYellow = 103,
  ColorBackgroundLightBlue = 104,
  ColorBackgroundLightMagenta = 105,
  ColorBackgroundLightCyan = 106,
  ColorBackgroundWhite = 107
};

/// Returns a string representation of the given logging level.
const char *to_string(Level LogLevel);

/// Returns the format options as escaped sequence, for the given logging level
std::string getFormatSequence(Level LogLevel = Level::Default);

/// Format the given message with the provided option(s) and return it.
/// Here formatting is only concerning control sequences using <Esc> character
/// which can be obtained using '\e' (on console), '\033' or '\x1B'.
std::string format(const std::string &Message, FormatOption Option);
std::string format(const std::string &Message, std::set<FormatOption> Options);

class Logger {
public:
  Logger(Level LogLevel = Level::Warning, std::ostream &OutStream = std::cerr,
         bool FormatOutput = true);
  ~Logger();

  /// Log the given message to the output.
  void log(const std::string &Message, Level LogLevel) const;

  /// Log a single event mismatch.
  void logEventMismatch(const std::string &Message,
                        const omptest::OmptAssertEvent &OffendingEvent,
                        Level LogLevel = Level::Error) const;

  /// Log an event-pair mismatch.
  void logEventMismatch(const std::string &Message,
                        const omptest::OmptAssertEvent &ExpectedEvent,
                        const omptest::OmptAssertEvent &ObservedEvent,
                        Level LogLevel = Level::Error) const;

  /// Set if output is being formatted (e.g. colored).
  void setFormatOutput(bool Enabled);

  /// Return the current (minimum) Logging Level.
  Level getLoggingLevel() const;

  /// Set the (minimum) Logging Level.
  void setLoggingLevel(Level LogLevel);

private:
  /// The minimum logging level that is considered by the logger instance.
  Level LoggingLevel;

  /// The output stream used by the logger instance.
  std::ostream &OutStream;

  /// Determine if log messages are formatted using control sequences.
  bool FormatOutput;

  /// Mutex to ensure serialized logging
  mutable std::mutex LogMutex;
};

} // namespace logging
} // namespace omptest

#endif
