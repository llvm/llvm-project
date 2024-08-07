//===--- ompTest/include/Logging.h - ompTest logging class ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef OFFLOAD_TEST_OMPTEST_INCLUDE_LOGGING_H
#define OFFLOAD_TEST_OMPTEST_INCLUDE_LOGGING_H

#include "OmptAssertEvent.h"

#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <string>

namespace omptest {
namespace logging {

enum class Level : uint32_t {
  // Levels (Note: DEBUG may already be reserved)
  DIAGNOSTIC = 10,
  INFO = 20,
  WARNING = 30,
  ERROR = 40,
  CRITICAL = 50,

  // Types used for formatting options
  Default,
  ExpectedEvent,
  ObservedEvent,
  OffendingEvent
};

enum class FormatOption : uint32_t {
  // General options
  // Note: BOLD is actually "BRIGHT" -- But it will be perceived as 'bold' font
  //       It is implicitly switching colors to the 'Light' variant
  //       Thus, it has -NO EFFECT- when already using a Light* color
  NONE = 0,
  BOLD = 1,
  DIM = 2,
  UNDERLINED = 4,
  BLINK = 5,
  INVERTED = 7,
  HIDDEN = 8,
  // Foreground colors
  COLOR_Default = 39,
  COLOR_Black = 30,
  COLOR_Red = 31,
  COLOR_Green = 32,
  COLOR_Yellow = 33,
  COLOR_Blue = 34,
  COLOR_Magenta = 35,
  COLOR_Cyan = 36,
  COLOR_LightGray = 37,
  COLOR_DarkGray = 90,
  COLOR_LightRed = 91,
  COLOR_LightGreen = 92,
  COLOR_LightYellow = 93,
  COLOR_LightBlue = 94,
  COLOR_LightMagenta = 95,
  COLOR_LightCyan = 96,
  COLOR_White = 97,
  // Background colors
  COLOR_BG_Default = 49,
  COLOR_BG_Black = 40,
  COLOR_BG_Red = 41,
  COLOR_BG_Green = 42,
  COLOR_BG_Yellow = 43,
  COLOR_BG_Blue = 44,
  COLOR_BG_Magenta = 45,
  COLOR_BG_Cyan = 46,
  COLOR_BG_LightGray = 47,
  COLOR_BG_DarkGray = 100,
  COLOR_BG_LightRed = 101,
  COLOR_BG_LightGreen = 102,
  COLOR_BG_LightYellow = 103,
  COLOR_BG_LightBlue = 104,
  COLOR_BG_LightMagenta = 105,
  COLOR_BG_LightCyan = 106,
  COLOR_BG_White = 107
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
  ~Logger();

  /// Retrieve the singleton logger (and initialize, if not done already)
  static Logger &get(Level LogLevel = Level::WARNING,
                     std::ostream &OutStream = std::cerr,
                     bool FormatOutput = true);

  /// Log the given message to the output.
  void log(Level LogLevel, const std::string &Message) const;

  /// Log a single event mismatch.
  void eventMismatch(const omptest::OmptAssertEvent &OffendingEvent,
                     const std::string &Message,
                     Level LogLevel = Level::ERROR) const;

  /// Log an event-pair mismatch.
  void eventMismatch(const omptest::OmptAssertEvent &ExpectedEvent,
                     const omptest::OmptAssertEvent &ObservedEvent,
                     const std::string &Message,
                     Level LogLevel = Level::ERROR) const;

  /// Set if output is being formatted.
  void setFormatOutput(bool Enabled);

  /// Return the current (minimum) Logging Level.
  Level getLoggingLevel() const;

  /// Set the (minimum) Logging Level.
  void setLoggingLevel(Level LogLevel);

private:
  Logger(Level LogLevel = Level::WARNING, std::ostream &OutStream = std::cerr,
         bool FormatOutput = true);

  /// The minimum logging level that is considered by the logger instance.
  Level LoggingLevel;

  /// The output stream used by the logger instance.
  std::ostream &OutStream;

  /// Determine if log messages are formatted using control sequences.
  bool FormatOutput;
};

} // namespace logging
} // namespace omptest

// Pointer to global logger
extern omptest::logging::Logger *Log;

#endif