//===- Logging.cpp - General logging class implementation -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Implements ompTest-tailored logging.
///
//===----------------------------------------------------------------------===//

#include "Logging.h"

using namespace omptest;
using namespace logging;

Logger::Logger(Level LogLevel, std::ostream &OutStream, bool FormatOutput)
    : LoggingLevel(LogLevel), OutStream(OutStream), FormatOutput(FormatOutput) {
  // Flush any buffered output
  OutStream << std::flush;
}

Logger::~Logger() {
  // Flush any buffered output
  OutStream << std::flush;
}

std::map<Level, std::set<FormatOption>> AggregatedFormatOptions{
    {Level::Diagnostic, {FormatOption::ColorLightBlue}},
    {Level::Info, {FormatOption::ColorLightGray}},
    {Level::Warning, {FormatOption::ColorLightYellow}},
    {Level::Error, {FormatOption::ColorRed}},
    {Level::Critical, {FormatOption::ColorLightRed}},
    {Level::Default, {FormatOption::None}},
    {Level::ExpectedEvent, {FormatOption::Bold, FormatOption::ColorCyan}},
    {Level::ObservedEvent, {FormatOption::ColorCyan}},
    {Level::OffendingEvent, {FormatOption::ColorYellow}}};

const char *logging::to_string(Level LogLevel) {
  switch (LogLevel) {
  case Level::Diagnostic:
    return "Diagnostic";
  case Level::Info:
    return "Info";
  case Level::Warning:
    return "Warning";
  case Level::Error:
    return "Error";
  case Level::Critical:
    return "Critical";
  case Level::Silent:
    return "Silent";
  default:
    assert(false && "Requested string representation for unknown LogLevel");
    return "UNKNOWN";
  }
}

std::string logging::getFormatSequence(Level LogLevel) {
  auto Options = AggregatedFormatOptions[LogLevel];
  std::stringstream SS{"\033["};
  SS << "\033[";
  if (!Options.empty()) {
    for (auto &Option : AggregatedFormatOptions[LogLevel])
      SS << int(Option) << ';';
    SS.seekp(-1, SS.cur);
    SS << 'm';
  } else {
    // Fallback to None / reset formatting
    SS << "0m";
  }
  return SS.str();
}

std::string logging::format(const std::string &Message, FormatOption Option) {
  std::stringstream SS{"\033["};
  SS << "\033[";
  SS << int(Option) << 'm' << Message << "\033[0m";
  return SS.str();
}

std::string logging::format(const std::string &Message,
                            std::set<FormatOption> Options) {
  std::stringstream SS{"\033["};
  SS << "\033[";
  for (auto &Option : Options)
    SS << int(Option) << ';';
  SS.seekp(-1, SS.cur);
  SS << 'm' << Message << "\033[0m";
  return SS.str();
}

void Logger::log(const std::string &Message, Level LogLevel) const {
  // Serialize logging
  std::lock_guard<std::mutex> Lock(LogMutex);

  if (LoggingLevel > LogLevel)
    return;

  if (FormatOutput) {
    OutStream << getFormatSequence(LogLevel) << '[' << to_string(LogLevel)
              << "] " << Message << getFormatSequence() << std::endl;
  } else {
    OutStream << '[' << to_string(LogLevel) << "] " << Message << std::endl;
  }
}

void Logger::logEventMismatch(const std::string &Message,
                              const omptest::OmptAssertEvent &OffendingEvent,
                              Level LogLevel) const {
  // Serialize logging
  std::lock_guard<std::mutex> Lock(LogMutex);
  if (LoggingLevel > LogLevel)
    return;

  if (FormatOutput) {
    OutStream << getFormatSequence(LogLevel) << '[' << to_string(LogLevel)
              << "] " << getFormatSequence()
              << format(Message, AggregatedFormatOptions[LogLevel])
              << "\n\tOffending event name='"
              << format(OffendingEvent.getEventName(),
                        AggregatedFormatOptions[Level::OffendingEvent])
              << "'\n\tOffending='"
              << format(OffendingEvent.toString(),
                        AggregatedFormatOptions[Level::OffendingEvent])
              << '\'' << std::endl;
  } else {
    OutStream << '[' << to_string(LogLevel) << "] " << Message
              << "\n\tOffending event name='" << OffendingEvent.getEventName()
              << "'\n\tOffending='" << OffendingEvent.toString() << '\''
              << std::endl;
  }
}

void Logger::logEventMismatch(const std::string &Message,
                              const omptest::OmptAssertEvent &ExpectedEvent,
                              const omptest::OmptAssertEvent &ObservedEvent,
                              Level LogLevel) const {
  // Serialize logging
  std::lock_guard<std::mutex> Lock(LogMutex);
  if (LoggingLevel > LogLevel)
    return;

  if (FormatOutput) {
    OutStream << getFormatSequence(LogLevel) << '[' << to_string(LogLevel)
              << "] " << Message << getFormatSequence()
              << "\n\tExpected event name='"
              << format(ExpectedEvent.getEventName(),
                        AggregatedFormatOptions[Level::ExpectedEvent])
              << "' observe='"
              << format(to_string(ExpectedEvent.getEventExpectedState()),
                        AggregatedFormatOptions[Level::ExpectedEvent])
              << "'\n\tObserved event name='"
              << format(ObservedEvent.getEventName(),
                        AggregatedFormatOptions[Level::ObservedEvent])
              << "'\n\tExpected='"
              << format(ExpectedEvent.toString(),
                        AggregatedFormatOptions[Level::ExpectedEvent])
              << "'\n\tObserved='"
              << format(ObservedEvent.toString(),
                        AggregatedFormatOptions[Level::ObservedEvent])
              << '\'' << std::endl;
  } else {
    OutStream << '[' << to_string(LogLevel) << "] " << Message
              << "\n\tExpected event name='" << ExpectedEvent.getEventName()
              << "' observe='"
              << to_string(ExpectedEvent.getEventExpectedState())
              << "'\n\tObserved event name='" << ObservedEvent.getEventName()
              << "'\n\tExpected='" << ExpectedEvent.toString()
              << "'\n\tObserved='" << ObservedEvent.toString() << '\''
              << std::endl;
  }
}

void Logger::setFormatOutput(bool Enabled) { FormatOutput = Enabled; }

Level Logger::getLoggingLevel() const { return LoggingLevel; }

void Logger::setLoggingLevel(Level LogLevel) { LoggingLevel = LogLevel; }
