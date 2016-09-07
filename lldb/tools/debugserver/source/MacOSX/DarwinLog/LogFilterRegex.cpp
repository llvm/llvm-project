//===-- LogFilterRegex.cpp --------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "LogFilterRegex.h"

#include "DNBLog.h"
#include "LogMessage.h"

//----------------------------------------------------------------------
// Enable enhanced mode if it is available. This allows for things like
// \d for digit, \s for space, and many more, but it isn't available
// everywhere.
//----------------------------------------------------------------------
#if defined(REG_ENHANCED)
#define DEFAULT_COMPILE_FLAGS (REG_ENHANCED | REG_EXTENDED)
#else
#define DEFAULT_COMPILE_FLAGS (REG_EXTENDED)
#endif

LogFilterRegex::LogFilterRegex(bool match_accepts, FilterTarget filter_target,
                               const std::string &regex)
    : LogFilter(match_accepts), m_filter_target(filter_target),
      m_regex_text(regex), m_regex(), m_is_valid(false), m_error_text() {
  // Clear it.
  memset(&m_regex, 0, sizeof(m_regex));

  // Compile it.
  if (!regex.empty()) {
    auto comp_err = ::regcomp(&m_regex, regex.c_str(), DEFAULT_COMPILE_FLAGS);
    m_is_valid = (comp_err == 0);
    if (!m_is_valid) {
      char buffer[256];
      buffer[0] = '\0';
      ::regerror(comp_err, &m_regex, buffer, sizeof(buffer));
      m_error_text = buffer;
    }
  }
}

LogFilterRegex::~LogFilterRegex() {
  if (m_is_valid) {
    // Free the regex internals.
    regfree(&m_regex);
  }
}

bool LogFilterRegex::DoesMatch(const LogMessage &message) const {
  switch (m_filter_target) {
  case eFilterTargetActivity:
    // Empty fields never match a condition.
    if (!message.HasActivity())
      return false;
    return ::regexec(&m_regex, message.GetActivity(), 0, nullptr, 0) == 0;
  case eFilterTargetActivityChain:
    // Empty fields never match a condition.
    if (!message.HasActivity())
      return false;
    return ::regexec(&m_regex, message.GetActivityChain().c_str(), 0, nullptr,
                     0) == 0;
  case eFilterTargetCategory:
    // Empty fields never match a condition.
    if (!message.HasCategory())
      return false;
    return ::regexec(&m_regex, message.GetCategory(), 0, nullptr, 0) == 0;
  case eFilterTargetMessage: {
    const char *message_text = message.GetMessage();
    if (!message_text) {
      DNBLogThreadedIf(LOG_DARWIN_LOG,
                       "LogFilterRegex: regex "
                       "\"%s\" no match due to nullptr message.",
                       m_regex_text.c_str());
      return false;
    }

    bool match = ::regexec(&m_regex, message_text, 0, nullptr, 0) == 0;
    DNBLogThreadedIf(LOG_DARWIN_LOG, "LogFilterRegex: regex "
                                     "\"%s\" %s message \"%s\".",
                     m_regex_text.c_str(), match ? "matches" : "does not match",
                     message_text);
    return match;
  }
  case eFilterTargetSubsystem:
    // Empty fields never match a condition.
    if (!message.HasSubsystem())
      return false;
    return ::regexec(&m_regex, message.GetSubsystem(), 0, nullptr, 0) == 0;
  default:
    // We don't know this type.
    return false;
  }
}
