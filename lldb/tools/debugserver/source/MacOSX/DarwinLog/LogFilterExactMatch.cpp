//===-- LogFilterExactMatch.cpp ---------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "LogFilterExactMatch.h"
#include "LogMessage.h"

LogFilterExactMatch::LogFilterExactMatch(bool match_accepts,
                                         FilterTarget filter_target,
                                         const std::string &match_text)
    : LogFilter(match_accepts), m_filter_target(filter_target),
      m_match_text(match_text) {}

bool LogFilterExactMatch::DoesMatch(const LogMessage &message) const {
  switch (m_filter_target) {
  case eFilterTargetActivity:
    // Empty fields never match a condition.
    if (!message.HasActivity())
      return false;
    return m_match_text == message.GetActivity();
  case eFilterTargetActivityChain:
    // Empty fields never match a condition.
    if (!message.HasActivity())
      return false;
    return m_match_text == message.GetActivityChain();
  case eFilterTargetCategory:
    // Empty fields never match a condition.
    if (!message.HasCategory())
      return false;
    return m_match_text == message.GetCategory();
  case eFilterTargetMessage: {
    const char *message_text = message.GetMessage();
    return (message_text != nullptr) && (m_match_text == message_text);
  }
  case eFilterTargetSubsystem:
    // Empty fields never match a condition.
    if (!message.HasSubsystem())
      return false;
    return m_match_text == message.GetSubsystem();
  default:
    // We don't know this type.
    return false;
  }
}
