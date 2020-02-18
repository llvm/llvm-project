//===-- LogFilterRegex.h ----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_TOOLS_DEBUGSERVER_SOURCE_MACOSX_DARWINLOG_LOGFILTERREGEX_H
#define LLDB_TOOLS_DEBUGSERVER_SOURCE_MACOSX_DARWINLOG_LOGFILTERREGEX_H

// C includes
#include <regex.h>

// C++ includes
#include <string>

#include "DarwinLogInterfaces.h"
#include "DarwinLogTypes.h"
#include "LogFilter.h"

class LogFilterRegex : public LogFilter {
public:
  LogFilterRegex(bool match_accepts, FilterTarget filter_target,
                 const std::string &regex);

  virtual ~LogFilterRegex();

  bool IsValid() const { return m_is_valid; }

  const char *GetErrorAsCString() const { return m_error_text.c_str(); }

  bool DoesMatch(const LogMessage &message) const override;

private:
  const FilterTarget m_filter_target;
  const std::string m_regex_text;
  regex_t m_regex;
  bool m_is_valid;
  std::string m_error_text;
};

#endif // LLDB_TOOLS_DEBUGSERVER_SOURCE_MACOSX_DARWINLOG_LOGFILTERREGEX_H
