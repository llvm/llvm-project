//===-- LogFilterRegex.h ----------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LogFilterRegex_h
#define LogFilterRegex_h

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

#endif /* LogFilterSubsystemRegex_hpp */
