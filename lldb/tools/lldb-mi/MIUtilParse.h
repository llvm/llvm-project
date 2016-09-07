//===-- MIUtilParse.h ------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#pragma once

// Third party headers:
#include "../lib/Support/regex_impl.h"

// In-house headers:
#include "MIUtilString.h"

namespace MIUtilParse {

//++
//============================================================================
// Details: MI common code utility class. Used to parse the output
//          returned from lldb commands using regex.
//--
class CRegexParser {
public:
  // Helper class for keeping track of regex matches.
  class Match {
    friend CRegexParser;

  public:
    /* ctor */ explicit Match(size_t nmatches)
        : m_matchStrs(nmatches), m_maxMatches(nmatches) {}
    size_t GetMatchCount() const { return m_matchStrs.size(); }
    CMIUtilString GetMatchAtIndex(size_t i) const {
      if (m_matchStrs.size() > i)
        return m_matchStrs[i];
      return CMIUtilString();
    }

  private:
    CMIUtilString::VecString_t m_matchStrs;
    const size_t m_maxMatches;
  };

  // Methods:
  // Compile the regular expression.
  /* ctor */ explicit CRegexParser(const char *regexStr);

  // Free the memory used by the regular expression.
  /* dtor */ ~CRegexParser();

  // No copies
  CRegexParser(const CRegexParser &) = delete;
  void operator=(CRegexParser &) = delete;

  // Return the match at the index.
  int GetMatchCount(const Match &match) const {
    if (m_isValid)
      return match.GetMatchCount();
    return 0;
  }

  bool IsValid() const { return m_isValid; }

  // Match the input against the regular expression.  Return an error
  // if the number of matches is less than minMatches.  If the default
  // minMatches value of 0 is passed, an error will be returned if
  // the number of matches is less than the maxMatches value used to
  // initialize Match.
  bool Execute(const char *input, Match &match, size_t minMatches = 0);

private:
  llvm_regex_t m_emma;
  const bool m_isValid;
};
}
