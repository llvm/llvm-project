//===-- LogFilter.h ---------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LogFilter_h
#define LogFilter_h

#include "DarwinLogInterfaces.h"

class LogFilter {
public:
  virtual ~LogFilter();

  virtual bool DoesMatch(const LogMessage &message) const = 0;

  bool MatchesAreAccepted() const { return m_matches_accept; }

protected:
  LogFilter(bool matches_accept) : m_matches_accept(matches_accept) {}

private:
  bool m_matches_accept;
};

#endif /* LogFilter_h */
