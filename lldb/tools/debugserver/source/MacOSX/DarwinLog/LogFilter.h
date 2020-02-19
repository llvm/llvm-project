//===-- LogFilter.h ---------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_TOOLS_DEBUGSERVER_SOURCE_MACOSX_DARWINLOG_LOGFILTER_H
#define LLDB_TOOLS_DEBUGSERVER_SOURCE_MACOSX_DARWINLOG_LOGFILTER_H

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

#endif // LLDB_TOOLS_DEBUGSERVER_SOURCE_MACOSX_DARWINLOG_LOGFILTER_H
