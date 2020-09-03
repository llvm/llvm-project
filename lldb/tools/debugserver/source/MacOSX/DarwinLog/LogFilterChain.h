//===-- LogFilterChain.h ----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_TOOLS_DEBUGSERVER_SOURCE_MACOSX_DARWINLOG_LOGFILTERCHAIN_H
#define LLDB_TOOLS_DEBUGSERVER_SOURCE_MACOSX_DARWINLOG_LOGFILTERCHAIN_H

#include <vector>

#include "DarwinLogInterfaces.h"

class LogFilterChain {
public:
  LogFilterChain(bool default_accept);

  void AppendFilter(const LogFilterSP &filter_sp);

  void ClearFilterChain();

  bool GetDefaultAccepts() const;

  void SetDefaultAccepts(bool default_accepts);

  bool GetAcceptMessage(const LogMessage &message) const;

private:
  using FilterVector = std::vector<LogFilterSP>;

  FilterVector m_filters;
  bool m_default_accept;
};

#endif // LLDB_TOOLS_DEBUGSERVER_SOURCE_MACOSX_DARWINLOG_LOGFILTERCHAIN_H
