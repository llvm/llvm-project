//===-- LogFilterExactMatch.h -----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LogFilterExactMatch_h
#define LogFilterExactMatch_h

#include <string>

#include "DarwinLogInterfaces.h"
#include "DarwinLogTypes.h"
#include "LogFilter.h"

class LogFilterExactMatch : public LogFilter {
public:
  LogFilterExactMatch(bool match_accepts, FilterTarget filter_target,
                      const std::string &match_text);

  bool DoesMatch(const LogMessage &message) const override;

private:
  const FilterTarget m_filter_target;
  const std::string m_match_text;
};

#endif
