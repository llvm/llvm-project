//===-- LogFilterExactMatch.h -----------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LogFilterExactMatch_h
#define LogFilterExactMatch_h

#include <string>

#include "DarwinLogInterfaces.h"
#include "DarwinLogTypes.h"
#include "LogFilter.h"

class LogFilterExactMatch : public LogFilter
{
public:

    LogFilterExactMatch(bool match_accepts, FilterTarget filter_target,
                        const std::string &match_text);

    bool
    DoesMatch(const LogMessage &message) const override;

private:

    const FilterTarget m_filter_target;
    const std::string m_match_text;

};

#endif
