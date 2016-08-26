//===-- LogFilterChain.cpp --------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "LogFilterChain.h"

#include "LogFilter.h"

LogFilterChain::LogFilterChain(bool default_accept) :
    m_filters(),
    m_default_accept(default_accept)
{
}

void
LogFilterChain::AppendFilter(const LogFilterSP &filter_sp)
{
    if (filter_sp)
        m_filters.push_back(filter_sp);
}

void
LogFilterChain::ClearFilterChain()
{
    m_filters.clear();
}

bool
LogFilterChain::GetDefaultAccepts() const
{
    return m_default_accept;
}

void
LogFilterChain::SetDefaultAccepts(bool default_accept)
{
    m_default_accept = default_accept;
}

bool
LogFilterChain::GetAcceptMessage(const LogMessage &message) const
{
    for (auto filter_sp : m_filters)
    {
        if (filter_sp->DoesMatch(message))
        {
            // This message matches this filter.  If the filter accepts matches,
            // this message matches; otherwise, it rejects matches.
            return filter_sp->MatchesAreAccepted();
        }
    }

    // None of the filters matched.  Therefore, we do whatever the
    // default fall-through rule says.
    return m_default_accept;
}
