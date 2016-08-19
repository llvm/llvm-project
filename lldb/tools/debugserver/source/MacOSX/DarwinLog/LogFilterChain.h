//===-- LogFilterChain.h ----------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//


#ifndef LogFilterChain_h
#define LogFilterChain_h

#include <vector>

#include "DarwinLogInterfaces.h"

class LogFilterChain
{
public:

    LogFilterChain(bool default_accept);

    void
    AppendFilter(const LogFilterSP &filter_sp);

    void
    ClearFilterChain();

    bool
    GetDefaultAccepts() const;

    void
    SetDefaultAccepts(bool default_accepts);

    bool
    GetAcceptMessage(const LogMessage &message) const;

private:

    using FilterVector = std::vector<LogFilterSP>;

    FilterVector m_filters;
    bool m_default_accept;

};

#endif /* LogFilterChain_hpp */
