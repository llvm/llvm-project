//===-- POSIXStopInfo.cpp ---------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "POSIXStopInfo.h"

using namespace lldb;
using namespace lldb_private;


//===----------------------------------------------------------------------===//
// POSIXLimboStopInfo

POSIXLimboStopInfo::~POSIXLimboStopInfo() { }

lldb::StopReason
POSIXLimboStopInfo::GetStopReason() const
{
    return lldb::eStopReasonTrace;
}

const char *
POSIXLimboStopInfo::GetDescription()
{
    return "thread exiting";
}

bool
POSIXLimboStopInfo::ShouldStop(Event *event_ptr)
{
    return true;
}

bool
POSIXLimboStopInfo::ShouldNotify(Event *event_ptr)
{
    return true;
}

//===----------------------------------------------------------------------===//
// POSIXCrashStopInfo

POSIXCrashStopInfo::~POSIXCrashStopInfo() { }

lldb::StopReason
POSIXCrashStopInfo::GetStopReason() const
{
    return lldb::eStopReasonException;
}

const char *
POSIXCrashStopInfo::GetDescription()
{
    return ProcessMessage::GetCrashReasonString(m_crash_reason);
}
