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

POSIXLimboStopInfo::~POSIXLimboStopInfo() {}

lldb::StopReason POSIXLimboStopInfo::GetStopReason() const {
  return lldb::eStopReasonThreadExiting;
}

const char *POSIXLimboStopInfo::GetDescription() { return "thread exiting"; }

bool POSIXLimboStopInfo::ShouldStop(Event *event_ptr) { return false; }

bool POSIXLimboStopInfo::ShouldNotify(Event *event_ptr) { return false; }

//===----------------------------------------------------------------------===//
// POSIXCrashStopInfo

POSIXCrashStopInfo::POSIXCrashStopInfo(FreeBSDThread &thread, uint32_t status,
                                       CrashReason reason,
                                       lldb::addr_t fault_addr)
    : POSIXStopInfo(thread, status) {
  m_description = ::GetCrashReasonString(reason, fault_addr);
}

POSIXCrashStopInfo::~POSIXCrashStopInfo() {}

lldb::StopReason POSIXCrashStopInfo::GetStopReason() const {
  return lldb::eStopReasonException;
}

//===----------------------------------------------------------------------===//
// POSIXNewThreadStopInfo

POSIXNewThreadStopInfo::~POSIXNewThreadStopInfo() {}

lldb::StopReason POSIXNewThreadStopInfo::GetStopReason() const {
  return lldb::eStopReasonNone;
}

const char *POSIXNewThreadStopInfo::GetDescription() {
  return "thread spawned";
}

bool POSIXNewThreadStopInfo::ShouldStop(Event *event_ptr) { return false; }

bool POSIXNewThreadStopInfo::ShouldNotify(Event *event_ptr) { return false; }
