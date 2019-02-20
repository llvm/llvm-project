//===-- POSIXStopInfo.cpp ---------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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
