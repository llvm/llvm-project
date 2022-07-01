//===-- TraceCursor.cpp -----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Target/TraceCursor.h"

#include "lldb/Target/ExecutionContext.h"

using namespace lldb;
using namespace lldb_private;
using namespace llvm;

TraceCursor::TraceCursor(lldb::ThreadSP thread_sp)
    : m_exe_ctx_ref(ExecutionContext(thread_sp)) {}

ExecutionContextRef &TraceCursor::GetExecutionContextRef() {
  return m_exe_ctx_ref;
}

void TraceCursor::SetForwards(bool forwards) { m_forwards = forwards; }

bool TraceCursor::IsForwards() const { return m_forwards; }

const char *trace_event_utils::EventToDisplayString(lldb::TraceEvents event) {
  switch (event) {
  case lldb::eTraceEventPaused:
    return "paused";
  }
  return nullptr;
}

void trace_event_utils::ForEachEvent(
    lldb::TraceEvents events,
    std::function<void(lldb::TraceEvents event)> callback) {
  while (events) {
    TraceEvents event = static_cast<TraceEvents>(events & ~(events - 1));
    callback(event);
    events &= ~event;
  }
}
