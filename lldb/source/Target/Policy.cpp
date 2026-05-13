//===-- Policy.cpp --------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Target/Policy.h"
#include "lldb/Utility/LLDBLog.h"
#include "lldb/Utility/Log.h"
#include "lldb/Utility/Stream.h"
#include "lldb/Utility/StreamString.h"

using namespace lldb_private;

Policy PolicyStack::Current() const {
  Policy p = m_stack.back();
  if (Log *log = GetLog(LLDBLog::Process)) {
    StreamString s;
    p.Dump(s);
    LLDB_LOG(log, "{0}", s.GetData());
  }
  return p;
}

void Policy::Dump(Stream &s) const {
  s << "policy: view=" << (view == View::Public ? "public" : "private");
  s << ", capabilities={";
  s << "eval_expr=" << capabilities.can_evaluate_expressions;
  s << " run_all=" << capabilities.can_run_all_threads;
  s << " try_all=" << capabilities.can_try_all_threads;
  s << " bp_actions=" << capabilities.can_run_breakpoint_actions;
  s << " frame_providers=" << capabilities.can_load_frame_providers;
  s << " frame_recognizers=" << capabilities.can_run_frame_recognizers;
  s << '}';
}

void PolicyStack::Dump(Stream &s) const {
  s.Printf("PolicyStack depth=%zu\n", m_stack.size());
  for (size_t i = 0; i < m_stack.size(); i++) {
    s.Printf("  [%zu] ", i);
    m_stack[i].Dump(s);
    s << '\n';
  }
}
