//===-- Policy.cpp --------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Utility/Policy.h"
#include "lldb/Utility/LLDBLog.h"
#include "lldb/Utility/Log.h"
#include "lldb/Utility/Stream.h"
#include "lldb/Utility/StreamString.h"
#include "llvm/Support/ErrorHandling.h"

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

// CreatePublicState is the baseline, not a transition. The stack returns to
// public state by popping the private-state guards, not by pushing a
// "public" policy on top. This factory exists only as a reference value
// (tests, dump comparisons); it never reads the current stack.
Policy Policy::CreatePublicState() { return {}; }

Policy Policy::CreatePrivateState(PrivateStatePurpose purpose) {
  Policy p = PolicyStack::Get().Current();
  p.view = View::Private;
  switch (purpose) {
  case PrivateStatePurpose::Default:
    break;
  case PrivateStatePurpose::RunningExpression:
    p.capabilities.can_load_frame_providers = false;
    p.capabilities.can_run_frame_recognizers = false;
    break;
  }
  return p;
}

Policy Policy::CreatePublicStateRunningExpression() {
  Policy p = PolicyStack::Get().Current();
  p.capabilities.can_run_breakpoint_actions = false;
  return p;
}

PolicyStack::Guard::~Guard() {
  if (!m_active)
    return;
  if (m_thread_id != std::this_thread::get_id())
    llvm::report_fatal_error(
        "PolicyStack::Guard destroyed on a different thread than the one "
        "that created it");
  Get().Pop();
}

PolicyStack::Guard::Guard(Guard &&other)
    : m_thread_id(other.m_thread_id), m_active(other.m_active) {
  if (m_active && m_thread_id != std::this_thread::get_id())
    llvm::report_fatal_error("PolicyStack::Guard moved across threads");
  other.m_active = false;
}

PolicyStack::Guard &PolicyStack::Guard::operator=(Guard &&other) {
  if (this != &other) {
    if (other.m_active && other.m_thread_id != std::this_thread::get_id())
      llvm::report_fatal_error("PolicyStack::Guard moved across threads");
    if (m_active) {
      if (m_thread_id != std::this_thread::get_id())
        llvm::report_fatal_error(
            "PolicyStack::Guard destroyed on a different thread than the "
            "one that created it");
      Get().Pop();
    }
    m_thread_id = other.m_thread_id;
    m_active = other.m_active;
    other.m_active = false;
  }
  return *this;
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
