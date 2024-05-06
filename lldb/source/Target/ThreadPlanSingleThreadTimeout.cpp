//===-- ThreadPlanStepOverRange.cpp ---------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Target/ThreadPlanSingleThreadTimeout.h"
#include "lldb/Symbol/Block.h"
#include "lldb/Symbol/CompileUnit.h"
#include "lldb/Symbol/Function.h"
#include "lldb/Symbol/LineTable.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/RegisterContext.h"
#include "lldb/Target/Target.h"
#include "lldb/Target/Thread.h"
#include "lldb/Target/ThreadPlanStepOut.h"
#include "lldb/Target/ThreadPlanStepThrough.h"
#include "lldb/Utility/LLDBLog.h"
#include "lldb/Utility/Log.h"
#include "lldb/Utility/Stream.h"

using namespace lldb_private;
using namespace lldb;

std::mutex ThreadPlanSingleThreadTimeout::s_mutex;
ThreadPlanSingleThreadTimeout *ThreadPlanSingleThreadTimeout::s_instance =
    nullptr;
ThreadPlanSingleThreadTimeout::State
    ThreadPlanSingleThreadTimeout::s_prev_state = State::WaitTimeout;

ThreadPlanSingleThreadTimeout::ThreadPlanSingleThreadTimeout(Thread &thread)
    : ThreadPlan(ThreadPlan::eKindSingleThreadTimeout, "Single thread timeout",
                 thread, eVoteNo, eVoteNoOpinion),
      m_state(State::WaitTimeout), m_exit_flag(false) {
  std::lock_guard<std::mutex> lock(s_mutex);
  m_timer_thread = std::thread(TimeoutThreadFunc, this);
  s_instance = this;
  m_state = s_prev_state;
}

ThreadPlanSingleThreadTimeout::~ThreadPlanSingleThreadTimeout() {
  std::lock_guard<std::mutex> lock(s_mutex);
  s_instance = nullptr;
  if (m_state == State::Done)
    m_state = State::WaitTimeout;
  s_prev_state = m_state;
}

void ThreadPlanSingleThreadTimeout::GetDescription(
    Stream *s, lldb::DescriptionLevel level) {
  s->Printf("Single thread timeout, state(%s)", StateToString(m_state).c_str());
}

std::string ThreadPlanSingleThreadTimeout::StateToString(State state) {
  switch (state) {
  case State::WaitTimeout:
    return "WaitTimeout";
  case State::AsyncInterrupt:
    return "AsyncInterrupt";
  case State::Done:
    return "Done";
  }
}

void ThreadPlanSingleThreadTimeout::CreateNew(Thread &thread) {
  uint64_t timeout_in_ms = thread.GetSingleThreadPlanTimeout();
  if (timeout_in_ms == 0)
    return;

  // Do not create timeout if we are not stopping other threads.
  if (!thread.GetCurrentPlan()->StopOthers())
    return;

  if (ThreadPlanSingleThreadTimeout::IsAlive())
    return;
  {
    std::lock_guard<std::mutex> lock(s_mutex);
    s_prev_state = State::WaitTimeout;
  }
  auto timeout_plan = new ThreadPlanSingleThreadTimeout(thread);
  ThreadPlanSP thread_plan_sp(timeout_plan);
  auto status = thread.QueueThreadPlan(thread_plan_sp,
                                       /*abort_other_plans*/ false);
  Log *log = GetLog(LLDBLog::Step);
  LLDB_LOGF(log, "ThreadPlanSingleThreadTimeout pushing a brand new one");
}

void ThreadPlanSingleThreadTimeout::ResetFromPrevState(Thread &thread) {
  uint64_t timeout_in_ms = thread.GetSingleThreadPlanTimeout();
  if (timeout_in_ms == 0)
    return;

  if (ThreadPlanSingleThreadTimeout::IsAlive())
    return;

  // Do not create timeout if we are not stopping other threads.
  if (!thread.GetCurrentPlan()->StopOthers())
    return;

  auto timeout_plan = new ThreadPlanSingleThreadTimeout(thread);
  ThreadPlanSP thread_plan_sp(timeout_plan);
  auto status = thread.QueueThreadPlan(thread_plan_sp,
                                       /*abort_other_plans*/ false);
  Log *log = GetLog(LLDBLog::Step);
  LLDB_LOGF(log, "ThreadPlanSingleThreadTimeout reset from previous state");
}

bool ThreadPlanSingleThreadTimeout::WillStop() {
  Log *log = GetLog(LLDBLog::Step);
  LLDB_LOGF(log, "ThreadPlanSingleThreadTimeout::WillStop().");

  // Reset the state during stop.
  std::lock_guard<std::mutex> lock(s_mutex);
  s_prev_state = State::WaitTimeout;
  return true;
}

void ThreadPlanSingleThreadTimeout::DidPop() {
  Log *log = GetLog(LLDBLog::Step);
  {
    std::lock_guard<std::mutex> lock(m_mutex);
    LLDB_LOGF(log, "ThreadPlanSingleThreadTimeout::DidPop().");
    // Tell timer thread to exit.
    m_exit_flag = true;
  }
  m_wakeup_cv.notify_one();
  // Wait for timer thread to exit.
  m_timer_thread.join();
}

bool ThreadPlanSingleThreadTimeout::DoPlanExplainsStop(Event *event_ptr) {
  lldb::StateType stop_state =
      Process::ProcessEventData::GetStateFromEvent(event_ptr);
  Log *log = GetLog(LLDBLog::Step);
  LLDB_LOGF(
      log,
      "ThreadPlanSingleThreadTimeout::DoPlanExplainsStop(): got event: %s.",
      StateAsCString(stop_state));
  return true;
}

lldb::StateType ThreadPlanSingleThreadTimeout::GetPlanRunState() {
  return GetPreviousPlan()->GetPlanRunState();
}

void ThreadPlanSingleThreadTimeout::TimeoutThreadFunc(
    ThreadPlanSingleThreadTimeout *self) {
  std::unique_lock<std::mutex> lock(self->m_mutex);
  uint64_t timeout_in_ms = self->GetThread().GetSingleThreadPlanTimeout();
  self->m_wakeup_cv.wait_for(lock, std::chrono::milliseconds(timeout_in_ms),
                             [self] { return self->m_exit_flag; });

  Log *log = GetLog(LLDBLog::Step);
  LLDB_LOGF(log,
            "ThreadPlanSingleThreadTimeout::TimeoutThreadFunc() called with "
            "m_exit_flag(%d).",
            self->m_exit_flag);
  if (self->m_exit_flag)
    return;

  self->HandleTimeout();
}

bool ThreadPlanSingleThreadTimeout::MischiefManaged() {
  Log *log = GetLog(LLDBLog::Step);
  LLDB_LOGF(log, "ThreadPlanSingleThreadTimeout::MischiefManaged() called.");
  // Need to reset timer on each internal stop/execution progress.
  return true;
}

bool ThreadPlanSingleThreadTimeout::ShouldStop(Event *event_ptr) {
  return HandleEvent(event_ptr);
}

void ThreadPlanSingleThreadTimeout::SetStopOthers(bool new_value) {
  GetPreviousPlan()->SetStopOthers(new_value);
}

bool ThreadPlanSingleThreadTimeout::StopOthers() {
  if (m_state == State::Done)
    return false;
  else
    return GetPreviousPlan()->StopOthers();
}

bool ThreadPlanSingleThreadTimeout::HandleEvent(Event *event_ptr) {
  lldb::StateType stop_state =
      Process::ProcessEventData::GetStateFromEvent(event_ptr);
  Log *log = GetLog(LLDBLog::Step);
  LLDB_LOGF(log, "ThreadPlanSingleThreadTimeout::HandleEvent(): got event: %s.",
            StateAsCString(stop_state));

  lldb::StopInfoSP stop_info = GetThread().GetStopInfo();
  if (m_state == State::AsyncInterrupt && stop_state == lldb::eStateStopped &&
      stop_info && stop_info->GetStopReason() == lldb::eStopReasonInterrupt) {
    if (Process::ProcessEventData::GetRestartedFromEvent(event_ptr)) {
      // If we were restarted, we just need to go back up to fetch
      // another event.
      LLDB_LOGF(log,
                "ThreadPlanSingleThreadTimeout::HandleEvent(): Got a stop and "
                "restart, so we'll continue waiting.");

    } else {
      LLDB_LOGF(
          log,
          "ThreadPlanSingleThreadTimeout::HandleEvent(): Got async interrupt "
          ", so we will resume all threads.");
      GetThread().GetCurrentPlan()->SetStopOthers(false);
      GetPreviousPlan()->SetStopOthers(false);
      m_state = State::Done;
    }
  }
  // Should not report stop.
  return false;
}

void ThreadPlanSingleThreadTimeout::HandleTimeout() {
  Log *log = GetLog(LLDBLog::Step);
  LLDB_LOGF(
      log,
      "ThreadPlanSingleThreadTimeout::HandleTimeout() send async interrupt.");
  m_state = State::AsyncInterrupt;

  // Private state thread will only send async interrupt
  // in running state so no need to check state here.
  m_process.SendAsyncInterrupt(&GetThread());
}

bool ThreadPlanSingleThreadTimeout::IsAlive() {
  std::lock_guard<std::mutex> lock(s_mutex);
  return s_instance != nullptr;
}
