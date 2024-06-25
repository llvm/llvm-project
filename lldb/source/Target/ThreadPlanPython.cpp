//===-- ThreadPlanPython.cpp ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Target/ThreadPlan.h"

#include "lldb/Core/Debugger.h"
#include "lldb/Interpreter/CommandInterpreter.h"
#include "lldb/Interpreter/ScriptInterpreter.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/RegisterContext.h"
#include "lldb/Target/Target.h"
#include "lldb/Target/Thread.h"
#include "lldb/Target/ThreadPlan.h"
#include "lldb/Target/ThreadPlanPython.h"
#include "lldb/Utility/LLDBLog.h"
#include "lldb/Utility/Log.h"
#include "lldb/Utility/State.h"

using namespace lldb;
using namespace lldb_private;

// ThreadPlanPython

ThreadPlanPython::ThreadPlanPython(Thread &thread, const char *class_name,
                                   const StructuredDataImpl &args_data)
    : ThreadPlan(ThreadPlan::eKindPython, "Python based Thread Plan", thread,
                 eVoteNoOpinion, eVoteNoOpinion),
      m_class_name(class_name), m_args_data(args_data), m_did_push(false),
      m_stop_others(false) {
  SetIsControllingPlan(true);
  SetOkayToDiscard(true);
  SetPrivate(false);
}

bool ThreadPlanPython::ValidatePlan(Stream *error) {
  if (!m_did_push)
    return true;

  if (!m_implementation_sp) {
    if (error)
      error->Printf("Error constructing Python ThreadPlan: %s",
          m_error_str.empty() ? "<unknown error>"
                                : m_error_str.c_str());
    return false;
  }

  return true;
}

ScriptInterpreter *ThreadPlanPython::GetScriptInterpreter() {
  return m_process.GetTarget().GetDebugger().GetScriptInterpreter();
}

void ThreadPlanPython::DidPush() {
  // We set up the script side in DidPush, so that it can push other plans in
  // the constructor, and doesn't have to care about the details of DidPush.
  m_did_push = true;
  if (!m_class_name.empty()) {
    ScriptInterpreter *script_interp = GetScriptInterpreter();
    if (script_interp) {
      m_implementation_sp = script_interp->CreateScriptedThreadPlan(
          m_class_name.c_str(), m_args_data, m_error_str, 
          this->shared_from_this());
    }
  }
}

bool ThreadPlanPython::ShouldStop(Event *event_ptr) {
  Log *log = GetLog(LLDBLog::Thread);
  LLDB_LOGF(log, "%s called on Python Thread Plan: %s )", LLVM_PRETTY_FUNCTION,
            m_class_name.c_str());

  bool should_stop = true;
  if (m_implementation_sp) {
    ScriptInterpreter *script_interp = GetScriptInterpreter();
    if (script_interp) {
      bool script_error;
      should_stop = script_interp->ScriptedThreadPlanShouldStop(
          m_implementation_sp, event_ptr, script_error);
      if (script_error)
        SetPlanComplete(false);
    }
  }
  return should_stop;
}

bool ThreadPlanPython::IsPlanStale() {
  Log *log = GetLog(LLDBLog::Thread);
  LLDB_LOGF(log, "%s called on Python Thread Plan: %s )", LLVM_PRETTY_FUNCTION,
            m_class_name.c_str());

  bool is_stale = true;
  if (m_implementation_sp) {
    ScriptInterpreter *script_interp = GetScriptInterpreter();
    if (script_interp) {
      bool script_error;
      is_stale = script_interp->ScriptedThreadPlanIsStale(m_implementation_sp,
                                                          script_error);
      if (script_error)
        SetPlanComplete(false);
    }
  }
  return is_stale;
}

bool ThreadPlanPython::DoPlanExplainsStop(Event *event_ptr) {
  Log *log = GetLog(LLDBLog::Thread);
  LLDB_LOGF(log, "%s called on Python Thread Plan: %s )", LLVM_PRETTY_FUNCTION,
            m_class_name.c_str());

  bool explains_stop = true;
  if (m_implementation_sp) {
    ScriptInterpreter *script_interp = GetScriptInterpreter();
    if (script_interp) {
      bool script_error;
      explains_stop = script_interp->ScriptedThreadPlanExplainsStop(
          m_implementation_sp, event_ptr, script_error);
      if (script_error)
        SetPlanComplete(false);
    }
  }
  return explains_stop;
}

bool ThreadPlanPython::MischiefManaged() {
  Log *log = GetLog(LLDBLog::Thread);
  LLDB_LOGF(log, "%s called on Python Thread Plan: %s )", LLVM_PRETTY_FUNCTION,
            m_class_name.c_str());
  bool mischief_managed = true;
  if (m_implementation_sp) {
    // I don't really need mischief_managed, since it's simpler to just call
    // SetPlanComplete in should_stop.
    mischief_managed = IsPlanComplete();
    if (mischief_managed) {
      // We need to cache the stop reason here we'll need it in GetDescription.
      GetDescription(&m_stop_description, eDescriptionLevelBrief);
      m_implementation_sp.reset();
    }
  }
  return mischief_managed;
}

lldb::StateType ThreadPlanPython::GetPlanRunState() {
  Log *log = GetLog(LLDBLog::Thread);
  LLDB_LOGF(log, "%s called on Python Thread Plan: %s )", LLVM_PRETTY_FUNCTION,
            m_class_name.c_str());
  lldb::StateType run_state = eStateRunning;
  if (m_implementation_sp) {
    ScriptInterpreter *script_interp = GetScriptInterpreter();
    if (script_interp) {
      bool script_error;
      run_state = script_interp->ScriptedThreadPlanGetRunState(
          m_implementation_sp, script_error);
    }
  }
  return run_state;
}

void ThreadPlanPython::GetDescription(Stream *s, lldb::DescriptionLevel level) {
  Log *log = GetLog(LLDBLog::Thread);
  LLDB_LOGF(log, "%s called on Python Thread Plan: %s )", LLVM_PRETTY_FUNCTION,
            m_class_name.c_str());
  if (m_implementation_sp) {
    ScriptInterpreter *script_interp = GetScriptInterpreter();
    if (script_interp) {
      bool script_error;
      bool added_desc = script_interp->ScriptedThreadPlanGetStopDescription(
          m_implementation_sp, s, script_error);
      if (script_error || !added_desc)
        s->Printf("Python thread plan implemented by class %s.",
            m_class_name.c_str());
    }
    return;
  }
  // It's an error not to have a description, so if we get here, we should
  // add something.
  if (m_stop_description.Empty())
    s->Printf("Python thread plan implemented by class %s.",
              m_class_name.c_str());
  s->PutCString(m_stop_description.GetData());
}

// The ones below are not currently exported to Python.
bool ThreadPlanPython::WillStop() {
  Log *log = GetLog(LLDBLog::Thread);
  LLDB_LOGF(log, "%s called on Python Thread Plan: %s )", LLVM_PRETTY_FUNCTION,
            m_class_name.c_str());
  return true;
}

bool ThreadPlanPython::DoWillResume(lldb::StateType resume_state, 
                                  bool current_plan) {
  m_stop_description.Clear();
  return true;                                  
}
