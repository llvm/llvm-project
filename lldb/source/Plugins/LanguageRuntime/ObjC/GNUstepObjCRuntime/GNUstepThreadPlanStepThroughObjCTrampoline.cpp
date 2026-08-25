//===-- GNUstepThreadPlanStepThroughObjCTrampoline.cpp --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "GNUstepThreadPlanStepThroughObjCTrampoline.h"
#include "GNUstepObjCRuntime.h"

#include "lldb/Expression/DiagnosticManager.h"
#include "lldb/Expression/FunctionCaller.h"
#include "lldb/Target/ABI.h"
#include "lldb/Target/ExecutionContext.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/Target.h"
#include "lldb/Target/Thread.h"
#include "lldb/Target/ThreadPlanRunToAddress.h"
#include "lldb/Utility/LLDBLog.h"
#include "lldb/Utility/Log.h"
#include "lldb/Utility/Stream.h"

using namespace lldb;
using namespace lldb_private;

GNUstepThreadPlanStepThroughObjCTrampoline::
    GNUstepThreadPlanStepThroughObjCTrampoline(Thread &thread,
                                               GNUstepObjCRuntime &runtime,
                                               ValueList &input_values,
                                               lldb::addr_t isa_addr,
                                               lldb::addr_t sel_addr)
    : ThreadPlan(ThreadPlan::eKindGeneric,
                 "GNUstep step through ObjC trampoline", thread, eVoteNoOpinion,
                 eVoteNoOpinion),
      m_runtime(runtime), m_input_values(input_values), m_isa_addr(isa_addr),
      m_sel_addr(sel_addr) {}

GNUstepThreadPlanStepThroughObjCTrampoline::
    ~GNUstepThreadPlanStepThroughObjCTrampoline() = default;

void GNUstepThreadPlanStepThroughObjCTrampoline::DidPush() {
  // Setting up the called function might require allocations in the
  // inferior, i.e. a nested function call. This needs to be done as a
  // PreResumeAction.
  m_process.AddPreResumeAction(PreResumeInitializeFunctionCaller, (void *)this);
}

void GNUstepThreadPlanStepThroughObjCTrampoline::DidPop() {
  // The action holds a bare pointer to this plan, so it must not outlive it -
  // the plan can be discarded before the process ever resumes.
  m_process.ClearPreResumeAction(PreResumeInitializeFunctionCaller,
                                 (void *)this);
}

bool GNUstepThreadPlanStepThroughObjCTrampoline::
    PreResumeInitializeFunctionCaller(void *void_myself) {
  auto *myself =
      static_cast<GNUstepThreadPlanStepThroughObjCTrampoline *>(void_myself);
  return myself->InitializeFunctionCaller();
}

bool GNUstepThreadPlanStepThroughObjCTrampoline::InitializeFunctionCaller() {
  if (m_func_sp)
    return true;

  m_lookup_function = m_runtime.GetMsgLookupFunctionCaller(GetThread());
  if (!m_lookup_function)
    return false;

  ExecutionContext exe_ctx;
  GetThread().CalculateExecutionContext(exe_ctx);

  // The wrapper was already compiled into the inferior when the caller was
  // built (GetMsgLookupFunctionCaller); only write a fresh argument struct
  // here. m_args_addr starts invalid so WriteFunctionArguments allocates one.
  DiagnosticManager diagnostics;
  m_args_addr = LLDB_INVALID_ADDRESS;
  if (!m_lookup_function->WriteFunctionArguments(exe_ctx, m_args_addr,
                                                 m_input_values, diagnostics))
    return false;

  EvaluateExpressionOptions options;
  options.SetUnwindOnError(true);
  options.SetIgnoreBreakpoints(true);
  options.SetStopOthers(false);

  m_func_sp = m_lookup_function->GetThreadPlanToCallFunction(
      exe_ctx, m_args_addr, options, diagnostics);
  if (!m_func_sp)
    return false;
  m_func_sp->SetOkayToDiscard(true);
  PushPlan(m_func_sp);
  return true;
}

void GNUstepThreadPlanStepThroughObjCTrampoline::GetDescription(
    Stream *s, lldb::DescriptionLevel level) {
  if (level == lldb::eDescriptionLevelBrief) {
    s->Printf("Step through GNUstep ObjC trampoline");
    return;
  }
  s->Printf("Stepping to implementation of ObjC method - obj: 0x%" PRIx64
            ", isa: 0x%" PRIx64 ", sel: 0x%" PRIx64,
            static_cast<uint64_t>(
                m_input_values.GetValueAtIndex(0)->GetScalar().ULongLong()),
            m_isa_addr, m_sel_addr);
}

bool GNUstepThreadPlanStepThroughObjCTrampoline::ShouldStop(Event *event_ptr) {
  // First stage: the nested "call objc_msg_lookup" plan is still running.
  if (m_func_sp) {
    if (!m_func_sp->IsPlanComplete())
      return false;
    if (!m_func_sp->PlanSucceeded()) {
      SetPlanComplete(false);
      return true;
    }
    m_func_sp.reset();
  }

  Log *log = GetLog(LLDBLog::Step);

  // Setting up the call can fail after the plan is already on the stack, in
  // which case there is nothing to collect a result from.
  if (!m_lookup_function || m_args_addr == LLDB_INVALID_ADDRESS) {
    LLDB_LOG(log, "objc_msg_lookup call was never set up, stopping.");
    SetPlanComplete(false);
    return true;
  }

  // Second stage: fetch the IMP the lookup returned and run to it.
  if (!m_run_to_sp) {
    Value target_addr_value;
    ExecutionContext exe_ctx;
    GetThread().CalculateExecutionContext(exe_ctx);
    m_lookup_function->FetchFunctionResults(exe_ctx, m_args_addr,
                                            target_addr_value);
    m_lookup_function->DeallocateFunctionResults(exe_ctx, m_args_addr);
    lldb::addr_t target_addr = target_addr_value.GetScalar().ULongLong();

    if (ABISP abi_sp = GetThread().GetProcess()->GetABI())
      target_addr = abi_sp->FixCodeAddress(target_addr);

    if (target_addr == 0 || target_addr == LLDB_INVALID_ADDRESS) {
      LLDB_LOG(log, "objc_msg_lookup returned {0:x}, stopping.", target_addr);
      SetPlanComplete();
      return true;
    }

    // A selector the class does not implement resolves to the runtime's
    // forwarding machinery, which lives inside libobjc itself - as do the
    // runtime's own internal method implementations. There is no user code to
    // step into in either case, so stop here instead.
    if (m_runtime.IsRuntimeInternalAddress(target_addr)) {
      LLDB_LOG(log, "objc_msg_lookup resolved into the runtime itself "
                    "(forwarding or an internal method), stopping.");
      SetPlanComplete();
      return true;
    }

    LLDB_LOG(log, "Running to GNUstep ObjC method implementation: {0:x}",
             target_addr);

    if (m_isa_addr != LLDB_INVALID_ADDRESS &&
        m_sel_addr != LLDB_INVALID_ADDRESS)
      m_runtime.AddToMethodCache(m_isa_addr, m_sel_addr, target_addr);

    Address target_so_addr;
    target_so_addr.SetOpcodeLoadAddress(target_addr, exe_ctx.GetTargetPtr());
    m_run_to_sp = std::make_shared<ThreadPlanRunToAddress>(
        GetThread(), target_so_addr, false);
    PushPlan(m_run_to_sp);
    return false;
  }

  // Third stage: wait for the run-to-implementation plan.
  if (GetThread().IsThreadPlanDone(m_run_to_sp.get())) {
    SetPlanComplete();
    return true;
  }
  return false;
}
