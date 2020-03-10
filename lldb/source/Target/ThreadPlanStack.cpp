//===-- ThreadPlanStack.cpp -------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Target/ThreadPlanStack.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/Target.h"
#include "lldb/Target/Thread.h"
#include "lldb/Target/ThreadPlan.h"
#include "lldb/Utility/Log.h"

using namespace lldb;
using namespace lldb_private;

static void PrintPlanElement(Stream *s, const ThreadPlanSP &plan,
                             lldb::DescriptionLevel desc_level,
                             int32_t elem_idx) {
  s->IndentMore();
  s->Indent();
  s->Printf("Element %d: ", elem_idx);
  plan->GetDescription(s, desc_level);
  s->EOL();
  s->IndentLess();
}

void ThreadPlanStack::DumpThreadPlans(Stream *s,
                                      lldb::DescriptionLevel desc_level,
                                      bool include_internal) const {

  uint32_t stack_size;

  s->IndentMore();
  s->Indent();
  s->Printf("Active plan stack:\n");
  int32_t print_idx = 0;
  for (auto plan : m_plans) {
    PrintPlanElement(s, plan, desc_level, print_idx++);
  }

  if (AnyCompletedPlans()) {
    print_idx = 0;
    s->Indent();
    s->Printf("Completed Plan Stack:\n");
    for (auto plan : m_completed_plans)
      PrintPlanElement(s, plan, desc_level, print_idx++);
  }

  if (AnyDiscardedPlans()) {
    print_idx = 0;
    s->Indent();
    s->Printf("Discarded Plan Stack:\n");
    for (auto plan : m_discarded_plans)
      PrintPlanElement(s, plan, desc_level, print_idx++);
  }

  s->IndentLess();
}

size_t ThreadPlanStack::CheckpointCompletedPlans() {
  m_completed_plan_checkpoint++;
  m_completed_plan_store.insert(
      std::make_pair(m_completed_plan_checkpoint, m_completed_plans));
  return m_completed_plan_checkpoint;
}

void ThreadPlanStack::RestoreCompletedPlanCheckpoint(size_t checkpoint) {
  auto result = m_completed_plan_store.find(checkpoint);
  assert(result != m_completed_plan_store.end() &&
         "Asked for a checkpoint that didn't exist");
  m_completed_plans.swap((*result).second);
  m_completed_plan_store.erase(result);
}

void ThreadPlanStack::DiscardCompletedPlanCheckpoint(size_t checkpoint) {
  m_completed_plan_store.erase(checkpoint);
}

void ThreadPlanStack::ThreadDestroyed(Thread *thread) {
  // Tell the plan stacks that this thread is going away:
  for (ThreadPlanSP plan : m_plans)
    plan->ThreadDestroyed();

  for (ThreadPlanSP plan : m_discarded_plans)
    plan->ThreadDestroyed();

  for (ThreadPlanSP plan : m_completed_plans)
    plan->ThreadDestroyed();

  // Now clear the current plan stacks:
  m_plans.clear();
  m_discarded_plans.clear();
  m_completed_plans.clear();

  // Push a ThreadPlanNull on the plan stack.  That way we can continue
  // assuming that the plan stack is never empty, but if somebody errantly asks
  // questions of a destroyed thread without checking first whether it is
  // destroyed, they won't crash.
  if (thread != nullptr) {
    lldb::ThreadPlanSP null_plan_sp(new ThreadPlanNull(*thread));
    m_plans.push_back(null_plan_sp);
  }
}

void ThreadPlanStack::EnableTracer(bool value, bool single_stepping) {
  for (ThreadPlanSP plan : m_plans) {
    if (plan->GetThreadPlanTracer()) {
      plan->GetThreadPlanTracer()->EnableTracing(value);
      plan->GetThreadPlanTracer()->EnableSingleStep(single_stepping);
    }
  }
}

void ThreadPlanStack::SetTracer(lldb::ThreadPlanTracerSP &tracer_sp) {
  for (ThreadPlanSP plan : m_plans)
    plan->SetThreadPlanTracer(tracer_sp);
}

void ThreadPlanStack::PushPlan(lldb::ThreadPlanSP new_plan_sp) {
  // If the thread plan doesn't already have a tracer, give it its parent's
  // tracer:
  // The first plan has to be a base plan:
  assert(m_plans.size() > 0 ||
         new_plan_sp->IsBasePlan() && "Zeroth plan must be a base plan");

  if (!new_plan_sp->GetThreadPlanTracer()) {
    assert(!m_plans.empty());
    new_plan_sp->SetThreadPlanTracer(m_plans.back()->GetThreadPlanTracer());
  }
  m_plans.push_back(new_plan_sp);
  new_plan_sp->DidPush();
}

lldb::ThreadPlanSP ThreadPlanStack::PopPlan() {
  assert(m_plans.size() > 1 && "Can't pop the base thread plan");

  lldb::ThreadPlanSP &plan_sp = m_plans.back();
  m_completed_plans.push_back(plan_sp);
  plan_sp->WillPop();
  m_plans.pop_back();
  return plan_sp;
}

lldb::ThreadPlanSP ThreadPlanStack::DiscardPlan() {
  assert(m_plans.size() > 1 && "Can't discard the base thread plan");

  lldb::ThreadPlanSP &plan_sp = m_plans.back();
  m_discarded_plans.push_back(plan_sp);
  plan_sp->WillPop();
  m_plans.pop_back();
  return plan_sp;
}

// If the input plan is nullptr, discard all plans.  Otherwise make sure this
// plan is in the stack, and if so discard up to and including it.
void ThreadPlanStack::DiscardPlansUpToPlan(ThreadPlan *up_to_plan_ptr) {
  int stack_size = m_plans.size();

  if (up_to_plan_ptr == nullptr) {
    for (int i = stack_size - 1; i > 0; i--)
      DiscardPlan();
    return;
  }

  bool found_it = false;
  for (int i = stack_size - 1; i > 0; i--) {
    if (m_plans[i].get() == up_to_plan_ptr) {
      found_it = true;
      break;
    }
  }

  if (found_it) {
    bool last_one = false;
    for (int i = stack_size - 1; i > 0 && !last_one; i--) {
      if (GetCurrentPlan().get() == up_to_plan_ptr)
        last_one = true;
      DiscardPlan();
    }
  }
}

void ThreadPlanStack::DiscardAllPlans() {
  int stack_size = m_plans.size();
  for (int i = stack_size - 1; i > 0; i--) {
    DiscardPlan();
  }
  return;
}

void ThreadPlanStack::DiscardConsultingMasterPlans() {
  while (true) {
    int master_plan_idx;
    bool discard = true;

    // Find the first master plan, see if it wants discarding, and if yes
    // discard up to it.
    for (master_plan_idx = m_plans.size() - 1; master_plan_idx >= 0;
         master_plan_idx--) {
      if (m_plans[master_plan_idx]->IsMasterPlan()) {
        discard = m_plans[master_plan_idx]->OkayToDiscard();
        break;
      }
    }

    // If the master plan doesn't want to get discarded, then we're done.
    if (!discard)
      return;

    // First pop all the dependent plans:
    for (int i = m_plans.size() - 1; i > master_plan_idx; i--) {
      DiscardPlan();
    }

    // Now discard the master plan itself.
    // The bottom-most plan never gets discarded.  "OkayToDiscard" for it
    // means discard it's dependent plans, but not it...
    if (master_plan_idx > 0) {
      DiscardPlan();
    }
  }
}

lldb::ThreadPlanSP ThreadPlanStack::GetCurrentPlan() const {
  assert(m_plans.size() != 0 && "There will always be a base plan.");
  return m_plans.back();
}

lldb::ThreadPlanSP ThreadPlanStack::GetCompletedPlan(bool skip_private) const {
  if (m_completed_plans.empty())
    return {};

  if (!skip_private)
    return m_completed_plans.back();

  for (int i = m_completed_plans.size() - 1; i >= 0; i--) {
    lldb::ThreadPlanSP completed_plan_sp;
    completed_plan_sp = m_completed_plans[i];
    if (!completed_plan_sp->GetPrivate())
      return completed_plan_sp;
  }
  return {};
}

lldb::ThreadPlanSP ThreadPlanStack::GetPlanByIndex(uint32_t plan_idx,
                                                   bool skip_private) const {
  uint32_t idx = 0;
  ThreadPlan *up_to_plan_ptr = nullptr;

  for (lldb::ThreadPlanSP plan_sp : m_plans) {
    if (skip_private && plan_sp->GetPrivate())
      continue;
    if (idx == plan_idx)
      return plan_sp;
    idx++;
  }
  return {};
}

lldb::ValueObjectSP ThreadPlanStack::GetReturnValueObject() const {
  if (m_completed_plans.empty())
    return {};

  for (int i = m_completed_plans.size() - 1; i >= 0; i--) {
    lldb::ValueObjectSP return_valobj_sp;
    return_valobj_sp = m_completed_plans[i]->GetReturnValueObject();
    if (return_valobj_sp)
      return return_valobj_sp;
  }
  return {};
}

lldb::ExpressionVariableSP ThreadPlanStack::GetExpressionVariable() const {
  if (m_completed_plans.empty())
    return {};

  for (int i = m_completed_plans.size() - 1; i >= 0; i--) {
    lldb::ExpressionVariableSP expression_variable_sp;
    expression_variable_sp = m_completed_plans[i]->GetExpressionVariable();
    if (expression_variable_sp)
      return expression_variable_sp;
  }
  return {};
}
bool ThreadPlanStack::AnyPlans() const {
  // There is always a base plan...
  return m_plans.size() > 1;
}

bool ThreadPlanStack::AnyCompletedPlans() const {
  return !m_completed_plans.empty();
}

bool ThreadPlanStack::AnyDiscardedPlans() const {
  return !m_discarded_plans.empty();
}

bool ThreadPlanStack::IsPlanDone(ThreadPlan *in_plan) const {
  for (auto plan : m_completed_plans) {
    if (plan.get() == in_plan)
      return true;
  }
  return false;
}

bool ThreadPlanStack::WasPlanDiscarded(ThreadPlan *in_plan) const {
  for (auto plan : m_discarded_plans) {
    if (plan.get() == in_plan)
      return true;
  }
  return false;
}

ThreadPlan *ThreadPlanStack::GetPreviousPlan(ThreadPlan *current_plan) const {
  if (current_plan == nullptr)
    return nullptr;

  // Look first in the completed plans, if the plan is here and there is
  // a completed plan above it, return that.
  int stack_size = m_completed_plans.size();
  for (int i = stack_size - 1; i > 0; i--) {
    if (current_plan == m_completed_plans[i].get())
      return m_completed_plans[i - 1].get();
  }

  // If this is the first completed plan, the previous one is the
  // bottom of the regular plan stack.
  if (stack_size > 0 && m_completed_plans[0].get() == current_plan) {
    return GetCurrentPlan().get();
  }

  // Otherwise look for it in the regular plans.
  stack_size = m_plans.size();
  for (int i = stack_size - 1; i > 0; i--) {
    if (current_plan == m_plans[i].get())
      return m_plans[i - 1].get();
  }
  return nullptr;
}

ThreadPlan *ThreadPlanStack::GetInnermostExpression() const {
  int stack_size = m_plans.size();

  for (int i = stack_size - 1; i > 0; i--) {
    if (m_plans[i]->GetKind() == ThreadPlan::eKindCallFunction)
      return m_plans[i].get();
  }
  return nullptr;
}

void ThreadPlanStack::WillResume() {
  m_completed_plans.clear();
  m_discarded_plans.clear();
}

const ThreadPlanStack::PlanStack &
ThreadPlanStack::GetStackOfKind(ThreadPlanStack::StackKind kind) const {
  switch (kind) {
  case ePlans:
    return m_plans;
  case eCompletedPlans:
    return m_completed_plans;
  case eDiscardedPlans:
    return m_discarded_plans;
  }
  llvm_unreachable("Invalid StackKind value");
}
