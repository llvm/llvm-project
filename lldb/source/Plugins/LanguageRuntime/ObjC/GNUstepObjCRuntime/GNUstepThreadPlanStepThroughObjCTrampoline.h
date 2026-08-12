//===-- GNUstepThreadPlanStepThroughObjCTrampoline.h ------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SOURCE_PLUGINS_LANGUAGERUNTIME_OBJC_GNUSTEPOBJCRUNTIME_GNUSTEPTHREADPLANSTEPTHROUGHOBJCTRAMPOLINE_H
#define LLDB_SOURCE_PLUGINS_LANGUAGERUNTIME_OBJC_GNUSTEPOBJCRUNTIME_GNUSTEPTHREADPLANSTEPTHROUGHOBJCTRAMPOLINE_H

#include "lldb/Core/Value.h"
#include "lldb/Target/ThreadPlan.h"
#include "lldb/lldb-types.h"

namespace lldb_private {

class GNUstepObjCRuntime;

/// Steps from a libobjc2 dispatch entry point (objc_msgSend,
/// objc_msg_lookup, ...) to the method implementation it is about to
/// dispatch to. The IMP is resolved by calling `objc_msg_lookup(receiver,
/// selector)` in the inferior - the same lookup the trampoline itself
/// performs - from a nested function-call plan, then running to the
/// returned address. This is the same shape as
/// AppleThreadPlanStepThroughObjCTrampoline.
class GNUstepThreadPlanStepThroughObjCTrampoline : public ThreadPlan {
public:
  GNUstepThreadPlanStepThroughObjCTrampoline(Thread &thread,
                                             GNUstepObjCRuntime &runtime,
                                             ValueList &input_values,
                                             lldb::addr_t isa_addr,
                                             lldb::addr_t sel_addr);

  ~GNUstepThreadPlanStepThroughObjCTrampoline() override;

  static bool PreResumeInitializeFunctionCaller(void *myself);

  void GetDescription(Stream *s, lldb::DescriptionLevel level) override;

  bool ValidatePlan(Stream *error) override { return true; }

  lldb::StateType GetPlanRunState() override { return lldb::eStateRunning; }

  bool ShouldStop(Event *event_ptr) override;

  // The lookup might have to fill dispatch caches, so it is not safe to run
  // only one thread.
  bool StopOthers() override { return false; }

  bool MischiefManaged() override { return IsPlanComplete(); }

  void DidPush() override;

  void DidPop() override;

  bool WillStop() override { return true; }

protected:
  bool DoPlanExplainsStop(Event *event_ptr) override { return true; }

private:
  bool InitializeFunctionCaller();

  GNUstepObjCRuntime &m_runtime;
  /// Address of the argument struct of the msg-lookup function call.
  lldb::addr_t m_args_addr = LLDB_INVALID_ADDRESS;
  ValueList m_input_values;
  /// Keys for the method cache filled in when the lookup completes.
  lldb::addr_t m_isa_addr;
  lldb::addr_t m_sel_addr;
  /// The nested function-call plan; reset once it completes.
  lldb::ThreadPlanSP m_func_sp;
  /// The run-to-implementation plan queued after the lookup.
  lldb::ThreadPlanSP m_run_to_sp;
  FunctionCaller *m_lookup_function = nullptr;
};

} // namespace lldb_private

#endif // LLDB_SOURCE_PLUGINS_LANGUAGERUNTIME_OBJC_GNUSTEPOBJCRUNTIME_GNUSTEPTHREADPLANSTEPTHROUGHOBJCTRAMPOLINE_H
