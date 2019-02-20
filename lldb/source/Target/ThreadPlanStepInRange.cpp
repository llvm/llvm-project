//===-- ThreadPlanStepInRange.cpp -------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Target/ThreadPlanStepInRange.h"
#include "lldb/Breakpoint/BreakpointLocation.h"
#include "lldb/Core/Architecture.h"
#include "lldb/Core/Module.h"
#include "lldb/Symbol/Function.h"
#include "lldb/Symbol/Symbol.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/RegisterContext.h"
#include "lldb/Target/SwiftLanguageRuntime.h"
#include "lldb/Target/SectionLoadList.h"
#include "lldb/Target/Target.h"
#include "lldb/Target/Thread.h"
#include "lldb/Target/ThreadPlanStepOut.h"
#include "lldb/Target/ThreadPlanStepThrough.h"
#include "lldb/Utility/Log.h"
#include "lldb/Utility/RegularExpression.h"
#include "lldb/Utility/Stream.h"

using namespace lldb;
using namespace lldb_private;

uint32_t ThreadPlanStepInRange::s_default_flag_values =
    ThreadPlanShouldStopHere::eStepInAvoidNoDebug;

//----------------------------------------------------------------------
// ThreadPlanStepInRange: Step through a stack range, either stepping over or
// into based on the value of \a type.
//----------------------------------------------------------------------

ThreadPlanStepInRange::ThreadPlanStepInRange(
    Thread &thread, const AddressRange &range,
    const SymbolContext &addr_context, lldb::RunMode stop_others,
    LazyBool step_in_avoids_code_without_debug_info,
    LazyBool step_out_avoids_code_without_debug_info)
    : ThreadPlanStepRange(ThreadPlan::eKindStepInRange,
                          "Step Range stepping in", thread, range, addr_context,
                          stop_others),
      ThreadPlanShouldStopHere(this), m_step_past_prologue(true),
      m_virtual_step(false) {
  SetCallbacks();
  SetFlagsToDefault();
  SetupAvoidNoDebug(step_in_avoids_code_without_debug_info,
                    step_out_avoids_code_without_debug_info);
}

ThreadPlanStepInRange::ThreadPlanStepInRange(
    Thread &thread, const AddressRange &range,
    const SymbolContext &addr_context, const char *step_into_target,
    lldb::RunMode stop_others, LazyBool step_in_avoids_code_without_debug_info,
    LazyBool step_out_avoids_code_without_debug_info)
    : ThreadPlanStepRange(ThreadPlan::eKindStepInRange,
                          "Step Range stepping in", thread, range, addr_context,
                          stop_others),
      ThreadPlanShouldStopHere(this), m_step_past_prologue(true),
      m_virtual_step(false), m_step_into_target(step_into_target) {
  SetCallbacks();
  SetFlagsToDefault();
  SetupAvoidNoDebug(step_in_avoids_code_without_debug_info,
                    step_out_avoids_code_without_debug_info);
}

ThreadPlanStepInRange::~ThreadPlanStepInRange() {
  ClearStepInDeepBreakpoints();
}

void ThreadPlanStepInRange::SetupAvoidNoDebug(
    LazyBool step_in_avoids_code_without_debug_info,
    LazyBool step_out_avoids_code_without_debug_info) {
  bool avoid_nodebug = true;

  switch (step_in_avoids_code_without_debug_info) {
  case eLazyBoolYes:
    avoid_nodebug = true;
    break;
  case eLazyBoolNo:
    avoid_nodebug = false;
    break;
  case eLazyBoolCalculate:
    avoid_nodebug = m_thread.GetStepInAvoidsNoDebug();
    break;
  }
  if (avoid_nodebug)
    GetFlags().Set(ThreadPlanShouldStopHere::eStepInAvoidNoDebug);
  else
    GetFlags().Clear(ThreadPlanShouldStopHere::eStepInAvoidNoDebug);

  switch (step_out_avoids_code_without_debug_info) {
  case eLazyBoolYes:
    avoid_nodebug = true;
    break;
  case eLazyBoolNo:
    avoid_nodebug = false;
    break;
  case eLazyBoolCalculate:
    avoid_nodebug = m_thread.GetStepOutAvoidsNoDebug();
    break;
  }
  if (avoid_nodebug)
    GetFlags().Set(ThreadPlanShouldStopHere::eStepOutAvoidNoDebug);
  else
    GetFlags().Clear(ThreadPlanShouldStopHere::eStepOutAvoidNoDebug);
}

void ThreadPlanStepInRange::GetDescription(Stream *s,
                                           lldb::DescriptionLevel level) {

  auto PrintFailureIfAny = [&]() {
    if (m_status.Success())
      return;
    s->Printf(" failed (%s)", m_status.AsCString());
  };

  if (level == lldb::eDescriptionLevelBrief) {
    s->Printf("step in");
    PrintFailureIfAny();
    return;
  }

  s->Printf("Stepping in");
  bool printed_line_info = false;
  if (m_addr_context.line_entry.IsValid()) {
    s->Printf(" through line ");
    m_addr_context.line_entry.DumpStopContext(s, false);
    printed_line_info = true;
  }

  const char *step_into_target = m_step_into_target.AsCString();
  if (step_into_target && step_into_target[0] != '\0')
    s->Printf(" targeting %s", m_step_into_target.AsCString());

  if (!printed_line_info || level == eDescriptionLevelVerbose) {
    s->Printf(" using ranges:");
    DumpRanges(s);
  }

  PrintFailureIfAny();

  s->PutChar('.');
}

bool ThreadPlanStepInRange::ShouldStop(Event *event_ptr) {
  Log *log(lldb_private::GetLogIfAllCategoriesSet(LIBLLDB_LOG_STEP));

  if (log) {
    StreamString s;
    s.Address(
        m_thread.GetRegisterContext()->GetPC(),
        m_thread.CalculateTarget()->GetArchitecture().GetAddressByteSize());
    log->Printf("ThreadPlanStepInRange reached %s.", s.GetData());
  }

  if (IsPlanComplete())
    return true;

  m_no_more_plans = false;
  if (m_sub_plan_sp && m_sub_plan_sp->IsPlanComplete()) {
    if (!m_sub_plan_sp->PlanSucceeded()) {
      SetPlanComplete();
      m_no_more_plans = true;
      return true;
    } else
      m_sub_plan_sp.reset();
  }

  if (m_virtual_step) {
    // If we've just completed a virtual step, all we need to do is check for a
    // ShouldStopHere plan, and otherwise we're done.
    // FIXME - This can be both a step in and a step out.  Probably should
    // record which in the m_virtual_step.
    m_sub_plan_sp =
        CheckShouldStopHereAndQueueStepOut(eFrameCompareYounger, m_status);
  } else {
    // Stepping through should be done running other threads in general, since
    // we're setting a breakpoint and continuing.  So only stop others if we
    // are explicitly told to do so.

    bool stop_others = (m_stop_others == lldb::eOnlyThisThread);

    FrameComparison frame_order = CompareCurrentFrameToStartFrame();

    if (frame_order == eFrameCompareOlder ||
        frame_order == eFrameCompareSameParent) {
      // If we're in an older frame then we should stop.
      //
      // A caveat to this is if we think the frame is older but we're actually
      // in a trampoline.
      // I'm going to make the assumption that you wouldn't RETURN to a
      // trampoline.  So if we are in a trampoline we think the frame is older
      // because the trampoline confused the backtracer.
      m_sub_plan_sp = m_thread.QueueThreadPlanForStepThrough(
          m_stack_id, false, stop_others, m_status);
      if (!m_sub_plan_sp) {
        // Otherwise check the ShouldStopHere for step out:
        m_sub_plan_sp =
            CheckShouldStopHereAndQueueStepOut(frame_order, m_status);
        if (log) {
          if (m_sub_plan_sp)
            log->Printf("ShouldStopHere found plan to step out of this frame.");
          else
            log->Printf("ShouldStopHere no plan to step out of this frame.");
        }
      } else if (log) {
        log->Printf(
            "Thought I stepped out, but in fact arrived at a trampoline.");
      }
    } else if (frame_order == eFrameCompareEqual && InSymbol()) {
      // If we are not in a place we should step through, we're done. One
      // tricky bit here is that some stubs don't push a frame, so we have to
      // check both the case of a frame that is younger, or the same as this
      // frame. However, if the frame is the same, and we are still in the
      // symbol we started in, the we don't need to do this.  This first check
      // isn't strictly necessary, but it is more efficient.

      // If we're still in the range, keep going, either by running to the next
      // branch breakpoint, or by stepping.
      if (InRange()) {
        SetNextBranchBreakpoint();
        return false;
      }

      SetPlanComplete();
      m_no_more_plans = true;
      return true;
    }

    // If we get to this point, we're not going to use a previously set "next
    // branch" breakpoint, so delete it:
    ClearNextBranchBreakpoint();

    // We may have set the plan up above in the FrameIsOlder section:

    if (!m_sub_plan_sp)
      m_sub_plan_sp = m_thread.QueueThreadPlanForStepThrough(
          m_stack_id, false, stop_others, m_status);

    if (log) {
      if (m_sub_plan_sp)
        log->Printf("Found a step through plan: %s", m_sub_plan_sp->GetName());
      else
        log->Printf("No step through plan found.");
    }

    // If not, give the "should_stop" callback a chance to push a plan to get
    // us out of here. But only do that if we actually have stepped in.
    if (!m_sub_plan_sp && frame_order == eFrameCompareYounger)
      m_sub_plan_sp = CheckShouldStopHereAndQueueStepOut(frame_order, m_status);

    // If we've stepped in and we are going to stop here, check to see if we
    // were asked to run past the prologue, and if so do that.

    if (!m_sub_plan_sp && frame_order == eFrameCompareYounger &&
        m_step_past_prologue) {
      lldb::StackFrameSP curr_frame = m_thread.GetStackFrameAtIndex(0);
      if (curr_frame) {
        size_t bytes_to_skip = 0;
        lldb::addr_t curr_addr = m_thread.GetRegisterContext()->GetPC();
        Address func_start_address;

        SymbolContext sc = curr_frame->GetSymbolContext(eSymbolContextFunction |
                                                        eSymbolContextSymbol);

        if (sc.function) {
          func_start_address = sc.function->GetAddressRange().GetBaseAddress();
          if (curr_addr ==
              func_start_address.GetLoadAddress(
                  m_thread.CalculateTarget().get()))
            bytes_to_skip = sc.function->GetPrologueByteSize();
        } else if (sc.symbol) {
          func_start_address = sc.symbol->GetAddress();
          if (curr_addr ==
              func_start_address.GetLoadAddress(
                  m_thread.CalculateTarget().get()))
            bytes_to_skip = sc.symbol->GetPrologueByteSize();
        }

        if (bytes_to_skip == 0 && sc.symbol) {
          TargetSP target = m_thread.CalculateTarget();
          const Architecture *arch = target->GetArchitecturePlugin();
          if (arch) {
            Address curr_sec_addr;
            target->GetSectionLoadList().ResolveLoadAddress(curr_addr,
                                                            curr_sec_addr);
            bytes_to_skip = arch->GetBytesToSkip(*sc.symbol, curr_sec_addr);
          }
        }

        if (bytes_to_skip != 0) {
          func_start_address.Slide(bytes_to_skip);
          log = lldb_private::GetLogIfAllCategoriesSet(LIBLLDB_LOG_STEP);
          if (log)
            log->Printf("Pushing past prologue ");

          m_sub_plan_sp = m_thread.QueueThreadPlanForRunToAddress(
              false, func_start_address, true, m_status);
        }
      }
    }
  }

  if (!m_sub_plan_sp) {
    m_no_more_plans = true;
    SetPlanComplete();
    return true;
  } else {
    m_no_more_plans = false;
    m_sub_plan_sp->SetPrivate(true);
    return false;
  }
}

bool ThreadPlanStepInRange::MischiefManaged() {
  bool return_value = ThreadPlanStepRange::MischiefManaged();
  if (return_value)
    ClearStepInDeepBreakpoints();
  return return_value;
}

void ThreadPlanStepInRange::SetAvoidRegexp(const char *name) {
  auto name_ref = llvm::StringRef::withNullAsEmpty(name);
  if (!m_avoid_regexp_ap)
    m_avoid_regexp_ap.reset(new RegularExpression(name_ref));

  m_avoid_regexp_ap->Compile(name_ref);
}

void ThreadPlanStepInRange::SetDefaultFlagValue(uint32_t new_value) {
  // TODO: Should we test this for sanity?
  ThreadPlanStepInRange::s_default_flag_values = new_value;
}

bool ThreadPlanStepInRange::StepInDeepBreakpointExplainsStop(
    lldb::StopInfoSP stop_info_sp) {
  Log *log(lldb_private::GetLogIfAllCategoriesSet(LIBLLDB_LOG_STEP));
  size_t num_step_in_deep_bps = m_step_in_deep_bps.size();
  if (num_step_in_deep_bps == 0)
    return false;

  break_id_t bp_site_id = stop_info_sp->GetValue();
  BreakpointSiteSP bp_site_sp =
      m_thread.GetProcess()->GetBreakpointSiteList().FindByID(bp_site_id);
  if (!bp_site_sp)
    return false;

  bool explains_stop = false;
  bool hit_step_in_deep_bp = false;
  for (size_t i = 0; i < num_step_in_deep_bps && hit_step_in_deep_bp == false;
       i++) {
    if (bp_site_sp->IsBreakpointAtThisSite(m_step_in_deep_bps[i])) {
      // We've hit a step in deep breakpoint, see if it is the only breakpoint
      // at this site:
      explains_stop = true;
      hit_step_in_deep_bp = true;
      size_t num_owners = bp_site_sp->GetNumberOfOwners();

      // If all the owners are internal, then we are probably just stepping over
      // this range from multiple threads,
      // or multiple frames, so we want to continue.  If one is not internal,
      // then we should not explain the stop,
      // and let the user breakpoint handle the stop.
      // Of course, if there's only one owner, it's us so we don't need to
      // check.

      if (num_owners == 1)
        continue;

      for (size_t i = 0; i < num_owners; i++) {
        BreakpointLocationSP owner_loc_sp(bp_site_sp->GetOwnerAtIndex(i));
        Breakpoint &owner_bp(owner_loc_sp->GetBreakpoint());
        if (owner_loc_sp->ValidForThisThread(&GetThread()) &&
            !owner_bp.IsInternal()) {
          explains_stop = false;
          break;
        }
      }
      if (log)
        log->Printf("ThreadPlanStepRange::StepInDeepBreakpointExplainsStop - "
                    "Hit step in deep breakpoint %d which has %zu owners - "
                    "explains stop: %u.",
                    m_step_in_deep_bps[i], num_owners, explains_stop);
    }
  }

  // For now, if we trigger one of our "step in deep" breakpoints we delete them
  // all:
  if (hit_step_in_deep_bp) {
    ClearStepInDeepBreakpoints();
  }

  return explains_stop;
}

void ThreadPlanStepInRange::ClearStepInDeepBreakpoints() {
  size_t num_step_in_deep_bps = m_step_in_deep_bps.size();
  for (size_t i = 0; i < num_step_in_deep_bps; i++) {
    GetTarget().RemoveBreakpointByID(m_step_in_deep_bps[i]);
  }
  m_step_in_deep_bps.clear();
}

bool ThreadPlanStepInRange::FrameMatchesAvoidCriteria() {
  StackFrame *frame = GetThread().GetStackFrameAtIndex(0).get();

  // Check the library list first, as that's cheapest:
  FileSpecList libraries_to_avoid(GetThread().GetLibrariesToAvoid());
  size_t num_libraries = libraries_to_avoid.GetSize();
  bool libraries_say_avoid = false;
  SymbolContext sc(frame->GetSymbolContext(eSymbolContextModule));
  FileSpec frame_library(sc.module_sp->GetFileSpec());

  if (frame_library) {
    for (size_t i = 0; i < num_libraries; i++) {
      const FileSpec &file_spec(libraries_to_avoid.GetFileSpecAtIndex(i));
      if (FileSpec::Equal(file_spec, frame_library, false)) {
        libraries_say_avoid = true;
        break;
      }
    }
  }
  if (libraries_say_avoid)
    return true;

  const RegularExpression *avoid_regexp_to_use = m_avoid_regexp_ap.get();
  if (avoid_regexp_to_use == nullptr)
    avoid_regexp_to_use = GetThread().GetSymbolsToAvoidRegexp();

  if (avoid_regexp_to_use != nullptr) {
    SymbolContext sc = frame->GetSymbolContext(
        eSymbolContextFunction | eSymbolContextBlock | eSymbolContextSymbol);
    if (sc.symbol != nullptr) {
      const char *frame_function_name =
          sc.GetFunctionName(Mangled::ePreferDemangledWithoutArguments)
              .GetCString();
      if (frame_function_name) {
        size_t num_matches = 0;
        Log *log(lldb_private::GetLogIfAllCategoriesSet(LIBLLDB_LOG_STEP));
        if (log)
          num_matches = 1;

        RegularExpression::Match regex_match(num_matches);

        bool return_value =
            avoid_regexp_to_use->Execute(frame_function_name, &regex_match);
        if (return_value) {
          if (log) {
            std::string match;
            regex_match.GetMatchAtIndex(frame_function_name, 0, match);
            log->Printf("Stepping out of function \"%s\" because it matches "
                        "the avoid regexp \"%s\" - match substring: \"%s\".",
                        frame_function_name,
                        avoid_regexp_to_use->GetText().str().c_str(),
                        match.c_str());
          }
        }
        return return_value;
      }
    }
  }
  return false;
}

bool ThreadPlanStepInRange::DefaultShouldStopHereCallback(
    ThreadPlan *current_plan, Flags &flags, FrameComparison operation,
    Status &status, void *baton) {
  bool should_stop_here = true;

  // First see if the ThreadPlanShouldStopHere default implementation thinks we
  // should get out of here:
  should_stop_here = ThreadPlanShouldStopHere::DefaultShouldStopHereCallback(
      current_plan, flags, operation, status, baton);
  if (!should_stop_here)
    return false;

  if (should_stop_here && current_plan->GetKind() == eKindStepInRange &&
      operation == eFrameCompareYounger) {
    ThreadPlanStepInRange *step_in_range_plan =
        static_cast<ThreadPlanStepInRange *>(current_plan);
    should_stop_here =
        step_in_range_plan->DefaultShouldStopHereImpl(flags, !should_stop_here);

    //        if (should_stop_here)
    //        {
    //            ThreadPlanStepInRange *step_in_range_plan =
    //            static_cast<ThreadPlanStepInRange *> (current_plan);
    //            // Don't log the should_step_out here, it's easier to do it in
    //            FrameMatchesAvoidCriteria.
    //            should_stop_here =
    //            !step_in_range_plan->FrameMatchesAvoidCriteria ();
    //        }
  }

  return should_stop_here;
}

bool ThreadPlanStepInRange::DefaultShouldStopHereImpl(Flags &flags,
                                                      bool should_step_out) {
  StackFrame *frame = GetThread().GetStackFrameAtIndex(0).get();
  Log *log(lldb_private::GetLogIfAllCategoriesSet(LIBLLDB_LOG_STEP));

  if (m_step_into_target) {
    SymbolContext sc = frame->GetSymbolContext(
        eSymbolContextFunction | eSymbolContextBlock | eSymbolContextSymbol);
    if (sc.symbol != NULL) {
      SymbolContext sc = frame->GetSymbolContext(
          eSymbolContextFunction | eSymbolContextBlock | eSymbolContextSymbol);
      if (sc.symbol != nullptr) {
        // First try an exact match, since that's cheap with
        // ConstStrings.  Then do a strstr compare.
        if (m_step_into_target == sc.GetFunctionName()) {
          should_step_out = false;
        } else {
          const char *target_name = m_step_into_target.AsCString();
          const char *function_name = sc.GetFunctionName().AsCString();

          if (function_name == nullptr)
            should_step_out = true;
          else if (strstr(function_name, target_name) == nullptr)
            should_step_out = true;
        }
        if (log && should_step_out)
          log->Printf("Stepping out of frame %s which did not match step into "
                      "target %s.",
                      sc.GetFunctionName().AsCString(),
                      m_step_into_target.AsCString());
      } else {
        const char *target_name = m_step_into_target.AsCString();
        const char *function_name = sc.GetFunctionName().AsCString();

        if (function_name == NULL)
          should_step_out = true;
        else if (strstr(function_name, target_name) == NULL)
          should_step_out = true;
      }
      if (log && should_step_out)
        log->Printf(
            "Stepping out of frame %s which did not match step into target %s.",
            sc.GetFunctionName().AsCString(), m_step_into_target.AsCString());
    }
  }

  if (!should_step_out) {
    // Don't log the should_step_out here, it's easier to do it in
    // FrameMatchesAvoidRegexp.
    should_step_out = FrameMatchesAvoidCriteria();
  }

  if (should_step_out) {
    // We are going to step out, but first let's examine the function we are
    // stepping past to see if it tells us
    // about any interesting places we could stop while running it.  For
    // instance, if we can tell from the signature
    // that we're being passed a function pointer that points to user code,
    // we'll prospectively stop there.
    // We only know how to do this for Swift at present.
    // FIXME: We could probably do this for C++ mangled names as well, if we
    // could come up with some
    // good heuristic to identify function pointers in the mangled function
    // arguments.

    SymbolContext sc = frame->GetSymbolContext(eSymbolContextSymbol);
    if (sc.symbol) {
      Mangled mangled_name = sc.symbol->GetMangled();
      if (mangled_name.GuessLanguage() == lldb::eLanguageTypeSwift) {
        ProcessSP process_sp(GetThread().GetProcess());
        SwiftLanguageRuntime *swift_runtime =
            process_sp->GetSwiftLanguageRuntime();
        if (swift_runtime) {
          std::vector<Address> interesting_addresses;
          swift_runtime->FindFunctionPointersInCall(*frame,
                                                    interesting_addresses);
          size_t num_addresses = interesting_addresses.size();
          if (num_addresses) {
            // Run through the addresses we found, make sure they have debug
            // info, and if so set breakpoints
            // on all these addresses.
            for (size_t i = 0; i < num_addresses; i++) {
              LineEntry line_entry;
              if (interesting_addresses[i].CalculateSymbolContextLineEntry(
                      line_entry)) {
                // It has debug information, use it:
                const bool internal = true;
                const bool hardware = false;
                BreakpointSP bkpt_sp = GetTarget().CreateBreakpoint(
                    interesting_addresses[i], internal, hardware);
                bkpt_sp->SetThreadID(GetThread().GetID());
                m_step_in_deep_bps.push_back(bkpt_sp->GetID());
              }
            }
          }
        }
      }
    }
  }
  // We're returning an answer to "Should Stop Here" which is the opposite of
  // "should_step_out".
  return !should_step_out;
}

bool ThreadPlanStepInRange::DoPlanExplainsStop(Event *event_ptr) {
  // We always explain a stop.  Either we've just done a single step, in which
  // case we'll do our ordinary processing, or we stopped for some reason that
  // isn't handled by our sub-plans, in which case we want to just stop right
  // away. In general, we don't want to mark the plan as complete for
  // unexplained stops. For instance, if you step in to some code with no debug
  // info, so you step out and in the course of that hit a breakpoint, then you
  // want to stop & show the user the breakpoint, but not unship the step in
  // plan, since you still may want to complete that plan when you continue.
  // This is particularly true when doing "step in to target function."
  // stepping.
  //
  // The only variation is that if we are doing "step by running to next
  // branch" in which case if we hit our branch breakpoint we don't set the
  // plan to complete.

  bool return_value = false;

  if (m_virtual_step) {
    return_value = true;
  } else {
    StopInfoSP stop_info_sp = GetPrivateStopInfo();
    if (stop_info_sp) {
      StopReason reason = stop_info_sp->GetStopReason();

      if (reason == eStopReasonBreakpoint) {
        bool hit_next_range_bp = NextRangeBreakpointExplainsStop(stop_info_sp);
        bool hit_step_in_deep_bp =
            StepInDeepBreakpointExplainsStop(stop_info_sp);
        if (hit_next_range_bp || hit_step_in_deep_bp) {
          if (hit_step_in_deep_bp)
            SetPlanComplete();

          return_value = true;
        }
      } else if (IsUsuallyUnexplainedStopReason(reason)) {
        Log *log(lldb_private::GetLogIfAllCategoriesSet(LIBLLDB_LOG_STEP));
        if (log)
          log->PutCString("ThreadPlanStepInRange got asked if it explains the "
                          "stop for some reason other than step.");
        return_value = false;
      } else {
        return_value = true;
      }
    } else
      return_value = true;
  }

  return return_value;
}

bool ThreadPlanStepInRange::DoWillResume(lldb::StateType resume_state,
                                         bool current_plan) {
  m_virtual_step = false;
  if (resume_state == eStateStepping && current_plan) {
    // See if we are about to step over a virtual inlined call.
    bool step_without_resume = m_thread.DecrementCurrentInlinedDepth();
    if (step_without_resume) {
      Log *log(lldb_private::GetLogIfAllCategoriesSet(LIBLLDB_LOG_STEP));
      if (log)
        log->Printf("ThreadPlanStepInRange::DoWillResume: returning false, "
                    "inline_depth: %d",
                    m_thread.GetCurrentInlinedDepth());
      SetStopInfo(StopInfo::CreateStopReasonToTrace(m_thread));

      // FIXME: Maybe it would be better to create a InlineStep stop reason, but
      // then
      // the whole rest of the world would have to handle that stop reason.
      m_virtual_step = true;
    }
    return !step_without_resume;
  }
  return true;
}

bool ThreadPlanStepInRange::IsVirtualStep() { return m_virtual_step; }
