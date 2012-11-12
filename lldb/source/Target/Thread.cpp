//===-- Thread.cpp ----------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/lldb-private-log.h"
#include "lldb/Breakpoint/BreakpointLocation.h"
#include "lldb/Core/Debugger.h"
#include "lldb/Core/Log.h"
#include "lldb/Core/Stream.h"
#include "lldb/Core/StreamString.h"
#include "lldb/Core/RegularExpression.h"
#include "lldb/Host/Host.h"
#include "lldb/Symbol/Function.h"
#include "lldb/Target/DynamicLoader.h"
#include "lldb/Target/ExecutionContext.h"
#include "lldb/Target/ObjCLanguageRuntime.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/RegisterContext.h"
#include "lldb/Target/StopInfo.h"
#include "lldb/Target/Target.h"
#include "lldb/Target/Thread.h"
#include "lldb/Target/ThreadPlan.h"
#include "lldb/Target/ThreadPlanCallFunction.h"
#include "lldb/Target/ThreadPlanBase.h"
#include "lldb/Target/ThreadPlanStepInstruction.h"
#include "lldb/Target/ThreadPlanStepOut.h"
#include "lldb/Target/ThreadPlanStepOverBreakpoint.h"
#include "lldb/Target/ThreadPlanStepThrough.h"
#include "lldb/Target/ThreadPlanStepInRange.h"
#include "lldb/Target/ThreadPlanStepOverRange.h"
#include "lldb/Target/ThreadPlanRunToAddress.h"
#include "lldb/Target/ThreadPlanStepUntil.h"
#include "lldb/Target/ThreadSpec.h"
#include "lldb/Target/Unwind.h"
#include "Plugins/Process/Utility/UnwindLLDB.h"
#include "UnwindMacOSXFrameBackchain.h"


using namespace lldb;
using namespace lldb_private;


const ThreadPropertiesSP &
Thread::GetGlobalProperties()
{
    static ThreadPropertiesSP g_settings_sp;
    if (!g_settings_sp)
        g_settings_sp.reset (new ThreadProperties (true));
    return g_settings_sp;
}

static PropertyDefinition
g_properties[] =
{
    { "step-avoid-regexp",  OptionValue::eTypeRegex  , true , REG_EXTENDED, "^std::", NULL, "A regular expression defining functions step-in won't stop in." },
    { "trace-thread",       OptionValue::eTypeBoolean, false, false, NULL, NULL, "If true, this thread will single-step and log execution." },
    {  NULL               , OptionValue::eTypeInvalid, false, 0    , NULL, NULL, NULL  }
};

enum {
    ePropertyStepAvoidRegex,
    ePropertyEnableThreadTrace
};


class ThreadOptionValueProperties : public OptionValueProperties
{
public:
    ThreadOptionValueProperties (const ConstString &name) :
        OptionValueProperties (name)
    {
    }
    
    // This constructor is used when creating ThreadOptionValueProperties when it
    // is part of a new lldb_private::Thread instance. It will copy all current
    // global property values as needed
    ThreadOptionValueProperties (ThreadProperties *global_properties) :
        OptionValueProperties(*global_properties->GetValueProperties())
    {
    }
    
    virtual const Property *
    GetPropertyAtIndex (const ExecutionContext *exe_ctx, bool will_modify, uint32_t idx) const
    {
        // When gettings the value for a key from the thread options, we will always
        // try and grab the setting from the current thread if there is one. Else we just
        // use the one from this instance.
        if (exe_ctx)
        {
            Thread *thread = exe_ctx->GetThreadPtr();
            if (thread)
            {
                ThreadOptionValueProperties *instance_properties = static_cast<ThreadOptionValueProperties *>(thread->GetValueProperties().get());
                if (this != instance_properties)
                    return instance_properties->ProtectedGetPropertyAtIndex (idx);
            }
        }
        return ProtectedGetPropertyAtIndex (idx);
    }
};



ThreadProperties::ThreadProperties (bool is_global) :
    Properties ()
{
    if (is_global)
    {
        m_collection_sp.reset (new ThreadOptionValueProperties(ConstString("thread")));
        m_collection_sp->Initialize(g_properties);
    }
    else
        m_collection_sp.reset (new ThreadOptionValueProperties(Thread::GetGlobalProperties().get()));
}

ThreadProperties::~ThreadProperties()
{
}

const RegularExpression *
ThreadProperties::GetSymbolsToAvoidRegexp()
{
    const uint32_t idx = ePropertyStepAvoidRegex;
    return m_collection_sp->GetPropertyAtIndexAsOptionValueRegex (NULL, idx);
}

bool
ThreadProperties::GetTraceEnabledState() const
{
    const uint32_t idx = ePropertyEnableThreadTrace;
    return m_collection_sp->GetPropertyAtIndexAsBoolean (NULL, idx, g_properties[idx].default_uint_value != 0);
}

//------------------------------------------------------------------
// Thread Event Data
//------------------------------------------------------------------


const ConstString &
Thread::ThreadEventData::GetFlavorString ()
{
    static ConstString g_flavor ("Thread::ThreadEventData");
    return g_flavor;
}

Thread::ThreadEventData::ThreadEventData (const lldb::ThreadSP thread_sp) :
    m_thread_sp (thread_sp),
    m_stack_id ()
{
}

Thread::ThreadEventData::ThreadEventData (const lldb::ThreadSP thread_sp, const StackID &stack_id) :
    m_thread_sp (thread_sp),
    m_stack_id (stack_id)
{
}

Thread::ThreadEventData::ThreadEventData () :
    m_thread_sp (),
    m_stack_id ()
{
}

Thread::ThreadEventData::~ThreadEventData ()
{
}

void
Thread::ThreadEventData::Dump (Stream *s) const
{

}

const Thread::ThreadEventData *
Thread::ThreadEventData::GetEventDataFromEvent (const Event *event_ptr)
{
    if (event_ptr)
    {
        const EventData *event_data = event_ptr->GetData();
        if (event_data && event_data->GetFlavor() == ThreadEventData::GetFlavorString())
            return static_cast <const ThreadEventData *> (event_ptr->GetData());
    }
    return NULL;
}

ThreadSP
Thread::ThreadEventData::GetThreadFromEvent (const Event *event_ptr)
{
    ThreadSP thread_sp;
    const ThreadEventData *event_data = GetEventDataFromEvent (event_ptr);
    if (event_data)
        thread_sp = event_data->GetThread();
    return thread_sp;
}

StackID
Thread::ThreadEventData::GetStackIDFromEvent (const Event *event_ptr)
{
    StackID stack_id;
    const ThreadEventData *event_data = GetEventDataFromEvent (event_ptr);
    if (event_data)
        stack_id = event_data->GetStackID();
    return stack_id;
}

StackFrameSP
Thread::ThreadEventData::GetStackFrameFromEvent (const Event *event_ptr)
{
    const ThreadEventData *event_data = GetEventDataFromEvent (event_ptr);
    StackFrameSP frame_sp;
    if (event_data)
    {
        ThreadSP thread_sp = event_data->GetThread();
        if (thread_sp)
        {
            frame_sp = thread_sp->GetStackFrameList()->GetFrameWithStackID (event_data->GetStackID());
        }
    }
    return frame_sp;
}

//------------------------------------------------------------------
// Thread class
//------------------------------------------------------------------

ConstString &
Thread::GetStaticBroadcasterClass ()
{
    static ConstString class_name ("lldb.thread");
    return class_name;
}

Thread::Thread (Process &process, lldb::tid_t tid) :
    ThreadProperties (false),
    UserID (tid),
    Broadcaster(&process.GetTarget().GetDebugger(), Thread::GetStaticBroadcasterClass().AsCString()),
    m_process_wp (process.shared_from_this()),
    m_actual_stop_info_sp (),
    m_index_id (process.GetNextThreadIndexID ()),
    m_reg_context_sp (),
    m_state (eStateUnloaded),
    m_state_mutex (Mutex::eMutexTypeRecursive),
    m_plan_stack (),
    m_completed_plan_stack(),
    m_frame_mutex (Mutex::eMutexTypeRecursive),
    m_curr_frames_sp (),
    m_prev_frames_sp (),
    m_resume_signal (LLDB_INVALID_SIGNAL_NUMBER),
    m_resume_state (eStateRunning),
    m_temporary_resume_state (eStateRunning),
    m_unwinder_ap (),
    m_destroy_called (false),
    m_thread_stop_reason_stop_id (0)

{
    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_OBJECT));
    if (log)
        log->Printf ("%p Thread::Thread(tid = 0x%4.4llx)", this, GetID());

    CheckInWithManager();
    QueueFundamentalPlan(true);
}


Thread::~Thread()
{
    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_OBJECT));
    if (log)
        log->Printf ("%p Thread::~Thread(tid = 0x%4.4llx)", this, GetID());
    /// If you hit this assert, it means your derived class forgot to call DoDestroy in its destructor.
    assert (m_destroy_called);
}

void 
Thread::DestroyThread ()
{
    m_destroy_called = true;
    m_plan_stack.clear();
    m_discarded_plan_stack.clear();
    m_completed_plan_stack.clear();
    m_actual_stop_info_sp.reset();
    m_reg_context_sp.reset();
    m_unwinder_ap.reset();
    Mutex::Locker locker(m_frame_mutex);
    m_curr_frames_sp.reset();
    m_prev_frames_sp.reset();
}

void
Thread::BroadcastSelectedFrameChange(StackID &new_frame_id)
{
    if (EventTypeHasListeners(eBroadcastBitSelectedFrameChanged))
        BroadcastEvent(eBroadcastBitSelectedFrameChanged, new ThreadEventData (this->shared_from_this(), new_frame_id));
}

uint32_t
Thread::SetSelectedFrame (lldb_private::StackFrame *frame, bool broadcast)
{
    uint32_t ret_value = GetStackFrameList()->SetSelectedFrame(frame);
    if (broadcast)
        BroadcastSelectedFrameChange(frame->GetStackID());
    return ret_value;
}

bool
Thread::SetSelectedFrameByIndex (uint32_t frame_idx, bool broadcast)
{
    StackFrameSP frame_sp(GetStackFrameList()->GetFrameAtIndex (frame_idx));
    if (frame_sp)
    {
        GetStackFrameList()->SetSelectedFrame(frame_sp.get());
        if (broadcast)
            BroadcastSelectedFrameChange(frame_sp->GetStackID());
        return true;
    }
    else
        return false;
}


lldb::StopInfoSP
Thread::GetStopInfo ()
{
    ThreadPlanSP plan_sp (GetCompletedPlan());
    if (plan_sp && plan_sp->PlanSucceeded())
        return StopInfo::CreateStopReasonWithPlan (plan_sp, GetReturnValueObject());
    else
    {
        ProcessSP process_sp (GetProcess());
        if (process_sp 
            && m_actual_stop_info_sp 
            && m_actual_stop_info_sp->IsValid()
            && m_thread_stop_reason_stop_id == process_sp->GetStopID())
            return m_actual_stop_info_sp;
        else
            return GetPrivateStopReason ();
    }
}

lldb::StopReason
Thread::GetStopReason()
{
    lldb::StopInfoSP stop_info_sp (GetStopInfo ());
    if (stop_info_sp)
        return stop_info_sp->GetStopReason();
    return eStopReasonNone;
}



void
Thread::SetStopInfo (const lldb::StopInfoSP &stop_info_sp)
{
    m_actual_stop_info_sp = stop_info_sp;
    if (m_actual_stop_info_sp)
        m_actual_stop_info_sp->MakeStopInfoValid();
    ProcessSP process_sp (GetProcess());
    if (process_sp)
        m_thread_stop_reason_stop_id = process_sp->GetStopID();
    else
        m_thread_stop_reason_stop_id = UINT32_MAX;
}

void
Thread::SetStopInfoToNothing()
{
    // Note, we can't just NULL out the private reason, or the native thread implementation will try to
    // go calculate it again.  For now, just set it to a Unix Signal with an invalid signal number.
    SetStopInfo (StopInfo::CreateStopReasonWithSignal (*this,  LLDB_INVALID_SIGNAL_NUMBER));
}

bool
Thread::ThreadStoppedForAReason (void)
{
    return (bool) GetPrivateStopReason ();
}

bool
Thread::CheckpointThreadState (ThreadStateCheckpoint &saved_state)
{
    if (!SaveFrameZeroState(saved_state.register_backup))
        return false;

    saved_state.stop_info_sp = GetStopInfo();
    ProcessSP process_sp (GetProcess());
    if (process_sp)
        saved_state.orig_stop_id = process_sp->GetStopID();
    saved_state.current_inlined_depth = GetCurrentInlinedDepth();
    
    return true;
}

bool
Thread::RestoreThreadStateFromCheckpoint (ThreadStateCheckpoint &saved_state)
{
    RestoreSaveFrameZero(saved_state.register_backup);
    if (saved_state.stop_info_sp)
        saved_state.stop_info_sp->MakeStopInfoValid();
    SetStopInfo(saved_state.stop_info_sp);
    GetStackFrameList()->SetCurrentInlinedDepth (saved_state.current_inlined_depth);
    return true;
}

StateType
Thread::GetState() const
{
    // If any other threads access this we will need a mutex for it
    Mutex::Locker locker(m_state_mutex);
    return m_state;
}

void
Thread::SetState(StateType state)
{
    Mutex::Locker locker(m_state_mutex);
    m_state = state;
}

void
Thread::WillStop()
{
    ThreadPlan *current_plan = GetCurrentPlan();

    // FIXME: I may decide to disallow threads with no plans.  In which
    // case this should go to an assert.

    if (!current_plan)
        return;

    current_plan->WillStop();
}

void
Thread::SetupForResume ()
{
    if (GetResumeState() != eStateSuspended)
    {
    
        // If we're at a breakpoint push the step-over breakpoint plan.  Do this before
        // telling the current plan it will resume, since we might change what the current
        // plan is.

        StopReason stop_reason = lldb::eStopReasonInvalid;
        StopInfoSP stop_info_sp = GetStopInfo();
        if (stop_info_sp.get())
            stop_reason = stop_info_sp->GetStopReason();
        if (stop_reason == lldb::eStopReasonBreakpoint)
        {
            // Note, don't assume there's a ThreadPlanStepOverBreakpoint, the target may not require anything
            // special to step over a breakpoint.
                
            ThreadPlan *cur_plan = GetCurrentPlan();

            if (cur_plan->GetKind() != ThreadPlan::eKindStepOverBreakpoint)
            {
                ThreadPlanStepOverBreakpoint *step_bp_plan = new ThreadPlanStepOverBreakpoint (*this);
                if (step_bp_plan)
                {
                    ThreadPlanSP step_bp_plan_sp;
                    step_bp_plan->SetPrivate (true);

                    if (GetCurrentPlan()->RunState() != eStateStepping)
                    {
                        step_bp_plan->SetAutoContinue(true);
                    }
                    step_bp_plan_sp.reset (step_bp_plan);
                    QueueThreadPlan (step_bp_plan_sp, false);
                }
            }
        }
    }
}

bool
Thread::WillResume (StateType resume_state)
{
    // At this point clear the completed plan stack.
    m_completed_plan_stack.clear();
    m_discarded_plan_stack.clear();

    m_temporary_resume_state = resume_state;
    
    // This is a little dubious, but we are trying to limit how often we actually fetch stop info from
    // the target, 'cause that slows down single stepping.  So assume that if we got to the point where
    // we're about to resume, and we haven't yet had to fetch the stop reason, then it doesn't need to know
    // about the fact that we are resuming...
        const uint32_t process_stop_id = GetProcess()->GetStopID();
    if (m_thread_stop_reason_stop_id == process_stop_id &&
        (m_actual_stop_info_sp && m_actual_stop_info_sp->IsValid()))
    {
        StopInfo *stop_info = GetPrivateStopReason().get();
        if (stop_info)
            stop_info->WillResume (resume_state);
    }
    
    // Tell all the plans that we are about to resume in case they need to clear any state.
    // We distinguish between the plan on the top of the stack and the lower
    // plans in case a plan needs to do any special business before it runs.
    
    ThreadPlan *plan_ptr = GetCurrentPlan();
    bool need_to_resume = plan_ptr->WillResume(resume_state, true);

    while ((plan_ptr = GetPreviousPlan(plan_ptr)) != NULL)
    {
        plan_ptr->WillResume (resume_state, false);
    }
    
    // If the WillResume for the plan says we are faking a resume, then it will have set an appropriate stop info.
    // In that case, don't reset it here.
    
    if (need_to_resume && resume_state != eStateSuspended)
    {
        m_actual_stop_info_sp.reset();
    }

    return need_to_resume;
}

void
Thread::DidResume ()
{
    SetResumeSignal (LLDB_INVALID_SIGNAL_NUMBER);
}

bool
Thread::ShouldStop (Event* event_ptr)
{
    ThreadPlan *current_plan = GetCurrentPlan();
    bool should_stop = true;

    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_STEP));
    
    if (GetResumeState () == eStateSuspended)
    {
        if (log)
            log->Printf ("Thread::%s for tid = 0x%4.4llx, should_stop = 0 (ignore since thread was suspended)", 
                         __FUNCTION__, 
                         GetID ());
//            log->Printf ("Thread::%s for tid = 0x%4.4llx, pc = 0x%16.16llx, should_stop = 0 (ignore since thread was suspended)", 
//                         __FUNCTION__, 
//                         GetID (), 
//                         GetRegisterContext()->GetPC());
        return false;
    }
    
    if (GetTemporaryResumeState () == eStateSuspended)
    {
        if (log)
            log->Printf ("Thread::%s for tid = 0x%4.4llx, should_stop = 0 (ignore since thread was suspended)", 
                         __FUNCTION__, 
                         GetID ());
//            log->Printf ("Thread::%s for tid = 0x%4.4llx, pc = 0x%16.16llx, should_stop = 0 (ignore since thread was suspended)", 
//                         __FUNCTION__, 
//                         GetID (), 
//                         GetRegisterContext()->GetPC());
        return false;
    }
    
    if (ThreadStoppedForAReason() == false)
    {
        if (log)
            log->Printf ("Thread::%s for tid = 0x%4.4llx, pc = 0x%16.16llx, should_stop = 0 (ignore since no stop reason)", 
                         __FUNCTION__, 
                         GetID (), 
                         GetRegisterContext()->GetPC());
        return false;
    }
    
    if (log)
    {
        log->Printf ("Thread::%s for tid = 0x%4.4llx, pc = 0x%16.16llx", 
                     __FUNCTION__, 
                     GetID (), 
                     GetRegisterContext()->GetPC());
        log->Printf ("^^^^^^^^ Thread::ShouldStop Begin ^^^^^^^^");
        StreamString s;
        s.IndentMore();
        DumpThreadPlans(&s);
        log->Printf ("Plan stack initial state:\n%s", s.GetData());
    }
    
    // The top most plan always gets to do the trace log...
    current_plan->DoTraceLog ();
    
    // First query the stop info's ShouldStopSynchronous.  This handles "synchronous" stop reasons, for example the breakpoint
    // command on internal breakpoints.  If a synchronous stop reason says we should not stop, then we don't have to
    // do any more work on this stop.
    StopInfoSP private_stop_info (GetPrivateStopReason());
    if (private_stop_info && private_stop_info->ShouldStopSynchronous(event_ptr) == false)
    {
        if (log)
            log->Printf ("StopInfo::ShouldStop async callback says we should not stop, returning ShouldStop of false.");
        return false;
    }

    // If we've already been restarted, don't query the plans since the state they would examine is not current.
    if (Process::ProcessEventData::GetRestartedFromEvent(event_ptr))
        return false;

    // Before the plans see the state of the world, calculate the current inlined depth.
    GetStackFrameList()->CalculateCurrentInlinedDepth();

    // If the base plan doesn't understand why we stopped, then we have to find a plan that does.
    // If that plan is still working, then we don't need to do any more work.  If the plan that explains 
    // the stop is done, then we should pop all the plans below it, and pop it, and then let the plans above it decide
    // whether they still need to do more work.
    
    bool done_processing_current_plan = false;
    
    if (!current_plan->PlanExplainsStop())
    {
        if (current_plan->TracerExplainsStop())
        {
            done_processing_current_plan = true;
            should_stop = false;
        }
        else
        {
            // If the current plan doesn't explain the stop, then find one that
            // does and let it handle the situation.
            ThreadPlan *plan_ptr = current_plan;
            while ((plan_ptr = GetPreviousPlan(plan_ptr)) != NULL)
            {
                if (plan_ptr->PlanExplainsStop())
                {
                    should_stop = plan_ptr->ShouldStop (event_ptr);
                    
                    // plan_ptr explains the stop, next check whether plan_ptr is done, if so, then we should take it 
                    // and all the plans below it off the stack.
                    
                    if (plan_ptr->MischiefManaged())
                    {
                        // We're going to pop the plans up to and including the plan that explains the stop.
                        ThreadPlan *prev_plan_ptr = GetPreviousPlan (plan_ptr);
                        
                        do 
                        {
                            if (should_stop)
                                current_plan->WillStop();
                            PopPlan();
                        }
                        while ((current_plan = GetCurrentPlan()) != prev_plan_ptr);
                        // Now, if the responsible plan was not "Okay to discard" then we're done,
                        // otherwise we forward this to the next plan in the stack below.
                        if (plan_ptr->IsMasterPlan() && !plan_ptr->OkayToDiscard())
                            done_processing_current_plan = true;
                        else
                            done_processing_current_plan = false;
                    }
                    else
                        done_processing_current_plan = true;
                        
                    break;
                }

            }
        }
    }
    
    if (!done_processing_current_plan)
    {
        bool over_ride_stop = current_plan->ShouldAutoContinue(event_ptr);
        
        if (log)
            log->Printf("Plan %s explains stop, auto-continue %i.", current_plan->GetName(), over_ride_stop);
            
        // We're starting from the base plan, so just let it decide;
        if (PlanIsBasePlan(current_plan))
        {
            should_stop = current_plan->ShouldStop (event_ptr);
            if (log)
                log->Printf("Base plan says should stop: %i.", should_stop);
        }
        else
        {
            // Otherwise, don't let the base plan override what the other plans say to do, since
            // presumably if there were other plans they would know what to do...
            while (1)
            {
                if (PlanIsBasePlan(current_plan))
                    break;
                    
                should_stop = current_plan->ShouldStop(event_ptr);
                if (log)
                    log->Printf("Plan %s should stop: %d.", current_plan->GetName(), should_stop);
                if (current_plan->MischiefManaged())
                {
                    if (should_stop)
                        current_plan->WillStop();

                    // If a Master Plan wants to stop, and wants to stick on the stack, we let it.
                    // Otherwise, see if the plan's parent wants to stop.

                    if (should_stop && current_plan->IsMasterPlan() && !current_plan->OkayToDiscard())
                    {
                        PopPlan();
                        break;
                    }
                    else
                    {

                        PopPlan();

                        current_plan = GetCurrentPlan();
                        if (current_plan == NULL)
                        {
                            break;
                        }
                    }
                }
                else
                {
                    break;
                }
            }
        }
        
        if (over_ride_stop)
            should_stop = false;

        // One other potential problem is that we set up a master plan, then stop in before it is complete - for instance
        // by hitting a breakpoint during a step-over - then do some step/finish/etc operations that wind up
        // past the end point condition of the initial plan.  We don't want to strand the original plan on the stack,
        // This code clears stale plans off the stack.
        
        if (should_stop)
        {
            ThreadPlan *plan_ptr = GetCurrentPlan();
            while (!PlanIsBasePlan(plan_ptr))
            {
                bool stale = plan_ptr->IsPlanStale ();
                ThreadPlan *examined_plan = plan_ptr;
                plan_ptr = GetPreviousPlan (examined_plan);
                
                if (stale)
                {
                    if (log)
                        log->Printf("Plan %s being discarded in cleanup, it says it is already done.", examined_plan->GetName());
                    DiscardThreadPlansUpToPlan(examined_plan);
                }
            }
        }
        
    }

    if (log)
    {
        StreamString s;
        s.IndentMore();
        DumpThreadPlans(&s);
        log->Printf ("Plan stack final state:\n%s", s.GetData());
        log->Printf ("vvvvvvvv Thread::ShouldStop End (returning %i) vvvvvvvv", should_stop);
    }
    return should_stop;
}

Vote
Thread::ShouldReportStop (Event* event_ptr)
{
    StateType thread_state = GetResumeState ();
    StateType temp_thread_state = GetTemporaryResumeState();
    
    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_STEP));

    if (thread_state == eStateSuspended || thread_state == eStateInvalid)
    {
        if (log)
            log->Printf ("Thread::ShouldReportStop() tid = 0x%4.4llx: returning vote %i (state was suspended or invalid)\n", GetID(), eVoteNoOpinion);
        return eVoteNoOpinion;
    }

    if (temp_thread_state == eStateSuspended || temp_thread_state == eStateInvalid)
    {
        if (log)
            log->Printf ("Thread::ShouldReportStop() tid = 0x%4.4llx: returning vote %i (temporary state was suspended or invalid)\n", GetID(), eVoteNoOpinion);
        return eVoteNoOpinion;
    }

    if (!ThreadStoppedForAReason())
    {
        if (log)
            log->Printf ("Thread::ShouldReportStop() tid = 0x%4.4llx: returning vote %i (thread didn't stop for a reason.)\n", GetID(), eVoteNoOpinion);
        return eVoteNoOpinion;
    }

    if (m_completed_plan_stack.size() > 0)
    {
        // Don't use GetCompletedPlan here, since that suppresses private plans.
        if (log)
            log->Printf ("Thread::ShouldReportStop() tid = 0x%4.4llx: returning vote  for complete stack's back plan\n", GetID());
        return m_completed_plan_stack.back()->ShouldReportStop (event_ptr);
    }
    else
    {
        if (log)
            log->Printf ("Thread::ShouldReportStop() tid = 0x%4.4llx: returning vote  for current plan\n", GetID());
        return GetCurrentPlan()->ShouldReportStop (event_ptr);
    }
}

Vote
Thread::ShouldReportRun (Event* event_ptr)
{
    StateType thread_state = GetResumeState ();
    
    if (thread_state == eStateSuspended
            || thread_state == eStateInvalid)
    {
        return eVoteNoOpinion;
    }
    
    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_STEP));
    if (m_completed_plan_stack.size() > 0)
    {
        // Don't use GetCompletedPlan here, since that suppresses private plans.
        if (log)
            log->Printf ("Current Plan for thread %d (0x%4.4llx): %s being asked whether we should report run.", 
                         GetIndexID(), 
                         GetID(),
                         m_completed_plan_stack.back()->GetName());
                         
        return m_completed_plan_stack.back()->ShouldReportRun (event_ptr);
    }
    else
    {
        if (log)
            log->Printf ("Current Plan for thread %d (0x%4.4llx): %s being asked whether we should report run.", 
                         GetIndexID(), 
                         GetID(),
                         GetCurrentPlan()->GetName());
                         
        return GetCurrentPlan()->ShouldReportRun (event_ptr);
     }
}

bool
Thread::MatchesSpec (const ThreadSpec *spec)
{
    if (spec == NULL)
        return true;
        
    return spec->ThreadPassesBasicTests(*this);
}

void
Thread::PushPlan (ThreadPlanSP &thread_plan_sp)
{
    if (thread_plan_sp)
    {
        // If the thread plan doesn't already have a tracer, give it its parent's tracer:
        if (!thread_plan_sp->GetThreadPlanTracer())
            thread_plan_sp->SetThreadPlanTracer(m_plan_stack.back()->GetThreadPlanTracer());
        m_plan_stack.push_back (thread_plan_sp);
            
        thread_plan_sp->DidPush();

        LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_STEP));
        if (log)
        {
            StreamString s;
            thread_plan_sp->GetDescription (&s, lldb::eDescriptionLevelFull);
            log->Printf("Pushing plan: \"%s\", tid = 0x%4.4llx.",
                        s.GetData(),
                        thread_plan_sp->GetThread().GetID());
        }
    }
}

void
Thread::PopPlan ()
{
    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_STEP));

    if (m_plan_stack.size() <= 1)
        return;
    else
    {
        ThreadPlanSP &plan = m_plan_stack.back();
        if (log)
        {
            log->Printf("Popping plan: \"%s\", tid = 0x%4.4llx.", plan->GetName(), plan->GetThread().GetID());
        }
        m_completed_plan_stack.push_back (plan);
        plan->WillPop();
        m_plan_stack.pop_back();
    }
}

void
Thread::DiscardPlan ()
{
    if (m_plan_stack.size() > 1)
    {
        ThreadPlanSP &plan = m_plan_stack.back();
        m_discarded_plan_stack.push_back (plan);
        plan->WillPop();
        m_plan_stack.pop_back();
    }
}

ThreadPlan *
Thread::GetCurrentPlan ()
{
    // There will always be at least the base plan.  If somebody is mucking with a
    // thread with an empty plan stack, we should assert right away.
    assert (!m_plan_stack.empty());

    return m_plan_stack.back().get();
}

ThreadPlanSP
Thread::GetCompletedPlan ()
{
    ThreadPlanSP empty_plan_sp;
    if (!m_completed_plan_stack.empty())
    {
        for (int i = m_completed_plan_stack.size() - 1; i >= 0; i--)
        {
            ThreadPlanSP completed_plan_sp;
            completed_plan_sp = m_completed_plan_stack[i];
            if (!completed_plan_sp->GetPrivate ())
            return completed_plan_sp;
        }
    }
    return empty_plan_sp;
}

ValueObjectSP
Thread::GetReturnValueObject ()
{
    if (!m_completed_plan_stack.empty())
    {
        for (int i = m_completed_plan_stack.size() - 1; i >= 0; i--)
        {
            ValueObjectSP return_valobj_sp;
            return_valobj_sp = m_completed_plan_stack[i]->GetReturnValueObject();
            if (return_valobj_sp)
            return return_valobj_sp;
        }
    }
    return ValueObjectSP();
}

bool
Thread::IsThreadPlanDone (ThreadPlan *plan)
{
    if (!m_completed_plan_stack.empty())
    {
        for (int i = m_completed_plan_stack.size() - 1; i >= 0; i--)
        {
            if (m_completed_plan_stack[i].get() == plan)
                return true;
        }
    }
    return false;
}

bool
Thread::WasThreadPlanDiscarded (ThreadPlan *plan)
{
    if (!m_discarded_plan_stack.empty())
    {
        for (int i = m_discarded_plan_stack.size() - 1; i >= 0; i--)
        {
            if (m_discarded_plan_stack[i].get() == plan)
                return true;
        }
    }
    return false;
}

ThreadPlan *
Thread::GetPreviousPlan (ThreadPlan *current_plan)
{
    if (current_plan == NULL)
        return NULL;

    int stack_size = m_completed_plan_stack.size();
    for (int i = stack_size - 1; i > 0; i--)
    {
        if (current_plan == m_completed_plan_stack[i].get())
            return m_completed_plan_stack[i-1].get();
    }

    if (stack_size > 0 && m_completed_plan_stack[0].get() == current_plan)
    {
        if (m_plan_stack.size() > 0)
            return m_plan_stack.back().get();
        else
            return NULL;
    }

    stack_size = m_plan_stack.size();
    for (int i = stack_size - 1; i > 0; i--)
    {
        if (current_plan == m_plan_stack[i].get())
            return m_plan_stack[i-1].get();
    }
    return NULL;
}

void
Thread::QueueThreadPlan (ThreadPlanSP &thread_plan_sp, bool abort_other_plans)
{
    if (abort_other_plans)
       DiscardThreadPlans(true);

    PushPlan (thread_plan_sp);
}


void
Thread::EnableTracer (bool value, bool single_stepping)
{
    int stack_size = m_plan_stack.size();
    for (int i = 0; i < stack_size; i++)
    {
        if (m_plan_stack[i]->GetThreadPlanTracer())
        {
            m_plan_stack[i]->GetThreadPlanTracer()->EnableTracing(value);
            m_plan_stack[i]->GetThreadPlanTracer()->EnableSingleStep(single_stepping);
        }
    }
}

void
Thread::SetTracer (lldb::ThreadPlanTracerSP &tracer_sp)
{
    int stack_size = m_plan_stack.size();
    for (int i = 0; i < stack_size; i++)
        m_plan_stack[i]->SetThreadPlanTracer(tracer_sp);
}

void
Thread::DiscardThreadPlansUpToPlan (lldb::ThreadPlanSP &up_to_plan_sp)
{
    DiscardThreadPlansUpToPlan (up_to_plan_sp.get());
}

void
Thread::DiscardThreadPlansUpToPlan (ThreadPlan *up_to_plan_ptr)
{
    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_STEP));
    if (log)
    {
        log->Printf("Discarding thread plans for thread tid = 0x%4.4llx, up to %p", GetID(), up_to_plan_ptr);
    }

    int stack_size = m_plan_stack.size();
    
    // If the input plan is NULL, discard all plans.  Otherwise make sure this plan is in the
    // stack, and if so discard up to and including it.
    
    if (up_to_plan_ptr == NULL)
    {
        for (int i = stack_size - 1; i > 0; i--)
            DiscardPlan();
    }
    else
    {
        bool found_it = false;
        for (int i = stack_size - 1; i > 0; i--)
        {
            if (m_plan_stack[i].get() == up_to_plan_ptr)
                found_it = true;
        }
        if (found_it)
        {
            bool last_one = false;
            for (int i = stack_size - 1; i > 0 && !last_one ; i--)
            {
                if (GetCurrentPlan() == up_to_plan_ptr)
                    last_one = true;
                DiscardPlan();
            }
        }
    }
    return;
}

void
Thread::DiscardThreadPlans(bool force)
{
    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_STEP));
    if (log)
    {
        log->Printf("Discarding thread plans for thread (tid = 0x%4.4llx, force %d)", GetID(), force);
    }

    if (force)
    {
        int stack_size = m_plan_stack.size();
        for (int i = stack_size - 1; i > 0; i--)
        {
            DiscardPlan();
        }
        return;
    }

    while (1)
    {

        int master_plan_idx;
        bool discard = true;

        // Find the first master plan, see if it wants discarding, and if yes discard up to it.
        for (master_plan_idx = m_plan_stack.size() - 1; master_plan_idx >= 0; master_plan_idx--)
        {
            if (m_plan_stack[master_plan_idx]->IsMasterPlan())
            {
                discard = m_plan_stack[master_plan_idx]->OkayToDiscard();
                break;
            }
        }

        if (discard)
        {
            // First pop all the dependent plans:
            for (int i = m_plan_stack.size() - 1; i > master_plan_idx; i--)
            {

                // FIXME: Do we need a finalize here, or is the rule that "PrepareForStop"
                // for the plan leaves it in a state that it is safe to pop the plan
                // with no more notice?
                DiscardPlan();
            }

            // Now discard the master plan itself.
            // The bottom-most plan never gets discarded.  "OkayToDiscard" for it means
            // discard it's dependent plans, but not it...
            if (master_plan_idx > 0)
            {
                DiscardPlan();
            }
        }
        else
        {
            // If the master plan doesn't want to get discarded, then we're done.
            break;
        }

    }
}

bool
Thread::PlanIsBasePlan (ThreadPlan *plan_ptr)
{
    if (plan_ptr->IsBasePlan())
        return true;
    else if (m_plan_stack.size() == 0)
        return false;
    else
       return m_plan_stack[0].get() == plan_ptr;
}

ThreadPlan *
Thread::QueueFundamentalPlan (bool abort_other_plans)
{
    ThreadPlanSP thread_plan_sp (new ThreadPlanBase(*this));
    QueueThreadPlan (thread_plan_sp, abort_other_plans);
    return thread_plan_sp.get();
}

ThreadPlan *
Thread::QueueThreadPlanForStepSingleInstruction
(
    bool step_over, 
    bool abort_other_plans, 
    bool stop_other_threads
)
{
    ThreadPlanSP thread_plan_sp (new ThreadPlanStepInstruction (*this, step_over, stop_other_threads, eVoteNoOpinion, eVoteNoOpinion));
    QueueThreadPlan (thread_plan_sp, abort_other_plans);
    return thread_plan_sp.get();
}

ThreadPlan *
Thread::QueueThreadPlanForStepRange 
(
    bool abort_other_plans, 
    StepType type, 
    const AddressRange &range, 
    const SymbolContext &addr_context, 
    lldb::RunMode stop_other_threads,
    bool avoid_code_without_debug_info
)
{
    ThreadPlanSP thread_plan_sp;
    if (type == eStepTypeInto)
    {
        ThreadPlanStepInRange *plan = new ThreadPlanStepInRange (*this, range, addr_context, stop_other_threads);
        if (avoid_code_without_debug_info)
            plan->GetFlags().Set (ThreadPlanShouldStopHere::eAvoidNoDebug);
        else
            plan->GetFlags().Clear (ThreadPlanShouldStopHere::eAvoidNoDebug);
        thread_plan_sp.reset (plan);
    }
    else
        thread_plan_sp.reset (new ThreadPlanStepOverRange (*this, range, addr_context, stop_other_threads));

    QueueThreadPlan (thread_plan_sp, abort_other_plans);
    return thread_plan_sp.get();
}


ThreadPlan *
Thread::QueueThreadPlanForStepOverBreakpointPlan (bool abort_other_plans)
{
    ThreadPlanSP thread_plan_sp (new ThreadPlanStepOverBreakpoint (*this));
    QueueThreadPlan (thread_plan_sp, abort_other_plans);
    return thread_plan_sp.get();
}

ThreadPlan *
Thread::QueueThreadPlanForStepOut 
(
    bool abort_other_plans, 
    SymbolContext *addr_context, 
    bool first_insn,
    bool stop_other_threads, 
    Vote stop_vote, 
    Vote run_vote,
    uint32_t frame_idx
)
{
    ThreadPlanSP thread_plan_sp (new ThreadPlanStepOut (*this, 
                                                        addr_context, 
                                                        first_insn, 
                                                        stop_other_threads, 
                                                        stop_vote, 
                                                        run_vote, 
                                                        frame_idx));
    
    if (thread_plan_sp->ValidatePlan(NULL))
    {
        QueueThreadPlan (thread_plan_sp, abort_other_plans);
        return thread_plan_sp.get();
    }
    else
    {
        return NULL;
    }
}

ThreadPlan *
Thread::QueueThreadPlanForStepThrough (StackID &return_stack_id, bool abort_other_plans, bool stop_other_threads)
{
    ThreadPlanSP thread_plan_sp(new ThreadPlanStepThrough (*this, return_stack_id, stop_other_threads));
    if (!thread_plan_sp || !thread_plan_sp->ValidatePlan (NULL))
        return NULL;

    QueueThreadPlan (thread_plan_sp, abort_other_plans);
    return thread_plan_sp.get();
}

ThreadPlan *
Thread::QueueThreadPlanForCallFunction (bool abort_other_plans,
                                        Address& function,
                                        lldb::addr_t arg,
                                        bool stop_other_threads,
                                        bool discard_on_error)
{
    ThreadPlanSP thread_plan_sp (new ThreadPlanCallFunction (*this, function, ClangASTType(), arg, stop_other_threads, discard_on_error));
    QueueThreadPlan (thread_plan_sp, abort_other_plans);
    return thread_plan_sp.get();
}

ThreadPlan *
Thread::QueueThreadPlanForRunToAddress (bool abort_other_plans,
                                        Address &target_addr,
                                        bool stop_other_threads)
{
    ThreadPlanSP thread_plan_sp (new ThreadPlanRunToAddress (*this, target_addr, stop_other_threads));
    QueueThreadPlan (thread_plan_sp, abort_other_plans);
    return thread_plan_sp.get();
}

ThreadPlan *
Thread::QueueThreadPlanForStepUntil (bool abort_other_plans,
                                     lldb::addr_t *address_list,
                                     size_t num_addresses,
                                     bool stop_other_threads,
                                     uint32_t frame_idx)
{
    ThreadPlanSP thread_plan_sp (new ThreadPlanStepUntil (*this, address_list, num_addresses, stop_other_threads, frame_idx));
    QueueThreadPlan (thread_plan_sp, abort_other_plans);
    return thread_plan_sp.get();

}

uint32_t
Thread::GetIndexID () const
{
    return m_index_id;
}

void
Thread::DumpThreadPlans (lldb_private::Stream *s) const
{
    uint32_t stack_size = m_plan_stack.size();
    int i;
    s->Indent();
    s->Printf ("Plan Stack for thread #%u: tid = 0x%4.4llx, stack_size = %d\n", GetIndexID(), GetID(), stack_size);
    for (i = stack_size - 1; i >= 0; i--)
    {
        s->IndentMore();
        s->Indent();
        s->Printf ("Element %d: ", i);
        m_plan_stack[i]->GetDescription (s, eDescriptionLevelFull);
        s->EOL();
        s->IndentLess();
    }

    stack_size = m_completed_plan_stack.size();
    if (stack_size > 0)
    {
        s->Indent();
        s->Printf ("Completed Plan Stack: %d elements.\n", stack_size);
        for (i = stack_size - 1; i >= 0; i--)
        {
            s->IndentMore();
            s->Indent();
            s->Printf ("Element %d: ", i);
            m_completed_plan_stack[i]->GetDescription (s, eDescriptionLevelFull);
            s->EOL();
            s->IndentLess();
        }
    }

    stack_size = m_discarded_plan_stack.size();
    if (stack_size > 0)
    {
        s->Indent();
        s->Printf ("Discarded Plan Stack: %d elements.\n", stack_size);
        for (i = stack_size - 1; i >= 0; i--)
        {
            s->IndentMore();
            s->Indent();
            s->Printf ("Element %d: ", i);
            m_discarded_plan_stack[i]->GetDescription (s, eDescriptionLevelFull);
            s->EOL();
            s->IndentLess();
        }
    }

}

TargetSP
Thread::CalculateTarget ()
{
    TargetSP target_sp;
    ProcessSP process_sp(GetProcess());
    if (process_sp)
        target_sp = process_sp->CalculateTarget();
    return target_sp;
    
}

ProcessSP
Thread::CalculateProcess ()
{
    return GetProcess();
}

ThreadSP
Thread::CalculateThread ()
{
    return shared_from_this();
}

StackFrameSP
Thread::CalculateStackFrame ()
{
    return StackFrameSP();
}

void
Thread::CalculateExecutionContext (ExecutionContext &exe_ctx)
{
    exe_ctx.SetContext (shared_from_this());
}


StackFrameListSP
Thread::GetStackFrameList ()
{
    StackFrameListSP frame_list_sp;
    Mutex::Locker locker(m_frame_mutex);
    if (m_curr_frames_sp)
    {
        frame_list_sp = m_curr_frames_sp;
    }
    else
    {
        frame_list_sp.reset(new StackFrameList (*this, m_prev_frames_sp, true));
        m_curr_frames_sp = frame_list_sp;
    }
    return frame_list_sp;
}

void
Thread::ClearStackFrames ()
{
    Mutex::Locker locker(m_frame_mutex);

    // Only store away the old "reference" StackFrameList if we got all its frames:
    // FIXME: At some point we can try to splice in the frames we have fetched into
    // the new frame as we make it, but let's not try that now.
    if (m_curr_frames_sp && m_curr_frames_sp->GetAllFramesFetched())
        m_prev_frames_sp.swap (m_curr_frames_sp);
    m_curr_frames_sp.reset();
}

lldb::StackFrameSP
Thread::GetFrameWithConcreteFrameIndex (uint32_t unwind_idx)
{
    return GetStackFrameList()->GetFrameWithConcreteFrameIndex (unwind_idx);
}


Error
Thread::ReturnFromFrameWithIndex (uint32_t frame_idx, lldb::ValueObjectSP return_value_sp, bool broadcast)
{
    StackFrameSP frame_sp = GetStackFrameAtIndex (frame_idx);
    Error return_error;
    
    if (!frame_sp)
    {
        return_error.SetErrorStringWithFormat("Could not find frame with index %d in thread 0x%llx.", frame_idx, GetID());
    }
    
    return ReturnFromFrame(frame_sp, return_value_sp, broadcast);
}

Error
Thread::ReturnFromFrame (lldb::StackFrameSP frame_sp, lldb::ValueObjectSP return_value_sp, bool broadcast)
{
    Error return_error;
    
    if (!frame_sp)
    {
        return_error.SetErrorString("Can't return to a null frame.");
        return return_error;
    }
    
    Thread *thread = frame_sp->GetThread().get();
    uint32_t older_frame_idx = frame_sp->GetFrameIndex() + 1;
    StackFrameSP older_frame_sp = thread->GetStackFrameAtIndex(older_frame_idx);
    
    if (return_value_sp)
    {    
        lldb::ABISP abi = thread->GetProcess()->GetABI();
        if (!abi)
        {
            return_error.SetErrorString("Could not find ABI to set return value.");
            return return_error;
        }
        SymbolContext sc = frame_sp->GetSymbolContext(eSymbolContextFunction);
        
        // FIXME: ValueObject::Cast doesn't currently work correctly, at least not for scalars.
        // Turn that back on when that works.
        if (0 && sc.function != NULL)
        {
            Type *function_type = sc.function->GetType();
            if (function_type)
            {
                clang_type_t return_type = sc.function->GetReturnClangType();
                if (return_type)
                {
                    ClangASTType ast_type (function_type->GetClangAST(), return_type);
                    StreamString s;
                    ast_type.DumpTypeDescription(&s);
                    ValueObjectSP cast_value_sp = return_value_sp->Cast(ast_type);
                    if (cast_value_sp)
                    {
                        cast_value_sp->SetFormat(eFormatHex);
                        return_value_sp = cast_value_sp;
                    }
                }
            }
        }

        return_error = abi->SetReturnValueObject(older_frame_sp, return_value_sp);
        if (!return_error.Success())
            return return_error;
    }
    
    // Now write the return registers for the chosen frame:
    // Note, we can't use ReadAllRegisterValues->WriteAllRegisterValues, since the read & write
    // cook their data 
    bool copy_success = thread->GetStackFrameAtIndex(0)->GetRegisterContext()->CopyFromRegisterContext(older_frame_sp->GetRegisterContext());
    if (copy_success)
    {
        thread->DiscardThreadPlans(true);
        thread->ClearStackFrames();
        if (broadcast && EventTypeHasListeners(eBroadcastBitStackChanged))
            BroadcastEvent(eBroadcastBitStackChanged, new ThreadEventData (this->shared_from_this()));
        return return_error;
    }
    else
    {
        return_error.SetErrorString("Could not reset register values.");
        return return_error;
    }
}

void
Thread::DumpUsingSettingsFormat (Stream &strm, uint32_t frame_idx)
{
    ExecutionContext exe_ctx (shared_from_this());
    Process *process = exe_ctx.GetProcessPtr();
    if (process == NULL)
        return;

    StackFrameSP frame_sp;
    SymbolContext frame_sc;
    if (frame_idx != LLDB_INVALID_INDEX32)
    {
        frame_sp = GetStackFrameAtIndex (frame_idx);
        if (frame_sp)
        {
            exe_ctx.SetFrameSP(frame_sp);
            frame_sc = frame_sp->GetSymbolContext(eSymbolContextEverything);
        }
    }

    const char *thread_format = exe_ctx.GetTargetRef().GetDebugger().GetThreadFormat();
    assert (thread_format);
    const char *end = NULL;
    Debugger::FormatPrompt (thread_format, 
                            frame_sp ? &frame_sc : NULL,
                            &exe_ctx, 
                            NULL,
                            strm, 
                            &end);
}

void
Thread::SettingsInitialize ()
{
}

void
Thread::SettingsTerminate ()
{
}

lldb::StackFrameSP
Thread::GetStackFrameSPForStackFramePtr (StackFrame *stack_frame_ptr)
{
    return GetStackFrameList()->GetStackFrameSPForStackFramePtr (stack_frame_ptr);
}

const char *
Thread::StopReasonAsCString (lldb::StopReason reason)
{
    switch (reason)
    {
    case eStopReasonInvalid:      return "invalid";
    case eStopReasonNone:         return "none";
    case eStopReasonTrace:        return "trace";
    case eStopReasonBreakpoint:   return "breakpoint";
    case eStopReasonWatchpoint:   return "watchpoint";
    case eStopReasonSignal:       return "signal";
    case eStopReasonException:    return "exception";
    case eStopReasonPlanComplete: return "plan complete";
    }


    static char unknown_state_string[64];
    snprintf(unknown_state_string, sizeof (unknown_state_string), "StopReason = %i", reason);
    return unknown_state_string;
}

const char *
Thread::RunModeAsCString (lldb::RunMode mode)
{
    switch (mode)
    {
    case eOnlyThisThread:     return "only this thread";
    case eAllThreads:         return "all threads";
    case eOnlyDuringStepping: return "only during stepping";
    }

    static char unknown_state_string[64];
    snprintf(unknown_state_string, sizeof (unknown_state_string), "RunMode = %i", mode);
    return unknown_state_string;
}

size_t
Thread::GetStatus (Stream &strm, uint32_t start_frame, uint32_t num_frames, uint32_t num_frames_with_source)
{
    ExecutionContext exe_ctx (shared_from_this());
    Target *target = exe_ctx.GetTargetPtr();
    Process *process = exe_ctx.GetProcessPtr();
    size_t num_frames_shown = 0;
    strm.Indent();
    bool is_selected = false;
    if (process)
    {
        if (process->GetThreadList().GetSelectedThread().get() == this)
            is_selected = true;
    }
    strm.Printf("%c ", is_selected ? '*' : ' ');
    if (target && target->GetDebugger().GetUseExternalEditor())
    {
        StackFrameSP frame_sp = GetStackFrameAtIndex(start_frame);
        if (frame_sp)
        {
            SymbolContext frame_sc(frame_sp->GetSymbolContext (eSymbolContextLineEntry));
            if (frame_sc.line_entry.line != 0 && frame_sc.line_entry.file)
            {
                Host::OpenFileInExternalEditor (frame_sc.line_entry.file, frame_sc.line_entry.line);
            }
        }
    }
    
    DumpUsingSettingsFormat (strm, start_frame);
    
    if (num_frames > 0)
    {
        strm.IndentMore();
        
        const bool show_frame_info = true;
        strm.IndentMore ();
        num_frames_shown = GetStackFrameList ()->GetStatus (strm,
                                                            start_frame, 
                                                            num_frames, 
                                                            show_frame_info, 
                                                            num_frames_with_source);
        strm.IndentLess();
        strm.IndentLess();
    }
    return num_frames_shown;
}

size_t
Thread::GetStackFrameStatus (Stream& strm,
                             uint32_t first_frame,
                             uint32_t num_frames,
                             bool show_frame_info,
                             uint32_t num_frames_with_source)
{
    return GetStackFrameList()->GetStatus (strm,
                                           first_frame,
                                           num_frames,
                                           show_frame_info,
                                           num_frames_with_source);
}

bool
Thread::SaveFrameZeroState (RegisterCheckpoint &checkpoint)
{
    lldb::StackFrameSP frame_sp(GetStackFrameAtIndex (0));
    if (frame_sp)
    {
        checkpoint.SetStackID(frame_sp->GetStackID());
        return frame_sp->GetRegisterContext()->ReadAllRegisterValues (checkpoint.GetData());
    }
    return false;
}

bool
Thread::RestoreSaveFrameZero (const RegisterCheckpoint &checkpoint)
{
    return ResetFrameZeroRegisters (checkpoint.GetData());
}

bool
Thread::ResetFrameZeroRegisters (lldb::DataBufferSP register_data_sp)
{
    lldb::StackFrameSP frame_sp(GetStackFrameAtIndex (0));
    if (frame_sp)
    {
        bool ret = frame_sp->GetRegisterContext()->WriteAllRegisterValues (register_data_sp);

        // Clear out all stack frames as our world just changed.
        ClearStackFrames();
        frame_sp->GetRegisterContext()->InvalidateIfNeeded(true);

        return ret;
    }
    return false;
}

Unwind *
Thread::GetUnwinder ()
{
    if (m_unwinder_ap.get() == NULL)
    {
        const ArchSpec target_arch (CalculateTarget()->GetArchitecture ());
        const llvm::Triple::ArchType machine = target_arch.GetMachine();
        switch (machine)
        {
            case llvm::Triple::x86_64:
            case llvm::Triple::x86:
            case llvm::Triple::arm:
            case llvm::Triple::thumb:
                m_unwinder_ap.reset (new UnwindLLDB (*this));
                break;
                
            default:
                if (target_arch.GetTriple().getVendor() == llvm::Triple::Apple)
                    m_unwinder_ap.reset (new UnwindMacOSXFrameBackchain (*this));
                break;
        }
    }
    return m_unwinder_ap.get();
}


void
Thread::Flush ()
{
    ClearStackFrames ();
    m_reg_context_sp.reset();
}

const bool
Thread::IsStillAtLastBreakpointHit ()
{
    // If we are currently stopped at a breakpoint, always return that stopinfo and don't reset it.
    // This allows threads to maintain their breakpoint stopinfo, such as when thread-stepping in
    // multithreaded programs.
    if (m_actual_stop_info_sp) {
        StopReason stop_reason = m_actual_stop_info_sp->GetStopReason();
        if (stop_reason == lldb::eStopReasonBreakpoint) {
            uint64_t value = m_actual_stop_info_sp->GetValue();
            lldb::addr_t pc = GetRegisterContext()->GetPC();
            BreakpointSiteSP bp_site_sp = GetProcess()->GetBreakpointSiteList().FindByAddress(pc);
            if (bp_site_sp && value == bp_site_sp->GetID())
                return true;
        }
    }
    return false;
}
