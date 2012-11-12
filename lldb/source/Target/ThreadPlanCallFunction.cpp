//===-- ThreadPlanCallFunction.cpp ------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Target/ThreadPlanCallFunction.h"

// C Includes
// C++ Includes
// Other libraries and framework includes
#include "llvm/Support/MachO.h"
// Project includes
#include "lldb/lldb-private-log.h"
#include "lldb/Breakpoint/Breakpoint.h"
#include "lldb/Breakpoint/BreakpointLocation.h"
#include "lldb/Core/Address.h"
#include "lldb/Core/Log.h"
#include "lldb/Core/Module.h"
#include "lldb/Core/Stream.h"
#include "lldb/Symbol/ObjectFile.h"
#include "lldb/Target/LanguageRuntime.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/RegisterContext.h"
#include "lldb/Target/StopInfo.h"
#include "lldb/Target/Target.h"
#include "lldb/Target/Thread.h"
#include "lldb/Target/ThreadPlanRunToAddress.h"

using namespace lldb;
using namespace lldb_private;

//----------------------------------------------------------------------
// ThreadPlanCallFunction: Plan to call a single function
//----------------------------------------------------------------------
bool
ThreadPlanCallFunction::ConstructorSetup (Thread &thread,
                                          ABI *& abi,
                                          lldb::addr_t &start_load_addr,
                                          lldb::addr_t &function_load_addr)
{
    SetIsMasterPlan (true);
    SetOkayToDiscard (false);

    ProcessSP process_sp (thread.GetProcess());
    if (!process_sp)
        return false;
    
    abi = process_sp->GetABI().get();
    
    if (!abi)
        return false;
    
    TargetSP target_sp (thread.CalculateTarget());

    LogSP log(lldb_private::GetLogIfAnyCategoriesSet (LIBLLDB_LOG_STEP));
    
    SetBreakpoints();
    
    m_function_sp = thread.GetRegisterContext()->GetSP() - abi->GetRedZoneSize();
    // If we can't read memory at the point of the process where we are planning to put our function, we're
    // not going to get any further...
    Error error;
    process_sp->ReadUnsignedIntegerFromMemory(m_function_sp, 4, 0, error);
    if (!error.Success())
    {
        if (log)
            log->Printf ("ThreadPlanCallFunction(%p): Trying to put the stack in unreadable memory at: 0x%llx.", this, m_function_sp);
        return false;
    }
    
    Module *exe_module = target_sp->GetExecutableModulePointer();

    if (exe_module == NULL)
    {
        if (log)
            log->Printf ("ThreadPlanCallFunction(%p): Can't execute code without an executable module.", this);
        return false;
    }
    else
    {
        ObjectFile *objectFile = exe_module->GetObjectFile();
        if (!objectFile)
        {
            if (log)
                log->Printf ("ThreadPlanCallFunction(%p): Could not find object file for module \"%s\".", 
                             this, exe_module->GetFileSpec().GetFilename().AsCString());
            return false;
        }
        m_start_addr = objectFile->GetEntryPointAddress();
        if (!m_start_addr.IsValid())
        {
            if (log)
                log->Printf ("ThreadPlanCallFunction(%p): Could not find entry point address for executable module \"%s\".", 
                             this, exe_module->GetFileSpec().GetFilename().AsCString());
            return false;
        }
    }
    
    start_load_addr = m_start_addr.GetLoadAddress (target_sp.get());
    
    // Checkpoint the thread state so we can restore it later.
    if (log && log->GetVerbose())
        ReportRegisterState ("About to checkpoint thread before function call.  Original register state was:");

    if (!thread.CheckpointThreadState (m_stored_thread_state))
    {
        if (log)
            log->Printf ("ThreadPlanCallFunction(%p): Setting up ThreadPlanCallFunction, failed to checkpoint thread state.", this);
        return false;
    }
    // Now set the thread state to "no reason" so we don't run with whatever signal was outstanding...
    thread.SetStopInfoToNothing();
    
    function_load_addr = m_function_addr.GetLoadAddress (target_sp.get());
    
    return true;
}

ThreadPlanCallFunction::ThreadPlanCallFunction (Thread &thread,
                                                Address &function,
                                                const ClangASTType &return_type,
                                                addr_t arg,
                                                bool stop_other_threads,
                                                bool discard_on_error,
                                                addr_t *this_arg,
                                                addr_t *cmd_arg) :
    ThreadPlan (ThreadPlan::eKindCallFunction, "Call function plan", thread, eVoteNoOpinion, eVoteNoOpinion),
    m_valid (false),
    m_stop_other_threads (stop_other_threads),
    m_function_addr (function),
    m_function_sp (0),
    m_return_type (return_type),
    m_takedown_done (false),
    m_stop_address (LLDB_INVALID_ADDRESS),
    m_discard_on_error (discard_on_error)
{
    lldb::addr_t start_load_addr;
    ABI *abi;
    lldb::addr_t function_load_addr;
    if (!ConstructorSetup (thread, abi, start_load_addr, function_load_addr))
        return;
        
    if (this_arg && cmd_arg)
    {
        if (!abi->PrepareTrivialCall (thread, 
                                      m_function_sp, 
                                      function_load_addr, 
                                      start_load_addr, 
                                      this_arg,
                                      cmd_arg,
                                      &arg))
            return;
    }
    else if (this_arg)
    {
        if (!abi->PrepareTrivialCall (thread, 
                                      m_function_sp, 
                                      function_load_addr, 
                                      start_load_addr, 
                                      this_arg,
                                      &arg))
            return;
    }
    else
    {
        if (!abi->PrepareTrivialCall (thread, 
                                      m_function_sp, 
                                      function_load_addr, 
                                      start_load_addr, 
                                      &arg))
            return;
    }
    
    ReportRegisterState ("Function call was set up.  Register state was:");
    
    m_valid = true;    
}


ThreadPlanCallFunction::ThreadPlanCallFunction (Thread &thread,
                                                Address &function,
                                                const ClangASTType &return_type,
                                                bool stop_other_threads,
                                                bool discard_on_error,
                                                addr_t *arg1_ptr,
                                                addr_t *arg2_ptr,
                                                addr_t *arg3_ptr,
                                                addr_t *arg4_ptr,
                                                addr_t *arg5_ptr,
                                                addr_t *arg6_ptr) :
    ThreadPlan (ThreadPlan::eKindCallFunction, "Call function plan", thread, eVoteNoOpinion, eVoteNoOpinion),
    m_valid (false),
    m_stop_other_threads (stop_other_threads),
    m_function_addr (function),
    m_function_sp(0),
    m_return_type (return_type),
    m_takedown_done (false),
    m_stop_address (LLDB_INVALID_ADDRESS)
{
    lldb::addr_t start_load_addr;
    ABI *abi;
    lldb::addr_t function_load_addr;
    if (!ConstructorSetup (thread, abi, start_load_addr, function_load_addr))
        return;
    
    if (!abi->PrepareTrivialCall (thread, 
                                  m_function_sp,
                                  function_load_addr, 
                                  start_load_addr, 
                                  arg1_ptr,
                                  arg2_ptr,
                                  arg3_ptr,
                                  arg4_ptr,
                                  arg5_ptr,
                                  arg6_ptr))
    {
            return;
    }
    
    ReportRegisterState ("Function call was set up.  Register state was:");
    
    m_valid = true;    
}

ThreadPlanCallFunction::~ThreadPlanCallFunction ()
{
    DoTakedown(true);
}

void
ThreadPlanCallFunction::ReportRegisterState (const char *message)
{
    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_STEP | LIBLLDB_LOG_VERBOSE));
    if (log)
    {
        StreamString strm;
        RegisterContext *reg_ctx = m_thread.GetRegisterContext().get();

        log->PutCString(message);

        RegisterValue reg_value;

        for (uint32_t reg_idx = 0, num_registers = reg_ctx->GetRegisterCount();
             reg_idx < num_registers;
             ++reg_idx)
        {
            const RegisterInfo *reg_info = reg_ctx->GetRegisterInfoAtIndex (reg_idx);
            if (reg_ctx->ReadRegister(reg_info, reg_value))
            {
                reg_value.Dump(&strm, reg_info, true, false, eFormatDefault);
                strm.EOL();
            }
        }
        log->PutCString(strm.GetData());
    }
}

void
ThreadPlanCallFunction::DoTakedown (bool success)
{
    LogSP log(lldb_private::GetLogIfAnyCategoriesSet (LIBLLDB_LOG_STEP));
    
    if (!m_valid)
    {
        //Don't call DoTakedown if we were never valid to begin with.
        if (log)
            log->Printf ("ThreadPlanCallFunction(%p): Log called on ThreadPlanCallFunction that was never valid.", this);
        return;
    }
    
    if (!m_takedown_done)
    {
        if (success)
        {
            ProcessSP process_sp (m_thread.GetProcess());
            const ABI *abi = process_sp ? process_sp->GetABI().get() : NULL;
            if (abi && m_return_type.IsValid())
            {
                const bool persistent = false;
                m_return_valobj_sp = abi->GetReturnValueObject (m_thread, m_return_type, persistent);
            }
        }
        if (log)
            log->Printf ("ThreadPlanCallFunction(%p): DoTakedown called for thread 0x%4.4llx, m_valid: %d complete: %d.\n", this, m_thread.GetID(), m_valid, IsPlanComplete());
        m_takedown_done = true;
        m_stop_address = m_thread.GetStackFrameAtIndex(0)->GetRegisterContext()->GetPC();
        m_real_stop_info_sp = GetPrivateStopReason();
        m_thread.RestoreThreadStateFromCheckpoint(m_stored_thread_state);
        SetPlanComplete(success);
        ClearBreakpoints();
        if (log && log->GetVerbose())
            ReportRegisterState ("Restoring thread state after function call.  Restored register state:");

    }
    else
    {
        if (log)
            log->Printf ("ThreadPlanCallFunction(%p): DoTakedown called as no-op for thread 0x%4.4llx, m_valid: %d complete: %d.\n", this, m_thread.GetID(), m_valid, IsPlanComplete());
    }
}

void
ThreadPlanCallFunction::WillPop ()
{
    DoTakedown(true);
}

void
ThreadPlanCallFunction::GetDescription (Stream *s, DescriptionLevel level)
{
    if (level == eDescriptionLevelBrief)
    {
        s->Printf("Function call thread plan");
    }
    else
    {
        TargetSP target_sp (m_thread.CalculateTarget());
        s->Printf("Thread plan to call 0x%llx", m_function_addr.GetLoadAddress(target_sp.get()));
    }
}

bool
ThreadPlanCallFunction::ValidatePlan (Stream *error)
{
    if (!m_valid)
        return false;

    return true;
}

bool
ThreadPlanCallFunction::PlanExplainsStop ()
{    
    m_real_stop_info_sp = GetPrivateStopReason();
    
    // If our subplan knows why we stopped, even if it's done (which would forward the question to us)
    // we answer yes.
    if (m_subplan_sp.get() != NULL && m_subplan_sp->PlanExplainsStop())
        return true;
    
    // Check if the breakpoint is one of ours.
    
    StopReason stop_reason;
    if (!m_real_stop_info_sp)
        stop_reason = eStopReasonNone;
    else
        stop_reason = m_real_stop_info_sp->GetStopReason();

    if (stop_reason == eStopReasonBreakpoint && BreakpointsExplainStop())
        return true;
    
    // If we don't want to discard this plan, than any stop we don't understand should be propagated up the stack.
    if (!m_discard_on_error)
        return false;
            
    // Otherwise, check the case where we stopped for an internal breakpoint, in that case, continue on.
    // If it is not an internal breakpoint, consult OkayToDiscard.
    
    
    if (stop_reason == eStopReasonBreakpoint)
    {
        ProcessSP process_sp (m_thread.CalculateProcess());
        uint64_t break_site_id = m_real_stop_info_sp->GetValue();
        BreakpointSiteSP bp_site_sp;
        if (process_sp)
            bp_site_sp = process_sp->GetBreakpointSiteList().FindByID(break_site_id);
        if (bp_site_sp)
        {
            uint32_t num_owners = bp_site_sp->GetNumberOfOwners();
            bool is_internal = true;
            for (uint32_t i = 0; i < num_owners; i++)
            {
                Breakpoint &bp = bp_site_sp->GetOwnerAtIndex(i)->GetBreakpoint();
                
                if (!bp.IsInternal())
                {
                    is_internal = false;
                    break;
                }
            }
            if (is_internal)
                return false;
        }
        
        if (m_discard_on_error)
        {
            DoTakedown(false);
            return true;
        }
        else
            return false;
    }
    else
    {
        // If the subplan is running, any crashes are attributable to us.
        // If we want to discard the plan, then we say we explain the stop
        // but if we are going to be discarded, let whoever is above us
        // explain the stop.
        if (m_subplan_sp)
        {
            if (m_discard_on_error)
            {
                DoTakedown(false);
                return true;
            }
            else
                return false;
        }
        else
            return false;
    }
}

bool
ThreadPlanCallFunction::ShouldStop (Event *event_ptr)
{
    if (IsPlanComplete() || PlanExplainsStop())
    {
        ReportRegisterState ("Function completed.  Register state was:");
        
        DoTakedown(true);
        
        return true;
    }
    else
    {
        return false;
    }
}

bool
ThreadPlanCallFunction::StopOthers ()
{
    return m_stop_other_threads;
}

void
ThreadPlanCallFunction::SetStopOthers (bool new_value)
{
    if (m_subplan_sp)
    {
        ThreadPlanRunToAddress *address_plan = static_cast<ThreadPlanRunToAddress *>(m_subplan_sp.get());
        address_plan->SetStopOthers(new_value);
    }
    m_stop_other_threads = new_value;
}

StateType
ThreadPlanCallFunction::GetPlanRunState ()
{
    return eStateRunning;
}

void
ThreadPlanCallFunction::DidPush ()
{
//#define SINGLE_STEP_EXPRESSIONS
    
#ifndef SINGLE_STEP_EXPRESSIONS
    m_subplan_sp.reset(new ThreadPlanRunToAddress(m_thread, m_start_addr, m_stop_other_threads));
    
    m_thread.QueueThreadPlan(m_subplan_sp, false);
    m_subplan_sp->SetPrivate (true);
#endif
}

bool
ThreadPlanCallFunction::WillStop ()
{
    return true;
}

bool
ThreadPlanCallFunction::MischiefManaged ()
{
    if (IsPlanComplete())
    {
        LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_STEP));

        if (log)
            log->Printf("ThreadPlanCallFunction(%p): Completed call function plan.", this);

        ThreadPlan::MischiefManaged ();
        return true;
    }
    else
    {
        return false;
    }
}

void
ThreadPlanCallFunction::SetBreakpoints ()
{
    ProcessSP process_sp (m_thread.CalculateProcess());
    if (process_sp)
    {
        m_cxx_language_runtime = process_sp->GetLanguageRuntime(eLanguageTypeC_plus_plus);
        m_objc_language_runtime = process_sp->GetLanguageRuntime(eLanguageTypeObjC);
    
        if (m_cxx_language_runtime)
            m_cxx_language_runtime->SetExceptionBreakpoints();
        if (m_objc_language_runtime)
            m_objc_language_runtime->SetExceptionBreakpoints();
    }
}

void
ThreadPlanCallFunction::ClearBreakpoints ()
{
    if (m_cxx_language_runtime)
        m_cxx_language_runtime->ClearExceptionBreakpoints();
    if (m_objc_language_runtime)
        m_objc_language_runtime->ClearExceptionBreakpoints();
}

bool
ThreadPlanCallFunction::BreakpointsExplainStop()
{
    StopInfoSP stop_info_sp = GetPrivateStopReason();
    
    if (m_cxx_language_runtime &&
        m_cxx_language_runtime->ExceptionBreakpointsExplainStop(stop_info_sp))
        return true;
    
    if (m_objc_language_runtime &&
        m_objc_language_runtime->ExceptionBreakpointsExplainStop(stop_info_sp))
        return true;
    
    return false;
}
