//===-- ThreadPlanCallUserExpression.cpp ------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Target/ThreadPlanCallUserExpression.h"

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
#include "lldb/Core/Stream.h"
#include "lldb/Expression/ClangUserExpression.h"
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
// ThreadPlanCallUserExpression: Plan to call a single function
//----------------------------------------------------------------------

ThreadPlanCallUserExpression::ThreadPlanCallUserExpression (Thread &thread,
                                                Address &function,
                                                lldb::addr_t arg,
                                                bool stop_other_threads,
                                                bool unwind_on_error,
                                                bool ignore_breakpoints,
                                                lldb::addr_t *this_arg,
                                                lldb::addr_t *cmd_arg,
                                                ClangUserExpression::ClangUserExpressionSP &user_expression_sp) :
    ThreadPlanCallFunction (thread, function, ClangASTType(), arg, stop_other_threads, unwind_on_error, ignore_breakpoints, this_arg, cmd_arg),
    m_user_expression_sp (user_expression_sp)
{
    // User expressions are generally "User generated" so we should set them up to stop when done.
    SetIsMasterPlan (true);
    SetOkayToDiscard(false);
}

ThreadPlanCallUserExpression::~ThreadPlanCallUserExpression ()
{
}

void
ThreadPlanCallUserExpression::GetDescription (Stream *s, lldb::DescriptionLevel level)
{        
    ThreadPlanCallFunction::GetDescription (s, level);
}

StopInfoSP
ThreadPlanCallUserExpression::GetRealStopInfo()
{
    StopInfoSP stop_info_sp = ThreadPlanCallFunction::GetRealStopInfo();
    
    if (stop_info_sp)
    {
        lldb::addr_t addr = GetStopAddress();
        DynamicCheckerFunctions *checkers = m_thread.GetProcess()->GetDynamicCheckers();
        StreamString s;
        
        if (checkers && checkers->DoCheckersExplainStop(addr, s))
            stop_info_sp->SetDescription(s.GetData());
    }

    return stop_info_sp;
}
