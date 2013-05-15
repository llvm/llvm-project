//===-- ThreadPlanShouldStopHere.cpp ----------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Target/RegisterContext.h"
#include "lldb/Target/Thread.h"
#include "lldb/Target/ThreadPlanShouldStopHere.h"
#include "lldb/Core/Log.h"

using namespace lldb;
using namespace lldb_private;

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes

//----------------------------------------------------------------------
// ThreadPlanShouldStopHere constructor
//----------------------------------------------------------------------
ThreadPlanShouldStopHere::ThreadPlanShouldStopHere(ThreadPlan *owner, ThreadPlanShouldStopHereCallback callback, void *baton) :
    m_callback (callback),
    m_baton (baton),
    m_owner (owner),
    m_flags (ThreadPlanShouldStopHere::eNone)
{
}

//----------------------------------------------------------------------
// Destructor
//----------------------------------------------------------------------
ThreadPlanShouldStopHere::~ThreadPlanShouldStopHere()
{
}

void
ThreadPlanShouldStopHere::SetShouldStopHereCallback (ThreadPlanShouldStopHereCallback callback, void *baton)
{
    m_callback = callback;
    m_baton = baton;
}

ThreadPlan *
ThreadPlanShouldStopHere::InvokeShouldStopHereCallback ()
{
    if (m_callback)
    {
        ThreadPlan *return_plan = m_callback (m_owner, m_flags, m_baton);
        Log *log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_STEP));
        if (log)
        {
            lldb::addr_t current_addr = m_owner->GetThread().GetRegisterContext()->GetPC(0);

            if (return_plan)
            {
                StreamString s;
                return_plan->GetDescription (&s, lldb::eDescriptionLevelFull);
                log->Printf ("ShouldStopHere callback found a step out plan from 0x%" PRIx64 ": %s.", current_addr, s.GetData());
            }
            else
            {
                log->Printf ("ShouldStopHere callback didn't find a step out plan from: 0x%" PRIx64 ".", current_addr);
            }
        }
        return return_plan;
    }
    else
        return NULL;
}
