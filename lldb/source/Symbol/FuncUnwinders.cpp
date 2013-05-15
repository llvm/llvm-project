//===-- FuncUnwinders.cpp ----------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Core/AddressRange.h"
#include "lldb/Core/Address.h"
#include "lldb/Symbol/FuncUnwinders.h"
#include "lldb/Symbol/DWARFCallFrameInfo.h"
#include "lldb/Symbol/ObjectFile.h"
#include "lldb/Symbol/UnwindPlan.h"
#include "lldb/Symbol/UnwindTable.h"
#include "lldb/Target/ABI.h"
#include "lldb/Target/ExecutionContext.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/Thread.h"
#include "lldb/Target/Target.h"
#include "lldb/Target/UnwindAssembly.h"

using namespace lldb;
using namespace lldb_private;


FuncUnwinders::FuncUnwinders
(
    UnwindTable& unwind_table, 
    UnwindAssembly *assembly_profiler, 
    AddressRange range
) : 
    m_unwind_table(unwind_table), 
    m_assembly_profiler(assembly_profiler), 
    m_range(range), 
    m_mutex (Mutex::eMutexTypeNormal),
    m_unwind_plan_call_site_sp (), 
    m_unwind_plan_non_call_site_sp (), 
    m_unwind_plan_fast_sp (), 
    m_unwind_plan_arch_default_sp (), 
    m_tried_unwind_at_call_site (false),
    m_tried_unwind_at_non_call_site (false),
    m_tried_unwind_fast (false),
    m_tried_unwind_arch_default (false),
    m_tried_unwind_arch_default_at_func_entry (false),
    m_first_non_prologue_insn() 
{
}

FuncUnwinders::~FuncUnwinders () 
{ 
}

UnwindPlanSP
FuncUnwinders::GetUnwindPlanAtCallSite (int current_offset)
{
    // Lock the mutex to ensure we can always give out the most appropriate
    // information. We want to make sure if someone requests a call site unwind
    // plan, that they get one and don't run into a race condition where one
    // thread has started to create the unwind plan and has put it into 
    // m_unwind_plan_call_site_sp, and have another thread enter this function
    // and return the partially filled in m_unwind_plan_call_site_sp pointer.
    // We also want to make sure that we lock out other unwind plans from
    // being accessed until this one is done creating itself in case someone
    // had some code like:
    //  UnwindPlan *best_unwind_plan = ...GetUnwindPlanAtCallSite (...)
    //  if (best_unwind_plan == NULL)
    //      best_unwind_plan = GetUnwindPlanAtNonCallSite (...)
    Mutex::Locker locker (m_mutex);
    if (m_tried_unwind_at_call_site == false && m_unwind_plan_call_site_sp.get() == NULL)
    {
        m_tried_unwind_at_call_site = true;
        // We have cases (e.g. with _sigtramp on Mac OS X) where the hand-written eh_frame unwind info for a
        // function does not cover the entire range of the function and so the FDE only lists a subset of the
        // address range.  If we try to look up the unwind info by the starting address of the function 
        // (i.e. m_range.GetBaseAddress()) we may not find the eh_frame FDE.  We need to use the actual byte offset
        // into the function when looking it up.

        if (m_range.GetBaseAddress().IsValid())
        {
            Address current_pc (m_range.GetBaseAddress ());
            if (current_offset != -1)
                current_pc.SetOffset (current_pc.GetOffset() + current_offset);

            DWARFCallFrameInfo *eh_frame = m_unwind_table.GetEHFrameInfo();
            if (eh_frame)
            {
                m_unwind_plan_call_site_sp.reset (new UnwindPlan (lldb::eRegisterKindGeneric));
                if (!eh_frame->GetUnwindPlan (current_pc, *m_unwind_plan_call_site_sp))
                    m_unwind_plan_call_site_sp.reset();
            }
        }
    }
    return m_unwind_plan_call_site_sp;
}

UnwindPlanSP
FuncUnwinders::GetUnwindPlanAtNonCallSite (Thread& thread)
{
    // Lock the mutex to ensure we can always give out the most appropriate
    // information. We want to make sure if someone requests an unwind
    // plan, that they get one and don't run into a race condition where one
    // thread has started to create the unwind plan and has put it into 
    // the unique pointer member variable, and have another thread enter this function
    // and return the partially filled pointer contained in the unique pointer.
    // We also want to make sure that we lock out other unwind plans from
    // being accessed until this one is done creating itself in case someone
    // had some code like:
    //  UnwindPlan *best_unwind_plan = ...GetUnwindPlanAtCallSite (...)
    //  if (best_unwind_plan == NULL)
    //      best_unwind_plan = GetUnwindPlanAtNonCallSite (...)
    Mutex::Locker locker (m_mutex);
    if (m_tried_unwind_at_non_call_site == false && m_unwind_plan_non_call_site_sp.get() == NULL)
    {
        m_tried_unwind_at_non_call_site = true;
        m_unwind_plan_non_call_site_sp.reset (new UnwindPlan (lldb::eRegisterKindGeneric));
        if (!m_assembly_profiler->GetNonCallSiteUnwindPlanFromAssembly (m_range, thread, *m_unwind_plan_non_call_site_sp))
            m_unwind_plan_non_call_site_sp.reset();
    }
    return m_unwind_plan_non_call_site_sp;
}

UnwindPlanSP
FuncUnwinders::GetUnwindPlanFastUnwind (Thread& thread)
{
    // Lock the mutex to ensure we can always give out the most appropriate
    // information. We want to make sure if someone requests an unwind
    // plan, that they get one and don't run into a race condition where one
    // thread has started to create the unwind plan and has put it into 
    // the unique pointer member variable, and have another thread enter this function
    // and return the partially filled pointer contained in the unique pointer.
    // We also want to make sure that we lock out other unwind plans from
    // being accessed until this one is done creating itself in case someone
    // had some code like:
    //  UnwindPlan *best_unwind_plan = ...GetUnwindPlanAtCallSite (...)
    //  if (best_unwind_plan == NULL)
    //      best_unwind_plan = GetUnwindPlanAtNonCallSite (...)
    Mutex::Locker locker (m_mutex);
    if (m_tried_unwind_fast == false && m_unwind_plan_fast_sp.get() == NULL)
    {
        m_tried_unwind_fast = true;
        m_unwind_plan_fast_sp.reset (new UnwindPlan (lldb::eRegisterKindGeneric));
        if (!m_assembly_profiler->GetFastUnwindPlan (m_range, thread, *m_unwind_plan_fast_sp))
            m_unwind_plan_fast_sp.reset();
    }
    return m_unwind_plan_fast_sp;
}

UnwindPlanSP
FuncUnwinders::GetUnwindPlanArchitectureDefault (Thread& thread)
{
    // Lock the mutex to ensure we can always give out the most appropriate
    // information. We want to make sure if someone requests an unwind
    // plan, that they get one and don't run into a race condition where one
    // thread has started to create the unwind plan and has put it into 
    // the unique pointer member variable, and have another thread enter this function
    // and return the partially filled pointer contained in the unique pointer.
    // We also want to make sure that we lock out other unwind plans from
    // being accessed until this one is done creating itself in case someone
    // had some code like:
    //  UnwindPlan *best_unwind_plan = ...GetUnwindPlanAtCallSite (...)
    //  if (best_unwind_plan == NULL)
    //      best_unwind_plan = GetUnwindPlanAtNonCallSite (...)
    Mutex::Locker locker (m_mutex);
    if (m_tried_unwind_arch_default == false && m_unwind_plan_arch_default_sp.get() == NULL)
    {
        m_tried_unwind_arch_default = true;
        Address current_pc;
        ProcessSP process_sp (thread.CalculateProcess());
        if (process_sp)
        {
            ABI *abi = process_sp->GetABI().get();
            if (abi)
            {
                m_unwind_plan_arch_default_sp.reset (new UnwindPlan (lldb::eRegisterKindGeneric));
                if (m_unwind_plan_arch_default_sp)
                    abi->CreateDefaultUnwindPlan(*m_unwind_plan_arch_default_sp);
            }
        }
    }

    return m_unwind_plan_arch_default_sp;
}

UnwindPlanSP
FuncUnwinders::GetUnwindPlanArchitectureDefaultAtFunctionEntry (Thread& thread)
{
    // Lock the mutex to ensure we can always give out the most appropriate
    // information. We want to make sure if someone requests an unwind
    // plan, that they get one and don't run into a race condition where one
    // thread has started to create the unwind plan and has put it into 
    // the unique pointer member variable, and have another thread enter this function
    // and return the partially filled pointer contained in the unique pointer.
    // We also want to make sure that we lock out other unwind plans from
    // being accessed until this one is done creating itself in case someone
    // had some code like:
    //  UnwindPlan *best_unwind_plan = ...GetUnwindPlanAtCallSite (...)
    //  if (best_unwind_plan == NULL)
    //      best_unwind_plan = GetUnwindPlanAtNonCallSite (...)
    Mutex::Locker locker (m_mutex);
    if (m_tried_unwind_arch_default_at_func_entry == false && m_unwind_plan_arch_default_at_func_entry_sp.get() == NULL)
    {
        m_tried_unwind_arch_default_at_func_entry = true;
        Address current_pc;
        ProcessSP process_sp (thread.CalculateProcess());
        if (process_sp)
        {
            ABI *abi = process_sp->GetABI().get();
            if (abi)
            {
                m_unwind_plan_arch_default_at_func_entry_sp.reset (new UnwindPlan (lldb::eRegisterKindGeneric));
                if (m_unwind_plan_arch_default_at_func_entry_sp)
                    abi->CreateFunctionEntryUnwindPlan(*m_unwind_plan_arch_default_at_func_entry_sp);
            }
        }
    }

    return m_unwind_plan_arch_default_sp;
}


Address&
FuncUnwinders::GetFirstNonPrologueInsn (Target& target)
{
    if (m_first_non_prologue_insn.IsValid())
        return m_first_non_prologue_insn;
    ExecutionContext exe_ctx (target.shared_from_this(), false);
    m_assembly_profiler->FirstNonPrologueInsn (m_range, exe_ctx, m_first_non_prologue_insn);
    return m_first_non_prologue_insn;
}

const Address&
FuncUnwinders::GetFunctionStartAddress () const
{
    return m_range.GetBaseAddress();
}

void
FuncUnwinders::InvalidateNonCallSiteUnwindPlan (lldb_private::Thread& thread)
{
    UnwindPlanSP arch_default = GetUnwindPlanArchitectureDefault (thread);
    if (arch_default && m_tried_unwind_at_call_site)
    {
        m_unwind_plan_call_site_sp = arch_default;
    }
}
