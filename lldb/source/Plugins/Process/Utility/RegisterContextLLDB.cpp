//===-- RegisterContextLLDB.cpp --------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//


#include "lldb/lldb-private.h"
#include "lldb/Core/Address.h"
#include "lldb/Core/AddressRange.h"
#include "lldb/Core/DataBufferHeap.h"
#include "lldb/Core/Log.h"
#include "lldb/Core/Module.h"
#include "lldb/Core/RegisterValue.h"
#include "lldb/Core/Value.h"
#include "lldb/Expression/DWARFExpression.h"
#include "lldb/Symbol/FuncUnwinders.h"
#include "lldb/Symbol/Function.h"
#include "lldb/Symbol/ObjectFile.h"
#include "lldb/Symbol/SymbolContext.h"
#include "lldb/Symbol/Symbol.h"
#include "lldb/Target/ABI.h"
#include "lldb/Target/ExecutionContext.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/StackFrame.h"
#include "lldb/Target/Target.h"
#include "lldb/Target/Thread.h"
#include "lldb/Target/DynamicLoader.h"

#include "RegisterContextLLDB.h"

using namespace lldb;
using namespace lldb_private;

RegisterContextLLDB::RegisterContextLLDB
(
    Thread& thread,
    const SharedPtr &next_frame,
    SymbolContext& sym_ctx,
    uint32_t frame_number,
    UnwindLLDB& unwind_lldb
) :
    RegisterContext (thread, frame_number),
    m_thread(thread),
    m_fast_unwind_plan_sp (),
    m_full_unwind_plan_sp (),
    m_all_registers_available(false),
    m_frame_type (-1),
    m_cfa (LLDB_INVALID_ADDRESS),
    m_start_pc (),
    m_current_pc (),
    m_current_offset (0),
    m_current_offset_backed_up_one (0),
    m_sym_ctx(sym_ctx),
    m_sym_ctx_valid (false),
    m_frame_number (frame_number),
    m_registers(),
    m_parent_unwind (unwind_lldb)
{
    m_sym_ctx.Clear();
    m_sym_ctx_valid = false;

    if (IsFrameZero ())
    {
        InitializeZerothFrame ();
    }
    else
    {
        InitializeNonZerothFrame ();
    }

    // This same code exists over in the GetFullUnwindPlanForFrame() but it may not have been executed yet
    if (IsFrameZero()
        || next_frame->m_frame_type == eSigtrampFrame
        || next_frame->m_frame_type == eDebuggerFrame)
    {
        m_all_registers_available = true;
    }
}

// Initialize a RegisterContextLLDB which is the first frame of a stack -- the zeroth frame or currently
// executing frame.

void
RegisterContextLLDB::InitializeZerothFrame()
{
    ExecutionContext exe_ctx(m_thread.shared_from_this());
    RegisterContextSP reg_ctx_sp = m_thread.GetRegisterContext();

    if (reg_ctx_sp.get() == NULL)
    {
        m_frame_type = eNotAValidFrame;
        return;
    }

    addr_t current_pc = reg_ctx_sp->GetPC();

    if (current_pc == LLDB_INVALID_ADDRESS)
    {
        m_frame_type = eNotAValidFrame;
        return;
    }

    Process *process = exe_ctx.GetProcessPtr();

    // Let ABIs fixup code addresses to make sure they are valid. In ARM ABIs
    // this will strip bit zero in case we read a PC from memory or from the LR.
    // (which would be a no-op in frame 0 where we get it from the register set,
    // but still a good idea to make the call here for other ABIs that may exist.)
    ABI *abi = process->GetABI().get();
    if (abi)
        current_pc = abi->FixCodeAddress(current_pc);

    // Initialize m_current_pc, an Address object, based on current_pc, an addr_t.
    process->GetTarget().GetSectionLoadList().ResolveLoadAddress (current_pc, m_current_pc);

    // If we don't have a Module for some reason, we're not going to find symbol/function information - just
    // stick in some reasonable defaults and hope we can unwind past this frame.
    ModuleSP pc_module_sp (m_current_pc.GetModule());
    if (!m_current_pc.IsValid() || !pc_module_sp)
    {
        UnwindLogMsg ("using architectural default unwind method");
    }

    // We require that eSymbolContextSymbol be successfully filled in or this context is of no use to us.
    if (pc_module_sp.get()
        && (pc_module_sp->ResolveSymbolContextForAddress (m_current_pc, eSymbolContextFunction| eSymbolContextSymbol, m_sym_ctx) & eSymbolContextSymbol) == eSymbolContextSymbol)
    {
        m_sym_ctx_valid = true;
    }

    AddressRange addr_range;
    m_sym_ctx.GetAddressRange (eSymbolContextFunction | eSymbolContextSymbol, 0, false, addr_range);

    static ConstString g_sigtramp_name ("_sigtramp");
    if ((m_sym_ctx.function && m_sym_ctx.function->GetName() == g_sigtramp_name) ||
        (m_sym_ctx.symbol   && m_sym_ctx.symbol->GetName()   == g_sigtramp_name))
    {
        m_frame_type = eSigtrampFrame;
    }
    else
    {
        // FIXME:  Detect eDebuggerFrame here.
        m_frame_type = eNormalFrame;
    }

    // If we were able to find a symbol/function, set addr_range to the bounds of that symbol/function.
    // else treat the current pc value as the start_pc and record no offset.
    if (addr_range.GetBaseAddress().IsValid())
    {
        m_start_pc = addr_range.GetBaseAddress();
        if (m_current_pc.GetSection() == m_start_pc.GetSection())
        {
            m_current_offset = m_current_pc.GetOffset() - m_start_pc.GetOffset();
        }
        else if (m_current_pc.GetModule() == m_start_pc.GetModule())
        {
            // This means that whatever symbol we kicked up isn't really correct
            // --- we should not cross section boundaries ... We really should NULL out
            // the function/symbol in this case unless there is a bad assumption
            // here due to inlined functions?
            m_current_offset = m_current_pc.GetFileAddress() - m_start_pc.GetFileAddress();
        }
        m_current_offset_backed_up_one = m_current_offset;
    }
    else
    {
        m_start_pc = m_current_pc;
        m_current_offset = -1;
        m_current_offset_backed_up_one = -1;
    }

    // We've set m_frame_type and m_sym_ctx before these calls.

    m_fast_unwind_plan_sp = GetFastUnwindPlanForFrame ();
    m_full_unwind_plan_sp = GetFullUnwindPlanForFrame ();

    UnwindPlan::RowSP active_row;
    int cfa_offset = 0;
    int row_register_kind = -1;
    if (m_full_unwind_plan_sp && m_full_unwind_plan_sp->PlanValidAtAddress (m_current_pc))
    {
        active_row = m_full_unwind_plan_sp->GetRowForFunctionOffset (m_current_offset);
        row_register_kind = m_full_unwind_plan_sp->GetRegisterKind ();
        if (active_row.get() && log)
        {
            StreamString active_row_strm;
            active_row->Dump(active_row_strm, m_full_unwind_plan_sp.get(), &m_thread, m_start_pc.GetLoadAddress(exe_ctx.GetTargetPtr()));
            UnwindLogMsg ("%s", active_row_strm.GetString().c_str());
        }
    }

    if (!active_row.get())
    {
        m_frame_type = eNotAValidFrame;
        return;
    }


    addr_t cfa_regval;
    if (!ReadGPRValue (row_register_kind, active_row->GetCFARegister(), cfa_regval))
    {
        m_frame_type = eNotAValidFrame;
        return;
    }
    else
    {
    }
    cfa_offset = active_row->GetCFAOffset ();

    m_cfa = cfa_regval + cfa_offset;

    UnwindLogMsg ("cfa_regval = 0x%16.16llx (cfa_regval = 0x%16.16llx, cfa_offset = %i)", m_cfa, cfa_regval, cfa_offset);
    UnwindLogMsg ("initialized frame current pc is 0x%llx cfa is 0x%llx using %s UnwindPlan",
            (uint64_t) m_current_pc.GetLoadAddress (exe_ctx.GetTargetPtr()),
            (uint64_t) m_cfa,
            m_full_unwind_plan_sp->GetSourceName().GetCString());
}

// Initialize a RegisterContextLLDB for the non-zeroth frame -- rely on the RegisterContextLLDB "below" it
// to provide things like its current pc value.

void
RegisterContextLLDB::InitializeNonZerothFrame()
{
    LogSP log(GetLogIfAllCategoriesSet (LIBLLDB_LOG_UNWIND));
    if (IsFrameZero ())
    {
        m_frame_type = eNotAValidFrame;
        return;
    }

    if (!GetNextFrame().get() || !GetNextFrame()->IsValid())
    {
        m_frame_type = eNotAValidFrame;
        return;
    }
    if (!m_thread.GetRegisterContext())
    {
        m_frame_type = eNotAValidFrame;
        return;
    }

    addr_t pc;
    if (!ReadGPRValue (eRegisterKindGeneric, LLDB_REGNUM_GENERIC_PC, pc))
    {
        UnwindLogMsg ("could not get pc value");
        m_frame_type = eNotAValidFrame;
        return;
    }

    if (log)
    {
        UnwindLogMsg ("pc = 0x%16.16llx", pc);
        addr_t reg_val;
        if (ReadGPRValue (eRegisterKindGeneric, LLDB_REGNUM_GENERIC_FP, reg_val))
            UnwindLogMsg ("fp = 0x%16.16llx", reg_val);
        if (ReadGPRValue (eRegisterKindGeneric, LLDB_REGNUM_GENERIC_SP, reg_val))
            UnwindLogMsg ("sp = 0x%16.16llx", reg_val);
    }

    // A pc of 0x0 means it's the end of the stack crawl
    if (pc == 0)
    {
        m_frame_type = eNotAValidFrame;
        return;
    }

    ExecutionContext exe_ctx(m_thread.shared_from_this());
    Process *process = exe_ctx.GetProcessPtr();
    // Let ABIs fixup code addresses to make sure they are valid. In ARM ABIs
    // this will strip bit zero in case we read a PC from memory or from the LR.
    ABI *abi = process->GetABI().get();
    if (abi)
        pc = abi->FixCodeAddress(pc);

    process->GetTarget().GetSectionLoadList().ResolveLoadAddress (pc, m_current_pc);

    // If we don't have a Module for some reason, we're not going to find symbol/function information - just
    // stick in some reasonable defaults and hope we can unwind past this frame.
    ModuleSP pc_module_sp (m_current_pc.GetModule());
    if (!m_current_pc.IsValid() || !pc_module_sp)
    {
        UnwindLogMsg ("using architectural default unwind method");

        // Test the pc value to see if we know it's in an unmapped/non-executable region of memory.
        uint32_t permissions;
        if (process->GetLoadAddressPermissions(pc, permissions)
            && (permissions & ePermissionsExecutable) == 0)
        {
            // If this is the second frame off the stack, we may have unwound the first frame
            // incorrectly.  But using the architecture default unwind plan may get us back on
            // track -- albeit possibly skipping a real frame.  Give this frame a clearly-invalid
            // pc and see if we can get any further.
            if (GetNextFrame().get() && GetNextFrame()->IsValid() && GetNextFrame()->IsFrameZero())
            {
                UnwindLogMsg ("had a pc of 0x%llx which is not in executable memory but on frame 1 -- allowing it once.",
                         (uint64_t) pc);
                m_frame_type = eSkipFrame;
            }
            else
            {
                // anywhere other than the second frame, a non-executable pc means we're off in the weeds -- stop now.
                m_frame_type = eNotAValidFrame;
                return;
            }
        }

        if (abi)
        {
            m_fast_unwind_plan_sp.reset ();
            m_full_unwind_plan_sp.reset (new UnwindPlan (lldb::eRegisterKindGeneric));
            abi->CreateDefaultUnwindPlan(*m_full_unwind_plan_sp);
            if (m_frame_type != eSkipFrame)  // don't override eSkipFrame
            {
                m_frame_type = eNormalFrame;
            }
            m_all_registers_available = false;
            m_current_offset = -1;
            m_current_offset_backed_up_one = -1;
            addr_t cfa_regval;
            int row_register_kind = m_full_unwind_plan_sp->GetRegisterKind ();
            UnwindPlan::RowSP row = m_full_unwind_plan_sp->GetRowForFunctionOffset(0);
            if (row.get())
            {
                uint32_t cfa_regnum = row->GetCFARegister();
                int cfa_offset = row->GetCFAOffset();
                if (!ReadGPRValue (row_register_kind, cfa_regnum, cfa_regval))
                {
                    UnwindLogMsg ("failed to get cfa value");
                    if (m_frame_type != eSkipFrame)   // don't override eSkipFrame
                    {
                        m_frame_type = eNormalFrame;
                    }
                    return;
                }
                m_cfa = cfa_regval + cfa_offset;

                // A couple of sanity checks..
                if (cfa_regval == LLDB_INVALID_ADDRESS || cfa_regval == 0 || cfa_regval == 1)
                {
                    UnwindLogMsg ("could not find a valid cfa address");
                    m_frame_type = eNotAValidFrame;
                    return;
                }

                // cfa_regval should point into the stack memory; if we can query memory region permissions,
                // see if the memory is allocated & readable.
                if (process->GetLoadAddressPermissions(cfa_regval, permissions)
                    && (permissions & ePermissionsReadable) == 0)
                {
                    m_frame_type = eNotAValidFrame;
                    return;
                }
            }
            else
            {
                UnwindLogMsg ("could not find a row for function offset zero");
                m_frame_type = eNotAValidFrame;
                return;
            }

            UnwindLogMsg ("initialized frame cfa is 0x%llx", (uint64_t) m_cfa);
            return;
        }
        m_frame_type = eNotAValidFrame;
        return;
    }

    // We require that eSymbolContextSymbol be successfully filled in or this context is of no use to us.
    if ((pc_module_sp->ResolveSymbolContextForAddress (m_current_pc, eSymbolContextFunction| eSymbolContextSymbol, m_sym_ctx) & eSymbolContextSymbol) == eSymbolContextSymbol)
    {
        m_sym_ctx_valid = true;
    }

    AddressRange addr_range;
    if (!m_sym_ctx.GetAddressRange (eSymbolContextFunction | eSymbolContextSymbol, 0, false, addr_range))
    {
        m_sym_ctx_valid = false;
    }

    bool decr_pc_and_recompute_addr_range = false;

    // If the symbol lookup failed...
    if (m_sym_ctx_valid == false)
       decr_pc_and_recompute_addr_range = true;

    // Or if we're in the middle of the stack (and not "above" an asynchronous event like sigtramp),
    // and our "current" pc is the start of a function...
    if (m_sym_ctx_valid
        && GetNextFrame()->m_frame_type != eSigtrampFrame
        && GetNextFrame()->m_frame_type != eDebuggerFrame
        && addr_range.GetBaseAddress().IsValid()
        && addr_range.GetBaseAddress().GetSection() == m_current_pc.GetSection()
        && addr_range.GetBaseAddress().GetOffset() == m_current_pc.GetOffset())
    {
        decr_pc_and_recompute_addr_range = true;
    }

    // We need to back up the pc by 1 byte and re-search for the Symbol to handle the case where the "saved pc"
    // value is pointing to the next function, e.g. if a function ends with a CALL instruction.
    // FIXME this may need to be an architectural-dependent behavior; if so we'll need to add a member function
    // to the ABI plugin and consult that.
    if (decr_pc_and_recompute_addr_range)
    {
        Address temporary_pc(m_current_pc);
        temporary_pc.SetOffset(m_current_pc.GetOffset() - 1);
        m_sym_ctx.Clear();
        m_sym_ctx_valid = false;
        if ((pc_module_sp->ResolveSymbolContextForAddress (temporary_pc, eSymbolContextFunction| eSymbolContextSymbol, m_sym_ctx) & eSymbolContextSymbol) == eSymbolContextSymbol)
        {
            m_sym_ctx_valid = true;
        }
        if (!m_sym_ctx.GetAddressRange (eSymbolContextFunction | eSymbolContextSymbol, 0, false,  addr_range))
        {
            m_sym_ctx_valid = false;
        }
    }

    // If we were able to find a symbol/function, set addr_range_ptr to the bounds of that symbol/function.
    // else treat the current pc value as the start_pc and record no offset.
    if (addr_range.GetBaseAddress().IsValid())
    {
        m_start_pc = addr_range.GetBaseAddress();
        m_current_offset = m_current_pc.GetOffset() - m_start_pc.GetOffset();
        m_current_offset_backed_up_one = m_current_offset;
        if (decr_pc_and_recompute_addr_range && m_current_offset_backed_up_one > 0)
            m_current_offset_backed_up_one--;
    }
    else
    {
        m_start_pc = m_current_pc;
        m_current_offset = -1;
        m_current_offset_backed_up_one = -1;
    }

    static ConstString sigtramp_name ("_sigtramp");
    if ((m_sym_ctx.function && m_sym_ctx.function->GetMangled().GetMangledName() == sigtramp_name)
        || (m_sym_ctx.symbol && m_sym_ctx.symbol->GetMangled().GetMangledName() == sigtramp_name))
    {
        m_frame_type = eSigtrampFrame;
    }
    else
    {
        // FIXME:  Detect eDebuggerFrame here.
        if (m_frame_type != eSkipFrame) // don't override eSkipFrame
        {
            m_frame_type = eNormalFrame;
        }
    }

    // We've set m_frame_type and m_sym_ctx before this call.
    m_fast_unwind_plan_sp = GetFastUnwindPlanForFrame ();

    UnwindPlan::RowSP active_row;
    int cfa_offset = 0;
    int row_register_kind = -1;

    // Try to get by with just the fast UnwindPlan if possible - the full UnwindPlan may be expensive to get
    // (e.g. if we have to parse the entire eh_frame section of an ObjectFile for the first time.)

    if (m_fast_unwind_plan_sp && m_fast_unwind_plan_sp->PlanValidAtAddress (m_current_pc))
    {
        active_row = m_fast_unwind_plan_sp->GetRowForFunctionOffset (m_current_offset);
        row_register_kind = m_fast_unwind_plan_sp->GetRegisterKind ();
        if (active_row.get() && log)
        {
            StreamString active_row_strm;
            active_row->Dump(active_row_strm, m_fast_unwind_plan_sp.get(), &m_thread, m_start_pc.GetLoadAddress(exe_ctx.GetTargetPtr()));
            UnwindLogMsg ("active row: %s", active_row_strm.GetString().c_str());
        }
    }
    else
    {
        m_full_unwind_plan_sp = GetFullUnwindPlanForFrame ();
        if (m_full_unwind_plan_sp && m_full_unwind_plan_sp->PlanValidAtAddress (m_current_pc))
        {
            active_row = m_full_unwind_plan_sp->GetRowForFunctionOffset (m_current_offset);
            row_register_kind = m_full_unwind_plan_sp->GetRegisterKind ();
            if (active_row.get() && log)
            {
                StreamString active_row_strm;
                active_row->Dump(active_row_strm, m_full_unwind_plan_sp.get(), &m_thread, m_start_pc.GetLoadAddress(exe_ctx.GetTargetPtr()));
                UnwindLogMsg ("active row: %s", active_row_strm.GetString().c_str());
            }
        }
    }

    if (!active_row.get())
    {
        m_frame_type = eNotAValidFrame;
        return;
    }

    addr_t cfa_regval;
    if (!ReadGPRValue (row_register_kind, active_row->GetCFARegister(), cfa_regval))
    {
        UnwindLogMsg ("failed to get cfa reg %d/%d", row_register_kind, active_row->GetCFARegister());
        m_frame_type = eNotAValidFrame;
        return;
    }
    cfa_offset = active_row->GetCFAOffset ();

    m_cfa = cfa_regval + cfa_offset;

    UnwindLogMsg ("cfa_regval = 0x%16.16llx (cfa_regval = 0x%16.16llx, cfa_offset = %i)", m_cfa, cfa_regval, cfa_offset);

    // A couple of sanity checks..
    if (cfa_regval == LLDB_INVALID_ADDRESS || cfa_regval == 0 || cfa_regval == 1)
    {
        UnwindLogMsg ("could not find a valid cfa address");
        m_frame_type = eNotAValidFrame;
        return;
    }

    // If we have a bad stack setup, we can get the same CFA value multiple times -- or even
    // more devious, we can actually oscillate between two CFA values.  Detect that here and
    // break out to avoid a possible infinite loop in lldb trying to unwind the stack.
    addr_t next_frame_cfa;
    addr_t next_next_frame_cfa = LLDB_INVALID_ADDRESS;
    if (GetNextFrame().get() && GetNextFrame()->GetCFA(next_frame_cfa))
    {
        bool repeating_frames = false;
        if (next_frame_cfa == m_cfa)
        {
            repeating_frames = true;
        }
        else
        {
            if (GetNextFrame()->GetNextFrame() && GetNextFrame()->GetNextFrame()->GetCFA(next_next_frame_cfa)
                && next_next_frame_cfa == m_cfa)
            {
                repeating_frames = true;
            }
        }
        if (repeating_frames && abi->FunctionCallsChangeCFA())
        {
            UnwindLogMsg ("same CFA address as next frame, assuming the unwind is looping - stopping");
            m_frame_type = eNotAValidFrame;
            return;
        }
    }

    UnwindLogMsg ("initialized frame current pc is 0x%llx cfa is 0x%llx",
            (uint64_t) m_current_pc.GetLoadAddress (exe_ctx.GetTargetPtr()), (uint64_t) m_cfa);
}


bool
RegisterContextLLDB::IsFrameZero () const
{
    return m_frame_number == 0;
}


// Find a fast unwind plan for this frame, if possible.
//
// On entry to this method,
//
//   1. m_frame_type should already be set to eSigtrampFrame/eDebuggerFrame if either of those are correct,
//   2. m_sym_ctx should already be filled in, and
//   3. m_current_pc should have the current pc value for this frame
//   4. m_current_offset_backed_up_one should have the current byte offset into the function, maybe backed up by 1, -1 if unknown

UnwindPlanSP
RegisterContextLLDB::GetFastUnwindPlanForFrame ()
{
    UnwindPlanSP unwind_plan_sp;
    ModuleSP pc_module_sp (m_current_pc.GetModule());

    if (!m_current_pc.IsValid() || !pc_module_sp || pc_module_sp->GetObjectFile() == NULL)
        return unwind_plan_sp;

    if (IsFrameZero ())
        return unwind_plan_sp;

    FuncUnwindersSP func_unwinders_sp (pc_module_sp->GetObjectFile()->GetUnwindTable().GetFuncUnwindersContainingAddress (m_current_pc, m_sym_ctx));
    if (!func_unwinders_sp)
        return unwind_plan_sp;

    // If we're in _sigtramp(), unwinding past this frame requires special knowledge.
    if (m_frame_type == eSigtrampFrame || m_frame_type == eDebuggerFrame)
        return unwind_plan_sp;

    unwind_plan_sp = func_unwinders_sp->GetUnwindPlanFastUnwind (m_thread);
    if (unwind_plan_sp)
    {
        if (unwind_plan_sp->PlanValidAtAddress (m_current_pc))
        {
            LogSP log(GetLogIfAllCategoriesSet (LIBLLDB_LOG_UNWIND));
            if (log && log->GetVerbose())
            {
                if (m_fast_unwind_plan_sp)
                    UnwindLogMsgVerbose ("frame, and has a fast UnwindPlan");
                else
                    UnwindLogMsgVerbose ("frame");
            }
            m_frame_type = eNormalFrame;
            return unwind_plan_sp;
        }
        else
        {
            unwind_plan_sp.reset();
        }
    }
    return unwind_plan_sp;
}

// On entry to this method,
//
//   1. m_frame_type should already be set to eSigtrampFrame/eDebuggerFrame if either of those are correct,
//   2. m_sym_ctx should already be filled in, and
//   3. m_current_pc should have the current pc value for this frame
//   4. m_current_offset_backed_up_one should have the current byte offset into the function, maybe backed up by 1, -1 if unknown

UnwindPlanSP
RegisterContextLLDB::GetFullUnwindPlanForFrame ()
{
    UnwindPlanSP unwind_plan_sp;
    UnwindPlanSP arch_default_unwind_plan_sp;
    ExecutionContext exe_ctx(m_thread.shared_from_this());
    Process *process = exe_ctx.GetProcessPtr();
    ABI *abi = process ? process->GetABI().get() : NULL;
    if (abi)
    {
        arch_default_unwind_plan_sp.reset (new UnwindPlan (lldb::eRegisterKindGeneric));
        abi->CreateDefaultUnwindPlan(*arch_default_unwind_plan_sp);
    }

    bool behaves_like_zeroth_frame = false;
    if (IsFrameZero ()
        || GetNextFrame()->m_frame_type == eSigtrampFrame
        || GetNextFrame()->m_frame_type == eDebuggerFrame)
    {
        behaves_like_zeroth_frame = true;
        // If this frame behaves like a 0th frame (currently executing or
        // interrupted asynchronously), all registers can be retrieved.
        m_all_registers_available = true;
    }

    // If we've done a jmp 0x0 / bl 0x0 (called through a null function pointer) so the pc is 0x0
    // in the zeroth frame, we need to use the "unwind at first instruction" arch default UnwindPlan
    // Also, if this Process can report on memory region attributes, any non-executable region means
    // we jumped through a bad function pointer - handle the same way as 0x0.
    // Note, if the symbol context has a function for the symbol, then we don't need to do this check.

    if ((!m_sym_ctx_valid  || m_sym_ctx.function == NULL) && behaves_like_zeroth_frame && m_current_pc.IsValid())
    {
        uint32_t permissions;
        addr_t current_pc_addr = m_current_pc.GetLoadAddress (exe_ctx.GetTargetPtr());
        if (current_pc_addr == 0
            || (process->GetLoadAddressPermissions(current_pc_addr, permissions)
                && (permissions & ePermissionsExecutable) == 0))
        {
            unwind_plan_sp.reset (new UnwindPlan (lldb::eRegisterKindGeneric));
            abi->CreateFunctionEntryUnwindPlan(*unwind_plan_sp);
            m_frame_type = eNormalFrame;
            return unwind_plan_sp;
        }
    }

    // No Module for the current pc, try using the architecture default unwind.
    ModuleSP pc_module_sp (m_current_pc.GetModule());
    if (!m_current_pc.IsValid() || !pc_module_sp || pc_module_sp->GetObjectFile() == NULL)
    {
        m_frame_type = eNormalFrame;
        return arch_default_unwind_plan_sp;
    }

    FuncUnwindersSP func_unwinders_sp;
    if (m_sym_ctx_valid)
    {
        func_unwinders_sp = pc_module_sp->GetObjectFile()->GetUnwindTable().GetFuncUnwindersContainingAddress (m_current_pc, m_sym_ctx);
    }

    // No FuncUnwinders available for this pc, try using architectural default unwind.
    if (!func_unwinders_sp)
    {
        m_frame_type = eNormalFrame;
        return arch_default_unwind_plan_sp;
    }

    // If we're in _sigtramp(), unwinding past this frame requires special knowledge.  On Mac OS X this knowledge
    // is properly encoded in the eh_frame section, so prefer that if available.
    // On other platforms we may need to provide a platform-specific UnwindPlan which encodes the details of
    // how to unwind out of sigtramp.
    if (m_frame_type == eSigtrampFrame)
    {
        m_fast_unwind_plan_sp.reset();
        unwind_plan_sp = func_unwinders_sp->GetUnwindPlanAtCallSite (m_current_offset_backed_up_one);
        if (unwind_plan_sp && unwind_plan_sp->PlanValidAtAddress (m_current_pc))
            return unwind_plan_sp;
    }

    // Ask the DynamicLoader if the eh_frame CFI should be trusted in this frame even when it's frame zero
    // This comes up if we have hand-written functions in a Module and hand-written eh_frame.  The assembly
    // instruction inspection may fail and the eh_frame CFI were probably written with some care to do the
    // right thing.  It'd be nice if there was a way to ask the eh_frame directly if it is asynchronous
    // (can be trusted at every instruction point) or synchronous (the normal case - only at call sites).
    // But there is not.
    if (process && process->GetDynamicLoader() && process->GetDynamicLoader()->AlwaysRelyOnEHUnwindInfo (m_sym_ctx))
    {
        unwind_plan_sp = func_unwinders_sp->GetUnwindPlanAtCallSite (m_current_offset_backed_up_one);
        if (unwind_plan_sp && unwind_plan_sp->PlanValidAtAddress (m_current_pc))
        {
            UnwindLogMsgVerbose ("frame uses %s for full UnwindPlan because the DynamicLoader suggested we prefer it",
                           unwind_plan_sp->GetSourceName().GetCString());
            return unwind_plan_sp;
        }
    }

    // Typically the NonCallSite UnwindPlan is the unwind created by inspecting the assembly language instructions
    if (behaves_like_zeroth_frame)
    {
        unwind_plan_sp = func_unwinders_sp->GetUnwindPlanAtNonCallSite (m_thread);
        if (unwind_plan_sp && unwind_plan_sp->PlanValidAtAddress (m_current_pc))
        {
            UnwindLogMsgVerbose ("frame uses %s for full UnwindPlan", unwind_plan_sp->GetSourceName().GetCString());
            return unwind_plan_sp;
        }
    }

    // Typically this is unwind info from an eh_frame section intended for exception handling; only valid at call sites
    unwind_plan_sp = func_unwinders_sp->GetUnwindPlanAtCallSite (m_current_offset_backed_up_one);
    if (unwind_plan_sp && unwind_plan_sp->PlanValidAtAddress (m_current_pc))
    {
        UnwindLogMsgVerbose ("frame uses %s for full UnwindPlan", unwind_plan_sp->GetSourceName().GetCString());
        return unwind_plan_sp;
    }

    // We'd prefer to use an UnwindPlan intended for call sites when we're at a call site but if we've
    // struck out on that, fall back to using the non-call-site assembly inspection UnwindPlan if possible.
    unwind_plan_sp = func_unwinders_sp->GetUnwindPlanAtNonCallSite (m_thread);
    if (unwind_plan_sp && unwind_plan_sp->PlanValidAtAddress (m_current_pc))
    {
        UnwindLogMsgVerbose ("frame uses %s for full UnwindPlan", unwind_plan_sp->GetSourceName().GetCString());
        return unwind_plan_sp;
    }

    // If nothing else, use the architectural default UnwindPlan and hope that does the job.
    UnwindLogMsgVerbose ("frame uses %s for full UnwindPlan", arch_default_unwind_plan_sp->GetSourceName().GetCString());
    return arch_default_unwind_plan_sp;
}


void
RegisterContextLLDB::InvalidateAllRegisters ()
{
    m_frame_type = eNotAValidFrame;
}

size_t
RegisterContextLLDB::GetRegisterCount ()
{
    return m_thread.GetRegisterContext()->GetRegisterCount();
}

const RegisterInfo *
RegisterContextLLDB::GetRegisterInfoAtIndex (uint32_t reg)
{
    return m_thread.GetRegisterContext()->GetRegisterInfoAtIndex (reg);
}

size_t
RegisterContextLLDB::GetRegisterSetCount ()
{
    return m_thread.GetRegisterContext()->GetRegisterSetCount ();
}

const RegisterSet *
RegisterContextLLDB::GetRegisterSet (uint32_t reg_set)
{
    return m_thread.GetRegisterContext()->GetRegisterSet (reg_set);
}

uint32_t
RegisterContextLLDB::ConvertRegisterKindToRegisterNumber (uint32_t kind, uint32_t num)
{
    return m_thread.GetRegisterContext()->ConvertRegisterKindToRegisterNumber (kind, num);
}

bool
RegisterContextLLDB::ReadRegisterValueFromRegisterLocation (lldb_private::UnwindLLDB::RegisterLocation regloc,
                                                            const RegisterInfo *reg_info,
                                                            RegisterValue &value)
{
    if (!IsValid())
        return false;
    bool success = false;

    switch (regloc.type)
    {
    case UnwindLLDB::RegisterLocation::eRegisterInRegister:
        {
            const RegisterInfo *other_reg_info = GetRegisterInfoAtIndex(regloc.location.register_number);

            if (!other_reg_info)
                return false;

            if (IsFrameZero ())
            {
                success = m_thread.GetRegisterContext()->ReadRegister (other_reg_info, value);
            }
            else
            {
                success = GetNextFrame()->ReadRegister (other_reg_info, value);
            }
        }
        break;
    case UnwindLLDB::RegisterLocation::eRegisterValueInferred:
        success = value.SetUInt (regloc.location.inferred_value, reg_info->byte_size);
        break;

    case UnwindLLDB::RegisterLocation::eRegisterNotSaved:
        break;
    case UnwindLLDB::RegisterLocation::eRegisterSavedAtHostMemoryLocation:
        assert ("FIXME debugger inferior function call unwind");
        break;
    case UnwindLLDB::RegisterLocation::eRegisterSavedAtMemoryLocation:
        {
            Error error (ReadRegisterValueFromMemory(reg_info,
                                                     regloc.location.target_memory_location,
                                                     reg_info->byte_size,
                                                     value));
            success = error.Success();
        }
        break;
    default:
        assert ("Unknown RegisterLocation type.");
        break;
    }
    return success;
}

bool
RegisterContextLLDB::WriteRegisterValueToRegisterLocation (lldb_private::UnwindLLDB::RegisterLocation regloc,
                                                           const RegisterInfo *reg_info,
                                                           const RegisterValue &value)
{
    if (!IsValid())
        return false;

    bool success = false;

    switch (regloc.type)
    {
        case UnwindLLDB::RegisterLocation::eRegisterInRegister:
            {
                const RegisterInfo *other_reg_info = GetRegisterInfoAtIndex(regloc.location.register_number);
                if (IsFrameZero ())
                {
                    success = m_thread.GetRegisterContext()->WriteRegister (other_reg_info, value);
                }
                else
                {
                    success = GetNextFrame()->WriteRegister (other_reg_info, value);
                }
            }
            break;
        case UnwindLLDB::RegisterLocation::eRegisterValueInferred:
        case UnwindLLDB::RegisterLocation::eRegisterNotSaved:
            break;
        case UnwindLLDB::RegisterLocation::eRegisterSavedAtHostMemoryLocation:
            assert ("FIXME debugger inferior function call unwind");
            break;
        case UnwindLLDB::RegisterLocation::eRegisterSavedAtMemoryLocation:
            {
                Error error (WriteRegisterValueToMemory (reg_info,
                                                         regloc.location.target_memory_location,
                                                         reg_info->byte_size,
                                                         value));
                success = error.Success();
            }
            break;
        default:
            assert ("Unknown RegisterLocation type.");
            break;
    }
    return success;
}


bool
RegisterContextLLDB::IsValid () const
{
    return m_frame_type != eNotAValidFrame;
}

// A skip frame is a bogus frame on the stack -- but one where we're likely to find a real frame farther
// up the stack if we keep looking.  It's always the second frame in an unwind (i.e. the first frame after
// frame zero) where unwinding can be the trickiest.  Ideally we'll mark up this frame in some way so the
// user knows we're displaying bad data and we may have skipped one frame of their real program in the
// process of getting back on track.

bool
RegisterContextLLDB::IsSkipFrame () const
{
    return m_frame_type == eSkipFrame;
}

// Answer the question: Where did THIS frame save the CALLER frame ("previous" frame)'s register value?

bool
RegisterContextLLDB::SavedLocationForRegister (uint32_t lldb_regnum, lldb_private::UnwindLLDB::RegisterLocation &regloc)
{
    // Have we already found this register location?
    if (!m_registers.empty())
    {
        std::map<uint32_t, lldb_private::UnwindLLDB::RegisterLocation>::const_iterator iterator;
        iterator = m_registers.find (lldb_regnum);
        if (iterator != m_registers.end())
        {
            regloc = iterator->second;
            return true;
        }
    }

    static uint32_t sp_regnum = LLDB_INVALID_REGNUM;
    static uint32_t pc_regnum = LLDB_INVALID_REGNUM;
    static bool generic_registers_initialized = false;
    if (!generic_registers_initialized)
    {
        m_thread.GetRegisterContext()->ConvertBetweenRegisterKinds (eRegisterKindGeneric, LLDB_REGNUM_GENERIC_SP, eRegisterKindLLDB, sp_regnum);
        m_thread.GetRegisterContext()->ConvertBetweenRegisterKinds (eRegisterKindGeneric, LLDB_REGNUM_GENERIC_PC, eRegisterKindLLDB, pc_regnum);
        generic_registers_initialized = true;
    }

    // Are we looking for the CALLER's stack pointer?  The stack pointer is defined to be the same as THIS frame's
    // CFA so just return the CFA value.  This is true on x86-32/x86-64 at least.
    if (sp_regnum != LLDB_INVALID_REGNUM && sp_regnum == lldb_regnum)
    {
        // make sure we won't lose precision copying an addr_t (m_cfa) into a uint64_t (.inferred_value)
        assert (sizeof (addr_t) <= sizeof (uint64_t));
        regloc.type = UnwindLLDB::RegisterLocation::eRegisterValueInferred;
        regloc.location.inferred_value = m_cfa;
        m_registers[lldb_regnum] = regloc;
        return true;
    }

    // Look through the available UnwindPlans for the register location.

    UnwindPlan::Row::RegisterLocation unwindplan_regloc;
    bool have_unwindplan_regloc = false;
    RegisterKind unwindplan_registerkind = (RegisterKind)-1;

    if (m_fast_unwind_plan_sp)
    {
        UnwindPlan::RowSP active_row = m_fast_unwind_plan_sp->GetRowForFunctionOffset (m_current_offset);
        unwindplan_registerkind = m_fast_unwind_plan_sp->GetRegisterKind ();
        uint32_t row_regnum;
        if (!m_thread.GetRegisterContext()->ConvertBetweenRegisterKinds (eRegisterKindLLDB, lldb_regnum, unwindplan_registerkind, row_regnum))
        {
            UnwindLogMsg ("could not convert lldb regnum %d into %d RegisterKind reg numbering scheme",
                    lldb_regnum, (int) unwindplan_registerkind);
            return false;
        }
        if (active_row->GetRegisterInfo (row_regnum, unwindplan_regloc))
        {
            UnwindLogMsg ("supplying caller's saved reg %d's location using FastUnwindPlan", lldb_regnum);
            have_unwindplan_regloc = true;
        }
    }

    if (!have_unwindplan_regloc)
    {
        // m_full_unwind_plan_sp being NULL means that we haven't tried to find a full UnwindPlan yet
        if (!m_full_unwind_plan_sp)
            m_full_unwind_plan_sp = GetFullUnwindPlanForFrame ();

        if (m_full_unwind_plan_sp)
        {
            UnwindPlan::RowSP active_row = m_full_unwind_plan_sp->GetRowForFunctionOffset (m_current_offset);
            unwindplan_registerkind = m_full_unwind_plan_sp->GetRegisterKind ();
            uint32_t row_regnum;

            // If we're fetching the saved pc and this UnwindPlan defines a ReturnAddress register (e.g. lr on arm),
            // look for the return address register number in the UnwindPlan's row.
            if (lldb_regnum == pc_regnum && m_full_unwind_plan_sp->GetReturnAddressRegister() != LLDB_INVALID_REGNUM)
            {
               row_regnum = m_full_unwind_plan_sp->GetReturnAddressRegister();
               UnwindLogMsg ("requested caller's saved PC but this UnwindPlan uses a RA reg; getting reg %d instead",
                       row_regnum);
            }
            else
            {
                if (!m_thread.GetRegisterContext()->ConvertBetweenRegisterKinds (eRegisterKindLLDB, lldb_regnum, unwindplan_registerkind, row_regnum))
                {
                    if (unwindplan_registerkind == eRegisterKindGeneric)
                        UnwindLogMsg ("could not convert lldb regnum %d into eRegisterKindGeneric reg numbering scheme", lldb_regnum);
                    else
                        UnwindLogMsg ("could not convert lldb regnum %d into %d RegisterKind reg numbering scheme",
                                lldb_regnum, (int) unwindplan_registerkind);
                    return false;
                }
            }

            if (active_row->GetRegisterInfo (row_regnum, unwindplan_regloc))
            {
                have_unwindplan_regloc = true;
                UnwindLogMsg ("supplying caller's saved reg %d's location using %s UnwindPlan", lldb_regnum,
                              m_full_unwind_plan_sp->GetSourceName().GetCString());
            }

            // If this architecture stores the return address in a register (it defines a Return Address register)
            // and we're on a non-zero stack frame and the Full UnwindPlan says that the pc is stored in the
            // RA registers (e.g. lr on arm), then we know that the full unwindplan is not trustworthy -- this
            // is an impossible situation and the instruction emulation code has likely been misled.  
            // If this stack frame meets those criteria, we need to throw away the Full UnwindPlan that the 
            // instruction emulation came up with and fall back to the architecture's Default UnwindPlan so
            // the stack walk can get past this point.

            // Special note:  If the Full UnwindPlan was generated from the compiler, don't second-guess it 
            // when we're at a call site location.

            // arch_default_ra_regnum is the return address register # in the Full UnwindPlan register numbering
            uint32_t arch_default_ra_regnum = LLDB_INVALID_REGNUM; 
            if (m_thread.GetRegisterContext()->ConvertBetweenRegisterKinds (eRegisterKindGeneric, LLDB_REGNUM_GENERIC_RA, unwindplan_registerkind, arch_default_ra_regnum)
                && arch_default_ra_regnum != LLDB_INVALID_REGNUM
                && pc_regnum != LLDB_INVALID_REGNUM
                && pc_regnum == lldb_regnum
                && unwindplan_regloc.IsInOtherRegister()
                && unwindplan_regloc.GetRegisterNumber() == arch_default_ra_regnum
                && m_full_unwind_plan_sp->GetSourcedFromCompiler() != eLazyBoolYes
                && !m_all_registers_available)
            {
                UnwindLogMsg ("%s UnwindPlan tried to restore the pc from the link register but this is a non-zero frame",
                              m_full_unwind_plan_sp->GetSourceName().GetCString());

                // Throw away the full unwindplan; install the arch default unwindplan
                InvalidateFullUnwindPlan();

                // Now re-fetch the pc value we're searching for
                uint32_t arch_default_pc_reg = LLDB_INVALID_REGNUM;
                UnwindPlan::RowSP active_row = m_full_unwind_plan_sp->GetRowForFunctionOffset (m_current_offset);
                if (m_thread.GetRegisterContext()->ConvertBetweenRegisterKinds (eRegisterKindGeneric, LLDB_REGNUM_GENERIC_PC, m_full_unwind_plan_sp->GetRegisterKind(), arch_default_pc_reg)
                    && arch_default_pc_reg != LLDB_INVALID_REGNUM
                    && active_row
                    && active_row->GetRegisterInfo (arch_default_pc_reg, unwindplan_regloc))
                {
                    have_unwindplan_regloc = true;
                }
                else
                {
                    have_unwindplan_regloc = false;
                }
            }
        }
    }


    ExecutionContext exe_ctx(m_thread.shared_from_this());
    Process *process = exe_ctx.GetProcessPtr();
    if (have_unwindplan_regloc == false)
    {
        // If a volatile register is being requested, we don't want to forward the next frame's register contents
        // up the stack -- the register is not retrievable at this frame.
        ABI *abi = process ? process->GetABI().get() : NULL;
        if (abi)
        {
            const RegisterInfo *reg_info = GetRegisterInfoAtIndex(lldb_regnum);
            if (reg_info && abi->RegisterIsVolatile (reg_info))
            {
                UnwindLogMsg ("did not supply reg location for %d because it is volatile", lldb_regnum);
                return false;
            }
        }

        if (IsFrameZero ())
        {
            // This is frame 0 - we should return the actual live register context value
            lldb_private::UnwindLLDB::RegisterLocation new_regloc;
            new_regloc.type = UnwindLLDB::RegisterLocation::eRegisterInRegister;
            new_regloc.location.register_number = lldb_regnum;
            m_registers[lldb_regnum] = new_regloc;
            regloc = new_regloc;
            return true;
        }
        else
        UnwindLogMsg ("could not supply caller's reg %d location", lldb_regnum);
        return false;
    }

    // unwindplan_regloc has valid contents about where to retrieve the register
    if (unwindplan_regloc.IsUnspecified())
    {
        lldb_private::UnwindLLDB::RegisterLocation new_regloc;
        new_regloc.type = UnwindLLDB::RegisterLocation::eRegisterNotSaved;
        m_registers[lldb_regnum] = new_regloc;
        UnwindLogMsg ("could not supply caller's reg %d location", lldb_regnum);
        return false;
    }

    if (unwindplan_regloc.IsSame())
    {
        if (IsFrameZero ())
        {
            UnwindLogMsg ("could not supply caller's reg %d location", lldb_regnum);
            return false;
        }
        else
        {
            return false;
        }
    }

    if (unwindplan_regloc.IsCFAPlusOffset())
    {
        int offset = unwindplan_regloc.GetOffset();
        regloc.type = UnwindLLDB::RegisterLocation::eRegisterValueInferred;
        regloc.location.inferred_value = m_cfa + offset;
        m_registers[lldb_regnum] = regloc;
        return true;
    }

    if (unwindplan_regloc.IsAtCFAPlusOffset())
    {
        int offset = unwindplan_regloc.GetOffset();
        regloc.type = UnwindLLDB::RegisterLocation::eRegisterSavedAtMemoryLocation;
        regloc.location.target_memory_location = m_cfa + offset;
        m_registers[lldb_regnum] = regloc;
        return true;
    }

    if (unwindplan_regloc.IsInOtherRegister())
    {
        uint32_t unwindplan_regnum = unwindplan_regloc.GetRegisterNumber();
        uint32_t row_regnum_in_lldb;
        if (!m_thread.GetRegisterContext()->ConvertBetweenRegisterKinds (unwindplan_registerkind, unwindplan_regnum, eRegisterKindLLDB, row_regnum_in_lldb))
        {
            UnwindLogMsg ("could not supply caller's reg %d location", lldb_regnum);
            return false;
        }
        regloc.type = UnwindLLDB::RegisterLocation::eRegisterInRegister;
        regloc.location.register_number = row_regnum_in_lldb;
        m_registers[lldb_regnum] = regloc;
        return true;
    }

    if (unwindplan_regloc.IsDWARFExpression() || unwindplan_regloc.IsAtDWARFExpression())
    {
        DataExtractor dwarfdata (unwindplan_regloc.GetDWARFExpressionBytes(),
                                 unwindplan_regloc.GetDWARFExpressionLength(),
                                 process->GetByteOrder(), process->GetAddressByteSize());
        DWARFExpression dwarfexpr (dwarfdata, 0, unwindplan_regloc.GetDWARFExpressionLength());
        dwarfexpr.SetRegisterKind (unwindplan_registerkind);
        Value result;
        Error error;
        if (dwarfexpr.Evaluate (&exe_ctx, NULL, NULL, NULL, this, 0, NULL, result, &error))
        {
            addr_t val;
            val = result.GetScalar().ULongLong();
            if (unwindplan_regloc.IsDWARFExpression())
             {
                regloc.type = UnwindLLDB::RegisterLocation::eRegisterValueInferred;
                regloc.location.inferred_value = val;
                m_registers[lldb_regnum] = regloc;
                return true;
            }
            else
            {
               regloc.type = UnwindLLDB::RegisterLocation::eRegisterSavedAtMemoryLocation;
               regloc.location.target_memory_location = val;
               m_registers[lldb_regnum] = regloc;
               return true;
            }
        }
        UnwindLogMsg ("tried to use IsDWARFExpression or IsAtDWARFExpression for reg %d but failed", lldb_regnum);
        return false;
    }

    UnwindLogMsg ("could not supply caller's reg %d location", lldb_regnum);

    // FIXME UnwindPlan::Row types atDWARFExpression and isDWARFExpression are unsupported.

    return false;
}

// If the Full unwindplan has been determined to be incorrect, this method will
// replace it with the architecture's default unwindplna, if one is defined.
// It will also find the FuncUnwinders object for this function and replace the
// Full unwind method for the function there so we don't use the errant Full unwindplan
// again in the future of this debug session.
// We're most likely doing this because the Full unwindplan was generated by assembly
// instruction profiling and the profiler got something wrong.

void
RegisterContextLLDB::InvalidateFullUnwindPlan ()
{
    UnwindPlan::Row::RegisterLocation unwindplan_regloc;
    ExecutionContext exe_ctx (m_thread.shared_from_this());
    Process *process = exe_ctx.GetProcessPtr();
    ABI *abi = process ? process->GetABI().get() : NULL;
    if (abi)
    {
        UnwindPlanSP original_full_unwind_plan_sp = m_full_unwind_plan_sp;
        UnwindPlanSP arch_default_unwind_plan_sp;
        arch_default_unwind_plan_sp.reset (new UnwindPlan (lldb::eRegisterKindGeneric));
        abi->CreateDefaultUnwindPlan(*arch_default_unwind_plan_sp);
        if (arch_default_unwind_plan_sp)
        {
            UnwindPlan::RowSP active_row = arch_default_unwind_plan_sp->GetRowForFunctionOffset (m_current_offset);
        
            if (active_row && active_row->GetCFARegister() != LLDB_INVALID_REGNUM)
            {
                FuncUnwindersSP func_unwinders_sp;
                if (m_sym_ctx_valid && m_current_pc.IsValid() && m_current_pc.GetModule())
                {
                    func_unwinders_sp = m_current_pc.GetModule()->GetObjectFile()->GetUnwindTable().GetFuncUnwindersContainingAddress (m_current_pc, m_sym_ctx);
                    if (func_unwinders_sp)
                    {
                        func_unwinders_sp->InvalidateNonCallSiteUnwindPlan (m_thread);
                    }
                }
                m_registers.clear();
                m_full_unwind_plan_sp = arch_default_unwind_plan_sp;
                addr_t cfa_regval;
                if (ReadGPRValue (arch_default_unwind_plan_sp->GetRegisterKind(), active_row->GetCFARegister(), cfa_regval))
                {
                    m_cfa = cfa_regval + active_row->GetCFAOffset ();
                }

                UnwindLogMsg ("full unwind plan '%s' has been replaced by architecture default unwind plan '%s' for this function from now on.",
                              original_full_unwind_plan_sp->GetSourceName().GetCString(), arch_default_unwind_plan_sp->GetSourceName().GetCString());
            }
        }
    }
}

// Retrieve a general purpose register value for THIS frame, as saved by the NEXT frame, i.e. the frame that
// this frame called.  e.g.
//
//  foo () { }
//  bar () { foo (); }
//  main () { bar (); }
//
//  stopped in foo() so
//     frame 0 - foo
//     frame 1 - bar
//     frame 2 - main
//  and this RegisterContext is for frame 1 (bar) - if we want to get the pc value for frame 1, we need to ask
//  where frame 0 (the "next" frame) saved that and retrieve the value.

bool
RegisterContextLLDB::ReadGPRValue (int register_kind, uint32_t regnum, addr_t &value)
{
    if (!IsValid())
        return false;

    uint32_t lldb_regnum;
    if (register_kind == eRegisterKindLLDB)
    {
        lldb_regnum = regnum;
    }
    else if (!m_thread.GetRegisterContext()->ConvertBetweenRegisterKinds (register_kind, regnum, eRegisterKindLLDB, lldb_regnum))
    {
        return false;
    }

    const RegisterInfo *reg_info = GetRegisterInfoAtIndex(lldb_regnum);
    RegisterValue reg_value;
    // if this is frame 0 (currently executing frame), get the requested reg contents from the actual thread registers
    if (IsFrameZero ())
    {
        if (m_thread.GetRegisterContext()->ReadRegister (reg_info, reg_value))
        {
            value = reg_value.GetAsUInt64();
            return true;
        }
        return false;
    }

    bool pc_or_return_address = false;
    uint32_t generic_regnum;
    if (register_kind == eRegisterKindGeneric
        && (regnum == LLDB_REGNUM_GENERIC_PC || regnum == LLDB_REGNUM_GENERIC_RA))
    {
        pc_or_return_address = true;
    }
    else if (m_thread.GetRegisterContext()->ConvertBetweenRegisterKinds (register_kind, regnum, eRegisterKindGeneric, generic_regnum)
             && (generic_regnum == LLDB_REGNUM_GENERIC_PC || generic_regnum == LLDB_REGNUM_GENERIC_RA))
    {
        pc_or_return_address = true;
    }

    lldb_private::UnwindLLDB::RegisterLocation regloc;
    if (!m_parent_unwind.SearchForSavedLocationForRegister (lldb_regnum, regloc, m_frame_number - 1, pc_or_return_address))
    {
        return false;
    }
    if (ReadRegisterValueFromRegisterLocation (regloc, reg_info, reg_value))
    {
        value = reg_value.GetAsUInt64();
        return true;
    }
    return false;
}

// Find the value of a register in THIS frame

bool
RegisterContextLLDB::ReadRegister (const RegisterInfo *reg_info, RegisterValue &value)
{
    if (!IsValid())
        return false;

    const uint32_t lldb_regnum = reg_info->kinds[eRegisterKindLLDB];
    UnwindLogMsgVerbose ("looking for register saved location for reg %d", lldb_regnum);

    // If this is the 0th frame, hand this over to the live register context
    if (IsFrameZero ())
    {
        UnwindLogMsgVerbose ("passing along to the live register context for reg %d", lldb_regnum);
        return m_thread.GetRegisterContext()->ReadRegister (reg_info, value);
    }

    lldb_private::UnwindLLDB::RegisterLocation regloc;
    // Find out where the NEXT frame saved THIS frame's register contents
    if (!m_parent_unwind.SearchForSavedLocationForRegister (lldb_regnum, regloc, m_frame_number - 1, false))
        return false;

    return ReadRegisterValueFromRegisterLocation (regloc, reg_info, value);
}

bool
RegisterContextLLDB::WriteRegister (const RegisterInfo *reg_info, const RegisterValue &value)
{
    if (!IsValid())
        return false;

    const uint32_t lldb_regnum = reg_info->kinds[eRegisterKindLLDB];
    UnwindLogMsgVerbose ("looking for register saved location for reg %d", lldb_regnum);

    // If this is the 0th frame, hand this over to the live register context
    if (IsFrameZero ())
    {
        UnwindLogMsgVerbose ("passing along to the live register context for reg %d", lldb_regnum);
        return m_thread.GetRegisterContext()->WriteRegister (reg_info, value);
    }

    lldb_private::UnwindLLDB::RegisterLocation regloc;
    // Find out where the NEXT frame saved THIS frame's register contents
    if (!m_parent_unwind.SearchForSavedLocationForRegister (lldb_regnum, regloc, m_frame_number - 1, false))
        return false;

    return WriteRegisterValueToRegisterLocation (regloc, reg_info, value);
}

// Don't need to implement this one
bool
RegisterContextLLDB::ReadAllRegisterValues (lldb::DataBufferSP &data_sp)
{
    return false;
}

// Don't need to implement this one
bool
RegisterContextLLDB::WriteAllRegisterValues (const lldb::DataBufferSP& data_sp)
{
    return false;
}

// Retrieve the pc value for THIS from

bool
RegisterContextLLDB::GetCFA (addr_t& cfa)
{
    if (!IsValid())
    {
        return false;
    }
    if (m_cfa == LLDB_INVALID_ADDRESS)
    {
        return false;
    }
    cfa = m_cfa;
    return true;
}


RegisterContextLLDB::SharedPtr
RegisterContextLLDB::GetNextFrame () const
{
    RegisterContextLLDB::SharedPtr regctx;
    if (m_frame_number == 0)
      return regctx;
    return m_parent_unwind.GetRegisterContextForFrameNum (m_frame_number - 1);
}

RegisterContextLLDB::SharedPtr
RegisterContextLLDB::GetPrevFrame () const
{
    RegisterContextLLDB::SharedPtr regctx;
    return m_parent_unwind.GetRegisterContextForFrameNum (m_frame_number + 1);
}

// Retrieve the address of the start of the function of THIS frame

bool
RegisterContextLLDB::GetStartPC (addr_t& start_pc)
{
    if (!IsValid())
        return false;

    if (!m_start_pc.IsValid())
    {
        return ReadPC (start_pc);
    }
    start_pc = m_start_pc.GetLoadAddress (CalculateTarget().get());
    return true;
}

// Retrieve the current pc value for THIS frame, as saved by the NEXT frame.

bool
RegisterContextLLDB::ReadPC (addr_t& pc)
{
    if (!IsValid())
        return false;

    if (ReadGPRValue (eRegisterKindGeneric, LLDB_REGNUM_GENERIC_PC, pc))
    {
        // A pc value of 0 or 1 is impossible in the middle of the stack -- it indicates the end of a stack walk.
        // On the currently executing frame (or such a frame interrupted asynchronously by sigtramp et al) this may
        // occur if code has jumped through a NULL pointer -- we want to be able to unwind past that frame to help
        // find the bug.

        if (m_all_registers_available == false
            && (pc == 0 || pc == 1))
        {
            return false;
        }
        else
        {
            return true;
        }
    }
    else
    {
        return false;
    }
}


void
RegisterContextLLDB::UnwindLogMsg (const char *fmt, ...)
{
    LogSP log(GetLogIfAllCategoriesSet (LIBLLDB_LOG_UNWIND));
    if (log)
    {
        va_list args;
        va_start (args, fmt);

        char *logmsg;
        if (vasprintf (&logmsg, fmt, args) == -1 || logmsg == NULL)
        {
            if (logmsg)
                free (logmsg);
            va_end (args);
            return;
        }
        va_end (args);

        log->Printf ("%*sth%d/fr%u %s",
                      m_frame_number < 100 ? m_frame_number : 100, "", m_thread.GetIndexID(), m_frame_number,
                      logmsg);
        free (logmsg);
    }
}

void
RegisterContextLLDB::UnwindLogMsgVerbose (const char *fmt, ...)
{
    LogSP log(GetLogIfAllCategoriesSet (LIBLLDB_LOG_UNWIND));
    if (log && log->GetVerbose())
    {
        va_list args;
        va_start (args, fmt);

        char *logmsg;
        if (vasprintf (&logmsg, fmt, args) == -1 || logmsg == NULL)
        {
            if (logmsg)
                free (logmsg);
            va_end (args);
            return;
        }
        va_end (args);

        log->Printf ("%*sth%d/fr%u %s",
                      m_frame_number < 100 ? m_frame_number : 100, "", m_thread.GetIndexID(), m_frame_number,
                      logmsg);
        free (logmsg);
    }
}

