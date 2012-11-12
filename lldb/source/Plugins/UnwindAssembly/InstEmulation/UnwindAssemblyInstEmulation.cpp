//===-- UnwindAssemblyInstEmulation.cpp --------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "UnwindAssemblyInstEmulation.h"

#include "llvm-c/EnhancedDisassembly.h"

#include "lldb/Core/Address.h"
#include "lldb/Core/ArchSpec.h"
#include "lldb/Core/DataBufferHeap.h"
#include "lldb/Core/DataExtractor.h"
#include "lldb/Core/Disassembler.h"
#include "lldb/Core/Error.h"
#include "lldb/Core/Log.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/Core/StreamString.h"
#include "lldb/Target/ExecutionContext.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/Thread.h"
#include "lldb/Target/Target.h"

using namespace lldb;
using namespace lldb_private;



//-----------------------------------------------------------------------------------------------
//  UnwindAssemblyInstEmulation method definitions 
//-----------------------------------------------------------------------------------------------

bool
UnwindAssemblyInstEmulation::GetNonCallSiteUnwindPlanFromAssembly (AddressRange& range, 
                                                                   Thread& thread, 
                                                                   UnwindPlan& unwind_plan)
{
    if (range.GetByteSize() > 0 && 
        range.GetBaseAddress().IsValid() &&
        m_inst_emulator_ap.get())
    {
     
        // The the instruction emulation subclass setup the unwind plan for the
        // first instruction.
        m_inst_emulator_ap->CreateFunctionEntryUnwind (unwind_plan);

        // CreateFunctionEntryUnwind should have created the first row. If it
        // doesn't, then we are done.
        if (unwind_plan.GetRowCount() == 0)
            return false;
        
        ExecutionContext exe_ctx;
        thread.CalculateExecutionContext(exe_ctx);
        DisassemblerSP disasm_sp (Disassembler::DisassembleRange (m_arch,
                                                                  NULL,
                                                                  exe_ctx,
                                                                  range));
        
        LogSP log(GetLogIfAllCategoriesSet (LIBLLDB_LOG_UNWIND));

        if (disasm_sp)
        {
            
            m_range_ptr = &range;
            m_thread_ptr = &thread;
            m_unwind_plan_ptr = &unwind_plan;

            const uint32_t addr_byte_size = m_arch.GetAddressByteSize();
            const bool show_address = true;
            const bool show_bytes = true;
            m_inst_emulator_ap->GetRegisterInfo (unwind_plan.GetRegisterKind(), 
                                                 unwind_plan.GetInitialCFARegister(), 
                                                 m_cfa_reg_info);
            
            m_fp_is_cfa = false;
            m_register_values.clear();
            m_pushed_regs.clear();

            // Initialize the CFA with a known value. In the 32 bit case
            // it will be 0x80000000, and in the 64 bit case 0x8000000000000000.
            // We use the address byte size to be safe for any future addresss sizes
            m_initial_sp = (1ull << ((addr_byte_size * 8) - 1));
            RegisterValue cfa_reg_value;
            cfa_reg_value.SetUInt (m_initial_sp, m_cfa_reg_info.byte_size);
            SetRegisterValue (m_cfa_reg_info, cfa_reg_value);

            const InstructionList &inst_list = disasm_sp->GetInstructionList ();
            const size_t num_instructions = inst_list.GetSize();

            if (num_instructions > 0)
            {
                Instruction *inst = inst_list.GetInstructionAtIndex (0).get();
                const addr_t base_addr = inst->GetAddress().GetFileAddress();

                // Make a copy of the current instruction Row and save it in m_curr_row
                // so we can add updates as we process the instructions.  
                UnwindPlan::RowSP last_row = unwind_plan.GetLastRow();
                UnwindPlan::Row *newrow = new UnwindPlan::Row;
                if (last_row.get())
                    *newrow = *last_row.get();
                m_curr_row.reset(newrow);

                // Once we've seen the initial prologue instructions complete, save a
                // copy of the CFI at that point into prologue_completed_row for possible
                // use later.
                int instructions_since_last_prologue_insn = 0;     // # of insns since last CFI was update

                bool reinstate_prologue_next_instruction = false;  // Next iteration, re-install the prologue row of CFI

                bool last_instruction_restored_return_addr_reg = false;  // re-install the prologue row of CFI if the next instruction is a branch immediate

                bool return_address_register_has_been_saved = false; // if we've seen the ra register get saved yet

                UnwindPlan::RowSP prologue_completed_row;          // copy of prologue row of CFI

                // cache the pc register number (in whatever register numbering this UnwindPlan uses) for
                // quick reference during instruction parsing.
                uint32_t pc_reg_num = LLDB_INVALID_REGNUM;
                RegisterInfo pc_reg_info;
                if (m_inst_emulator_ap->GetRegisterInfo (eRegisterKindGeneric, LLDB_REGNUM_GENERIC_PC, pc_reg_info))
                    pc_reg_num = pc_reg_info.kinds[unwind_plan.GetRegisterKind()];
                else
                    pc_reg_num = LLDB_INVALID_REGNUM;

                // cache the return address register number (in whatever register numbering this UnwindPlan uses) for
                // quick reference during instruction parsing.
                uint32_t ra_reg_num = LLDB_INVALID_REGNUM;
                RegisterInfo ra_reg_info;
                if (m_inst_emulator_ap->GetRegisterInfo (eRegisterKindGeneric, LLDB_REGNUM_GENERIC_RA, ra_reg_info))
                    ra_reg_num = ra_reg_info.kinds[unwind_plan.GetRegisterKind()];
                else
                    ra_reg_num = LLDB_INVALID_REGNUM;

                for (size_t idx=0; idx<num_instructions; ++idx)
                {
                    m_curr_row_modified = false;
                    m_curr_insn_restored_a_register = false;
                    inst = inst_list.GetInstructionAtIndex (idx).get();
                    if (inst)
                    {
                        if (log && log->GetVerbose ())
                        {
                            StreamString strm;
                            inst->Dump(&strm, inst_list.GetMaxOpcocdeByteSize (), show_address, show_bytes, NULL);
                            log->PutCString (strm.GetData());
                        }

                        m_inst_emulator_ap->SetInstruction (inst->GetOpcode(), 
                                                            inst->GetAddress(), 
                                                            exe_ctx.GetTargetPtr());

                        m_inst_emulator_ap->EvaluateInstruction (eEmulateInstructionOptionIgnoreConditions);

                        // Were there any changes to the CFI while evaluating this instruction?
                        if (m_curr_row_modified)
                        {
                            reinstate_prologue_next_instruction = false;
                            m_curr_row->SetOffset (inst->GetAddress().GetFileAddress() + inst->GetOpcode().GetByteSize() - base_addr);
                            // Append the new row
                            unwind_plan.AppendRow (m_curr_row);

                            // Allocate a new Row for m_curr_row, copy the current state into it
                            UnwindPlan::Row *newrow = new UnwindPlan::Row;
                            *newrow = *m_curr_row.get();
                            m_curr_row.reset(newrow);

                            // If m_curr_insn_restored_a_register == true, we're looking at an epilogue instruction.
                            // Set instructions_since_last_prologue_insn to a very high number so we don't append 
                            // any of these epilogue instructions to our prologue_complete row.
                            if (m_curr_insn_restored_a_register == false && instructions_since_last_prologue_insn < 8)
                              instructions_since_last_prologue_insn = 0;
                            else
                              instructions_since_last_prologue_insn = 99;

                            UnwindPlan::Row::RegisterLocation pc_regloc;
                            UnwindPlan::Row::RegisterLocation ra_regloc;

                            // While parsing the instructions of this function, if we've ever
                            // seen the return address register (aka lr on arm) in a non-IsSame() state,
                            // it has been saved on the stack.  If it's evern back to IsSame(), we've
                            // executed an epilogue.
                            if (ra_reg_num != LLDB_INVALID_REGNUM
                                && m_curr_row->GetRegisterInfo (ra_reg_num, ra_regloc)
                                && !ra_regloc.IsSame())
                            {
                                return_address_register_has_been_saved = true;
                            }

                            // If the caller's pc is "same", we've just executed an epilogue and we return to the caller
                            // after this instruction completes executing.
                            // If there are any instructions past this, there must have been flow control over this
                            // epilogue so we'll reinstate the original prologue setup instructions.
                            if (prologue_completed_row.get()
                                && pc_reg_num != LLDB_INVALID_REGNUM 
                                && m_curr_row->GetRegisterInfo (pc_reg_num, pc_regloc)
                                && pc_regloc.IsSame())
                            {
                                if (log && log->GetVerbose())
                                    log->Printf("UnwindAssemblyInstEmulation::GetNonCallSiteUnwindPlanFromAssembly -- pc is <same>, restore prologue instructions.");
                                reinstate_prologue_next_instruction = true;
                            }
                            else if (prologue_completed_row.get()
                                     && return_address_register_has_been_saved
                                     && ra_reg_num != LLDB_INVALID_REGNUM
                                     && m_curr_row->GetRegisterInfo (ra_reg_num, ra_regloc)
                                     && ra_regloc.IsSame())
                            {
                                if (log && log->GetVerbose())
                                    log->Printf("UnwindAssemblyInstEmulation::GetNonCallSiteUnwindPlanFromAssembly -- lr is <same>, restore prologue instruction if the next instruction is a branch immediate.");
                                last_instruction_restored_return_addr_reg = true;
                            }
                        }
                        else
                        {
                            // If the previous instruction was a return-to-caller (epilogue), and we're still executing
                            // instructions in this function, there must be a code path that jumps over that epilogue.
                            // Also detect the case where we epilogue & branch imm to another function (tail-call opt)
                            // instead of a normal pop lr-into-pc exit.
                            // Reinstate the frame setup from the prologue.
                            if (reinstate_prologue_next_instruction
                                || (m_curr_insn_is_branch_immediate && last_instruction_restored_return_addr_reg))
                            {
                                if (log && log->GetVerbose())
                                    log->Printf("UnwindAssemblyInstEmulation::GetNonCallSiteUnwindPlanFromAssembly -- Reinstating prologue instruction set");
                                UnwindPlan::Row *newrow = new UnwindPlan::Row;
                                *newrow = *prologue_completed_row.get();
                                m_curr_row.reset(newrow);
                                m_curr_row->SetOffset (inst->GetAddress().GetFileAddress() + inst->GetOpcode().GetByteSize() - base_addr);
                                unwind_plan.AppendRow(m_curr_row);

                                newrow = new UnwindPlan::Row;
                                *newrow = *m_curr_row.get();
                                m_curr_row.reset(newrow);

                                reinstate_prologue_next_instruction = false;
                                last_instruction_restored_return_addr_reg = false; 
                                m_curr_insn_is_branch_immediate = false;
                            }

                            // clear both of these if either one wasn't set
                            if (last_instruction_restored_return_addr_reg)
                            {
                                last_instruction_restored_return_addr_reg = false;
                            }
                            if (m_curr_insn_is_branch_immediate)
                            {
                                m_curr_insn_is_branch_immediate = false;
                            }
 
                            // Stop updating the prologue instructions if we've seen 8 non-prologue instructions
                            // in a row.
                            if (instructions_since_last_prologue_insn++ < 8)
                            {
                                UnwindPlan::Row *newrow = new UnwindPlan::Row;
                                *newrow = *m_curr_row.get();
                                prologue_completed_row.reset(newrow);
                                if (log && log->GetVerbose())
                                    log->Printf("UnwindAssemblyInstEmulation::GetNonCallSiteUnwindPlanFromAssembly -- saving a copy of the current row as the prologue row.");
                            }
                        }
                    }
                }
            }
        }
        
        if (log && log->GetVerbose ())
        {
            StreamString strm;
            lldb::addr_t base_addr = range.GetBaseAddress().GetLoadAddress(thread.CalculateTarget().get());
            strm.Printf ("Resulting unwind rows for [0x%llx - 0x%llx):", base_addr, base_addr + range.GetByteSize());
            unwind_plan.Dump(strm, &thread, base_addr);
            log->PutCString (strm.GetData());
        }
        return unwind_plan.GetRowCount() > 0;
    }
    return false;
}

bool
UnwindAssemblyInstEmulation::GetFastUnwindPlan (AddressRange& func, 
                                                Thread& thread, 
                                                UnwindPlan &unwind_plan)
{
    return false;
}

bool
UnwindAssemblyInstEmulation::FirstNonPrologueInsn (AddressRange& func, 
                                                   const ExecutionContext &exe_ctx, 
                                                   Address& first_non_prologue_insn)
{
    return false;
}

UnwindAssembly *
UnwindAssemblyInstEmulation::CreateInstance (const ArchSpec &arch)
{
    std::auto_ptr<EmulateInstruction> inst_emulator_ap (EmulateInstruction::FindPlugin (arch, eInstructionTypePrologueEpilogue, NULL));
    // Make sure that all prologue instructions are handled
    if (inst_emulator_ap.get())
        return new UnwindAssemblyInstEmulation (arch, inst_emulator_ap.release());
    return NULL;
}


//------------------------------------------------------------------
// PluginInterface protocol in UnwindAssemblyParser_x86
//------------------------------------------------------------------

const char *
UnwindAssemblyInstEmulation::GetPluginName()
{
    return "UnwindAssemblyInstEmulation";
}

const char *
UnwindAssemblyInstEmulation::GetShortPluginName()
{
    return "unwindassembly.inst-emulation";
}


uint32_t
UnwindAssemblyInstEmulation::GetPluginVersion()
{
    return 1;
}

void
UnwindAssemblyInstEmulation::Initialize()
{
    PluginManager::RegisterPlugin (GetPluginNameStatic(),
                                   GetPluginDescriptionStatic(),
                                   CreateInstance);
}

void
UnwindAssemblyInstEmulation::Terminate()
{
    PluginManager::UnregisterPlugin (CreateInstance);
}


const char *
UnwindAssemblyInstEmulation::GetPluginNameStatic()
{
    return "UnwindAssemblyInstEmulation";
}

const char *
UnwindAssemblyInstEmulation::GetPluginDescriptionStatic()
{
    return "Instruction emulation based unwind information.";
}


uint64_t 
UnwindAssemblyInstEmulation::MakeRegisterKindValuePair (const RegisterInfo &reg_info)
{
    uint32_t reg_kind, reg_num;
    if (EmulateInstruction::GetBestRegisterKindAndNumber (&reg_info, reg_kind, reg_num))
        return (uint64_t)reg_kind << 24 | reg_num;
    return 0ull;
}

void
UnwindAssemblyInstEmulation::SetRegisterValue (const RegisterInfo &reg_info, const RegisterValue &reg_value)
{
    m_register_values[MakeRegisterKindValuePair (reg_info)] = reg_value;
}

bool
UnwindAssemblyInstEmulation::GetRegisterValue (const RegisterInfo &reg_info, RegisterValue &reg_value)
{
    const uint64_t reg_id = MakeRegisterKindValuePair (reg_info);
    RegisterValueMap::const_iterator pos = m_register_values.find(reg_id);
    if (pos != m_register_values.end())
    {
        reg_value = pos->second;
        return true; // We had a real value that comes from an opcode that wrote
                     // to it...
    }
    // We are making up a value that is recognizable...
    reg_value.SetUInt(reg_id, reg_info.byte_size);
    return false;
}


size_t
UnwindAssemblyInstEmulation::ReadMemory (EmulateInstruction *instruction,
                                         void *baton,
                                         const EmulateInstruction::Context &context, 
                                         lldb::addr_t addr, 
                                         void *dst,
                                         size_t dst_len)
{
    LogSP log(GetLogIfAllCategoriesSet (LIBLLDB_LOG_UNWIND));

    if (log && log->GetVerbose ())
    {
        StreamString strm;
        strm.Printf ("UnwindAssemblyInstEmulation::ReadMemory    (addr = 0x%16.16llx, dst = %p, dst_len = %llu, context = ", 
                     addr,
                     dst,
                     (uint64_t)dst_len);
        context.Dump(strm, instruction);
        log->PutCString (strm.GetData ());
    }
    memset (dst, 0, dst_len);
    return dst_len;
}

size_t
UnwindAssemblyInstEmulation::WriteMemory (EmulateInstruction *instruction,
                                          void *baton,
                                          const EmulateInstruction::Context &context, 
                                          lldb::addr_t addr, 
                                          const void *dst,
                                          size_t dst_len)
{
    if (baton && dst && dst_len)
        return ((UnwindAssemblyInstEmulation *)baton)->WriteMemory (instruction, context, addr, dst, dst_len);
    return 0;
}

size_t
UnwindAssemblyInstEmulation::WriteMemory (EmulateInstruction *instruction,
                                          const EmulateInstruction::Context &context, 
                                          lldb::addr_t addr, 
                                          const void *dst,
                                          size_t dst_len)
{
    DataExtractor data (dst, 
                        dst_len, 
                        instruction->GetArchitecture ().GetByteOrder(), 
                        instruction->GetArchitecture ().GetAddressByteSize());

    LogSP log(GetLogIfAllCategoriesSet (LIBLLDB_LOG_UNWIND));

    if (log && log->GetVerbose ())
    {
        StreamString strm;

        strm.PutCString ("UnwindAssemblyInstEmulation::WriteMemory   (");
        data.Dump(&strm, 0, eFormatBytes, 1, dst_len, UINT32_MAX, addr, 0, 0);
        strm.PutCString (", context = ");
        context.Dump(strm, instruction);
        log->PutCString (strm.GetData());
    }
    
    const bool can_replace = true;
    const bool cant_replace = false;

    switch (context.type)
    {
        default:
        case EmulateInstruction::eContextInvalid:
        case EmulateInstruction::eContextReadOpcode:
        case EmulateInstruction::eContextImmediate:
        case EmulateInstruction::eContextAdjustBaseRegister:
        case EmulateInstruction::eContextRegisterPlusOffset:
        case EmulateInstruction::eContextAdjustPC:
        case EmulateInstruction::eContextRegisterStore:
        case EmulateInstruction::eContextRegisterLoad:  
        case EmulateInstruction::eContextRelativeBranchImmediate:
        case EmulateInstruction::eContextAbsoluteBranchRegister:
        case EmulateInstruction::eContextSupervisorCall:
        case EmulateInstruction::eContextTableBranchReadMemory:
        case EmulateInstruction::eContextWriteRegisterRandomBits:
        case EmulateInstruction::eContextWriteMemoryRandomBits:
        case EmulateInstruction::eContextArithmetic:
        case EmulateInstruction::eContextAdvancePC:    
        case EmulateInstruction::eContextReturnFromException:
        case EmulateInstruction::eContextPopRegisterOffStack:
        case EmulateInstruction::eContextAdjustStackPointer:
            break;
            
        case EmulateInstruction::eContextPushRegisterOnStack:
            {
                uint32_t reg_num = LLDB_INVALID_REGNUM;
                bool is_return_address_reg = false;
                const uint32_t unwind_reg_kind = m_unwind_plan_ptr->GetRegisterKind();
                if (context.info_type == EmulateInstruction::eInfoTypeRegisterToRegisterPlusOffset)
                {
                    reg_num = context.info.RegisterToRegisterPlusOffset.data_reg.kinds[unwind_reg_kind];
                    if (context.info.RegisterToRegisterPlusOffset.data_reg.kinds[eRegisterKindGeneric] == LLDB_REGNUM_GENERIC_RA)
                        is_return_address_reg = true;
                }
                else
                {
                    assert (!"unhandled case, add code to handle this!");
                }
                
                if (reg_num != LLDB_INVALID_REGNUM)
                {
                    if (m_pushed_regs.find (reg_num) == m_pushed_regs.end())
                    {
                        m_pushed_regs[reg_num] = addr;
                        const int32_t offset = addr - m_initial_sp;
                        m_curr_row->SetRegisterLocationToAtCFAPlusOffset (reg_num, offset, cant_replace);
                        m_curr_row_modified = true;
                        if (is_return_address_reg)
                        {
                            // This push was pushing the return address register,
                            // so this is also how we will unwind the PC...
                            RegisterInfo pc_reg_info;
                            if (instruction->GetRegisterInfo (eRegisterKindGeneric, LLDB_REGNUM_GENERIC_PC, pc_reg_info))
                            {
                                uint32_t pc_reg_num = pc_reg_info.kinds[unwind_reg_kind];
                                if (pc_reg_num != LLDB_INVALID_REGNUM)
                                {
                                    m_curr_row->SetRegisterLocationToAtCFAPlusOffset (pc_reg_num, offset, can_replace);
                                    m_curr_row_modified = true;
                                }
                            }
                        }
                    }
                }
            }
            break;
            
    }

    return dst_len;
}

bool
UnwindAssemblyInstEmulation::ReadRegister (EmulateInstruction *instruction,
                                           void *baton,
                                           const RegisterInfo *reg_info,
                                           RegisterValue &reg_value)
{
    
    if (baton && reg_info)
        return ((UnwindAssemblyInstEmulation *)baton)->ReadRegister (instruction, reg_info, reg_value);
    return false;
}
bool
UnwindAssemblyInstEmulation::ReadRegister (EmulateInstruction *instruction,
                                           const RegisterInfo *reg_info,
                                           RegisterValue &reg_value)
{
    bool synthetic = GetRegisterValue (*reg_info, reg_value);

    LogSP log(GetLogIfAllCategoriesSet (LIBLLDB_LOG_UNWIND));
    
    if (log && log->GetVerbose ())
    {
        
        StreamString strm;
        strm.Printf ("UnwindAssemblyInstEmulation::ReadRegister  (name = \"%s\") => synthetic_value = %i, value = ", reg_info->name, synthetic);
        reg_value.Dump(&strm, reg_info, false, false, eFormatDefault);
        log->PutCString(strm.GetData());
    }
    return true;
}

bool
UnwindAssemblyInstEmulation::WriteRegister (EmulateInstruction *instruction,
                                            void *baton,
                                            const EmulateInstruction::Context &context, 
                                            const RegisterInfo *reg_info,
                                            const RegisterValue &reg_value)
{
    if (baton && reg_info)
        return ((UnwindAssemblyInstEmulation *)baton)->WriteRegister (instruction, context, reg_info, reg_value);
    return false;
}
bool
UnwindAssemblyInstEmulation::WriteRegister (EmulateInstruction *instruction,
                                            const EmulateInstruction::Context &context, 
                                            const RegisterInfo *reg_info,
                                            const RegisterValue &reg_value)
{
    LogSP log(GetLogIfAllCategoriesSet (LIBLLDB_LOG_UNWIND));

    if (log && log->GetVerbose ())
    {
        
        StreamString strm;
        strm.Printf ("UnwindAssemblyInstEmulation::WriteRegister (name = \"%s\", value = ", reg_info->name);
        reg_value.Dump(&strm, reg_info, false, false, eFormatDefault);
        strm.PutCString (", context = ");
        context.Dump(strm, instruction);
        log->PutCString(strm.GetData());
    }

    const bool must_replace = true;
    SetRegisterValue (*reg_info, reg_value);

    switch (context.type)
    {
        default:
        case EmulateInstruction::eContextInvalid:
        case EmulateInstruction::eContextReadOpcode:
        case EmulateInstruction::eContextImmediate:
        case EmulateInstruction::eContextAdjustBaseRegister:
        case EmulateInstruction::eContextRegisterPlusOffset:
        case EmulateInstruction::eContextAdjustPC:
        case EmulateInstruction::eContextRegisterStore:
        case EmulateInstruction::eContextRegisterLoad:  
        case EmulateInstruction::eContextAbsoluteBranchRegister:
        case EmulateInstruction::eContextSupervisorCall:
        case EmulateInstruction::eContextTableBranchReadMemory:
        case EmulateInstruction::eContextWriteRegisterRandomBits:
        case EmulateInstruction::eContextWriteMemoryRandomBits:
        case EmulateInstruction::eContextArithmetic:
        case EmulateInstruction::eContextAdvancePC:    
        case EmulateInstruction::eContextReturnFromException:
        case EmulateInstruction::eContextPushRegisterOnStack:
//            {
//                const uint32_t reg_num = reg_info->kinds[m_unwind_plan_ptr->GetRegisterKind()];
//                if (reg_num != LLDB_INVALID_REGNUM)
//                {
//                    const bool can_replace_only_if_unspecified = true;
//
//                    m_curr_row.SetRegisterLocationToUndefined (reg_num, 
//                                                               can_replace_only_if_unspecified, 
//                                                               can_replace_only_if_unspecified);
//                    m_curr_row_modified = true;
//                }
//            }
            break;

        case EmulateInstruction::eContextRelativeBranchImmediate:
            {
                
                {
                    m_curr_insn_is_branch_immediate = true;
                }
            }
            break;

        case EmulateInstruction::eContextPopRegisterOffStack:
            {
                const uint32_t reg_num = reg_info->kinds[m_unwind_plan_ptr->GetRegisterKind()];
                if (reg_num != LLDB_INVALID_REGNUM)
                {
                    m_curr_row->SetRegisterLocationToSame (reg_num, must_replace);
                    m_curr_row_modified = true;
                    m_curr_insn_restored_a_register = true;
                }
            }
            break;

        case EmulateInstruction::eContextSetFramePointer:
            if (!m_fp_is_cfa)
            {
                m_fp_is_cfa = true;
                m_cfa_reg_info = *reg_info;
                const uint32_t cfa_reg_num = reg_info->kinds[m_unwind_plan_ptr->GetRegisterKind()];
                assert (cfa_reg_num != LLDB_INVALID_REGNUM);
                m_curr_row->SetCFARegister(cfa_reg_num);
                m_curr_row->SetCFAOffset(m_initial_sp - reg_value.GetAsUInt64());
                m_curr_row_modified = true;
            }
            break;

        case EmulateInstruction::eContextAdjustStackPointer:
            // If we have created a frame using the frame pointer, don't follow
            // subsequent adjustments to the stack pointer.
            if (!m_fp_is_cfa)
            {
                m_curr_row->SetCFAOffset (m_initial_sp - reg_value.GetAsUInt64());
                m_curr_row_modified = true;
            }
            break;
    }
    return true;
}


