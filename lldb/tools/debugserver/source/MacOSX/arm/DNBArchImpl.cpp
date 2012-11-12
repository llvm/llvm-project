//===-- DNBArchImpl.cpp -----------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  Created by Greg Clayton on 6/25/07.
//
//===----------------------------------------------------------------------===//

#if defined (__arm__)

#include "MacOSX/arm/DNBArchImpl.h"
#include "MacOSX/MachProcess.h"
#include "MacOSX/MachThread.h"
#include "DNBBreakpoint.h"
#include "DNBLog.h"
#include "DNBRegisterInfo.h"
#include "DNB.h"
#include "ARM_GCC_Registers.h"
#include "ARM_DWARF_Registers.h"

#include <sys/sysctl.h>

// BCR address match type
#define BCR_M_IMVA_MATCH        ((uint32_t)(0u << 21))
#define BCR_M_CONTEXT_ID_MATCH  ((uint32_t)(1u << 21))
#define BCR_M_IMVA_MISMATCH     ((uint32_t)(2u << 21))
#define BCR_M_RESERVED          ((uint32_t)(3u << 21))

// Link a BVR/BCR or WVR/WCR pair to another
#define E_ENABLE_LINKING        ((uint32_t)(1u << 20))

// Byte Address Select
#define BAS_IMVA_PLUS_0         ((uint32_t)(1u << 5))
#define BAS_IMVA_PLUS_1         ((uint32_t)(1u << 6))
#define BAS_IMVA_PLUS_2         ((uint32_t)(1u << 7))
#define BAS_IMVA_PLUS_3         ((uint32_t)(1u << 8))
#define BAS_IMVA_0_1            ((uint32_t)(3u << 5))
#define BAS_IMVA_2_3            ((uint32_t)(3u << 7))
#define BAS_IMVA_ALL            ((uint32_t)(0xfu << 5))

// Break only in priveleged or user mode
#define S_RSVD                  ((uint32_t)(0u << 1))
#define S_PRIV                  ((uint32_t)(1u << 1))
#define S_USER                  ((uint32_t)(2u << 1))
#define S_PRIV_USER             ((S_PRIV) | (S_USER))

#define BCR_ENABLE              ((uint32_t)(1u))
#define WCR_ENABLE              ((uint32_t)(1u))

// Watchpoint load/store
#define WCR_LOAD                ((uint32_t)(1u << 3))
#define WCR_STORE               ((uint32_t)(1u << 4))

// Definitions for the Debug Status and Control Register fields:
// [5:2] => Method of debug entry
//#define WATCHPOINT_OCCURRED     ((uint32_t)(2u))
// I'm seeing this, instead.
#define WATCHPOINT_OCCURRED     ((uint32_t)(10u))

static const uint8_t g_arm_breakpoint_opcode[] = { 0xFE, 0xDE, 0xFF, 0xE7 };
static const uint8_t g_thumb_breakpooint_opcode[] = { 0xFE, 0xDE };

// ARM constants used during decoding
#define REG_RD          0
#define LDM_REGLIST     1
#define PC_REG          15
#define PC_REGLIST_BIT  0x8000

// ARM conditions
#define COND_EQ     0x0
#define COND_NE     0x1
#define COND_CS     0x2
#define COND_HS     0x2
#define COND_CC     0x3
#define COND_LO     0x3
#define COND_MI     0x4
#define COND_PL     0x5
#define COND_VS     0x6
#define COND_VC     0x7
#define COND_HI     0x8
#define COND_LS     0x9
#define COND_GE     0xA
#define COND_LT     0xB
#define COND_GT     0xC
#define COND_LE     0xD
#define COND_AL     0xE
#define COND_UNCOND 0xF

#define MASK_CPSR_T (1u << 5)
#define MASK_CPSR_J (1u << 24)

#define MNEMONIC_STRING_SIZE 32
#define OPERAND_STRING_SIZE 128


void
DNBArchMachARM::Initialize()
{
    DNBArchPluginInfo arch_plugin_info = 
    {
        CPU_TYPE_ARM, 
        DNBArchMachARM::Create, 
        DNBArchMachARM::GetRegisterSetInfo,
        DNBArchMachARM::SoftwareBreakpointOpcode
    };
    
    // Register this arch plug-in with the main protocol class
    DNBArchProtocol::RegisterArchPlugin (arch_plugin_info);
}


DNBArchProtocol *
DNBArchMachARM::Create (MachThread *thread)
{
    DNBArchMachARM *obj = new DNBArchMachARM (thread);

    // When new thread comes along, it tries to inherit from the global debug state, if it is valid.
    if (Valid_Global_Debug_State)
    {
        obj->m_state.dbg = Global_Debug_State;
        kern_return_t kret = obj->SetDBGState();
        DNBLogThreadedIf(LOG_WATCHPOINTS,
                         "DNBArchMachARM::Create() Inherit and SetDBGState() => 0x%8.8x.", kret);
    }
    return obj;
}

const uint8_t * const
DNBArchMachARM::SoftwareBreakpointOpcode (nub_size_t byte_size)
{
    switch (byte_size)
    {
    case 2: return g_thumb_breakpooint_opcode;
    case 4: return g_arm_breakpoint_opcode;
    }
    return NULL;
}

uint32_t
DNBArchMachARM::GetCPUType()
{
    return CPU_TYPE_ARM;
}

uint64_t
DNBArchMachARM::GetPC(uint64_t failValue)
{
    // Get program counter
    if (GetGPRState(false) == KERN_SUCCESS)
        return m_state.context.gpr.__pc;
    return failValue;
}

kern_return_t
DNBArchMachARM::SetPC(uint64_t value)
{
    // Get program counter
    kern_return_t err = GetGPRState(false);
    if (err == KERN_SUCCESS)
    {
        m_state.context.gpr.__pc = value;
        err = SetGPRState();
    }
    return err == KERN_SUCCESS;
}

uint64_t
DNBArchMachARM::GetSP(uint64_t failValue)
{
    // Get stack pointer
    if (GetGPRState(false) == KERN_SUCCESS)
        return m_state.context.gpr.__sp;
    return failValue;
}

kern_return_t
DNBArchMachARM::GetGPRState(bool force)
{
    int set = e_regSetGPR;
    // Check if we have valid cached registers
    if (!force && m_state.GetError(set, Read) == KERN_SUCCESS)
        return KERN_SUCCESS;

    // Read the registers from our thread
    mach_msg_type_number_t count = ARM_THREAD_STATE_COUNT;
    kern_return_t kret = ::thread_get_state(m_thread->ThreadID(), ARM_THREAD_STATE, (thread_state_t)&m_state.context.gpr, &count);
    uint32_t *r = &m_state.context.gpr.__r[0];
    DNBLogThreadedIf(LOG_THREAD, "thread_get_state(0x%4.4x, %u, &gpr, %u) => 0x%8.8x (count = %u) regs r0=%8.8x r1=%8.8x r2=%8.8x r3=%8.8x r4=%8.8x r5=%8.8x r6=%8.8x r7=%8.8x r8=%8.8x r9=%8.8x r10=%8.8x r11=%8.8x s12=%8.8x sp=%8.8x lr=%8.8x pc=%8.8x cpsr=%8.8x", 
                     m_thread->ThreadID(), 
                     ARM_THREAD_STATE, 
                     ARM_THREAD_STATE_COUNT, 
                     kret,
                     count,
                     r[0], 
                     r[1], 
                     r[2], 
                     r[3], 
                     r[4], 
                     r[5], 
                     r[6], 
                     r[7], 
                     r[8], 
                     r[9], 
                     r[10], 
                     r[11], 
                     r[12], 
                     r[13], 
                     r[14], 
                     r[15], 
                     r[16]);
    m_state.SetError(set, Read, kret);
    return kret;
}

kern_return_t
DNBArchMachARM::GetVFPState(bool force)
{
    int set = e_regSetVFP;
    // Check if we have valid cached registers
    if (!force && m_state.GetError(set, Read) == KERN_SUCCESS)
        return KERN_SUCCESS;

    // Read the registers from our thread
    mach_msg_type_number_t count = ARM_VFP_STATE_COUNT;
    kern_return_t kret = ::thread_get_state(m_thread->ThreadID(), ARM_VFP_STATE, (thread_state_t)&m_state.context.vfp, &count);
    if (DNBLogEnabledForAny (LOG_THREAD))
    {
        uint32_t *r = &m_state.context.vfp.__r[0];
        DNBLogThreaded ("thread_get_state(0x%4.4x, %u, &gpr, %u) => 0x%8.8x (count => %u)",
                        m_thread->ThreadID(), 
                        ARM_THREAD_STATE, 
                        ARM_THREAD_STATE_COUNT, 
                        kret,
                        count);
        DNBLogThreaded("   s0=%8.8x  s1=%8.8x  s2=%8.8x  s3=%8.8x  s4=%8.8x  s5=%8.8x  s6=%8.8x  s7=%8.8x",r[ 0],r[ 1],r[ 2],r[ 3],r[ 4],r[ 5],r[ 6],r[ 7]);
        DNBLogThreaded("   s8=%8.8x  s9=%8.8x s10=%8.8x s11=%8.8x s12=%8.8x s13=%8.8x s14=%8.8x s15=%8.8x",r[ 8],r[ 9],r[10],r[11],r[12],r[13],r[14],r[15]);
        DNBLogThreaded("  s16=%8.8x s17=%8.8x s18=%8.8x s19=%8.8x s20=%8.8x s21=%8.8x s22=%8.8x s23=%8.8x",r[16],r[17],r[18],r[19],r[20],r[21],r[22],r[23]);
        DNBLogThreaded("  s24=%8.8x s25=%8.8x s26=%8.8x s27=%8.8x s28=%8.8x s29=%8.8x s30=%8.8x s31=%8.8x",r[24],r[25],r[26],r[27],r[28],r[29],r[30],r[31]);
        DNBLogThreaded("  s32=%8.8x s33=%8.8x s34=%8.8x s35=%8.8x s36=%8.8x s37=%8.8x s38=%8.8x s39=%8.8x",r[32],r[33],r[34],r[35],r[36],r[37],r[38],r[39]);
        DNBLogThreaded("  s40=%8.8x s41=%8.8x s42=%8.8x s43=%8.8x s44=%8.8x s45=%8.8x s46=%8.8x s47=%8.8x",r[40],r[41],r[42],r[43],r[44],r[45],r[46],r[47]);
        DNBLogThreaded("  s48=%8.8x s49=%8.8x s50=%8.8x s51=%8.8x s52=%8.8x s53=%8.8x s54=%8.8x s55=%8.8x",r[48],r[49],r[50],r[51],r[52],r[53],r[54],r[55]);
        DNBLogThreaded("  s56=%8.8x s57=%8.8x s58=%8.8x s59=%8.8x s60=%8.8x s61=%8.8x s62=%8.8x s63=%8.8x fpscr=%8.8x",r[56],r[57],r[58],r[59],r[60],r[61],r[62],r[63],r[64]);
    }
    m_state.SetError(set, Read, kret);
    return kret;
}

kern_return_t
DNBArchMachARM::GetEXCState(bool force)
{
    int set = e_regSetEXC;
    // Check if we have valid cached registers
    if (!force && m_state.GetError(set, Read) == KERN_SUCCESS)
        return KERN_SUCCESS;

    // Read the registers from our thread
    mach_msg_type_number_t count = ARM_EXCEPTION_STATE_COUNT;
    kern_return_t kret = ::thread_get_state(m_thread->ThreadID(), ARM_EXCEPTION_STATE, (thread_state_t)&m_state.context.exc, &count);
    m_state.SetError(set, Read, kret);
    return kret;
}

static void
DumpDBGState(const DNBArchMachARM::DBG& dbg)
{
    uint32_t i = 0;
    for (i=0; i<16; i++) {
        DNBLogThreadedIf(LOG_STEP, "BVR%-2u/BCR%-2u = { 0x%8.8x, 0x%8.8x } WVR%-2u/WCR%-2u = { 0x%8.8x, 0x%8.8x }",
            i, i, dbg.__bvr[i], dbg.__bcr[i],
            i, i, dbg.__wvr[i], dbg.__wcr[i]);
    }
}

kern_return_t
DNBArchMachARM::GetDBGState(bool force)
{
    int set = e_regSetDBG;

    // Check if we have valid cached registers
    if (!force && m_state.GetError(set, Read) == KERN_SUCCESS)
        return KERN_SUCCESS;

    // Read the registers from our thread
    mach_msg_type_number_t count = ARM_DEBUG_STATE_COUNT;
    kern_return_t kret = ::thread_get_state(m_thread->ThreadID(), ARM_DEBUG_STATE, (thread_state_t)&m_state.dbg, &count);
    m_state.SetError(set, Read, kret);
    return kret;
}

kern_return_t
DNBArchMachARM::SetGPRState()
{
    int set = e_regSetGPR;
    kern_return_t kret = ::thread_set_state(m_thread->ThreadID(), ARM_THREAD_STATE, (thread_state_t)&m_state.context.gpr, ARM_THREAD_STATE_COUNT);
    m_state.SetError(set, Write, kret);         // Set the current write error for this register set
    m_state.InvalidateRegisterSetState(set);    // Invalidate the current register state in case registers are read back differently
    return kret;                                // Return the error code
}

kern_return_t
DNBArchMachARM::SetVFPState()
{
    int set = e_regSetVFP;
    kern_return_t kret = ::thread_set_state (m_thread->ThreadID(), ARM_VFP_STATE, (thread_state_t)&m_state.context.vfp, ARM_VFP_STATE_COUNT);
    m_state.SetError(set, Write, kret);         // Set the current write error for this register set
    m_state.InvalidateRegisterSetState(set);    // Invalidate the current register state in case registers are read back differently
    return kret;                                // Return the error code
}

kern_return_t
DNBArchMachARM::SetEXCState()
{
    int set = e_regSetEXC;
    kern_return_t kret = ::thread_set_state (m_thread->ThreadID(), ARM_EXCEPTION_STATE, (thread_state_t)&m_state.context.exc, ARM_EXCEPTION_STATE_COUNT);
    m_state.SetError(set, Write, kret);         // Set the current write error for this register set
    m_state.InvalidateRegisterSetState(set);    // Invalidate the current register state in case registers are read back differently
    return kret;                                // Return the error code
}

kern_return_t
DNBArchMachARM::SetDBGState()
{
    int set = e_regSetDBG;
    kern_return_t kret = ::thread_set_state (m_thread->ThreadID(), ARM_DEBUG_STATE, (thread_state_t)&m_state.dbg, ARM_DEBUG_STATE_COUNT);
    m_state.SetError(set, Write, kret);         // Set the current write error for this register set
    m_state.InvalidateRegisterSetState(set);    // Invalidate the current register state in case registers are read back differently
    return kret;                                // Return the error code
}

void
DNBArchMachARM::ThreadWillResume()
{
    // Do we need to step this thread? If so, let the mach thread tell us so.
    if (m_thread->IsStepping())
    {
        // This is the primary thread, let the arch do anything it needs
        if (NumSupportedHardwareBreakpoints() > 0)
        {
            if (EnableHardwareSingleStep(true) != KERN_SUCCESS)
            {
                DNBLogThreaded("DNBArchMachARM::ThreadWillResume() failed to enable hardware single step");
            }
        }
        else
        {
            if (SetSingleStepSoftwareBreakpoints() != KERN_SUCCESS)
            {
                DNBLogThreaded("DNBArchMachARM::ThreadWillResume() failed to enable software single step");
            }
        }
    }

    // Disable the triggered watchpoint temporarily before we resume.
    // Plus, we try to enable hardware single step to execute past the instruction which triggered our watchpoint.
    if (m_watchpoint_did_occur)
    {
        if (m_watchpoint_hw_index >= 0)
        {
            kern_return_t kret = GetDBGState(false);
            if (kret == KERN_SUCCESS && !IsWatchpointEnabled(m_state.dbg, m_watchpoint_hw_index)) {
                // The watchpoint might have been disabled by the user.  We don't need to do anything at all
                // to enable hardware single stepping.
                m_watchpoint_did_occur = false;
                m_watchpoint_hw_index = -1;
                return;
            }

            DisableHardwareWatchpoint0(m_watchpoint_hw_index, true);
            DNBLogThreadedIf(LOG_WATCHPOINTS, "DNBArchMachARM::ThreadWillResume() DisableHardwareWatchpoint(%d) called",
                             m_watchpoint_hw_index);

            // Enable hardware single step to move past the watchpoint-triggering instruction.
            m_watchpoint_resume_single_step_enabled = (EnableHardwareSingleStep(true) == KERN_SUCCESS);

            // If we are not able to enable single step to move past the watchpoint-triggering instruction,
            // at least we should reset the two watchpoint member variables so that the next time around
            // this callback function is invoked, the enclosing logical branch is skipped.
            if (!m_watchpoint_resume_single_step_enabled) {
                // Reset the two watchpoint member variables.
                m_watchpoint_did_occur = false;
                m_watchpoint_hw_index = -1;
                DNBLogThreadedIf(LOG_WATCHPOINTS, "DNBArchMachARM::ThreadWillResume() failed to enable single step");
            }
            else
                DNBLogThreadedIf(LOG_WATCHPOINTS, "DNBArchMachARM::ThreadWillResume() succeeded to enable single step");
        }
    }
}

bool
DNBArchMachARM::ThreadDidStop()
{
    bool success = true;

    m_state.InvalidateRegisterSetState (e_regSetALL);

    if (m_watchpoint_resume_single_step_enabled)
    {
        // Great!  We now disable the hardware single step as well as re-enable the hardware watchpoint.
        // See also ThreadWillResume().
        if (EnableHardwareSingleStep(false) == KERN_SUCCESS)
        {
            if (m_watchpoint_did_occur && m_watchpoint_hw_index >= 0)
            {
                EnableHardwareWatchpoint0(m_watchpoint_hw_index, true);
                m_watchpoint_resume_single_step_enabled = false;
                m_watchpoint_did_occur = false;
                m_watchpoint_hw_index = -1;
            }
            else
            {
                DNBLogError("internal error detected: m_watchpoint_resume_step_enabled is true but (m_watchpoint_did_occur && m_watchpoint_hw_index >= 0) does not hold!");
            }
        }
        else
        {
            DNBLogError("internal error detected: m_watchpoint_resume_step_enabled is true but unable to disable single step!");
        }
    }

    // Are we stepping a single instruction?
    if (GetGPRState(true) == KERN_SUCCESS)
    {
        // We are single stepping, was this the primary thread?
        if (m_thread->IsStepping())
        {
            // Are we software single stepping?
            if (NUB_BREAK_ID_IS_VALID(m_sw_single_step_break_id) || m_sw_single_step_itblock_break_count)
            {
                // Remove any software single stepping breakpoints that we have set

                // Do we have a normal software single step breakpoint?
                if (NUB_BREAK_ID_IS_VALID(m_sw_single_step_break_id))
                {
                    DNBLogThreadedIf(LOG_STEP, "%s: removing software single step breakpoint (breakID=%d)", __FUNCTION__, m_sw_single_step_break_id);
                    success = m_thread->Process()->DisableBreakpoint(m_sw_single_step_break_id, true);
                    m_sw_single_step_break_id = INVALID_NUB_BREAK_ID;
                }

                // Do we have any Thumb IT breakpoints?
                if (m_sw_single_step_itblock_break_count > 0)
                {
                    // See if we hit one of our Thumb IT breakpoints?
                    DNBBreakpoint *step_bp = m_thread->Process()->Breakpoints().FindByAddress(m_state.context.gpr.__pc);

                    if (step_bp)
                    {
                        // We did hit our breakpoint, tell the breakpoint it was
                        // hit so that it can run its callback routine and fixup
                        // the PC.
                        DNBLogThreadedIf(LOG_STEP, "%s: IT software single step breakpoint hit (breakID=%u)", __FUNCTION__, step_bp->GetID());
                        step_bp->BreakpointHit(m_thread->Process()->ProcessID(), m_thread->ThreadID());
                    }

                    // Remove all Thumb IT breakpoints
                    for (int i = 0; i < m_sw_single_step_itblock_break_count; i++)
                    {
                        if (NUB_BREAK_ID_IS_VALID(m_sw_single_step_itblock_break_id[i]))
                        {
                            DNBLogThreadedIf(LOG_STEP, "%s: removing IT software single step breakpoint (breakID=%d)", __FUNCTION__, m_sw_single_step_itblock_break_id[i]);
                            success = m_thread->Process()->DisableBreakpoint(m_sw_single_step_itblock_break_id[i], true);
                            m_sw_single_step_itblock_break_id[i] = INVALID_NUB_BREAK_ID;
                        }
                    }
                    m_sw_single_step_itblock_break_count = 0;

#if defined (USE_ARM_DISASSEMBLER_FRAMEWORK)

                    // Decode instructions up to the current PC to ensure the internal decoder state is valid for the IT block
                    // The decoder has to decode each instruction in the IT block even if it is not executed so that
                    // the fields are correctly updated
                    DecodeITBlockInstructions(m_state.context.gpr.__pc);
#endif
                }

            }
            else
                success = EnableHardwareSingleStep(false) == KERN_SUCCESS;
        }
        else
        {
            // The MachThread will automatically restore the suspend count
            // in ThreadDidStop(), so we don't need to do anything here if
            // we weren't the primary thread the last time
        }
    }
    return success;
}

bool
DNBArchMachARM::NotifyException(MachException::Data& exc)
{
    switch (exc.exc_type)
    {
        default:
            break;
        case EXC_BREAKPOINT:
            if (exc.exc_data.size() == 2 && exc.exc_data[0] == EXC_ARM_DA_DEBUG)
            {
                // exc_code = EXC_ARM_DA_DEBUG
                //
                // Check whether this corresponds to a watchpoint hit event.
                // If yes, retrieve the exc_sub_code as the data break address.
                if (!HasWatchpointOccurred())
                    break;

                // The data break address is passed as exc_data[1].
                nub_addr_t addr = exc.exc_data[1];
                // Find the hardware index with the side effect of possibly massaging the
                // addr to return the starting address as seen from the debugger side.
                uint32_t hw_index = GetHardwareWatchpointHit(addr);
                if (hw_index != INVALID_NUB_HW_INDEX)
                {
                    m_watchpoint_did_occur = true;
                    m_watchpoint_hw_index = hw_index;
                    exc.exc_data[1] = addr;
                    // Piggyback the hw_index in the exc.data.
                    exc.exc_data.push_back(hw_index);
                }

                return true;
            }
            break;
    }
    return false;
}

bool
DNBArchMachARM::StepNotComplete ()
{
    if (m_hw_single_chained_step_addr != INVALID_NUB_ADDRESS)
    {
        kern_return_t kret = KERN_INVALID_ARGUMENT;
        kret = GetGPRState(false);
        if (kret == KERN_SUCCESS)
        {
            if (m_state.context.gpr.__pc == m_hw_single_chained_step_addr)
            {
                DNBLogThreadedIf(LOG_STEP, "Need to step some more at 0x%8.8x", m_hw_single_chained_step_addr);
                return true;
            }
        }
    }

    m_hw_single_chained_step_addr = INVALID_NUB_ADDRESS;
    return false;
}


#if defined (USE_ARM_DISASSEMBLER_FRAMEWORK)

void
DNBArchMachARM::DecodeITBlockInstructions(nub_addr_t curr_pc)

{
    uint16_t opcode16;
    uint32_t opcode32;
    nub_addr_t next_pc_in_itblock;
    nub_addr_t pc_in_itblock = m_last_decode_pc;

    DNBLogThreadedIf(LOG_STEP | LOG_VERBOSE, "%s: last_decode_pc=0x%8.8x", __FUNCTION__, m_last_decode_pc);

    // Decode IT block instruction from the instruction following the m_last_decoded_instruction at
    // PC m_last_decode_pc upto and including the instruction at curr_pc
    if (m_thread->Process()->Task().ReadMemory(pc_in_itblock, 2, &opcode16) == 2)
    {
        opcode32 = opcode16;
        pc_in_itblock += 2;
        // Check for 32 bit thumb opcode and read the upper 16 bits if needed
        if (((opcode32 & 0xE000) == 0xE000) && opcode32 & 0x1800)
        {
            // Adjust 'next_pc_in_itblock' to point to the default next Thumb instruction for
            // a 32 bit Thumb opcode
            // Read bits 31:16 of a 32 bit Thumb opcode
            if (m_thread->Process()->Task().ReadMemory(pc_in_itblock, 2, &opcode16) == 2)
            {
                pc_in_itblock += 2;
                // 32 bit thumb opcode
                opcode32 = (opcode32 << 16) | opcode16;
            }
            else
            {
                DNBLogError("%s: Unable to read opcode bits 31:16 for a 32 bit thumb opcode at pc=0x%8.8llx", __FUNCTION__, (uint64_t)pc_in_itblock);
            }
        }
    }
    else
    {
        DNBLogError("%s: Error reading 16-bit Thumb instruction at pc=0x%8.8x", __FUNCTION__, pc_in_itblock);
    }

    DNBLogThreadedIf(LOG_STEP | LOG_VERBOSE, "%s: pc_in_itblock=0x%8.8x, curr_pc=0x%8.8x", __FUNCTION__, pc_in_itblock, curr_pc);

    next_pc_in_itblock = pc_in_itblock;
    while (next_pc_in_itblock <= curr_pc)
    {
        arm_error_t decodeError;

        m_last_decode_pc = pc_in_itblock;
        decodeError = DecodeInstructionUsingDisassembler(pc_in_itblock, m_state.context.gpr.__cpsr, &m_last_decode_arm, &m_last_decode_thumb, &next_pc_in_itblock);

        pc_in_itblock = next_pc_in_itblock;
        DNBLogThreadedIf(LOG_STEP | LOG_VERBOSE, "%s: next_pc_in_itblock=0x%8.8x", __FUNCTION__, next_pc_in_itblock);
    }
}
#endif

// Set the single step bit in the processor status register.
kern_return_t
DNBArchMachARM::EnableHardwareSingleStep (bool enable)
{
    DNBError err;
    DNBLogThreadedIf(LOG_STEP, "%s( enable = %d )", __FUNCTION__, enable);

    err = GetGPRState(false);

    if (err.Fail())
    {
        err.LogThreaded("%s: failed to read the GPR registers", __FUNCTION__);
        return err.Error();
    }

    err = GetDBGState(false);

    if (err.Fail())
    {
        err.LogThreaded("%s: failed to read the DBG registers", __FUNCTION__);
        return err.Error();
    }

    const uint32_t i = 0;
    if (enable)
    {
        m_hw_single_chained_step_addr = INVALID_NUB_ADDRESS;

        // Save our previous state
        m_dbg_save = m_state.dbg;
        // Set a breakpoint that will stop when the PC doesn't match the current one!
        m_state.dbg.__bvr[i] = m_state.context.gpr.__pc & 0xFFFFFFFCu;      // Set the current PC as the breakpoint address
        m_state.dbg.__bcr[i] = BCR_M_IMVA_MISMATCH |    // Stop on address mismatch
                               S_USER |                 // Stop only in user mode
                               BCR_ENABLE;              // Enable this breakpoint
        if (m_state.context.gpr.__cpsr & 0x20)
        {
            // Thumb breakpoint
            if (m_state.context.gpr.__pc & 2)
                m_state.dbg.__bcr[i] |= BAS_IMVA_2_3;
            else
                m_state.dbg.__bcr[i] |= BAS_IMVA_0_1;

            uint16_t opcode;
            if (sizeof(opcode) == m_thread->Process()->Task().ReadMemory(m_state.context.gpr.__pc, sizeof(opcode), &opcode))
            {
                if (((opcode & 0xE000) == 0xE000) && opcode & 0x1800)
                {
                    // 32 bit thumb opcode...
                    if (m_state.context.gpr.__pc & 2)
                    {
                        // We can't take care of a 32 bit thumb instruction single step
                        // with just IVA mismatching. We will need to chain an extra
                        // hardware single step in order to complete this single step...
                        m_hw_single_chained_step_addr = m_state.context.gpr.__pc + 2;
                    }
                    else
                    {
                        // Extend the number of bits to ignore for the mismatch
                        m_state.dbg.__bcr[i] |= BAS_IMVA_ALL;
                    }
                }
            }
        }
        else
        {
            // ARM breakpoint
            m_state.dbg.__bcr[i] |= BAS_IMVA_ALL; // Stop when any address bits change
        }

        DNBLogThreadedIf(LOG_STEP, "%s: BVR%u=0x%8.8x  BCR%u=0x%8.8x", __FUNCTION__, i, m_state.dbg.__bvr[i], i, m_state.dbg.__bcr[i]);

        for (uint32_t j=i+1; j<16; ++j)
        {
            // Disable all others
            m_state.dbg.__bvr[j] = 0;
            m_state.dbg.__bcr[j] = 0;
        }
    }
    else
    {
        // Just restore the state we had before we did single stepping
        m_state.dbg = m_dbg_save;
    }

    return SetDBGState();
}

// return 1 if bit "BIT" is set in "value"
static inline uint32_t bit(uint32_t value, uint32_t bit)
{
    return (value >> bit) & 1u;
}

// return the bitfield "value[msbit:lsbit]".
static inline uint32_t bits(uint32_t value, uint32_t msbit, uint32_t lsbit)
{
    assert(msbit >= lsbit);
    uint32_t shift_left = sizeof(value) * 8 - 1 - msbit;
    value <<= shift_left;           // shift anything above the msbit off of the unsigned edge
    value >>= (shift_left + lsbit); // shift it back again down to the lsbit (including undoing any shift from above)
    return value;                   // return our result
}

bool
DNBArchMachARM::ConditionPassed(uint8_t condition, uint32_t cpsr)
{
    uint32_t cpsr_n = bit(cpsr, 31); // Negative condition code flag
    uint32_t cpsr_z = bit(cpsr, 30); // Zero condition code flag
    uint32_t cpsr_c = bit(cpsr, 29); // Carry condition code flag
    uint32_t cpsr_v = bit(cpsr, 28); // Overflow condition code flag

    switch (condition) {
        case COND_EQ: // (0x0)
            if (cpsr_z == 1) return true;
            break;
        case COND_NE: // (0x1)
            if (cpsr_z == 0) return true;
            break;
        case COND_CS: // (0x2)
            if (cpsr_c == 1) return true;
            break;
        case COND_CC: // (0x3)
            if (cpsr_c == 0) return true;
            break;
        case COND_MI: // (0x4)
            if (cpsr_n == 1) return true;
            break;
        case COND_PL: // (0x5)
            if (cpsr_n == 0) return true;
            break;
        case COND_VS: // (0x6)
            if (cpsr_v == 1) return true;
            break;
        case COND_VC: // (0x7)
            if (cpsr_v == 0) return true;
            break;
        case COND_HI: // (0x8)
            if ((cpsr_c == 1) && (cpsr_z == 0)) return true;
            break;
        case COND_LS: // (0x9)
            if ((cpsr_c == 0) || (cpsr_z == 1)) return true;
            break;
        case COND_GE: // (0xA)
            if (cpsr_n == cpsr_v) return true;
            break;
        case COND_LT: // (0xB)
            if (cpsr_n != cpsr_v) return true;
            break;
        case COND_GT: // (0xC)
            if ((cpsr_z == 0) && (cpsr_n == cpsr_v)) return true;
            break;
        case COND_LE: // (0xD)
            if ((cpsr_z == 1) || (cpsr_n != cpsr_v)) return true;
            break;
        default:
            return true;
            break;
    }

    return false;
}

#if defined (USE_ARM_DISASSEMBLER_FRAMEWORK)

bool
DNBArchMachARM::ComputeNextPC(nub_addr_t currentPC, arm_decoded_instruction_t decodedInstruction, bool currentPCIsThumb, nub_addr_t *targetPC)
{
    nub_addr_t myTargetPC, addressWherePCLives;
    pid_t mypid;

    uint32_t cpsr_c = bit(m_state.context.gpr.__cpsr, 29); // Carry condition code flag

    uint32_t firstOperand=0, secondOperand=0, shiftAmount=0, secondOperandAfterShift=0, immediateValue=0;
    uint32_t halfwords=0, baseAddress=0, immediateOffset=0, addressOffsetFromRegister=0, addressOffsetFromRegisterAfterShift;
    uint32_t baseAddressIndex=INVALID_NUB_HW_INDEX;
    uint32_t firstOperandIndex=INVALID_NUB_HW_INDEX;
    uint32_t secondOperandIndex=INVALID_NUB_HW_INDEX;
    uint32_t addressOffsetFromRegisterIndex=INVALID_NUB_HW_INDEX;
    uint32_t shiftRegisterIndex=INVALID_NUB_HW_INDEX;
    uint16_t registerList16, registerList16NoPC;
    uint8_t registerList8;
    uint32_t numRegistersToLoad=0;

    DNBLogThreadedIf(LOG_STEP | LOG_VERBOSE, "%s: instruction->code=%d", __FUNCTION__, decodedInstruction.instruction->code);

    // Get the following in this switch statement:
    //   - firstOperand, secondOperand, immediateValue, shiftAmount: For arithmetic, logical and move instructions
    //   - baseAddress, immediateOffset, shiftAmount: For LDR
    //   - numRegistersToLoad: For LDM and POP instructions
    switch (decodedInstruction.instruction->code)
    {
            // Arithmetic operations that can change the PC
        case ARM_INST_ADC:
        case ARM_INST_ADCS:
        case ARM_INST_ADD:
        case ARM_INST_ADDS:
        case ARM_INST_AND:
        case ARM_INST_ANDS:
        case ARM_INST_ASR:
        case ARM_INST_ASRS:
        case ARM_INST_BIC:
        case ARM_INST_BICS:
        case ARM_INST_EOR:
        case ARM_INST_EORS:
        case ARM_INST_ORR:
        case ARM_INST_ORRS:
        case ARM_INST_RSB:
        case ARM_INST_RSBS:
        case ARM_INST_RSC:
        case ARM_INST_RSCS:
        case ARM_INST_SBC:
        case ARM_INST_SBCS:
        case ARM_INST_SUB:
        case ARM_INST_SUBS:
            switch (decodedInstruction.addressMode)
            {
                case ARM_ADDR_DATA_IMM:
                    if (decodedInstruction.numOperands != 3)
                    {
                        DNBLogError("Expected 3 operands in decoded instruction structure. numOperands is %d!", decodedInstruction.numOperands);
                        return false;
                    }

                    if (decodedInstruction.op[0].value != PC_REG)
                    {
                        DNBLogError("Destination register is not a PC! %s routine should be called on on instructions that modify the PC. Destination register is R%d!", __FUNCTION__, decodedInstruction.op[0].value);
                        return false;
                    }

                    // Get firstOperand register value (at index=1)
                    firstOperandIndex = decodedInstruction.op[1].value; // first operand register index
                    firstOperand = m_state.context.gpr.__r[firstOperandIndex];

                    // Get immediateValue (at index=2)
                    immediateValue = decodedInstruction.op[2].value;

                    break;

                case ARM_ADDR_DATA_REG:
                    if (decodedInstruction.numOperands != 3)
                    {
                        DNBLogError("Expected 3 operands in decoded instruction structure. numOperands is %d!", decodedInstruction.numOperands);
                        return false;
                    }

                    if (decodedInstruction.op[0].value != PC_REG)
                    {
                        DNBLogError("Destination register is not a PC! %s routine should be called on on instructions that modify the PC. Destination register is R%d!", __FUNCTION__, decodedInstruction.op[0].value);
                        return false;
                    }

                    // Get firstOperand register value (at index=1)
                    firstOperandIndex = decodedInstruction.op[1].value; // first operand register index
                    firstOperand = m_state.context.gpr.__r[firstOperandIndex];

                    // Get secondOperand register value (at index=2)
                    secondOperandIndex = decodedInstruction.op[2].value; // second operand register index
                    secondOperand = m_state.context.gpr.__r[secondOperandIndex];

                    break;

                case ARM_ADDR_DATA_SCALED_IMM:
                    if (decodedInstruction.numOperands != 4)
                    {
                        DNBLogError("Expected 4 operands in decoded instruction structure. numOperands is %d!", decodedInstruction.numOperands);
                        return false;
                    }

                    if (decodedInstruction.op[0].value != PC_REG)
                    {
                        DNBLogError("Destination register is not a PC! %s routine should be called on on instructions that modify the PC. Destination register is R%d!", __FUNCTION__, decodedInstruction.op[0].value);
                        return false;
                    }

                    // Get firstOperand register value (at index=1)
                    firstOperandIndex = decodedInstruction.op[1].value; // first operand register index
                    firstOperand = m_state.context.gpr.__r[firstOperandIndex];

                    // Get secondOperand register value (at index=2)
                    secondOperandIndex = decodedInstruction.op[2].value; // second operand register index
                    secondOperand = m_state.context.gpr.__r[secondOperandIndex];

                    // Get shiftAmount as immediate value (at index=3)
                    shiftAmount = decodedInstruction.op[3].value;

                    break;


                case ARM_ADDR_DATA_SCALED_REG:
                    if (decodedInstruction.numOperands != 4)
                    {
                        DNBLogError("Expected 4 operands in decoded instruction structure. numOperands is %d!", decodedInstruction.numOperands);
                        return false;
                    }

                    if (decodedInstruction.op[0].value != PC_REG)
                    {
                        DNBLogError("Destination register is not a PC! %s routine should be called on on instructions that modify the PC. Destination register is R%d!", __FUNCTION__, decodedInstruction.op[0].value);
                        return false;
                    }

                    // Get firstOperand register value (at index=1)
                    firstOperandIndex = decodedInstruction.op[1].value; // first operand register index
                    firstOperand = m_state.context.gpr.__r[firstOperandIndex];

                    // Get secondOperand register value (at index=2)
                    secondOperandIndex = decodedInstruction.op[2].value; // second operand register index
                    secondOperand = m_state.context.gpr.__r[secondOperandIndex];

                    // Get shiftAmount from register (at index=3)
                    shiftRegisterIndex = decodedInstruction.op[3].value; // second operand register index
                    shiftAmount = m_state.context.gpr.__r[shiftRegisterIndex];

                    break;

                case THUMB_ADDR_HR_HR:
                    if (decodedInstruction.numOperands != 2)
                    {
                        DNBLogError("Expected 2 operands in decoded instruction structure. numOperands is %d!", decodedInstruction.numOperands);
                        return false;
                    }

                    if (decodedInstruction.op[0].value != PC_REG)
                    {
                        DNBLogError("Destination register is not a PC! %s routine should be called on on instructions that modify the PC. Destination register is R%d!", __FUNCTION__, decodedInstruction.op[0].value);
                        return false;
                    }

                    // Get firstOperand register value (at index=0)
                    firstOperandIndex = decodedInstruction.op[0].value; // first operand register index
                    firstOperand = m_state.context.gpr.__r[firstOperandIndex];

                    // Get secondOperand register value (at index=1)
                    secondOperandIndex = decodedInstruction.op[1].value; // second operand register index
                    secondOperand = m_state.context.gpr.__r[secondOperandIndex];

                    break;

                default:
                    break;
            }
            break;

            // Logical shifts and move operations that can change the PC
        case ARM_INST_LSL:
        case ARM_INST_LSLS:
        case ARM_INST_LSR:
        case ARM_INST_LSRS:
        case ARM_INST_MOV:
        case ARM_INST_MOVS:
        case ARM_INST_MVN:
        case ARM_INST_MVNS:
        case ARM_INST_ROR:
        case ARM_INST_RORS:
        case ARM_INST_RRX:
        case ARM_INST_RRXS:
            // In these cases, the firstOperand is always 0, as if it does not exist
            switch (decodedInstruction.addressMode)
            {
                case ARM_ADDR_DATA_IMM:
                    if (decodedInstruction.numOperands != 2)
                    {
                        DNBLogError("Expected 2 operands in decoded instruction structure. numOperands is %d!", decodedInstruction.numOperands);
                        return false;
                    }

                    if (decodedInstruction.op[0].value != PC_REG)
                    {
                        DNBLogError("Destination register is not a PC! %s routine should be called on on instructions that modify the PC. Destination register is R%d!", __FUNCTION__, decodedInstruction.op[0].value);
                        return false;
                    }

                    // Get immediateValue (at index=1)
                    immediateValue = decodedInstruction.op[1].value;

                    break;

                case ARM_ADDR_DATA_REG:
                    if (decodedInstruction.numOperands != 2)
                    {
                        DNBLogError("Expected 2 operands in decoded instruction structure. numOperands is %d!", decodedInstruction.numOperands);
                        return false;
                    }

                    if (decodedInstruction.op[0].value != PC_REG)
                    {
                        DNBLogError("Destination register is not a PC! %s routine should be called on on instructions that modify the PC. Destination register is R%d!", __FUNCTION__, decodedInstruction.op[0].value);
                        return false;
                    }

                    // Get secondOperand register value (at index=1)
                    secondOperandIndex = decodedInstruction.op[1].value; // second operand register index
                    secondOperand = m_state.context.gpr.__r[secondOperandIndex];

                    break;

                case ARM_ADDR_DATA_SCALED_IMM:
                    if (decodedInstruction.numOperands != 3)
                    {
                        DNBLogError("Expected 4 operands in decoded instruction structure. numOperands is %d!", decodedInstruction.numOperands);
                        return false;
                    }

                    if (decodedInstruction.op[0].value != PC_REG)
                    {
                        DNBLogError("Destination register is not a PC! %s routine should be called on on instructions that modify the PC. Destination register is R%d!", __FUNCTION__, decodedInstruction.op[0].value);
                        return false;
                    }

                    // Get secondOperand register value (at index=1)
                    secondOperandIndex = decodedInstruction.op[2].value; // second operand register index
                    secondOperand = m_state.context.gpr.__r[secondOperandIndex];

                    // Get shiftAmount as immediate value (at index=2)
                    shiftAmount = decodedInstruction.op[2].value;

                    break;


                case ARM_ADDR_DATA_SCALED_REG:
                    if (decodedInstruction.numOperands != 3)
                    {
                        DNBLogError("Expected 3 operands in decoded instruction structure. numOperands is %d!", decodedInstruction.numOperands);
                        return false;
                    }

                    if (decodedInstruction.op[0].value != PC_REG)
                    {
                        DNBLogError("Destination register is not a PC! %s routine should be called on on instructions that modify the PC. Destination register is R%d!", __FUNCTION__, decodedInstruction.op[0].value);
                        return false;
                    }

                    // Get secondOperand register value (at index=1)
                    secondOperandIndex = decodedInstruction.op[1].value; // second operand register index
                    secondOperand = m_state.context.gpr.__r[secondOperandIndex];

                    // Get shiftAmount from register (at index=2)
                    shiftRegisterIndex = decodedInstruction.op[2].value; // second operand register index
                    shiftAmount = m_state.context.gpr.__r[shiftRegisterIndex];

                    break;

                case THUMB_ADDR_HR_HR:
                    if (decodedInstruction.numOperands != 2)
                    {
                        DNBLogError("Expected 2 operands in decoded instruction structure. numOperands is %d!", decodedInstruction.numOperands);
                        return false;
                    }

                    if (decodedInstruction.op[0].value != PC_REG)
                    {
                        DNBLogError("Destination register is not a PC! %s routine should be called on on instructions that modify the PC. Destination register is R%d!", __FUNCTION__, decodedInstruction.op[0].value);
                        return false;
                    }

                    // Get secondOperand register value (at index=1)
                    secondOperandIndex = decodedInstruction.op[1].value; // second operand register index
                    secondOperand = m_state.context.gpr.__r[secondOperandIndex];

                    break;

                default:
                    break;
            }

            break;

            // Simple branches, used to hop around within a routine
        case ARM_INST_B:
            *targetPC = decodedInstruction.targetPC; // Known targetPC
            return true;
            break;

            // Branch-and-link, used to call ARM subroutines
        case ARM_INST_BL:
            *targetPC = decodedInstruction.targetPC; // Known targetPC
            return true;
            break;

            // Branch-and-link with exchange, used to call opposite-mode subroutines
        case ARM_INST_BLX:
            if ((decodedInstruction.addressMode == ARM_ADDR_BRANCH_IMM) ||
                (decodedInstruction.addressMode == THUMB_ADDR_UNCOND))
            {
                *targetPC = decodedInstruction.targetPC; // Known targetPC
                return true;
            }
            else    // addressMode == ARM_ADDR_BRANCH_REG
            {
                // Unknown target unless we're branching to the PC itself,
                //  although this may not work properly with BLX
                if (decodedInstruction.op[REG_RD].value == PC_REG)
                {
                    // this should (almost) never happen
                    *targetPC = decodedInstruction.targetPC; // Known targetPC
                    return true;
                }

                // Get the branch address and return
                if (decodedInstruction.numOperands != 1)
                {
                    DNBLogError("Expected 1 operands in decoded instruction structure. numOperands is %d!", decodedInstruction.numOperands);
                    return false;
                }

                // Get branch address in register (at index=0)
                *targetPC = m_state.context.gpr.__r[decodedInstruction.op[0].value];
                return true;
            }
            break;

            // Branch with exchange, used to hop to opposite-mode code
            // Branch to Jazelle code, used to execute Java; included here since it
            //  acts just like BX unless the Jazelle unit is active and JPC is
            //  already loaded into it.
        case ARM_INST_BX:
        case ARM_INST_BXJ:
            // Unknown target unless we're branching to the PC itself,
            //  although this can never switch to Thumb mode and is
            //  therefore pretty much useless
            if (decodedInstruction.op[REG_RD].value == PC_REG)
            {
                // this should (almost) never happen
                *targetPC = decodedInstruction.targetPC; // Known targetPC
                return true;
            }

            // Get the branch address and return
            if (decodedInstruction.numOperands != 1)
            {
                DNBLogError("Expected 1 operands in decoded instruction structure. numOperands is %d!", decodedInstruction.numOperands);
                return false;
            }

            // Get branch address in register (at index=0)
            *targetPC = m_state.context.gpr.__r[decodedInstruction.op[0].value];
            return true;
            break;

            // Compare and branch on zero/non-zero (Thumb-16 only)
            // Unusual condition check built into the instruction
        case ARM_INST_CBZ:
        case ARM_INST_CBNZ:
            // Branch address is known at compile time
            // Get the branch address and return
            if (decodedInstruction.numOperands != 2)
            {
                DNBLogError("Expected 2 operands in decoded instruction structure. numOperands is %d!", decodedInstruction.numOperands);
                return false;
            }

            // Get branch address as an immediate value (at index=1)
            *targetPC = decodedInstruction.op[1].value;
            return true;
            break;

            // Load register can be used to load PC, usually with a function pointer
        case ARM_INST_LDR:
            if (decodedInstruction.op[REG_RD].value != PC_REG)
            {
                DNBLogError("Destination register is not a PC! %s routine should be called on on instructions that modify the PC. Destination register is R%d!", __FUNCTION__, decodedInstruction.op[0].value);
                return false;
            }
            switch (decodedInstruction.addressMode)
            {
                case ARM_ADDR_LSWUB_IMM:
                case ARM_ADDR_LSWUB_IMM_PRE:
                case ARM_ADDR_LSWUB_IMM_POST:
                    if (decodedInstruction.numOperands != 3)
                    {
                        DNBLogError("Expected 3 operands in decoded instruction structure. numOperands is %d!", decodedInstruction.numOperands);
                        return false;
                    }

                    // Get baseAddress from register (at index=1)
                    baseAddressIndex = decodedInstruction.op[1].value;
                    baseAddress = m_state.context.gpr.__r[baseAddressIndex];

                    // Get immediateOffset (at index=2)
                    immediateOffset = decodedInstruction.op[2].value;
                    break;

                case ARM_ADDR_LSWUB_REG:
                case ARM_ADDR_LSWUB_REG_PRE:
                case ARM_ADDR_LSWUB_REG_POST:
                    if (decodedInstruction.numOperands != 3)
                    {
                        DNBLogError("Expected 3 operands in decoded instruction structure. numOperands is %d!", decodedInstruction.numOperands);
                        return false;
                    }

                    // Get baseAddress from register (at index=1)
                    baseAddressIndex = decodedInstruction.op[1].value;
                    baseAddress = m_state.context.gpr.__r[baseAddressIndex];

                    // Get immediateOffset from register (at index=2)
                    addressOffsetFromRegisterIndex = decodedInstruction.op[2].value;
                    addressOffsetFromRegister = m_state.context.gpr.__r[addressOffsetFromRegisterIndex];

                    break;

                case ARM_ADDR_LSWUB_SCALED:
                case ARM_ADDR_LSWUB_SCALED_PRE:
                case ARM_ADDR_LSWUB_SCALED_POST:
                    if (decodedInstruction.numOperands != 4)
                    {
                        DNBLogError("Expected 4 operands in decoded instruction structure. numOperands is %d!", decodedInstruction.numOperands);
                        return false;
                    }

                    // Get baseAddress from register (at index=1)
                    baseAddressIndex = decodedInstruction.op[1].value;
                    baseAddress = m_state.context.gpr.__r[baseAddressIndex];

                    // Get immediateOffset from register (at index=2)
                    addressOffsetFromRegisterIndex = decodedInstruction.op[2].value;
                    addressOffsetFromRegister = m_state.context.gpr.__r[addressOffsetFromRegisterIndex];

                    // Get shiftAmount (at index=3)
                    shiftAmount = decodedInstruction.op[3].value;

                    break;

                default:
                    break;
            }
            break;

            // 32b load multiple operations can load the PC along with everything else,
            //  usually to return from a function call
        case ARM_INST_LDMDA:
        case ARM_INST_LDMDB:
        case ARM_INST_LDMIA:
        case ARM_INST_LDMIB:
            if (decodedInstruction.op[LDM_REGLIST].value & PC_REGLIST_BIT)
            {
                if (decodedInstruction.numOperands != 2)
                {
                    DNBLogError("Expected 2 operands in decoded instruction structure. numOperands is %d!", decodedInstruction.numOperands);
                    return false;
                }

                // Get baseAddress from register (at index=0)
                baseAddressIndex = decodedInstruction.op[0].value;
                baseAddress = m_state.context.gpr.__r[baseAddressIndex];

                // Get registerList from register (at index=1)
                registerList16 = (uint16_t)decodedInstruction.op[1].value;

                // Count number of registers to load in the multiple register list excluding the PC
                registerList16NoPC = registerList16&0x3FFF; // exclude the PC
                numRegistersToLoad=0;
                for (int i = 0; i < 16; i++)
                {
                    if (registerList16NoPC & 0x1) numRegistersToLoad++;
                    registerList16NoPC = registerList16NoPC >> 1;
                }
            }
            else
            {
                DNBLogError("Destination register is not a PC! %s routine should be called on on instructions that modify the PC. Destination register is R%d!", __FUNCTION__, decodedInstruction.op[0].value);
                return false;
            }
            break;

            // Normal 16-bit LD multiple can't touch R15, but POP can
        case ARM_INST_POP:  // Can also get the PC & updates SP
            // Get baseAddress from SP (at index=0)
            baseAddress = m_state.context.gpr.__sp;

            if (decodedInstruction.thumb16b)
            {
                // Get registerList from register (at index=0)
                registerList8 = (uint8_t)decodedInstruction.op[0].value;

                // Count number of registers to load in the multiple register list
                numRegistersToLoad=0;
                for (int i = 0; i < 8; i++)
                {
                    if (registerList8 & 0x1) numRegistersToLoad++;
                    registerList8 = registerList8 >> 1;
                }
            }
            else
            {
                // Get registerList from register (at index=0)
                registerList16 = (uint16_t)decodedInstruction.op[0].value;

                // Count number of registers to load in the multiple register list excluding the PC
                registerList16NoPC = registerList16&0x3FFF; // exclude the PC
                numRegistersToLoad=0;
                for (int i = 0; i < 16; i++)
                {
                    if (registerList16NoPC & 0x1) numRegistersToLoad++;
                    registerList16NoPC = registerList16NoPC >> 1;
                }
            }
            break;

            // 16b TBB and TBH instructions load a jump address from a table
        case ARM_INST_TBB:
        case ARM_INST_TBH:
            // Get baseAddress from register (at index=0)
            baseAddressIndex = decodedInstruction.op[0].value;
            baseAddress = m_state.context.gpr.__r[baseAddressIndex];

            // Get immediateOffset from register (at index=1)
            addressOffsetFromRegisterIndex = decodedInstruction.op[1].value;
            addressOffsetFromRegister = m_state.context.gpr.__r[addressOffsetFromRegisterIndex];
            break;

            // ThumbEE branch-to-handler instructions: Jump to handlers at some offset
            //  from a special base pointer register (which is unknown at disassembly time)
        case ARM_INST_HB:
        case ARM_INST_HBP:
//          TODO: ARM_INST_HB, ARM_INST_HBP
            break;

        case ARM_INST_HBL:
        case ARM_INST_HBLP:
//          TODO: ARM_INST_HBL, ARM_INST_HBLP
            break;

            // Breakpoint and software interrupt jump to interrupt handler (always ARM)
        case ARM_INST_BKPT:
        case ARM_INST_SMC:
        case ARM_INST_SVC:

            // Return from exception, obviously modifies PC [interrupt only!]
        case ARM_INST_RFEDA:
        case ARM_INST_RFEDB:
        case ARM_INST_RFEIA:
        case ARM_INST_RFEIB:

            // Other instructions either can't change R15 or are "undefined" if you do,
            //  so no sane compiler should ever generate them & we don't care here.
            //  Also, R15 can only legally be used in a read-only manner for the
            //  various ARM addressing mode (to get PC-relative addressing of constants),
            //  but can NOT be used with any of the update modes.
        default:
            DNBLogError("%s should not be called for instruction code %d!", __FUNCTION__, decodedInstruction.instruction->code);
            return false;
            break;
    }

    // Adjust PC if PC is one of the input operands
    if (baseAddressIndex == PC_REG)
    {
        if (currentPCIsThumb)
            baseAddress += 4;
        else
            baseAddress += 8;
    }

    if (firstOperandIndex == PC_REG)
    {
        if (currentPCIsThumb)
            firstOperand += 4;
        else
            firstOperand += 8;
    }

    if (secondOperandIndex == PC_REG)
    {
        if (currentPCIsThumb)
            secondOperand += 4;
        else
            secondOperand += 8;
    }

    if (addressOffsetFromRegisterIndex == PC_REG)
    {
        if (currentPCIsThumb)
            addressOffsetFromRegister += 4;
        else
            addressOffsetFromRegister += 8;
    }

    DNBLogThreadedIf(LOG_STEP | LOG_VERBOSE,
        "%s: firstOperand=%8.8x, secondOperand=%8.8x, immediateValue = %d, shiftAmount = %d, baseAddress = %8.8x, addressOffsetFromRegister = %8.8x, immediateOffset = %d, numRegistersToLoad = %d",
        __FUNCTION__,
        firstOperand,
        secondOperand,
        immediateValue,
        shiftAmount,
        baseAddress,
        addressOffsetFromRegister,
        immediateOffset,
        numRegistersToLoad);


    // Calculate following values after applying shiftAmount:
    //   - immediateOffsetAfterShift, secondOperandAfterShift

    switch (decodedInstruction.scaleMode)
    {
        case ARM_SCALE_NONE:
            addressOffsetFromRegisterAfterShift = addressOffsetFromRegister;
            secondOperandAfterShift = secondOperand;
            break;

        case ARM_SCALE_LSL:             // Logical shift left
            addressOffsetFromRegisterAfterShift = addressOffsetFromRegister << shiftAmount;
            secondOperandAfterShift = secondOperand << shiftAmount;
            break;

        case ARM_SCALE_LSR:             // Logical shift right
            addressOffsetFromRegisterAfterShift = addressOffsetFromRegister >> shiftAmount;
            secondOperandAfterShift = secondOperand >> shiftAmount;
            break;

        case ARM_SCALE_ASR:             // Arithmetic shift right
            asm("mov %0, %1, asr %2" : "=r" (addressOffsetFromRegisterAfterShift) : "r" (addressOffsetFromRegister), "r" (shiftAmount));
            asm("mov %0, %1, asr %2" : "=r" (secondOperandAfterShift) : "r" (secondOperand), "r" (shiftAmount));
            break;

        case ARM_SCALE_ROR:             // Rotate right
            asm("mov %0, %1, ror %2" : "=r" (addressOffsetFromRegisterAfterShift) : "r" (addressOffsetFromRegister), "r" (shiftAmount));
            asm("mov %0, %1, ror %2" : "=r" (secondOperandAfterShift) : "r" (secondOperand), "r" (shiftAmount));
            break;

        case ARM_SCALE_RRX:             // Rotate right, pulling in carry (1-bit shift only)
            asm("mov %0, %1, rrx" : "=r" (addressOffsetFromRegisterAfterShift) : "r" (addressOffsetFromRegister));
            asm("mov %0, %1, rrx" : "=r" (secondOperandAfterShift) : "r" (secondOperand));
            break;
    }

    // Emulate instruction to calculate targetPC
    // All branches are already handled in the first switch statement. A branch should not reach this switch
    switch (decodedInstruction.instruction->code)
    {
            // Arithmetic operations that can change the PC
        case ARM_INST_ADC:
        case ARM_INST_ADCS:
            // Add with Carry
            *targetPC = firstOperand + (secondOperandAfterShift + immediateValue) + cpsr_c;
            break;

        case ARM_INST_ADD:
        case ARM_INST_ADDS:
            *targetPC = firstOperand + (secondOperandAfterShift + immediateValue);
            break;

        case ARM_INST_AND:
        case ARM_INST_ANDS:
            *targetPC = firstOperand & (secondOperandAfterShift + immediateValue);
            break;

        case ARM_INST_ASR:
        case ARM_INST_ASRS:
            asm("mov %0, %1, asr %2" : "=r" (myTargetPC) : "r" (firstOperand), "r" (secondOperandAfterShift + immediateValue));
            *targetPC = myTargetPC;
            break;

        case ARM_INST_BIC:
        case ARM_INST_BICS:
            asm("bic %0, %1, %2" : "=r" (myTargetPC) : "r" (firstOperand), "r" (secondOperandAfterShift + immediateValue));
            *targetPC = myTargetPC;
            break;

        case ARM_INST_EOR:
        case ARM_INST_EORS:
            asm("eor %0, %1, %2" : "=r" (myTargetPC) : "r" (firstOperand), "r" (secondOperandAfterShift + immediateValue));
            *targetPC = myTargetPC;
            break;

        case ARM_INST_ORR:
        case ARM_INST_ORRS:
            asm("orr %0, %1, %2" : "=r" (myTargetPC) : "r" (firstOperand), "r" (secondOperandAfterShift + immediateValue));
            *targetPC = myTargetPC;
            break;

        case ARM_INST_RSB:
        case ARM_INST_RSBS:
            asm("rsb %0, %1, %2" : "=r" (myTargetPC) : "r" (firstOperand), "r" (secondOperandAfterShift + immediateValue));
            *targetPC = myTargetPC;
            break;

        case ARM_INST_RSC:
        case ARM_INST_RSCS:
            myTargetPC = secondOperandAfterShift - (firstOperand + !cpsr_c);
            *targetPC = myTargetPC;
            break;

        case ARM_INST_SBC:
        case ARM_INST_SBCS:
            asm("sbc %0, %1, %2" : "=r" (myTargetPC) : "r" (firstOperand), "r" (secondOperandAfterShift + immediateValue  + !cpsr_c));
            *targetPC = myTargetPC;
            break;

        case ARM_INST_SUB:
        case ARM_INST_SUBS:
            asm("sub %0, %1, %2" : "=r" (myTargetPC) : "r" (firstOperand), "r" (secondOperandAfterShift + immediateValue));
            *targetPC = myTargetPC;
            break;

            // Logical shifts and move operations that can change the PC
        case ARM_INST_LSL:
        case ARM_INST_LSLS:
        case ARM_INST_LSR:
        case ARM_INST_LSRS:
        case ARM_INST_MOV:
        case ARM_INST_MOVS:
        case ARM_INST_ROR:
        case ARM_INST_RORS:
        case ARM_INST_RRX:
        case ARM_INST_RRXS:
            myTargetPC = secondOperandAfterShift + immediateValue;
            *targetPC = myTargetPC;
            break;

        case ARM_INST_MVN:
        case ARM_INST_MVNS:
            myTargetPC = !(secondOperandAfterShift + immediateValue);
            *targetPC = myTargetPC;
            break;

            // Load register can be used to load PC, usually with a function pointer
        case ARM_INST_LDR:
            switch (decodedInstruction.addressMode) {
                case ARM_ADDR_LSWUB_IMM_POST:
                case ARM_ADDR_LSWUB_REG_POST:
                case ARM_ADDR_LSWUB_SCALED_POST:
                    addressWherePCLives = baseAddress;
                    break;

                case ARM_ADDR_LSWUB_IMM:
                case ARM_ADDR_LSWUB_REG:
                case ARM_ADDR_LSWUB_SCALED:
                case ARM_ADDR_LSWUB_IMM_PRE:
                case ARM_ADDR_LSWUB_REG_PRE:
                case ARM_ADDR_LSWUB_SCALED_PRE:
                    addressWherePCLives = baseAddress + (addressOffsetFromRegisterAfterShift + immediateOffset);
                    break;

                default:
                    break;
            }

            mypid = m_thread->ProcessID();
            if (DNBProcessMemoryRead(mypid, addressWherePCLives, sizeof(nub_addr_t), &myTargetPC) !=  sizeof(nub_addr_t))
            {
                DNBLogError("Could not read memory at %8.8x to get targetPC when processing the pop instruction!", addressWherePCLives);
                return false;
            }

            *targetPC = myTargetPC;
            break;

            // 32b load multiple operations can load the PC along with everything else,
            //  usually to return from a function call
        case ARM_INST_LDMDA:
            mypid = m_thread->ProcessID();
            addressWherePCLives = baseAddress;
            if (DNBProcessMemoryRead(mypid, addressWherePCLives, sizeof(nub_addr_t), &myTargetPC) !=  sizeof(nub_addr_t))
            {
                DNBLogError("Could not read memory at %8.8x to get targetPC when processing the pop instruction!", addressWherePCLives);
                return false;
            }

            *targetPC = myTargetPC;
            break;

        case ARM_INST_LDMDB:
            mypid = m_thread->ProcessID();
            addressWherePCLives = baseAddress - 4;
            if (DNBProcessMemoryRead(mypid, addressWherePCLives, sizeof(nub_addr_t), &myTargetPC) !=  sizeof(nub_addr_t))
            {
                DNBLogError("Could not read memory at %8.8x to get targetPC when processing the pop instruction!", addressWherePCLives);
                return false;
            }

            *targetPC = myTargetPC;
            break;

        case ARM_INST_LDMIB:
            mypid = m_thread->ProcessID();
            addressWherePCLives = baseAddress + numRegistersToLoad*4 + 4;
            if (DNBProcessMemoryRead(mypid, addressWherePCLives, sizeof(nub_addr_t), &myTargetPC) !=  sizeof(nub_addr_t))
            {
                DNBLogError("Could not read memory at %8.8x to get targetPC when processing the pop instruction!", addressWherePCLives);
                return false;
            }

            *targetPC = myTargetPC;
            break;

        case ARM_INST_LDMIA: // same as pop
            // Normal 16-bit LD multiple can't touch R15, but POP can
        case ARM_INST_POP:  // Can also get the PC & updates SP
            mypid = m_thread->ProcessID();
            addressWherePCLives = baseAddress + numRegistersToLoad*4;
            if (DNBProcessMemoryRead(mypid, addressWherePCLives, sizeof(nub_addr_t), &myTargetPC) !=  sizeof(nub_addr_t))
            {
                DNBLogError("Could not read memory at %8.8x to get targetPC when processing the pop instruction!", addressWherePCLives);
                return false;
            }

            *targetPC = myTargetPC;
            break;

            // 16b TBB and TBH instructions load a jump address from a table
        case ARM_INST_TBB:
            mypid = m_thread->ProcessID();
            addressWherePCLives = baseAddress + addressOffsetFromRegisterAfterShift;
            if (DNBProcessMemoryRead(mypid, addressWherePCLives, 1, &halfwords) !=  1)
            {
                DNBLogError("Could not read memory at %8.8x to get targetPC when processing the TBB instruction!", addressWherePCLives);
                return false;
            }
            // add 4 to currentPC since we are in Thumb mode and then add 2*halfwords
            *targetPC = (currentPC + 4) + 2*halfwords;
            break;

        case ARM_INST_TBH:
            mypid = m_thread->ProcessID();
            addressWherePCLives = ((baseAddress + (addressOffsetFromRegisterAfterShift << 1)) & ~0x1);
            if (DNBProcessMemoryRead(mypid, addressWherePCLives, 2, &halfwords) !=  2)
            {
                DNBLogError("Could not read memory at %8.8x to get targetPC when processing the TBH instruction!", addressWherePCLives);
                return false;
            }
            // add 4 to currentPC since we are in Thumb mode and then add 2*halfwords
            *targetPC = (currentPC + 4) + 2*halfwords;
            break;

            // ThumbEE branch-to-handler instructions: Jump to handlers at some offset
            //  from a special base pointer register (which is unknown at disassembly time)
        case ARM_INST_HB:
        case ARM_INST_HBP:
            //          TODO: ARM_INST_HB, ARM_INST_HBP
            break;

        case ARM_INST_HBL:
        case ARM_INST_HBLP:
            //          TODO: ARM_INST_HBL, ARM_INST_HBLP
            break;

            // Breakpoint and software interrupt jump to interrupt handler (always ARM)
        case ARM_INST_BKPT:
        case ARM_INST_SMC:
        case ARM_INST_SVC:
            //          TODO: ARM_INST_BKPT, ARM_INST_SMC, ARM_INST_SVC
            break;

            // Return from exception, obviously modifies PC [interrupt only!]
        case ARM_INST_RFEDA:
        case ARM_INST_RFEDB:
        case ARM_INST_RFEIA:
        case ARM_INST_RFEIB:
            //          TODO: ARM_INST_RFEDA, ARM_INST_RFEDB, ARM_INST_RFEIA, ARM_INST_RFEIB
            break;

            // Other instructions either can't change R15 or are "undefined" if you do,
            //  so no sane compiler should ever generate them & we don't care here.
            //  Also, R15 can only legally be used in a read-only manner for the
            //  various ARM addressing mode (to get PC-relative addressing of constants),
            //  but can NOT be used with any of the update modes.
        default:
            DNBLogError("%s should not be called for instruction code %d!", __FUNCTION__, decodedInstruction.instruction->code);
            return false;
            break;
    }

    return true;
}

void
DNBArchMachARM::EvaluateNextInstructionForSoftwareBreakpointSetup(nub_addr_t currentPC, uint32_t cpsr, bool currentPCIsThumb, nub_addr_t *nextPC, bool *nextPCIsThumb)
{
    DNBLogThreadedIf(LOG_STEP | LOG_VERBOSE, "DNBArchMachARM::EvaluateNextInstructionForSoftwareBreakpointSetup() called");

    nub_addr_t targetPC = INVALID_NUB_ADDRESS;
    uint32_t registerValue;
    arm_error_t decodeError;
    nub_addr_t currentPCInITBlock, nextPCInITBlock;
    int i;
    bool last_decoded_instruction_executes = true;

    DNBLogThreadedIf(LOG_STEP | LOG_VERBOSE, "%s: default nextPC=0x%8.8x (%s)", __FUNCTION__, *nextPC, *nextPCIsThumb ? "Thumb" : "ARM");

    // Update *nextPC and *nextPCIsThumb for special cases
    if (m_last_decode_thumb.itBlockRemaining) // we are in an IT block
    {
        // Set the nextPC to the PC of the instruction which will execute in the IT block
        // If none of the instruction execute in the IT block based on the condition flags,
        // then point to the instruction immediately following the IT block
        const int itBlockRemaining = m_last_decode_thumb.itBlockRemaining;
        DNBLogThreadedIf(LOG_STEP | LOG_VERBOSE, "%s: itBlockRemaining=%8.8x", __FUNCTION__, itBlockRemaining);

        // Determine the PC at which the next instruction resides
        if (m_last_decode_arm.thumb16b)
            currentPCInITBlock = currentPC + 2;
        else
            currentPCInITBlock = currentPC + 4;

        for (i = 0; i < itBlockRemaining; i++)
        {
            DNBLogThreadedIf(LOG_STEP | LOG_VERBOSE, "%s: currentPCInITBlock=%8.8x", __FUNCTION__, currentPCInITBlock);
            decodeError = DecodeInstructionUsingDisassembler(currentPCInITBlock, cpsr, &m_last_decode_arm, &m_last_decode_thumb, &nextPCInITBlock);

            if (decodeError != ARM_SUCCESS)
                DNBLogError("unable to disassemble instruction at 0x%8.8llx", (uint64_t)currentPCInITBlock);

            DNBLogThreadedIf(LOG_STEP | LOG_VERBOSE, "%s: condition=%d", __FUNCTION__, m_last_decode_arm.condition);
            if (ConditionPassed(m_last_decode_arm.condition, cpsr))
            {
                DNBLogThreadedIf(LOG_STEP | LOG_VERBOSE, "%s: Condition codes matched for instruction %d", __FUNCTION__, i);
                break; // break from the for loop
            }
            else
            {
                DNBLogThreadedIf(LOG_STEP | LOG_VERBOSE, "%s: Condition codes DID NOT matched for instruction %d", __FUNCTION__, i);
            }

            // update currentPC and nextPCInITBlock
            currentPCInITBlock = nextPCInITBlock;
        }

        if (i == itBlockRemaining) // We came out of the IT block without executing any instructions
            last_decoded_instruction_executes = false;

        *nextPC = currentPCInITBlock;
        DNBLogThreadedIf(LOG_STEP | LOG_VERBOSE, "%s: After IT block step-through: *nextPC=%8.8x", __FUNCTION__, *nextPC);
    }

    DNBLogThreadedIf(LOG_STEP | LOG_VERBOSE,
                    "%s: cpsr = %8.8x, thumb16b = %d, thumb = %d, branch = %d, conditional = %d, knownTarget = %d, links = %d, canSwitchMode = %d, doesSwitchMode = %d",
                    __FUNCTION__,
                    cpsr,
                    m_last_decode_arm.thumb16b,
                    m_last_decode_arm.thumb,
                    m_last_decode_arm.branch,
                    m_last_decode_arm.conditional,
                    m_last_decode_arm.knownTarget,
                    m_last_decode_arm.links,
                    m_last_decode_arm.canSwitchMode,
                    m_last_decode_arm.doesSwitchMode);


    if (last_decoded_instruction_executes &&                    // Was this a conditional instruction that did execute?
        m_last_decode_arm.branch &&                             // Can this instruction change the PC?
        (m_last_decode_arm.instruction->code != ARM_INST_SVC))  // If this instruction is not an SVC instruction
    {
        // Set targetPC. Compute if needed.
        if (m_last_decode_arm.knownTarget)
        {
            // Fixed, known PC-relative
            targetPC = m_last_decode_arm.targetPC;
        }
        else
        {
            // if targetPC is not known at compile time (PC-relative target), compute targetPC
            if (!ComputeNextPC(currentPC, m_last_decode_arm, currentPCIsThumb, &targetPC))
            {
                DNBLogError("%s: Unable to compute targetPC for instruction at 0x%8.8llx", __FUNCTION__, (uint64_t)currentPC);
                targetPC = INVALID_NUB_ADDRESS;
            }
        }

        DNBLogThreadedIf(LOG_STEP | LOG_VERBOSE, "%s: targetPC=0x%8.8x, cpsr=0x%8.8x, condition=0x%hhx", __FUNCTION__, targetPC, cpsr, m_last_decode_arm.condition);

        // Refine nextPC computation
        if ((m_last_decode_arm.instruction->code == ARM_INST_CBZ) ||
            (m_last_decode_arm.instruction->code == ARM_INST_CBNZ))
        {
            // Compare and branch on zero/non-zero (Thumb-16 only)
            // Unusual condition check built into the instruction
            registerValue = m_state.context.gpr.__r[m_last_decode_arm.op[REG_RD].value];

            if (m_last_decode_arm.instruction->code == ARM_INST_CBZ)
            {
                if (registerValue == 0)
                    *nextPC = targetPC;
            }
            else
            {
                if (registerValue != 0)
                    *nextPC = targetPC;
            }
        }
        else if (m_last_decode_arm.conditional) // Is the change conditional on flag results?
        {
            if (ConditionPassed(m_last_decode_arm.condition, cpsr)) // conditions match
            {
                DNBLogThreadedIf(LOG_STEP | LOG_VERBOSE, "%s: Condition matched!", __FUNCTION__);
                *nextPC = targetPC;
            }
            else
            {
                DNBLogThreadedIf(LOG_STEP | LOG_VERBOSE, "%s: Condition did not match!", __FUNCTION__);
            }
        }
        else
        {
            *nextPC = targetPC;
        }

        // Refine nextPCIsThumb computation
        if (m_last_decode_arm.doesSwitchMode)
        {
            *nextPCIsThumb = !currentPCIsThumb;
        }
        else if (m_last_decode_arm.canSwitchMode)
        {
            // Legal to switch ARM <--> Thumb mode with this branch
            // dependent on bit[0] of targetPC
            *nextPCIsThumb = (*nextPC & 1u) != 0;
        }
        else
        {
            *nextPCIsThumb = currentPCIsThumb;
        }
    }

    DNBLogThreadedIf(LOG_STEP, "%s: calculated nextPC=0x%8.8x (%s)", __FUNCTION__, *nextPC, *nextPCIsThumb ? "Thumb" : "ARM");
}


arm_error_t
DNBArchMachARM::DecodeInstructionUsingDisassembler(nub_addr_t curr_pc, uint32_t curr_cpsr, arm_decoded_instruction_t *decodedInstruction, thumb_static_data_t *thumbStaticData, nub_addr_t *next_pc)
{

    DNBLogThreadedIf(LOG_STEP | LOG_VERBOSE, "%s: pc=0x%8.8x, cpsr=0x%8.8x", __FUNCTION__, curr_pc, curr_cpsr);

    const uint32_t isetstate_mask = MASK_CPSR_T | MASK_CPSR_J;
    const uint32_t curr_isetstate = curr_cpsr & isetstate_mask;
    uint32_t opcode32;
    nub_addr_t nextPC = curr_pc;
    arm_error_t decodeReturnCode = ARM_SUCCESS;

    m_last_decode_pc = curr_pc;
    DNBLogThreadedIf(LOG_STEP | LOG_VERBOSE, "%s: last_decode_pc=0x%8.8x", __FUNCTION__, m_last_decode_pc);

    switch (curr_isetstate) {
        case 0x0: // ARM Instruction
            // Read the ARM opcode
            if (m_thread->Process()->Task().ReadMemory(curr_pc, 4, &opcode32) != 4)
            {
                DNBLogError("unable to read opcode bits 31:0 for an ARM opcode at 0x%8.8llx", (uint64_t)curr_pc);
                decodeReturnCode = ARM_ERROR;
            }
            else
            {
                nextPC += 4;
                decodeReturnCode = ArmDisassembler((uint64_t)curr_pc, opcode32, false, decodedInstruction, NULL, 0, NULL, 0);

                if (decodeReturnCode != ARM_SUCCESS)
                    DNBLogError("Unable to decode ARM instruction 0x%8.8x at 0x%8.8llx", opcode32, (uint64_t)curr_pc);
            }
            break;

        case 0x20: // Thumb Instruction
            uint16_t opcode16;
            // Read the a 16 bit Thumb opcode
            if (m_thread->Process()->Task().ReadMemory(curr_pc, 2, &opcode16) != 2)
            {
                DNBLogError("unable to read opcode bits 15:0 for a thumb opcode at 0x%8.8llx", (uint64_t)curr_pc);
                decodeReturnCode = ARM_ERROR;
            }
            else
            {
                nextPC += 2;
                opcode32 = opcode16;

                decodeReturnCode = ThumbDisassembler((uint64_t)curr_pc, opcode16, false, false, thumbStaticData, decodedInstruction, NULL, 0, NULL, 0);

                switch (decodeReturnCode) {
                    case ARM_SKIP:
                        // 32 bit thumb opcode
                        nextPC += 2;
                        if (m_thread->Process()->Task().ReadMemory(curr_pc+2, 2, &opcode16) != 2)
                        {
                            DNBLogError("unable to read opcode bits 15:0 for a thumb opcode at 0x%8.8llx", (uint64_t)curr_pc+2);
                        }
                        else
                        {
                            opcode32 = (opcode32 << 16) | opcode16;

                            decodeReturnCode = ThumbDisassembler((uint64_t)(curr_pc+2), opcode16, false, false, thumbStaticData, decodedInstruction, NULL, 0, NULL, 0);

                            if (decodeReturnCode != ARM_SUCCESS)
                                DNBLogError("Unable to decode 2nd half of Thumb instruction 0x%8.4hx at 0x%8.8llx", opcode16, (uint64_t)curr_pc+2);
                            break;
                        }
                        break;

                    case ARM_SUCCESS:
                        // 16 bit thumb opcode; at this point we are done decoding the opcode
                        break;

                    default:
                        DNBLogError("Unable to decode Thumb instruction 0x%8.4hx at 0x%8.8llx", opcode16, (uint64_t)curr_pc);
                        decodeReturnCode = ARM_ERROR;
                        break;
                }
            }
            break;

        default:
            break;
    }

    if (next_pc)
        *next_pc = nextPC;

    return decodeReturnCode;
}

#endif

nub_bool_t
DNBArchMachARM::BreakpointHit (nub_process_t pid, nub_thread_t tid, nub_break_t breakID, void *baton)
{
    nub_addr_t bkpt_pc = (nub_addr_t)baton;
    DNBLogThreadedIf(LOG_STEP | LOG_VERBOSE, "%s(pid = %i, tid = %4.4x, breakID = %u, baton = %p): Setting PC to 0x%8.8x", __FUNCTION__, pid, tid, breakID, baton, bkpt_pc);
    
    DNBRegisterValue pc_value;
    DNBThreadGetRegisterValueByID (pid, tid, REGISTER_SET_GENERIC, GENERIC_REGNUM_PC, &pc_value);
    pc_value.value.uint32 = bkpt_pc;
    return DNBThreadSetRegisterValueByID (pid, tid, REGISTER_SET_GENERIC, GENERIC_REGNUM_PC, &pc_value);
}

// Set the single step bit in the processor status register.
kern_return_t
DNBArchMachARM::SetSingleStepSoftwareBreakpoints()
{
    DNBError err;

#if defined (USE_ARM_DISASSEMBLER_FRAMEWORK)
    err = GetGPRState(false);

    if (err.Fail())
    {
        err.LogThreaded("%s: failed to read the GPR registers", __FUNCTION__);
        return err.Error();
    }

    nub_addr_t curr_pc = m_state.context.gpr.__pc;
    uint32_t curr_cpsr = m_state.context.gpr.__cpsr;
    nub_addr_t next_pc = curr_pc;

    bool curr_pc_is_thumb = (m_state.context.gpr.__cpsr & 0x20) != 0;
    bool next_pc_is_thumb = curr_pc_is_thumb;

    uint32_t curr_itstate = ((curr_cpsr & 0x6000000) >> 25) | ((curr_cpsr & 0xFC00) >> 8);
    bool inITBlock = (curr_itstate & 0xF) ? 1 : 0;
    bool lastInITBlock = ((curr_itstate & 0xF) == 0x8) ? 1 : 0;

    DNBLogThreadedIf(LOG_STEP | LOG_VERBOSE, "%s: curr_pc=0x%8.8x (%s), curr_itstate=0x%x, inITBlock=%d, lastInITBlock=%d", __FUNCTION__, curr_pc, curr_pc_is_thumb ? "Thumb" : "ARM", curr_itstate, inITBlock, lastInITBlock);

    // If the instruction is not in the IT block, then decode using the Disassembler and compute next_pc
    if (!inITBlock)
    {
        DNBLogThreadedIf(LOG_STEP | LOG_VERBOSE, "%s: Decoding an instruction NOT in the IT block", __FUNCTION__);

        arm_error_t decodeReturnCode =  DecodeInstructionUsingDisassembler(curr_pc, curr_cpsr, &m_last_decode_arm, &m_last_decode_thumb, &next_pc);

        if (decodeReturnCode != ARM_SUCCESS)
        {
            err = KERN_INVALID_ARGUMENT;
            DNBLogError("DNBArchMachARM::SetSingleStepSoftwareBreakpoints: Unable to disassemble instruction at 0x%8.8llx", (uint64_t)curr_pc);
        }
    }
    else
    {
        next_pc = curr_pc + ((m_last_decode_arm.thumb16b) ? 2 : 4);
    }

    // Instruction is NOT in the IT block OR
    if (!inITBlock)
    {
        DNBLogThreadedIf(LOG_STEP | LOG_VERBOSE, "%s: normal instruction", __FUNCTION__);
        EvaluateNextInstructionForSoftwareBreakpointSetup(curr_pc, m_state.context.gpr.__cpsr, curr_pc_is_thumb, &next_pc, &next_pc_is_thumb);
    }
    else if (inITBlock && !m_last_decode_arm.setsFlags)
    {
        DNBLogThreadedIf(LOG_STEP | LOG_VERBOSE, "%s: IT instruction that doesn't set flags", __FUNCTION__);
        EvaluateNextInstructionForSoftwareBreakpointSetup(curr_pc, m_state.context.gpr.__cpsr, curr_pc_is_thumb, &next_pc, &next_pc_is_thumb);
    }
    else if (lastInITBlock && m_last_decode_arm.branch)
    {
        DNBLogThreadedIf(LOG_STEP | LOG_VERBOSE, "%s: IT instruction which last in the IT block and is a branch", __FUNCTION__);
        EvaluateNextInstructionForSoftwareBreakpointSetup(curr_pc, m_state.context.gpr.__cpsr, curr_pc_is_thumb, &next_pc, &next_pc_is_thumb);
    }
    else
    {
        // Instruction is in IT block and can modify the CPSR flags
        DNBLogThreadedIf(LOG_STEP | LOG_VERBOSE, "%s: IT instruction that sets flags", __FUNCTION__);

        // NOTE: When this point of code is reached, the instruction at curr_pc has already been decoded
        // inside the function ThreadDidStop(). Therefore m_last_decode_arm, m_last_decode_thumb
        // reflect the decoded instruction at curr_pc

        // If we find an instruction inside the IT block which will set/modify the condition flags (NZCV bits in CPSR),
        // we set breakpoints at all remaining instructions inside the IT block starting from the instruction immediately
        // following this one AND a breakpoint at the instruction immediately following the IT block. We do this because
        // we cannot determine the next_pc until the instruction at which we are currently stopped executes. Hence we
        // insert (m_last_decode_thumb.itBlockRemaining+1) 16-bit Thumb breakpoints at consecutive memory locations
        // starting at addrOfNextInstructionInITBlock. We record these breakpoints in class variable m_sw_single_step_itblock_break_id[],
        // and also record the total number of IT breakpoints set in the variable 'm_sw_single_step_itblock_break_count'.

        // The instructions inside the IT block, which are replaced by the 16-bit Thumb breakpoints (opcode=0xDEFE)
        // instructions, can be either Thumb-16 or Thumb-32. When a Thumb-32 instruction (say, inst#1) is replaced  Thumb
        // by a 16-bit breakpoint (OS only supports 16-bit breakpoints in Thumb mode and 32-bit breakpoints in ARM mode), the
        // breakpoint for the next instruction (say instr#2) is saved in the upper half of this Thumb-32 (instr#1)
        // instruction. Hence if the execution stops at Breakpoint2 corresponding to instr#2, the PC is offset by 16-bits.
        // We therefore have to keep track of PC of each instruction in the IT block that is being replaced with the 16-bit
        // Thumb breakpoint, to ensure that when the breakpoint is hit, the PC is adjusted to the correct value. We save
        // the actual PC corresponding to each instruction in the IT block by associating a call back with each breakpoint
        // we set and passing it as a baton. When the breakpoint hits and the callback routine is called, the routine
        // adjusts the PC based on the baton that is passed to it.

        nub_addr_t addrOfNextInstructionInITBlock, pcInITBlock, nextPCInITBlock, bpAddressInITBlock;
        uint16_t opcode16;
        uint32_t opcode32;

        addrOfNextInstructionInITBlock = (m_last_decode_arm.thumb16b) ? curr_pc + 2 : curr_pc + 4;

        pcInITBlock = addrOfNextInstructionInITBlock;

        DNBLogThreadedIf(LOG_STEP | LOG_VERBOSE, "%s: itBlockRemaining=%d", __FUNCTION__, m_last_decode_thumb.itBlockRemaining);

        m_sw_single_step_itblock_break_count = 0;
        for (int i = 0; i <= m_last_decode_thumb.itBlockRemaining; i++)
        {
            if (NUB_BREAK_ID_IS_VALID(m_sw_single_step_itblock_break_id[i]))
            {
                DNBLogError("FunctionProfiler::SetSingleStepSoftwareBreakpoints(): Array m_sw_single_step_itblock_break_id should not contain any valid breakpoint IDs at this point. But found a valid breakID=%d at index=%d", m_sw_single_step_itblock_break_id[i], i);
            }
            else
            {
                nextPCInITBlock = pcInITBlock;
                // Compute nextPCInITBlock based on opcode present at pcInITBlock
                if (m_thread->Process()->Task().ReadMemory(pcInITBlock, 2, &opcode16) == 2)
                {
                    opcode32 = opcode16;
                    nextPCInITBlock += 2;

                    // Check for 32 bit thumb opcode and read the upper 16 bits if needed
                    if (((opcode32 & 0xE000) == 0xE000) && (opcode32 & 0x1800))
                    {
                        // Adjust 'next_pc_in_itblock' to point to the default next Thumb instruction for
                        // a 32 bit Thumb opcode
                        // Read bits 31:16 of a 32 bit Thumb opcode
                        if (m_thread->Process()->Task().ReadMemory(pcInITBlock+2, 2, &opcode16) == 2)
                        {
                            // 32 bit thumb opcode
                            opcode32 = (opcode32 << 16) | opcode16;
                            nextPCInITBlock += 2;
                        }
                        else
                        {
                            DNBLogError("FunctionProfiler::SetSingleStepSoftwareBreakpoints(): Unable to read opcode bits 31:16 for a 32 bit thumb opcode at pc=0x%8.8llx", (uint64_t)nextPCInITBlock);
                        }
                    }
                }
                else
                {
                    DNBLogError("FunctionProfiler::SetSingleStepSoftwareBreakpoints(): Error reading 16-bit Thumb instruction at pc=0x%8.8x", nextPCInITBlock);
                }


                // Set breakpoint and associate a callback function with it
                bpAddressInITBlock = addrOfNextInstructionInITBlock + 2*i;
                DNBLogThreadedIf(LOG_STEP | LOG_VERBOSE, "%s: Setting IT breakpoint[%d] at address: 0x%8.8x", __FUNCTION__, i, bpAddressInITBlock);

                m_sw_single_step_itblock_break_id[i] = m_thread->Process()->CreateBreakpoint(bpAddressInITBlock, 2, false, m_thread->ThreadID());
                if (!NUB_BREAK_ID_IS_VALID(m_sw_single_step_itblock_break_id[i]))
                    err = KERN_INVALID_ARGUMENT;
                else
                {
                    DNBLogThreadedIf(LOG_STEP, "%s: Set IT breakpoint[%i]=%d set at 0x%8.8x for instruction at 0x%8.8x", __FUNCTION__, i, m_sw_single_step_itblock_break_id[i], bpAddressInITBlock, pcInITBlock);

                    // Set the breakpoint callback for these special IT breakpoints
                    // so that if one of these breakpoints gets hit, it knows to
                    // update the PC to the original address of the conditional
                    // IT instruction.
                    DNBBreakpointSetCallback(m_thread->ProcessID(), m_sw_single_step_itblock_break_id[i], DNBArchMachARM::BreakpointHit, (void*)pcInITBlock);
                    m_sw_single_step_itblock_break_count++;
                }
            }

            pcInITBlock = nextPCInITBlock;
        }

        DNBLogThreadedIf(LOG_STEP | LOG_VERBOSE, "%s: Set %u IT software single breakpoints.", __FUNCTION__, m_sw_single_step_itblock_break_count);

    }

    DNBLogThreadedIf(LOG_STEP, "%s: next_pc=0x%8.8x (%s)", __FUNCTION__, next_pc, next_pc_is_thumb ? "Thumb" : "ARM");

    if (next_pc & 0x1)
    {
        assert(next_pc_is_thumb);
    }

    if (next_pc_is_thumb)
    {
        next_pc &= ~0x1;
    }
    else
    {
        assert((next_pc & 0x3) == 0);
    }

    if (!inITBlock || (inITBlock && !m_last_decode_arm.setsFlags) || (lastInITBlock && m_last_decode_arm.branch))
    {
        err = KERN_SUCCESS;

        const DNBBreakpoint *bp = m_thread->Process()->Breakpoints().FindByAddress(next_pc);

        if (bp == NULL)
        {
            m_sw_single_step_break_id = m_thread->Process()->CreateBreakpoint(next_pc, next_pc_is_thumb ? 2 : 4, false, m_thread->ThreadID());
            if (!NUB_BREAK_ID_IS_VALID(m_sw_single_step_break_id))
                err = KERN_INVALID_ARGUMENT;
            DNBLogThreadedIf(LOG_STEP, "%s: software single step breakpoint with breakID=%d set at 0x%8.8x", __FUNCTION__, m_sw_single_step_break_id, next_pc);
        }
    }
#else
    err.LogThreaded("%s: ARMDisassembler.framework support is disabled", __FUNCTION__);
#endif
    return err.Error();
}

uint32_t
DNBArchMachARM::NumSupportedHardwareBreakpoints()
{
    // Set the init value to something that will let us know that we need to
    // autodetect how many breakpoints are supported dynamically...
    static uint32_t g_num_supported_hw_breakpoints = UINT_MAX;
    if (g_num_supported_hw_breakpoints == UINT_MAX)
    {
        // Set this to zero in case we can't tell if there are any HW breakpoints
        g_num_supported_hw_breakpoints = 0;

        size_t len;
        uint32_t n = 0;
        len = sizeof (n);
        if (::sysctlbyname("hw.optional.breakpoint", &n, &len, NULL, 0) == 0)
        {
            g_num_supported_hw_breakpoints = n;
            DNBLogThreadedIf(LOG_THREAD, "hw.optional.breakpoint=%u", n);
        }
        else
        {
            // Read the DBGDIDR to get the number of available hardware breakpoints
            // However, in some of our current armv7 processors, hardware
            // breakpoints/watchpoints were not properly connected. So detect those
            // cases using a field in a sysctl. For now we are using "hw.cpusubtype"
            // field to distinguish CPU architectures. This is a hack until we can
            // get <rdar://problem/6372672> fixed, at which point we will switch to
            // using a different sysctl string that will tell us how many BRPs
            // are available to us directly without having to read DBGDIDR.
            uint32_t register_DBGDIDR;

            asm("mrc p14, 0, %0, c0, c0, 0" : "=r" (register_DBGDIDR));
            uint32_t numBRPs = bits(register_DBGDIDR, 27, 24);
            // Zero is reserved for the BRP count, so don't increment it if it is zero
            if (numBRPs > 0)
                numBRPs++;
            DNBLogThreadedIf(LOG_THREAD, "DBGDIDR=0x%8.8x (number BRP pairs = %u)", register_DBGDIDR, numBRPs);

            if (numBRPs > 0)
            {
                uint32_t cpusubtype;
                len = sizeof(cpusubtype);
                // TODO: remove this hack and change to using hw.optional.xx when implmented
                if (::sysctlbyname("hw.cpusubtype", &cpusubtype, &len, NULL, 0) == 0)
                {
                    DNBLogThreadedIf(LOG_THREAD, "hw.cpusubtype=%d", cpusubtype);
                    if (cpusubtype == CPU_SUBTYPE_ARM_V7)
                        DNBLogThreadedIf(LOG_THREAD, "Hardware breakpoints disabled for armv7 (rdar://problem/6372672)");
                    else
                        g_num_supported_hw_breakpoints = numBRPs;
                }
            }
        }
    }
    return g_num_supported_hw_breakpoints;
}


uint32_t
DNBArchMachARM::NumSupportedHardwareWatchpoints()
{
    // Set the init value to something that will let us know that we need to
    // autodetect how many watchpoints are supported dynamically...
    static uint32_t g_num_supported_hw_watchpoints = UINT_MAX;
    if (g_num_supported_hw_watchpoints == UINT_MAX)
    {
        // Set this to zero in case we can't tell if there are any HW breakpoints
        g_num_supported_hw_watchpoints = 0;
        
        
        size_t len;
        uint32_t n = 0;
        len = sizeof (n);
        if (::sysctlbyname("hw.optional.watchpoint", &n, &len, NULL, 0) == 0)
        {
            g_num_supported_hw_watchpoints = n;
            DNBLogThreadedIf(LOG_THREAD, "hw.optional.watchpoint=%u", n);
        }
        else
        {
            // Read the DBGDIDR to get the number of available hardware breakpoints
            // However, in some of our current armv7 processors, hardware
            // breakpoints/watchpoints were not properly connected. So detect those
            // cases using a field in a sysctl. For now we are using "hw.cpusubtype"
            // field to distinguish CPU architectures. This is a hack until we can
            // get <rdar://problem/6372672> fixed, at which point we will switch to
            // using a different sysctl string that will tell us how many WRPs
            // are available to us directly without having to read DBGDIDR.

            uint32_t register_DBGDIDR;
            asm("mrc p14, 0, %0, c0, c0, 0" : "=r" (register_DBGDIDR));
            uint32_t numWRPs = bits(register_DBGDIDR, 31, 28) + 1;
            DNBLogThreadedIf(LOG_THREAD, "DBGDIDR=0x%8.8x (number WRP pairs = %u)", register_DBGDIDR, numWRPs);

            if (numWRPs > 0)
            {
                uint32_t cpusubtype;
                size_t len;
                len = sizeof(cpusubtype);
                // TODO: remove this hack and change to using hw.optional.xx when implmented
                if (::sysctlbyname("hw.cpusubtype", &cpusubtype, &len, NULL, 0) == 0)
                {
                    DNBLogThreadedIf(LOG_THREAD, "hw.cpusubtype=0x%d", cpusubtype);

                    if (cpusubtype == CPU_SUBTYPE_ARM_V7)
                        DNBLogThreadedIf(LOG_THREAD, "Hardware watchpoints disabled for armv7 (rdar://problem/6372672)");
                    else
                        g_num_supported_hw_watchpoints = numWRPs;
                }
            }
        }
    }
    return g_num_supported_hw_watchpoints;
}


uint32_t
DNBArchMachARM::EnableHardwareBreakpoint (nub_addr_t addr, nub_size_t size)
{
    // Make sure our address isn't bogus
    if (addr & 1)
        return INVALID_NUB_HW_INDEX;

    kern_return_t kret = GetDBGState(false);

    if (kret == KERN_SUCCESS)
    {
        const uint32_t num_hw_breakpoints = NumSupportedHardwareBreakpoints();
        uint32_t i;
        for (i=0; i<num_hw_breakpoints; ++i)
        {
            if ((m_state.dbg.__bcr[i] & BCR_ENABLE) == 0)
                break; // We found an available hw breakpoint slot (in i)
        }

        // See if we found an available hw breakpoint slot above
        if (i < num_hw_breakpoints)
        {
            // Make sure bits 1:0 are clear in our address
            m_state.dbg.__bvr[i] = addr & ~((nub_addr_t)3);

            if (size == 2 || addr & 2)
            {
                uint32_t byte_addr_select = (addr & 2) ? BAS_IMVA_2_3 : BAS_IMVA_0_1;

                // We have a thumb breakpoint
                // We have an ARM breakpoint
                m_state.dbg.__bcr[i] =  BCR_M_IMVA_MATCH |  // Stop on address mismatch
                                        byte_addr_select |  // Set the correct byte address select so we only trigger on the correct opcode
                                        S_USER |            // Which modes should this breakpoint stop in?
                                        BCR_ENABLE;         // Enable this hardware breakpoint
                DNBLogThreadedIf (LOG_BREAKPOINTS, "DNBArchMachARM::EnableHardwareBreakpoint( addr = 0x%8.8llx, size = %llu ) - BVR%u/BCR%u = 0x%8.8x / 0x%8.8x (Thumb)",
                                  (uint64_t)addr,
                                  (uint64_t)size,
                                  i,
                                  i,
                                  m_state.dbg.__bvr[i],
                                  m_state.dbg.__bcr[i]);
            }
            else if (size == 4)
            {
                // We have an ARM breakpoint
                m_state.dbg.__bcr[i] =  BCR_M_IMVA_MATCH |  // Stop on address mismatch
                                        BAS_IMVA_ALL |      // Stop on any of the four bytes following the IMVA
                                        S_USER |            // Which modes should this breakpoint stop in?
                                        BCR_ENABLE;         // Enable this hardware breakpoint
                DNBLogThreadedIf (LOG_BREAKPOINTS, "DNBArchMachARM::EnableHardwareBreakpoint( addr = 0x%8.8llx, size = %llu ) - BVR%u/BCR%u = 0x%8.8x / 0x%8.8x (ARM)",
                                  (uint64_t)addr,
                                  (uint64_t)size,
                                  i,
                                  i,
                                  m_state.dbg.__bvr[i],
                                  m_state.dbg.__bcr[i]);
            }

            kret = SetDBGState();
            DNBLogThreadedIf(LOG_BREAKPOINTS, "DNBArchMachARM::EnableHardwareBreakpoint() SetDBGState() => 0x%8.8x.", kret);

            if (kret == KERN_SUCCESS)
                return i;
        }
        else
        {
            DNBLogThreadedIf (LOG_BREAKPOINTS, "DNBArchMachARM::EnableHardwareBreakpoint(addr = 0x%8.8llx, size = %llu) => all hardware breakpoint resources are being used.", (uint64_t)addr, (uint64_t)size);
        }
    }

    return INVALID_NUB_HW_INDEX;
}

bool
DNBArchMachARM::DisableHardwareBreakpoint (uint32_t hw_index)
{
    kern_return_t kret = GetDBGState(false);

    const uint32_t num_hw_points = NumSupportedHardwareBreakpoints();
    if (kret == KERN_SUCCESS)
    {
        if (hw_index < num_hw_points)
        {
            m_state.dbg.__bcr[hw_index] = 0;
            DNBLogThreadedIf(LOG_BREAKPOINTS, "DNBArchMachARM::SetHardwareBreakpoint( %u ) - BVR%u = 0x%8.8x  BCR%u = 0x%8.8x",
                    hw_index,
                    hw_index,
                    m_state.dbg.__bvr[hw_index],
                    hw_index,
                    m_state.dbg.__bcr[hw_index]);

            kret = SetDBGState();

            if (kret == KERN_SUCCESS)
                return true;
        }
    }
    return false;
}

// This stores the lo->hi mappings.  It's safe to initialize to all 0's
// since hi > lo and therefore LoHi[i] cannot be 0.
static uint32_t LoHi[16] = { 0 };

uint32_t
DNBArchMachARM::EnableHardwareWatchpoint (nub_addr_t addr, nub_size_t size, bool read, bool write)
{
    DNBLogThreadedIf(LOG_WATCHPOINTS, "DNBArchMachARM::EnableHardwareWatchpoint(addr = 0x%8.8llx, size = %llu, read = %u, write = %u)", (uint64_t)addr, (uint64_t)size, read, write);

    const uint32_t num_hw_watchpoints = NumSupportedHardwareWatchpoints();

    // Can't watch zero bytes
    if (size == 0)
        return INVALID_NUB_HW_INDEX;

    // We must watch for either read or write
    if (read == false && write == false)
        return INVALID_NUB_HW_INDEX;

    // Divide-and-conquer for size == 8.
    if (size == 8)
    {
        uint32_t lo = EnableHardwareWatchpoint(addr, 4, read, write);
        if (lo == INVALID_NUB_HW_INDEX)
            return INVALID_NUB_HW_INDEX;
        uint32_t hi = EnableHardwareWatchpoint(addr+4, 4, read, write);
        if (hi == INVALID_NUB_HW_INDEX)
        {
            DisableHardwareWatchpoint(lo);
            return INVALID_NUB_HW_INDEX;
        }
        // Tag this lo->hi mapping in our database.
        LoHi[lo] = hi;
        return lo;
    }

    // Otherwise, can't watch more than 4 bytes per WVR/WCR pair
    if (size > 4)
        return INVALID_NUB_HW_INDEX;

    // We can only watch up to four bytes that follow a 4 byte aligned address
    // per watchpoint register pair. Since we can only watch until the next 4
    // byte boundary, we need to make sure we can properly encode this.

    // addr_word_offset = addr % 4, i.e, is in set([0, 1, 2, 3])
    //
    //     +---+---+---+---+
    //     | 0 | 1 | 2 | 3 |
    //     +---+---+---+---+
    //     ^
    //     |
    // word address (4-byte aligned) = addr & 0xFFFFFFFC => goes into WVR
    //
    // examples:
    // 1. addr_word_offset = 1, size = 1 to watch a uint_8 => byte_mask = (0b0001 << 1) = 0b0010
    // 2. addr_word_offset = 2, size = 2 to watch a uint_16 => byte_mask = (0b0011 << 2) = 0b1100
    //
    // where byte_mask goes into WCR[8:5]

    uint32_t addr_word_offset = addr % 4;
    DNBLogThreadedIf(LOG_WATCHPOINTS, "DNBArchMachARM::EnableHardwareWatchpoint() - addr_word_offset = 0x%8.8x", addr_word_offset);

    uint32_t byte_mask = ((1u << size) - 1u) << addr_word_offset;
    DNBLogThreadedIf(LOG_WATCHPOINTS, "DNBArchMachARM::EnableHardwareWatchpoint() - byte_mask = 0x%8.8x", byte_mask);
    if (byte_mask > 0xfu)
        return INVALID_NUB_HW_INDEX;

    // Read the debug state
    kern_return_t kret = GetDBGState(false);

    if (kret == KERN_SUCCESS)
    {
        // Check to make sure we have the needed hardware support
        uint32_t i = 0;

        for (i=0; i<num_hw_watchpoints; ++i)
        {
            if ((m_state.dbg.__wcr[i] & WCR_ENABLE) == 0)
                break; // We found an available hw watchpoint slot (in i)
        }

        // See if we found an available hw watchpoint slot above
        if (i < num_hw_watchpoints)
        {
            //DumpDBGState(m_state.dbg);

            // Make the byte_mask into a valid Byte Address Select mask
            uint32_t byte_address_select = byte_mask << 5;
            // Make sure bits 1:0 are clear in our address
            m_state.dbg.__wvr[i] = addr & ~((nub_addr_t)3);     // DVA (Data Virtual Address)
            m_state.dbg.__wcr[i] =  byte_address_select |       // Which bytes that follow the DVA that we will watch
                                    S_USER |                    // Stop only in user mode
                                    (read ? WCR_LOAD : 0) |     // Stop on read access?
                                    (write ? WCR_STORE : 0) |   // Stop on write access?
                                    WCR_ENABLE;                 // Enable this watchpoint;

            kret = SetDBGState();
            //DumpDBGState(m_state.dbg);

            DNBLogThreadedIf(LOG_WATCHPOINTS, "DNBArchMachARM::EnableHardwareWatchpoint() SetDBGState() => 0x%8.8x.", kret);

            if (kret == KERN_SUCCESS)
                return i;
        }
        else
        {
            DNBLogThreadedIf(LOG_WATCHPOINTS, "DNBArchMachARM::EnableHardwareWatchpoint(): All hardware resources (%u) are in use.", num_hw_watchpoints);
        }
    }
    return INVALID_NUB_HW_INDEX;
}

bool
DNBArchMachARM::EnableHardwareWatchpoint0 (uint32_t hw_index, bool Delegate)
{
    kern_return_t kret = GetDBGState(false);
    if (kret != KERN_SUCCESS)
        return false;

    const uint32_t num_hw_points = NumSupportedHardwareWatchpoints();
    if (hw_index >= num_hw_points)
        return false;

    if (Delegate && LoHi[hw_index]) {
        // Enable lo and hi watchpoint hardware indexes.
        return EnableHardwareWatchpoint0(hw_index, false) &&
            EnableHardwareWatchpoint0(LoHi[hw_index], false);
    }

    m_state.dbg.__wcr[hw_index] |= (nub_addr_t)WCR_ENABLE;
    DNBLogThreadedIf(LOG_WATCHPOINTS, "DNBArchMachARM::EnableHardwareWatchpoint( %u ) - WVR%u = 0x%8.8x  WCR%u = 0x%8.8x",
                     hw_index,
                     hw_index,
                     m_state.dbg.__wvr[hw_index],
                     hw_index,
                     m_state.dbg.__wcr[hw_index]);

    kret = SetDBGState();

    return (kret == KERN_SUCCESS);
}

bool
DNBArchMachARM::DisableHardwareWatchpoint (uint32_t hw_index)
{
    return DisableHardwareWatchpoint0(hw_index, true);
}
bool
DNBArchMachARM::DisableHardwareWatchpoint0 (uint32_t hw_index, bool Delegate)
{
    kern_return_t kret = GetDBGState(false);
    if (kret != KERN_SUCCESS)
        return false;

    const uint32_t num_hw_points = NumSupportedHardwareWatchpoints();
    if (hw_index >= num_hw_points)
        return false;

    if (Delegate && LoHi[hw_index]) {
        // Disable lo and hi watchpoint hardware indexes.
        return DisableHardwareWatchpoint0(hw_index, false) &&
            DisableHardwareWatchpoint0(LoHi[hw_index], false);
    }

    m_state.dbg.__wcr[hw_index] &= ~((nub_addr_t)WCR_ENABLE);
    DNBLogThreadedIf(LOG_WATCHPOINTS, "DNBArchMachARM::DisableHardwareWatchpoint( %u ) - WVR%u = 0x%8.8x  WCR%u = 0x%8.8x",
                     hw_index,
                     hw_index,
                     m_state.dbg.__wvr[hw_index],
                     hw_index,
                     m_state.dbg.__wcr[hw_index]);

    kret = SetDBGState();

    return (kret == KERN_SUCCESS);
}

// {0} -> __bvr[16], {0} -> __bcr[16], {0} --> __wvr[16], {0} -> __wcr{16}
DNBArchMachARM::DBG DNBArchMachARM::Global_Debug_State = {{0},{0},{0},{0}};
bool DNBArchMachARM::Valid_Global_Debug_State = false;

// Use this callback from MachThread, which in turn was called from MachThreadList, to update
// the global view of the hardware watchpoint state, so that when new thread comes along, they
// get to inherit the existing hardware watchpoint state.
void
DNBArchMachARM::HardwareWatchpointStateChanged ()
{
    Global_Debug_State = m_state.dbg;
    Valid_Global_Debug_State = true;
}

// Returns -1 if the trailing bit patterns are not one of:
// { 0b???1, 0b??10, 0b?100, 0b1000 }.
static inline
int32_t
LowestBitSet(uint32_t val)
{
    for (unsigned i = 0; i < 4; ++i) {
        if (bit(val, i))
            return i;
    }
    return -1;
}

// Iterate through the debug registers; return the index of the first watchpoint whose address matches.
// As a side effect, the starting address as understood by the debugger is returned which could be
// different from 'addr' passed as an in/out argument.
uint32_t
DNBArchMachARM::GetHardwareWatchpointHit(nub_addr_t &addr)
{
    // Read the debug state
    kern_return_t kret = GetDBGState(true);
    //DumpDBGState(m_state.dbg);
    DNBLogThreadedIf(LOG_WATCHPOINTS, "DNBArchMachARM::GetHardwareWatchpointHit() GetDBGState() => 0x%8.8x.", kret);
    DNBLogThreadedIf(LOG_WATCHPOINTS, "DNBArchMachARM::GetHardwareWatchpointHit() addr = 0x%llx", (uint64_t)addr);

    // This is the watchpoint value to match against, i.e., word address.
    nub_addr_t wp_val = addr & ~((nub_addr_t)3);
    if (kret == KERN_SUCCESS)
    {
        DBG &debug_state = m_state.dbg;
        uint32_t i, num = NumSupportedHardwareWatchpoints();
        for (i = 0; i < num; ++i)
        {
            nub_addr_t wp_addr = GetWatchAddress(debug_state, i);
            DNBLogThreadedIf(LOG_WATCHPOINTS,
                             "DNBArchMachARM::GetHardwareWatchpointHit() slot: %u (addr = 0x%llx).",
                             i, (uint64_t)wp_addr);
            if (wp_val == wp_addr) {
                uint32_t byte_mask = bits(debug_state.__wcr[i], 8, 5);

                // Sanity check the byte_mask, first.
                if (LowestBitSet(byte_mask) < 0)
                    continue;

                // Compute the starting address (from the point of view of the debugger).
                addr = wp_addr + LowestBitSet(byte_mask);
                return i;
            }
        }
    }
    return INVALID_NUB_HW_INDEX;
}

// ThreadWillResume() calls this to clear bits[5:2] (Method of entry bits) of
// the Debug Status and Control Register (DSCR).
// 
// b0010 = a watchpoint occurred
// b0000 is the reset value
void
DNBArchMachARM::ClearWatchpointOccurred()
{
    uint32_t register_DBGDSCR;
    asm("mrc p14, 0, %0, c0, c1, 0" : "=r" (register_DBGDSCR));
    if (bits(register_DBGDSCR, 5, 2) == WATCHPOINT_OCCURRED)
    {
        uint32_t mask = ~(0xF << 2);
        register_DBGDSCR &= mask;
        asm("mcr p14, 0, %0, c0, c1, 0" : "=r" (register_DBGDSCR));
    }
    return;
}

// NotifyException() calls this to double check that a watchpoint has occurred
// by inspecting the bits[5:2] field of the Debug Status and Control Register
// (DSCR).
// 
// b0010 = a watchpoint occurred
bool
DNBArchMachARM::HasWatchpointOccurred()
{
    uint32_t register_DBGDSCR;
    asm("mrc p14, 0, %0, c0, c1, 0" : "=r" (register_DBGDSCR));
    return (bits(register_DBGDSCR, 5, 2) == WATCHPOINT_OCCURRED);
}

bool
DNBArchMachARM::IsWatchpointEnabled(const DBG &debug_state, uint32_t hw_index)
{
    // Watchpoint Control Registers, bitfield definitions
    // ...
    // Bits    Value    Description
    // [0]     0        Watchpoint disabled
    //         1        Watchpoint enabled.
    return (debug_state.__wcr[hw_index] & 1u);
}

nub_addr_t
DNBArchMachARM::GetWatchAddress(const DBG &debug_state, uint32_t hw_index)
{
    // Watchpoint Value Registers, bitfield definitions
    // Bits        Description
    // [31:2]      Watchpoint value (word address, i.e., 4-byte aligned)
    // [1:0]       RAZ/SBZP
    return bits(debug_state.__wvr[hw_index], 31, 0);
}

//----------------------------------------------------------------------
// Register information defintions for 32 bit ARMV6.
//----------------------------------------------------------------------
enum gpr_regnums
{
    gpr_r0 = 0,
    gpr_r1,
    gpr_r2,
    gpr_r3,
    gpr_r4,
    gpr_r5,
    gpr_r6,
    gpr_r7,
    gpr_r8,
    gpr_r9,
    gpr_r10,
    gpr_r11,
    gpr_r12,
    gpr_sp,
    gpr_lr,
    gpr_pc,
    gpr_cpsr
};

enum 
{
    vfp_s0 = 0,
    vfp_s1,
    vfp_s2,
    vfp_s3,
    vfp_s4,
    vfp_s5,
    vfp_s6,
    vfp_s7,
    vfp_s8,
    vfp_s9,
    vfp_s10,
    vfp_s11,
    vfp_s12,
    vfp_s13,
    vfp_s14,
    vfp_s15,
    vfp_s16,
    vfp_s17,
    vfp_s18,
    vfp_s19,
    vfp_s20,
    vfp_s21,
    vfp_s22,
    vfp_s23,
    vfp_s24,
    vfp_s25,
    vfp_s26,
    vfp_s27,
    vfp_s28,
    vfp_s29,
    vfp_s30,
    vfp_s31,
    vfp_d0,
    vfp_d1,
    vfp_d2,
    vfp_d3,
    vfp_d4,
    vfp_d5,
    vfp_d6,
    vfp_d7,
    vfp_d8,
    vfp_d9,
    vfp_d10,
    vfp_d11,
    vfp_d12,
    vfp_d13,
    vfp_d14,
    vfp_d15,
    vfp_d16,
    vfp_d17,
    vfp_d18,
    vfp_d19,
    vfp_d20,
    vfp_d21,
    vfp_d22,
    vfp_d23,
    vfp_d24,
    vfp_d25,
    vfp_d26,
    vfp_d27,
    vfp_d28,
    vfp_d29,
    vfp_d30,
    vfp_d31,
    vfp_fpscr
};

enum
{
    exc_exception,
	exc_fsr,
	exc_far,
};

enum
{
    gdb_r0 = 0,
    gdb_r1,
    gdb_r2,
    gdb_r3,
    gdb_r4,
    gdb_r5,
    gdb_r6,
    gdb_r7,
    gdb_r8,
    gdb_r9,
    gdb_r10,
    gdb_r11,
    gdb_r12,
    gdb_sp,
    gdb_lr,
    gdb_pc,
    gdb_f0,
    gdb_f1,
    gdb_f2,
    gdb_f3,
    gdb_f4,
    gdb_f5,
    gdb_f6,
    gdb_f7,
    gdb_f8,
    gdb_cpsr,
    gdb_s0,
    gdb_s1,
    gdb_s2,
    gdb_s3,
    gdb_s4,
    gdb_s5,
    gdb_s6,
    gdb_s7,
    gdb_s8,
    gdb_s9,
    gdb_s10,
    gdb_s11,
    gdb_s12,
    gdb_s13,
    gdb_s14,
    gdb_s15,
    gdb_s16,
    gdb_s17,
    gdb_s18,
    gdb_s19,
    gdb_s20,
    gdb_s21,
    gdb_s22,
    gdb_s23,
    gdb_s24,
    gdb_s25,
    gdb_s26,
    gdb_s27,
    gdb_s28,
    gdb_s29,
    gdb_s30,
    gdb_s31,
    gdb_fpscr,
    gdb_d0,
    gdb_d1,
    gdb_d2,
    gdb_d3,
    gdb_d4,
    gdb_d5,
    gdb_d6,
    gdb_d7,
    gdb_d8,
    gdb_d9,
    gdb_d10,
    gdb_d11,
    gdb_d12,
    gdb_d13,
    gdb_d14,
    gdb_d15
};

#define GPR_OFFSET_IDX(idx) (offsetof (DNBArchMachARM::GPR, __r[idx]))
#define GPR_OFFSET_NAME(reg) (offsetof (DNBArchMachARM::GPR, __##reg))
#define VFP_S_OFFSET_IDX(idx) (offsetof (DNBArchMachARM::FPU, __r[(idx)]) + offsetof (DNBArchMachARM::Context, vfp))
#define VFP_D_OFFSET_IDX(idx) (VFP_S_OFFSET_IDX ((idx) * 2))
#define VFP_OFFSET_NAME(reg) (offsetof (DNBArchMachARM::FPU, __##reg) + offsetof (DNBArchMachARM::Context, vfp))
#define EXC_OFFSET(reg)      (offsetof (DNBArchMachARM::EXC, __##reg)  + offsetof (DNBArchMachARM::Context, exc))

// These macros will auto define the register name, alt name, register size,
// register offset, encoding, format and native register. This ensures that
// the register state structures are defined correctly and have the correct
// sizes and offsets.
#define DEFINE_GPR_IDX(idx, reg, alt, gen) { e_regSetGPR, gpr_##reg, #reg, alt, Uint, Hex, 4, GPR_OFFSET_IDX(idx), gcc_##reg, dwarf_##reg, gen, gdb_##reg }
#define DEFINE_GPR_NAME(reg, alt, gen) { e_regSetGPR, gpr_##reg, #reg, alt, Uint, Hex, 4, GPR_OFFSET_NAME(reg), gcc_##reg, dwarf_##reg, gen, gdb_##reg }
//#define FLOAT_FORMAT Float
#define FLOAT_FORMAT Hex
#define DEFINE_VFP_S_IDX(idx) { e_regSetVFP, vfp_s##idx, "s" #idx, NULL, IEEE754, FLOAT_FORMAT, 4, VFP_S_OFFSET_IDX(idx), INVALID_NUB_REGNUM, dwarf_s##idx, INVALID_NUB_REGNUM, gdb_s##idx }
//#define DEFINE_VFP_D_IDX(idx) { e_regSetVFP, vfp_d##idx, "d" #idx, NULL, IEEE754, Float, 8, VFP_D_OFFSET_IDX(idx), INVALID_NUB_REGNUM, dwarf_d##idx, INVALID_NUB_REGNUM, gdb_d##idx }
#define DEFINE_VFP_D_IDX(idx) { e_regSetVFP, vfp_d##idx, "d" #idx, NULL, IEEE754, FLOAT_FORMAT, 8, VFP_D_OFFSET_IDX(idx), INVALID_NUB_REGNUM, dwarf_d##idx, INVALID_NUB_REGNUM, INVALID_NUB_REGNUM }

// General purpose registers
const DNBRegisterInfo
DNBArchMachARM::g_gpr_registers[] =
{
    DEFINE_GPR_IDX ( 0,  r0,"arg1", GENERIC_REGNUM_ARG1  ),
    DEFINE_GPR_IDX ( 1,  r1,"arg2", GENERIC_REGNUM_ARG2  ),
    DEFINE_GPR_IDX ( 2,  r2,"arg3", GENERIC_REGNUM_ARG3  ),
    DEFINE_GPR_IDX ( 3,  r3,"arg4", GENERIC_REGNUM_ARG4  ),
    DEFINE_GPR_IDX ( 4,  r4,  NULL, INVALID_NUB_REGNUM   ),
    DEFINE_GPR_IDX ( 5,  r5,  NULL, INVALID_NUB_REGNUM   ),
    DEFINE_GPR_IDX ( 6,  r6,  NULL, INVALID_NUB_REGNUM   ),
    DEFINE_GPR_IDX ( 7,  r7,  "fp", GENERIC_REGNUM_FP    ),
    DEFINE_GPR_IDX ( 8,  r8,  NULL, INVALID_NUB_REGNUM   ),
    DEFINE_GPR_IDX ( 9,  r9,  NULL, INVALID_NUB_REGNUM   ),
    DEFINE_GPR_IDX (10, r10,  NULL, INVALID_NUB_REGNUM   ),
    DEFINE_GPR_IDX (11, r11,  NULL, INVALID_NUB_REGNUM   ),
    DEFINE_GPR_IDX (12, r12,  NULL, INVALID_NUB_REGNUM   ),
    DEFINE_GPR_NAME (sp, "r13", GENERIC_REGNUM_SP    ),
    DEFINE_GPR_NAME (lr, "r14", GENERIC_REGNUM_RA    ),
    DEFINE_GPR_NAME (pc, "r15", GENERIC_REGNUM_PC    ),
    DEFINE_GPR_NAME (cpsr, "flags", GENERIC_REGNUM_FLAGS )
};

// Floating point registers
const DNBRegisterInfo
DNBArchMachARM::g_vfp_registers[] =
{
    DEFINE_VFP_S_IDX ( 0),
    DEFINE_VFP_S_IDX ( 1),
    DEFINE_VFP_S_IDX ( 2),
    DEFINE_VFP_S_IDX ( 3),
    DEFINE_VFP_S_IDX ( 4),
    DEFINE_VFP_S_IDX ( 5),
    DEFINE_VFP_S_IDX ( 6),
    DEFINE_VFP_S_IDX ( 7),
    DEFINE_VFP_S_IDX ( 8),
    DEFINE_VFP_S_IDX ( 9),
    DEFINE_VFP_S_IDX (10),
    DEFINE_VFP_S_IDX (11),
    DEFINE_VFP_S_IDX (12),
    DEFINE_VFP_S_IDX (13),
    DEFINE_VFP_S_IDX (14),
    DEFINE_VFP_S_IDX (15),
    DEFINE_VFP_S_IDX (16),
    DEFINE_VFP_S_IDX (17),
    DEFINE_VFP_S_IDX (18),
    DEFINE_VFP_S_IDX (19),
    DEFINE_VFP_S_IDX (20),
    DEFINE_VFP_S_IDX (21),
    DEFINE_VFP_S_IDX (22),
    DEFINE_VFP_S_IDX (23),
    DEFINE_VFP_S_IDX (24),
    DEFINE_VFP_S_IDX (25),
    DEFINE_VFP_S_IDX (26),
    DEFINE_VFP_S_IDX (27),
    DEFINE_VFP_S_IDX (28),
    DEFINE_VFP_S_IDX (29),
    DEFINE_VFP_S_IDX (30),
    DEFINE_VFP_S_IDX (31),
    DEFINE_VFP_D_IDX (0),
    DEFINE_VFP_D_IDX (1),
    DEFINE_VFP_D_IDX (2),
    DEFINE_VFP_D_IDX (3),
    DEFINE_VFP_D_IDX (4),
    DEFINE_VFP_D_IDX (5),
    DEFINE_VFP_D_IDX (6),
    DEFINE_VFP_D_IDX (7),
    DEFINE_VFP_D_IDX (8),
    DEFINE_VFP_D_IDX (9),
    DEFINE_VFP_D_IDX (10),
    DEFINE_VFP_D_IDX (11),
    DEFINE_VFP_D_IDX (12),
    DEFINE_VFP_D_IDX (13),
    DEFINE_VFP_D_IDX (14),
    DEFINE_VFP_D_IDX (15),
    DEFINE_VFP_D_IDX (16),
    DEFINE_VFP_D_IDX (17),
    DEFINE_VFP_D_IDX (18),
    DEFINE_VFP_D_IDX (19),
    DEFINE_VFP_D_IDX (20),
    DEFINE_VFP_D_IDX (21),
    DEFINE_VFP_D_IDX (22),
    DEFINE_VFP_D_IDX (23),
    DEFINE_VFP_D_IDX (24),
    DEFINE_VFP_D_IDX (25),
    DEFINE_VFP_D_IDX (26),
    DEFINE_VFP_D_IDX (27),
    DEFINE_VFP_D_IDX (28),
    DEFINE_VFP_D_IDX (29),
    DEFINE_VFP_D_IDX (30),
    DEFINE_VFP_D_IDX (31),
    { e_regSetVFP, vfp_fpscr, "fpscr", NULL, Uint, Hex, 4, VFP_OFFSET_NAME(fpscr), INVALID_NUB_REGNUM, INVALID_NUB_REGNUM, INVALID_NUB_REGNUM, gdb_fpscr }
};

// Exception registers

const DNBRegisterInfo
DNBArchMachARM::g_exc_registers[] =
{
  { e_regSetVFP, exc_exception  , "exception"   , NULL, Uint, Hex, 4, EXC_OFFSET(exception) , INVALID_NUB_REGNUM, INVALID_NUB_REGNUM, INVALID_NUB_REGNUM, INVALID_NUB_REGNUM },
  { e_regSetVFP, exc_fsr        , "fsr"         , NULL, Uint, Hex, 4, EXC_OFFSET(fsr)       , INVALID_NUB_REGNUM, INVALID_NUB_REGNUM, INVALID_NUB_REGNUM, INVALID_NUB_REGNUM },
  { e_regSetVFP, exc_far        , "far"         , NULL, Uint, Hex, 4, EXC_OFFSET(far)       , INVALID_NUB_REGNUM, INVALID_NUB_REGNUM, INVALID_NUB_REGNUM, INVALID_NUB_REGNUM }
};

// Number of registers in each register set
const size_t DNBArchMachARM::k_num_gpr_registers = sizeof(g_gpr_registers)/sizeof(DNBRegisterInfo);
const size_t DNBArchMachARM::k_num_vfp_registers = sizeof(g_vfp_registers)/sizeof(DNBRegisterInfo);
const size_t DNBArchMachARM::k_num_exc_registers = sizeof(g_exc_registers)/sizeof(DNBRegisterInfo);
const size_t DNBArchMachARM::k_num_all_registers = k_num_gpr_registers + k_num_vfp_registers + k_num_exc_registers;

//----------------------------------------------------------------------
// Register set definitions. The first definitions at register set index
// of zero is for all registers, followed by other registers sets. The
// register information for the all register set need not be filled in.
//----------------------------------------------------------------------
const DNBRegisterSetInfo
DNBArchMachARM::g_reg_sets[] =
{
    { "ARM Registers",              NULL,               k_num_all_registers     },
    { "General Purpose Registers",  g_gpr_registers,    k_num_gpr_registers     },
    { "Floating Point Registers",   g_vfp_registers,    k_num_vfp_registers     },
    { "Exception State Registers",  g_exc_registers,    k_num_exc_registers     }
};
// Total number of register sets for this architecture
const size_t DNBArchMachARM::k_num_register_sets = sizeof(g_reg_sets)/sizeof(DNBRegisterSetInfo);


const DNBRegisterSetInfo *
DNBArchMachARM::GetRegisterSetInfo(nub_size_t *num_reg_sets)
{
    *num_reg_sets = k_num_register_sets;
    return g_reg_sets;
}

bool
DNBArchMachARM::GetRegisterValue(int set, int reg, DNBRegisterValue *value)
{
    if (set == REGISTER_SET_GENERIC)
    {
        switch (reg)
        {
        case GENERIC_REGNUM_PC:     // Program Counter
            set = e_regSetGPR;
            reg = gpr_pc;
            break;

        case GENERIC_REGNUM_SP:     // Stack Pointer
            set = e_regSetGPR;
            reg = gpr_sp;
            break;

        case GENERIC_REGNUM_FP:     // Frame Pointer
            set = e_regSetGPR;
            reg = gpr_r7;   // is this the right reg?
            break;

        case GENERIC_REGNUM_RA:     // Return Address
            set = e_regSetGPR;
            reg = gpr_lr;
            break;

        case GENERIC_REGNUM_FLAGS:  // Processor flags register
            set = e_regSetGPR;
            reg = gpr_cpsr;
            break;

        default:
            return false;
        }
    }

    if (GetRegisterState(set, false) != KERN_SUCCESS)
        return false;

    const DNBRegisterInfo *regInfo = m_thread->GetRegisterInfo(set, reg);
    if (regInfo)
    {
        value->info = *regInfo;
        switch (set)
        {
        case e_regSetGPR:
            if (reg < k_num_gpr_registers)
            {
                value->value.uint32 = m_state.context.gpr.__r[reg];
                return true;
            }
            break;

        case e_regSetVFP:
            if (reg <= vfp_s31)
            {
                value->value.uint32 = m_state.context.vfp.__r[reg];
                return true;
            }
            else if (reg <= vfp_d31)
            {
                uint32_t d_reg_idx = reg - vfp_d0;
                uint32_t s_reg_idx = d_reg_idx * 2;
                value->value.v_sint32[0] = m_state.context.vfp.__r[s_reg_idx + 0];
                value->value.v_sint32[1] = m_state.context.vfp.__r[s_reg_idx + 1];
                return true;
            }
            else if (reg == vfp_fpscr)
            {
                value->value.uint32 = m_state.context.vfp.__fpscr;
                return true;
            }
            break;

        case e_regSetEXC:
            if (reg < k_num_exc_registers)
            {
                value->value.uint32 = (&m_state.context.exc.__exception)[reg];
                return true;
            }
            break;
        }
    }
    return false;
}

bool
DNBArchMachARM::SetRegisterValue(int set, int reg, const DNBRegisterValue *value)
{
    if (set == REGISTER_SET_GENERIC)
    {
        switch (reg)
        {
        case GENERIC_REGNUM_PC:     // Program Counter
            set = e_regSetGPR;
            reg = gpr_pc;
            break;

        case GENERIC_REGNUM_SP:     // Stack Pointer
            set = e_regSetGPR;
            reg = gpr_sp;
            break;

        case GENERIC_REGNUM_FP:     // Frame Pointer
            set = e_regSetGPR;
            reg = gpr_r7;
            break;

        case GENERIC_REGNUM_RA:     // Return Address
            set = e_regSetGPR;
            reg = gpr_lr;
            break;

        case GENERIC_REGNUM_FLAGS:  // Processor flags register
            set = e_regSetGPR;
            reg = gpr_cpsr;
            break;

        default:
            return false;
        }
    }

    if (GetRegisterState(set, false) != KERN_SUCCESS)
        return false;

    bool success = false;
    const DNBRegisterInfo *regInfo = m_thread->GetRegisterInfo(set, reg);
    if (regInfo)
    {
        switch (set)
        {
        case e_regSetGPR:
            if (reg < k_num_gpr_registers)
            {
                m_state.context.gpr.__r[reg] = value->value.uint32;
                success = true;
            }
            break;

        case e_regSetVFP:
            if (reg <= vfp_s31)
            {
                m_state.context.vfp.__r[reg] = value->value.uint32;
                success = true;
            }
            else if (reg <= vfp_d31)
            {
                uint32_t d_reg_idx = reg - vfp_d0;
                uint32_t s_reg_idx = d_reg_idx * 2;
                m_state.context.vfp.__r[s_reg_idx + 0] = value->value.v_sint32[0];
                m_state.context.vfp.__r[s_reg_idx + 1] = value->value.v_sint32[1];
                success = true;
            }
            else if (reg == vfp_fpscr)
            {
                m_state.context.vfp.__fpscr = value->value.uint32;
                success = true;
            }
            break;

        case e_regSetEXC:
            if (reg < k_num_exc_registers)
            {
                (&m_state.context.exc.__exception)[reg] = value->value.uint32;
                success = true;
            }
            break;
        }

    }
    if (success)
        return SetRegisterState(set) == KERN_SUCCESS;
    return false;
}

kern_return_t
DNBArchMachARM::GetRegisterState(int set, bool force)
{
    switch (set)
    {
    case e_regSetALL:   return GetGPRState(force) |
                               GetVFPState(force) |
                               GetEXCState(force) |
                               GetDBGState(force);
    case e_regSetGPR:   return GetGPRState(force);
    case e_regSetVFP:   return GetVFPState(force);
    case e_regSetEXC:   return GetEXCState(force);
    case e_regSetDBG:   return GetDBGState(force);
    default: break;
    }
    return KERN_INVALID_ARGUMENT;
}

kern_return_t
DNBArchMachARM::SetRegisterState(int set)
{
    // Make sure we have a valid context to set.
    kern_return_t err = GetRegisterState(set, false);
    if (err != KERN_SUCCESS)
        return err;

    switch (set)
    {
    case e_regSetALL:   return SetGPRState() |
                               SetVFPState() |
                               SetEXCState() |
                               SetDBGState();
    case e_regSetGPR:   return SetGPRState();
    case e_regSetVFP:   return SetVFPState();
    case e_regSetEXC:   return SetEXCState();
    case e_regSetDBG:   return SetDBGState();
    default: break;
    }
    return KERN_INVALID_ARGUMENT;
}

bool
DNBArchMachARM::RegisterSetStateIsValid (int set) const
{
    return m_state.RegsAreValid(set);
}


nub_size_t
DNBArchMachARM::GetRegisterContext (void *buf, nub_size_t buf_len)
{
    nub_size_t size = sizeof (m_state.context);
    
    if (buf && buf_len)
    {
        if (size > buf_len)
            size = buf_len;

        bool force = false;
        if (GetGPRState(force) | GetVFPState(force) | GetEXCState(force))
            return 0;
        ::memcpy (buf, &m_state.context, size);
    }
    DNBLogThreadedIf (LOG_THREAD, "DNBArchMachARM::GetRegisterContext (buf = %p, len = %llu) => %llu", buf, (uint64_t)buf_len, (uint64_t)size);
    // Return the size of the register context even if NULL was passed in
    return size;
}

nub_size_t
DNBArchMachARM::SetRegisterContext (const void *buf, nub_size_t buf_len)
{
    nub_size_t size = sizeof (m_state.context);
    if (buf == NULL || buf_len == 0)
        size = 0;
    
    if (size)
    {
        if (size > buf_len)
            size = buf_len;

        ::memcpy (&m_state.context, buf, size);
        SetGPRState();
        SetVFPState();
        SetEXCState();
    }
    DNBLogThreadedIf (LOG_THREAD, "DNBArchMachARM::SetRegisterContext (buf = %p, len = %llu) => %llu", buf, (uint64_t)buf_len, (uint64_t)size);
    return size;
}


#endif    // #if defined (__arm__)

