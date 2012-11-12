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

#if defined (__powerpc__) || defined (__ppc__) || defined (__ppc64__)

#if __DARWIN_UNIX03
#define PREFIX_DOUBLE_UNDERSCORE_DARWIN_UNIX03(reg) __##reg
#else
#define PREFIX_DOUBLE_UNDERSCORE_DARWIN_UNIX03(reg) reg
#endif

#include "MacOSX/ppc/DNBArchImpl.h"
#include "MacOSX/MachThread.h"
#include "DNBBreakpoint.h"
#include "DNBLog.h"
#include "DNBRegisterInfo.h"

static const uint8_t g_breakpoint_opcode[] = { 0x7F, 0xC0, 0x00, 0x08 };

const uint8_t * const
DNBArchMachPPC::SoftwareBreakpointOpcode (nub_size_t size)
{
    if (size == 4)
        return g_breakpoint_opcode;
    return NULL;
}

uint32_t
DNBArchMachPPC::GetCPUType()
{
    return CPU_TYPE_POWERPC;
}

uint64_t
DNBArchMachPPC::GetPC(uint64_t failValue)
{
    // Get program counter
    if (GetGPRState(false) == KERN_SUCCESS)
        return m_state.gpr.PREFIX_DOUBLE_UNDERSCORE_DARWIN_UNIX03(srr0);
    return failValue;
}

kern_return_t
DNBArchMachPPC::SetPC(uint64_t value)
{
    // Get program counter
    kern_return_t err = GetGPRState(false);
    if (err == KERN_SUCCESS)
    {
        m_state.gpr.PREFIX_DOUBLE_UNDERSCORE_DARWIN_UNIX03(srr0) = value;
        err = SetGPRState();
    }
    return err == KERN_SUCCESS;
}

uint64_t
DNBArchMachPPC::GetSP(uint64_t failValue)
{
    // Get stack pointer
    if (GetGPRState(false) == KERN_SUCCESS)
        return m_state.gpr.PREFIX_DOUBLE_UNDERSCORE_DARWIN_UNIX03(r1);
    return failValue;
}

kern_return_t
DNBArchMachPPC::GetGPRState(bool force)
{
    if (force || m_state.GetError(e_regSetGPR, Read))
    {
        mach_msg_type_number_t count = e_regSetWordSizeGPR;
        m_state.SetError(e_regSetGPR, Read, ::thread_get_state(m_thread->ThreadID(), e_regSetGPR, (thread_state_t)&m_state.gpr, &count));
    }
    return m_state.GetError(e_regSetGPR, Read);
}

kern_return_t
DNBArchMachPPC::GetFPRState(bool force)
{
    if (force || m_state.GetError(e_regSetFPR, Read))
    {
        mach_msg_type_number_t count = e_regSetWordSizeFPR;
        m_state.SetError(e_regSetFPR, Read, ::thread_get_state(m_thread->ThreadID(), e_regSetFPR, (thread_state_t)&m_state.fpr, &count));
    }
    return m_state.GetError(e_regSetFPR, Read);
}

kern_return_t
DNBArchMachPPC::GetEXCState(bool force)
{
    if (force || m_state.GetError(e_regSetEXC, Read))
    {
        mach_msg_type_number_t count = e_regSetWordSizeEXC;
        m_state.SetError(e_regSetEXC, Read, ::thread_get_state(m_thread->ThreadID(), e_regSetEXC, (thread_state_t)&m_state.exc, &count));
    }
    return m_state.GetError(e_regSetEXC, Read);
}

kern_return_t
DNBArchMachPPC::GetVECState(bool force)
{
    if (force || m_state.GetError(e_regSetVEC, Read))
    {
        mach_msg_type_number_t count = e_regSetWordSizeVEC;
        m_state.SetError(e_regSetVEC, Read, ::thread_get_state(m_thread->ThreadID(), e_regSetVEC, (thread_state_t)&m_state.vec, &count));
    }
    return m_state.GetError(e_regSetVEC, Read);
}

kern_return_t
DNBArchMachPPC::SetGPRState()
{
    m_state.SetError(e_regSetGPR, Write, ::thread_set_state(m_thread->ThreadID(), e_regSetGPR, (thread_state_t)&m_state.gpr, e_regSetWordSizeGPR));
    return m_state.GetError(e_regSetGPR, Write);
}

kern_return_t
DNBArchMachPPC::SetFPRState()
{
    m_state.SetError(e_regSetFPR, Write, ::thread_set_state(m_thread->ThreadID(), e_regSetFPR, (thread_state_t)&m_state.fpr, e_regSetWordSizeFPR));
    return m_state.GetError(e_regSetFPR, Write);
}

kern_return_t
DNBArchMachPPC::SetEXCState()
{
    m_state.SetError(e_regSetEXC, Write, ::thread_set_state(m_thread->ThreadID(), e_regSetEXC, (thread_state_t)&m_state.exc, e_regSetWordSizeEXC));
    return m_state.GetError(e_regSetEXC, Write);
}

kern_return_t
DNBArchMachPPC::SetVECState()
{
    m_state.SetError(e_regSetVEC, Write, ::thread_set_state(m_thread->ThreadID(), e_regSetVEC, (thread_state_t)&m_state.vec, e_regSetWordSizeVEC));
    return m_state.GetError(e_regSetVEC, Write);
}

bool
DNBArchMachPPC::ThreadWillResume()
{
    bool success = true;

    // Do we need to step this thread? If so, let the mach thread tell us so.
    if (m_thread->IsStepping())
    {
        // This is the primary thread, let the arch do anything it needs
        success = EnableHardwareSingleStep(true) == KERN_SUCCESS;
    }
    return success;
}

bool
DNBArchMachPPC::ThreadDidStop()
{
    bool success = true;

    m_state.InvalidateAllRegisterStates();

    // Are we stepping a single instruction?
    if (GetGPRState(true) == KERN_SUCCESS)
    {
        // We are single stepping, was this the primary thread?
        if (m_thread->IsStepping())
        {
            // This was the primary thread, we need to clear the trace
            // bit if so.
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


// Set the single step bit in the processor status register.
kern_return_t
DNBArchMachPPC::EnableHardwareSingleStep (bool enable)
{
    DNBLogThreadedIf(LOG_STEP, "DNBArchMachPPC::EnableHardwareSingleStep( enable = %d )", enable);
    if (GetGPRState(false) == KERN_SUCCESS)
    {
        const uint32_t trace_bit = 0x400;
        if (enable)
            m_state.gpr.PREFIX_DOUBLE_UNDERSCORE_DARWIN_UNIX03(srr1) |= trace_bit;
        else
            m_state.gpr.PREFIX_DOUBLE_UNDERSCORE_DARWIN_UNIX03(srr1) &= ~trace_bit;
        return SetGPRState();
    }
    return m_state.GetError(e_regSetGPR, Read);
}

//----------------------------------------------------------------------
// Register information defintions for 32 bit PowerPC.
//----------------------------------------------------------------------

enum gpr_regnums
{
    e_regNumGPR_srr0,
    e_regNumGPR_srr1,
    e_regNumGPR_r0,
    e_regNumGPR_r1,
    e_regNumGPR_r2,
    e_regNumGPR_r3,
    e_regNumGPR_r4,
    e_regNumGPR_r5,
    e_regNumGPR_r6,
    e_regNumGPR_r7,
    e_regNumGPR_r8,
    e_regNumGPR_r9,
    e_regNumGPR_r10,
    e_regNumGPR_r11,
    e_regNumGPR_r12,
    e_regNumGPR_r13,
    e_regNumGPR_r14,
    e_regNumGPR_r15,
    e_regNumGPR_r16,
    e_regNumGPR_r17,
    e_regNumGPR_r18,
    e_regNumGPR_r19,
    e_regNumGPR_r20,
    e_regNumGPR_r21,
    e_regNumGPR_r22,
    e_regNumGPR_r23,
    e_regNumGPR_r24,
    e_regNumGPR_r25,
    e_regNumGPR_r26,
    e_regNumGPR_r27,
    e_regNumGPR_r28,
    e_regNumGPR_r29,
    e_regNumGPR_r30,
    e_regNumGPR_r31,
    e_regNumGPR_cr,
    e_regNumGPR_xer,
    e_regNumGPR_lr,
    e_regNumGPR_ctr,
    e_regNumGPR_mq,
    e_regNumGPR_vrsave
};




// General purpose registers
static DNBRegisterInfo g_gpr_registers[] =
{
  { "srr0"  , Uint, 4, Hex },
  { "srr1"  , Uint, 4, Hex },
  { "r0"    , Uint, 4, Hex },
  { "r1"    , Uint, 4, Hex },
  { "r2"    , Uint, 4, Hex },
  { "r3"    , Uint, 4, Hex },
  { "r4"    , Uint, 4, Hex },
  { "r5"    , Uint, 4, Hex },
  { "r6"    , Uint, 4, Hex },
  { "r7"    , Uint, 4, Hex },
  { "r8"    , Uint, 4, Hex },
  { "r9"    , Uint, 4, Hex },
  { "r10"   , Uint, 4, Hex },
  { "r11"   , Uint, 4, Hex },
  { "r12"   , Uint, 4, Hex },
  { "r13"   , Uint, 4, Hex },
  { "r14"   , Uint, 4, Hex },
  { "r15"   , Uint, 4, Hex },
  { "r16"   , Uint, 4, Hex },
  { "r17"   , Uint, 4, Hex },
  { "r18"   , Uint, 4, Hex },
  { "r19"   , Uint, 4, Hex },
  { "r20"   , Uint, 4, Hex },
  { "r21"   , Uint, 4, Hex },
  { "r22"   , Uint, 4, Hex },
  { "r23"   , Uint, 4, Hex },
  { "r24"   , Uint, 4, Hex },
  { "r25"   , Uint, 4, Hex },
  { "r26"   , Uint, 4, Hex },
  { "r27"   , Uint, 4, Hex },
  { "r28"   , Uint, 4, Hex },
  { "r29"   , Uint, 4, Hex },
  { "r30"   , Uint, 4, Hex },
  { "r31"   , Uint, 4, Hex },
  { "cr"    , Uint, 4, Hex },
  { "xer"   , Uint, 4, Hex },
  { "lr"    , Uint, 4, Hex },
  { "ctr"   , Uint, 4, Hex },
  { "mq"    , Uint, 4, Hex },
  { "vrsave", Uint, 4, Hex },
};

// Floating point registers
static DNBRegisterInfo g_fpr_registers[] =
{
  { "fp0"   , IEEE754, 8, Float },
  { "fp1"   , IEEE754, 8, Float },
  { "fp2"   , IEEE754, 8, Float },
  { "fp3"   , IEEE754, 8, Float },
  { "fp4"   , IEEE754, 8, Float },
  { "fp5"   , IEEE754, 8, Float },
  { "fp6"   , IEEE754, 8, Float },
  { "fp7"   , IEEE754, 8, Float },
  { "fp8"   , IEEE754, 8, Float },
  { "fp9"   , IEEE754, 8, Float },
  { "fp10"  , IEEE754, 8, Float },
  { "fp11"  , IEEE754, 8, Float },
  { "fp12"  , IEEE754, 8, Float },
  { "fp13"  , IEEE754, 8, Float },
  { "fp14"  , IEEE754, 8, Float },
  { "fp15"  , IEEE754, 8, Float },
  { "fp16"  , IEEE754, 8, Float },
  { "fp17"  , IEEE754, 8, Float },
  { "fp18"  , IEEE754, 8, Float },
  { "fp19"  , IEEE754, 8, Float },
  { "fp20"  , IEEE754, 8, Float },
  { "fp21"  , IEEE754, 8, Float },
  { "fp22"  , IEEE754, 8, Float },
  { "fp23"  , IEEE754, 8, Float },
  { "fp24"  , IEEE754, 8, Float },
  { "fp25"  , IEEE754, 8, Float },
  { "fp26"  , IEEE754, 8, Float },
  { "fp27"  , IEEE754, 8, Float },
  { "fp28"  , IEEE754, 8, Float },
  { "fp29"  , IEEE754, 8, Float },
  { "fp30"  , IEEE754, 8, Float },
  { "fp31"  , IEEE754, 8, Float },
  { "fpscr" , Uint, 4, Hex }
};

// Exception registers

static DNBRegisterInfo g_exc_registers[] =
{
  { "dar"       , Uint, 4, Hex },
  { "dsisr"     , Uint, 4, Hex },
  { "exception" , Uint, 4, Hex }
};

// Altivec registers
static DNBRegisterInfo g_vec_registers[] =
{
  { "vr0"   , Vector, 16, VectorOfFloat32 },
  { "vr1"   , Vector, 16, VectorOfFloat32 },
  { "vr2"   , Vector, 16, VectorOfFloat32 },
  { "vr3"   , Vector, 16, VectorOfFloat32 },
  { "vr4"   , Vector, 16, VectorOfFloat32 },
  { "vr5"   , Vector, 16, VectorOfFloat32 },
  { "vr6"   , Vector, 16, VectorOfFloat32 },
  { "vr7"   , Vector, 16, VectorOfFloat32 },
  { "vr8"   , Vector, 16, VectorOfFloat32 },
  { "vr9"   , Vector, 16, VectorOfFloat32 },
  { "vr10"  , Vector, 16, VectorOfFloat32 },
  { "vr11"  , Vector, 16, VectorOfFloat32 },
  { "vr12"  , Vector, 16, VectorOfFloat32 },
  { "vr13"  , Vector, 16, VectorOfFloat32 },
  { "vr14"  , Vector, 16, VectorOfFloat32 },
  { "vr15"  , Vector, 16, VectorOfFloat32 },
  { "vr16"  , Vector, 16, VectorOfFloat32 },
  { "vr17"  , Vector, 16, VectorOfFloat32 },
  { "vr18"  , Vector, 16, VectorOfFloat32 },
  { "vr19"  , Vector, 16, VectorOfFloat32 },
  { "vr20"  , Vector, 16, VectorOfFloat32 },
  { "vr21"  , Vector, 16, VectorOfFloat32 },
  { "vr22"  , Vector, 16, VectorOfFloat32 },
  { "vr23"  , Vector, 16, VectorOfFloat32 },
  { "vr24"  , Vector, 16, VectorOfFloat32 },
  { "vr25"  , Vector, 16, VectorOfFloat32 },
  { "vr26"  , Vector, 16, VectorOfFloat32 },
  { "vr27"  , Vector, 16, VectorOfFloat32 },
  { "vr28"  , Vector, 16, VectorOfFloat32 },
  { "vr29"  , Vector, 16, VectorOfFloat32 },
  { "vr30"  , Vector, 16, VectorOfFloat32 },
  { "vr31"  , Vector, 16, VectorOfFloat32 },
  { "vscr"  , Uint, 16, Hex },
  { "vrvalid" , Uint, 4, Hex }
};

// Number of registers in each register set
const size_t k_num_gpr_registers = sizeof(g_gpr_registers)/sizeof(DNBRegisterInfo);
const size_t k_num_fpr_registers = sizeof(g_fpr_registers)/sizeof(DNBRegisterInfo);
const size_t k_num_exc_registers = sizeof(g_exc_registers)/sizeof(DNBRegisterInfo);
const size_t k_num_vec_registers = sizeof(g_vec_registers)/sizeof(DNBRegisterInfo);
// Total number of registers for this architecture
const size_t k_num_ppc_registers = k_num_gpr_registers + k_num_fpr_registers + k_num_exc_registers + k_num_vec_registers;

//----------------------------------------------------------------------
// Register set definitions. The first definitions at register set index
// of zero is for all registers, followed by other registers sets. The
// register information for the all register set need not be filled in.
//----------------------------------------------------------------------
static const DNBRegisterSetInfo g_reg_sets[] =
{
    { "PowerPC Registers",            NULL,             k_num_ppc_registers },
    { "General Purpose Registers",    g_gpr_registers, k_num_gpr_registers },
    { "Floating Point Registers",    g_fpr_registers, k_num_fpr_registers },
    { "Exception State Registers",    g_exc_registers, k_num_exc_registers },
    { "Altivec Registers",            g_vec_registers, k_num_vec_registers }
};
// Total number of register sets for this architecture
const size_t k_num_register_sets = sizeof(g_reg_sets)/sizeof(DNBRegisterSetInfo);


const DNBRegisterSetInfo *
DNBArchMachPPC::GetRegisterSetInfo(nub_size_t *num_reg_sets) const
{
    *num_reg_sets = k_num_register_sets;
    return g_reg_sets;
}

bool
DNBArchMachPPC::GetRegisterValue(int set, int reg, DNBRegisterValue *value) const
{
    if (set == REGISTER_SET_GENERIC)
    {
        switch (reg)
        {
        case GENERIC_REGNUM_PC:     // Program Counter
            set = e_regSetGPR;
            reg = e_regNumGPR_srr0;
            break;

        case GENERIC_REGNUM_SP:     // Stack Pointer
            set = e_regSetGPR;
            reg = e_regNumGPR_r1;
            break;

        case GENERIC_REGNUM_FP:     // Frame Pointer
            // Return false for now instead of returning r30 as gcc 3.x would
            // use a variety of registers for the FP and it takes inspecting
            // the stack to make sure there is a frame pointer before we can
            // determine the FP.
            return false;

        case GENERIC_REGNUM_RA:     // Return Address
            set = e_regSetGPR;
            reg = e_regNumGPR_lr;
            break;

        case GENERIC_REGNUM_FLAGS:  // Processor flags register
            set = e_regSetGPR;
            reg = e_regNumGPR_srr1;
            break;

        default:
            return false;
        }
    }

    if (!m_state.RegsAreValid(set))
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
                value->value.uint32 = (&m_state.gpr.PREFIX_DOUBLE_UNDERSCORE_DARWIN_UNIX03(srr0))[reg];
                return true;
            }
            break;

        case e_regSetFPR:
            if (reg < 32)
            {
                value->value.float64 = m_state.fpr.PREFIX_DOUBLE_UNDERSCORE_DARWIN_UNIX03(fpregs)[reg];
                return true;
            }
            else if (reg == 32)
            {
                value->value.uint32 = m_state.fpr.PREFIX_DOUBLE_UNDERSCORE_DARWIN_UNIX03(fpscr);
                return true;
            }
            break;

        case e_regSetEXC:
            if (reg < k_num_exc_registers)
            {
                value->value.uint32 = (&m_state.exc.PREFIX_DOUBLE_UNDERSCORE_DARWIN_UNIX03(dar))[reg];
                return true;
            }
            break;

        case e_regSetVEC:
            if (reg < k_num_vec_registers)
            {
                if (reg < 33)            // FP0 - FP31 and VSCR
                {
                    // Copy all 4 uint32 values for this vector register
                    value->value.v_uint32[0] = m_state.vec.PREFIX_DOUBLE_UNDERSCORE_DARWIN_UNIX03(save_vr)[reg][0];
                    value->value.v_uint32[1] = m_state.vec.PREFIX_DOUBLE_UNDERSCORE_DARWIN_UNIX03(save_vr)[reg][1];
                    value->value.v_uint32[2] = m_state.vec.PREFIX_DOUBLE_UNDERSCORE_DARWIN_UNIX03(save_vr)[reg][2];
                    value->value.v_uint32[3] = m_state.vec.PREFIX_DOUBLE_UNDERSCORE_DARWIN_UNIX03(save_vr)[reg][3];
                    return true;
                }
                else if (reg == 34)    // VRVALID
                {
                    value->value.uint32 = m_state.vec.PREFIX_DOUBLE_UNDERSCORE_DARWIN_UNIX03(save_vrvalid);
                    return true;
                }
            }
            break;
        }
    }
    return false;
}


kern_return_t
DNBArchMachPPC::GetRegisterState(int set, bool force)
{
    switch (set)
    {
    case e_regSetALL:
        return  GetGPRState(force) |
                GetFPRState(force) |
                GetEXCState(force) |
                GetVECState(force);
    case e_regSetGPR:    return GetGPRState(force);
    case e_regSetFPR:    return GetFPRState(force);
    case e_regSetEXC:    return GetEXCState(force);
    case e_regSetVEC:    return GetVECState(force);
    default: break;
    }
    return KERN_INVALID_ARGUMENT;
}

kern_return_t
DNBArchMachPPC::SetRegisterState(int set)
{
    // Make sure we have a valid context to set.
    kern_return_t err = GetRegisterState(set, false);
    if (err != KERN_SUCCESS)
        return err;

    switch (set)
    {
    case e_regSetALL:    return SetGPRState() | SetFPRState() | SetEXCState() | SetVECState();
    case e_regSetGPR:    return SetGPRState();
    case e_regSetFPR:    return SetFPRState();
    case e_regSetEXC:    return SetEXCState();
    case e_regSetVEC:    return SetVECState();
    default: break;
    }
    return KERN_INVALID_ARGUMENT;
}

bool
DNBArchMachPPC::RegisterSetStateIsValid (int set) const
{
    return m_state.RegsAreValid(set);
}


#endif    // #if defined (__powerpc__) || defined (__ppc__) || defined (__ppc64__)

