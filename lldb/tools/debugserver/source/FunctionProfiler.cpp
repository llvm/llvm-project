//===-- FunctionProfiler.cpp ------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  Created by Greg Clayton on 10/8/08.
//
//===----------------------------------------------------------------------===//

#include "FunctionProfiler.h"

// C Includes
// C++ Includes
// Other libraries and framework includes
#include "DNB.h"

// Project includes

//----------------------------------------------------------------------
// FunctionProfiler constructor
//----------------------------------------------------------------------
FunctionProfiler::FunctionProfiler(nub_addr_t start_addr, nub_addr_t stop_addr) :
    m_pid(INVALID_NUB_PROCESS),
    m_start_addr(start_addr),
    m_stop_addr(stop_addr),
    m_start_break_id(INVALID_NUB_BREAK_ID),
    m_stop_break_id(INVALID_NUB_BREAK_ID),
    m_func_entered_count(0),
    m_last_pc(0),
    m_last_flags(0),
    m_consecutive_opcode_count(0),
    m_total_opcode_count(0)
{
}


FunctionProfiler::~FunctionProfiler()
{
    Clear();
}


void
FunctionProfiler::Clear()
{
    if (m_pid != INVALID_NUB_PROCESS)
    {
        if (m_start_break_id != INVALID_NUB_BREAK_ID)
            DNBBreakpointClear(m_pid, m_start_break_id);
        if (m_stop_break_id != INVALID_NUB_BREAK_ID)
            DNBBreakpointClear(m_pid, m_stop_break_id);
    }
    m_start_break_id = INVALID_NUB_BREAK_ID;
    m_stop_break_id = INVALID_NUB_BREAK_ID;
    m_func_entered_count = 0;
    m_last_pc = 0;
    m_last_flags = 0;
    m_consecutive_opcode_count = 0;
}

void
FunctionProfiler::Initialize(nub_process_t pid)
{
    //printf("FunctionProfiler::%s(0x%4.4x)\n", __FUNCTION__, pid);
    Clear();
    m_pid = pid;
}

#include "DNBDataRef.h"

void
FunctionProfiler::SetBreakpoints()
{
#if defined (__i386__)
    nub_size_t bp_opcode_size = 1;
#elif defined (__powerpc__) || defined (__ppc__)
    nub_size_t bp_opcode_size = 4;
#endif
    if (m_start_addr != INVALID_NUB_ADDRESS && !NUB_BREAK_ID_IS_VALID(m_start_break_id))
    {
#if defined (__arm__)
        m_start_break_id = DNBBreakpointSet(m_pid, m_start_addr & 0xFFFFFFFEu, m_start_addr & 1 ? 2 : 4, false);
#else
        m_start_break_id = DNBBreakpointSet(m_pid, m_start_addr, bp_opcode_size, false);
#endif
        if (NUB_BREAK_ID_IS_VALID(m_start_break_id))
            DNBBreakpointSetCallback(m_pid, m_start_break_id, FunctionProfiler::BreakpointHitCallback, this);
    }
    if (m_stop_addr != INVALID_NUB_ADDRESS && !NUB_BREAK_ID_IS_VALID(m_stop_break_id))
    {
#if defined (__arm__)
        m_stop_break_id = DNBBreakpointSet(m_pid, m_stop_addr & 0xFFFFFFFEu, m_stop_addr & 1 ? 2 : 4, false);
#else
        m_stop_break_id = DNBBreakpointSet(m_pid, m_stop_addr, bp_opcode_size, false);
#endif
        if (NUB_BREAK_ID_IS_VALID(m_stop_break_id))
            DNBBreakpointSetCallback(m_pid, m_stop_break_id, FunctionProfiler::BreakpointHitCallback, this);
    }
}

nub_bool_t
FunctionProfiler::BreakpointHitCallback(nub_process_t pid, nub_thread_t tid, nub_break_t breakID, void *baton)
{
    printf("FunctionProfiler::%s(pid = %4.4x, tid = %4.4x, breakID = %u, baton = %p)\n", __FUNCTION__, pid, tid, breakID, baton);
    return ((FunctionProfiler*) baton)->BreakpointHit(pid, tid, breakID);
}

nub_bool_t
FunctionProfiler::BreakpointHit(nub_process_t pid, nub_thread_t tid, nub_break_t breakID)
{
    printf("FunctionProfiler::%s(pid = %4.4x, tid = %4.4x, breakID = %u)\n", __FUNCTION__, pid, tid, breakID);
    if (breakID == m_start_break_id)
    {
        m_func_entered_count++;
        printf("FunctionProfiler::%s(pid = %4.4x, tid = %4.4x, breakID = %u) START breakpoint hit (%u)\n", __FUNCTION__, pid, tid, breakID, m_func_entered_count);
    }
    else if (breakID == m_stop_break_id)
    {
        if (m_func_entered_count > 0)
            m_func_entered_count--;
        printf("FunctionProfiler::%s(pid = %4.4x, tid = %4.4x, breakID = %u) STOP breakpoint hit (%u)\n", __FUNCTION__, pid, tid, breakID, m_func_entered_count);
    }
    return true;
}

void
FunctionProfiler::ProcessStateChanged(nub_state_t state)
{
//    printf("FunctionProfiler::%s(%s)\n", __FUNCTION__, DNBStateAsString(state));

    switch (state)
    {
    case eStateInvalid:
    case eStateUnloaded:
    case eStateAttaching:
    case eStateLaunching:
        break;

    case eStateDetached:
    case eStateExited:
        // No sense is clearing out breakpoints if our process has exited...
        m_start_break_id = INVALID_NUB_BREAK_ID;
        m_stop_break_id = INVALID_NUB_BREAK_ID;
        printf("[0x%8.8x - 0x%8.8x) executed %u total opcodes.\n", m_total_opcode_count);
        break;

    case eStateStopped:
        // Keep trying find dyld each time we stop until we do
        if (!NUB_BREAK_ID_IS_VALID(m_start_break_id))
            SetBreakpoints();

        if (ShouldStepProcess())
        {

            // TODO: do logging/tracing here
            nub_thread_t tid = DNBProcessGetCurrentThread(m_pid);
            DNBRegisterValue reg;
            m_total_opcode_count++;

            if (DNBThreadGetRegisterValueByID(m_pid, tid, REGISTER_SET_GENERIC, GENERIC_REGNUM_PC, &reg))
            {
                const nub_addr_t pc = reg.value.uint32;

#if defined (__i386__)
                uint8_t buf[16];
                uint32_t bytes_read = DNBProcessMemoryRead(m_pid, pc, 1, buf);
                if (bytes_read == 1)
                    printf("0x%8.8x: %2.2x\n", pc, buf[0]);
                else
                    printf("0x%8.8x: error: can't read opcode byte.\n", pc);

//              if (bytes_read > 0)
//              {
//                  for (uint32_t i=0; i<bytes_read; ++i)
//                  {
//                      printf(" %2.2x", buf[i]);
//                  }
//              }
//              printf("\n");

#elif defined (__powerpc__) || defined (__ppc__)

                uint32_t opcode = 0;
                if (DNBProcessMemoryRead(m_pid, pc, 4, &opcode) == 4)
                {
                    printf("0x%8.8x: 0x%8.8x\n", pc, opcode);
                }

#elif defined (__arm__)
                #define CPSR_T (1u << 5)
                // Read the CPSR into flags
                if (DNBThreadGetRegisterValueByID(m_pid, tid, REGISTER_SET_GENERIC, GENERIC_REGNUM_FLAGS, &reg))
                {
                    const uint32_t flags = reg.value.uint32;

                    const bool curr_pc_is_thumb = (flags & CPSR_T) != 0; // check the CPSR T bit
                    const bool last_pc_was_thumb = (m_last_flags & CPSR_T) != 0; // check the CPSR T bit
                    bool opcode_is_sequential = false;

                    uint32_t opcode;
                    // Always read four bytes for the opcode
                    if (DNBProcessMemoryRead(m_pid, pc, 4, &opcode) == 4)
                    {
                        if (curr_pc_is_thumb)
                        {
                            // Trim off the high 16 bits if this is a 16 bit thumb instruction
                            if ((opcode & 0xe000) != 0xe000 || (opcode & 0x1800) == 0)
                            {
                                opcode &= 0xFFFFu;
                                printf("0x%8.8x: %4.4x     Thumb\n", pc, opcode);
                            }
                            else
                                printf("0x%8.8x: %8.8x Thumb\n", pc, opcode);
                        }
                        else
                            printf("0x%8.8x: %8.8x ARM\n", pc, opcode);
                    }

                    if (m_last_flags != 0 && curr_pc_is_thumb == last_pc_was_thumb)
                    {
                        if (curr_pc_is_thumb)
                        {
                            if (pc == m_last_pc + 2)
                            {
                                opcode_is_sequential = true;
                            }
                            else if (pc == m_last_pc + 4)
                            {
                                // Check for 32 bit thumb instruction...
                                uint16_t opcode16;
                                if (DNBProcessMemoryRead(m_pid, m_last_pc, 2, &opcode16) == 2)
                                {
                                    if ((opcode16 & 0xe000) == 0xe000 && (opcode16 & 0x1800) != 0)
                                    {
                                        // Last opcode was a 32 bit thumb instruction...
                                        opcode_is_sequential = true;
                                    }
                                }
                            }
                        }
                        else
                        {
                            if (pc == m_last_pc + 4)
                            {
                                opcode_is_sequential = true;
                            }
                        }
                    }


                    if (opcode_is_sequential)
                    {
                        m_consecutive_opcode_count++;
                    }
                    else
                    {
                        if (m_consecutive_opcode_count > 0)
                        {
                        //  printf(" x %u\n", m_consecutive_opcode_count);
                        }
                        m_consecutive_opcode_count = 1;
                        //printf("0x%8.8x: %-5s", pc, curr_pc_is_thumb ? "Thumb" : "ARM");
                        //fflush(stdout);
                    }
                    m_last_flags = flags;
                }
#else
#error undefined architecture
#endif
                m_last_pc = pc;
            }
        }
        break;

    case eStateRunning:
    case eStateStepping:
    case eStateCrashed:
    case eStateSuspended:
        break;

    default:
        break;
    }
}
