//===-- ProfileObjectiveC.cpp -----------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  Created by Greg Clayton on 10/4/07.
//
//===----------------------------------------------------------------------===//

#include "ProfileObjectiveC.h"
#include "DNB.h"
#include <objc/objc-runtime.h>
#include <map>

#if defined (__powerpc__) || defined (__ppc__)
#define OBJC_MSG_SEND_PPC32_COMM_PAGE_ADDR ((nub_addr_t)0xfffeff00)
#endif

//----------------------------------------------------------------------
// Constructor
//----------------------------------------------------------------------
ProfileObjectiveC::ProfileObjectiveC() :
    m_pid(INVALID_NUB_PROCESS),
    m_objcStats(),
    m_hit_count(0),
    m_dump_count(0xffff)
{
    memset(&m_begin_time, 0, sizeof(m_begin_time));
}

//----------------------------------------------------------------------
// Destructor
//----------------------------------------------------------------------
ProfileObjectiveC::~ProfileObjectiveC()
{
}

//----------------------------------------------------------------------
// Clear any counts that we may have had
//----------------------------------------------------------------------
void
ProfileObjectiveC::Clear()
{
    if (m_pid != INVALID_NUB_PROCESS)
    {
        DNBBreakpointClear(m_pid, m_objc_msgSend.breakID);
        DNBBreakpointClear(m_pid, m_objc_msgSendSuper.breakID);
#if defined (__powerpc__) || defined (__ppc__)
        DNBBreakpointClear(m_pid, m_objc_msgSend_rtp.breakID);
#endif
    }
    m_objc_msgSend.Clear();
    m_objc_msgSendSuper.Clear();
#if defined (__powerpc__) || defined (__ppc__)
    memset(m_objc_msgSend_opcode, 0, k_opcode_size);
    m_objc_msgSend_rtp.Clear();
#endif
    memset(&m_begin_time, 0, sizeof(m_begin_time));
    m_objcStats.clear();
}

void
ProfileObjectiveC::Initialize(nub_process_t pid)
{
    Clear();
    m_pid = pid;
}


void
ProfileObjectiveC::ProcessStateChanged(nub_state_t state)
{
    //printf("ProfileObjectiveC::%s(%s)\n", __FUNCTION__, DNBStateAsString(state));
    switch (state)
    {
    case eStateInvalid:
    case eStateUnloaded:
    case eStateExited:
    case eStateDetached:
        Clear();
        break;

    case eStateStopped:
#if defined (__powerpc__) || defined (__ppc__)
        if (NUB_BREAK_ID_IS_VALID(m_objc_msgSend.breakID) && !NUB_BREAK_ID_IS_VALID(m_objc_msgSend_rtp.breakID))
        {
            nub_thread_t tid = DNBProcessGetCurrentThread(m_pid);
            DNBRegisterValue pc_value;
            if (DNBThreadGetRegisterValueByName(m_pid, tid, REGISTER_SET_ALL, "srr0" , &pc_value))
            {
                nub_addr_t pc = pc_value.value.uint32;
                if (pc == OBJC_MSG_SEND_PPC32_COMM_PAGE_ADDR)
                {
                    // Restore previous first instruction to 0xfffeff00 in comm page
                    DNBProcessMemoryWrite(m_pid, OBJC_MSG_SEND_PPC32_COMM_PAGE_ADDR, k_opcode_size, m_objc_msgSend_opcode);
                    //printf("Setting breakpoint on _objc_msgSend_rtp...\n");
                    m_objc_msgSend_rtp.breakID = DNBBreakpointSet(m_pid, OBJC_MSG_SEND_PPC32_COMM_PAGE_ADDR);
                    if (NUB_BREAK_ID_IS_VALID(m_objc_msgSend_rtp.breakID))
                    {
                        DNBBreakpointSetCallback(m_pid, m_objc_msgSend_rtp.breakID, ProfileObjectiveC::MessageSendBreakpointCallback, this);
                    }
                }
            }
        }
#endif
        DumpStats(m_pid, stdout);
        break;

    case eStateAttaching:
    case eStateLaunching:
    case eStateRunning:
    case eStateStepping:
    case eStateCrashed:
    case eStateSuspended:
        break;

    default:
        break;
    }
}

void
ProfileObjectiveC::SharedLibraryStateChanged(DNBExecutableImageInfo *image_infos, nub_size_t num_image_infos)
{
    //printf("ProfileObjectiveC::%s(%p, %u)\n", __FUNCTION__, image_infos, num_image_infos);
    if (m_objc_msgSend.IsValid() && m_objc_msgSendSuper.IsValid())
        return;

    if (image_infos)
    {
        nub_process_t pid = m_pid;
        nub_size_t i;
        for (i = 0; i < num_image_infos; i++)
        {
            if (strcmp(image_infos[i].name, "/usr/lib/libobjc.A.dylib") == 0)
            {
                if (!NUB_BREAK_ID_IS_VALID(m_objc_msgSend.breakID))
                {
                    m_objc_msgSend.addr = DNBProcessLookupAddress(pid, "_objc_msgSend", image_infos[i].name);

                    if (m_objc_msgSend.addr != INVALID_NUB_ADDRESS)
                    {
#if defined (__powerpc__) || defined (__ppc__)
                        if (DNBProcessMemoryRead(pid, m_objc_msgSend.addr, k_opcode_size, m_objc_msgSend_opcode) != k_opcode_size)
                            memset(m_objc_msgSend_opcode, 0, sizeof(m_objc_msgSend_opcode));
#endif
                        m_objc_msgSend.breakID = DNBBreakpointSet(pid, m_objc_msgSend.addr, 4, false);
                        if (NUB_BREAK_ID_IS_VALID(m_objc_msgSend.breakID))
                            DNBBreakpointSetCallback(pid, m_objc_msgSend.breakID, ProfileObjectiveC::MessageSendBreakpointCallback, this);
                    }
                }

                if (!NUB_BREAK_ID_IS_VALID(m_objc_msgSendSuper.breakID))
                {
                    m_objc_msgSendSuper.addr = DNBProcessLookupAddress(pid, "_objc_msgSendSuper", image_infos[i].name);

                    if (m_objc_msgSendSuper.addr != INVALID_NUB_ADDRESS)
                    {
                        m_objc_msgSendSuper.breakID = DNBBreakpointSet(pid, m_objc_msgSendSuper.addr, 4, false);
                        if (NUB_BREAK_ID_IS_VALID(m_objc_msgSendSuper.breakID))
                            DNBBreakpointSetCallback(pid, m_objc_msgSendSuper.breakID, ProfileObjectiveC::MessageSendSuperBreakpointCallback, this);
                    }
                }
                break;
            }
        }
    }
}


void
ProfileObjectiveC::SetStartTime()
{
    gettimeofday(&m_begin_time, NULL);
}

void
ProfileObjectiveC::SelectorHit(objc_class_ptr_t isa, objc_selector_t sel)
{
    m_objcStats[isa][sel]++;
}

nub_bool_t
ProfileObjectiveC::MessageSendBreakpointCallback(nub_process_t pid, nub_thread_t tid, nub_break_t breakID, void *userData)
{
    ProfileObjectiveC *profile_objc = (ProfileObjectiveC*)userData;
    uint32_t hit_count = profile_objc->IncrementHitCount();
    if (hit_count == 1)
        profile_objc->SetStartTime();

    objc_class_ptr_t objc_self = 0;
    objc_selector_t objc_selector = 0;
#if defined (__i386__)
    DNBRegisterValue esp;
    if (DNBThreadGetRegisterValueByName(pid, tid, REGISTER_SET_ALL, "esp", &esp))
    {
        uint32_t uval32[2];
        if (DNBProcessMemoryRead(pid, esp.value.uint32 + 4, 8, &uval32) == 8)
        {
            objc_self = uval32[0];
            objc_selector = uval32[1];
        }
    }
#elif defined (__powerpc__) || defined (__ppc__)
    DNBRegisterValue r3;
    DNBRegisterValue r4;
    if (DNBThreadGetRegisterValueByName(pid, tid, REGISTER_SET_ALL, "r3", &r3) &&
        DNBThreadGetRegisterValueByName(pid, tid, REGISTER_SET_ALL, "r4", &r4))
    {
        objc_self = r3.value.uint32;
        objc_selector = r4.value.uint32;
    }
#elif defined (__arm__)
    DNBRegisterValue r0;
    DNBRegisterValue r1;
    if (DNBThreadGetRegisterValueByName(pid, tid, REGISTER_SET_ALL, "r0", &r0) &&
        DNBThreadGetRegisterValueByName(pid, tid, REGISTER_SET_ALL, "r1", &r1))
    {
        objc_self = r0.value.uint32;
        objc_selector = r1.value.uint32;
    }
#else
#error undefined architecture
#endif
    if (objc_selector != 0)
    {
        uint32_t isa = 0;
        if (objc_self == 0)
        {
            profile_objc->SelectorHit(0, objc_selector);
        }
        else
        if (DNBProcessMemoryRead(pid, (nub_addr_t)objc_self, sizeof(isa), &isa) == sizeof(isa))
        {
            if (isa)
            {
                profile_objc->SelectorHit(isa, objc_selector);
            }
            else
            {
                profile_objc->SelectorHit(0, objc_selector);
            }
        }
    }


    // Dump stats if we are supposed to
    if (profile_objc->ShouldDumpStats())
    {
        profile_objc->DumpStats(pid, stdout);
        return true;
    }

    // Just let the target run again by returning false;
    return false;
}

nub_bool_t
ProfileObjectiveC::MessageSendSuperBreakpointCallback(nub_process_t pid, nub_thread_t tid, nub_break_t breakID, void *userData)
{
    ProfileObjectiveC *profile_objc = (ProfileObjectiveC*)userData;

    uint32_t hit_count = profile_objc->IncrementHitCount();
    if (hit_count == 1)
        profile_objc->SetStartTime();

//    printf("BreakID %u hit count is = %u\n", breakID, hc);
    objc_class_ptr_t objc_super = 0;
    objc_selector_t objc_selector = 0;
#if defined (__i386__)
    DNBRegisterValue esp;
    if (DNBThreadGetRegisterValueByName(pid, tid, REGISTER_SET_ALL, "esp", &esp))
    {
        uint32_t uval32[2];
        if (DNBProcessMemoryRead(pid, esp.value.uint32 + 4, 8, &uval32) == 8)
        {
            objc_super = uval32[0];
            objc_selector = uval32[1];
        }
    }
#elif defined (__powerpc__) || defined (__ppc__)
    DNBRegisterValue r3;
    DNBRegisterValue r4;
    if (DNBThreadGetRegisterValueByName(pid, tid, REGISTER_SET_ALL, "r3", &r3) &&
        DNBThreadGetRegisterValueByName(pid, tid, REGISTER_SET_ALL, "r4", &r4))
    {
        objc_super = r3.value.uint32;
        objc_selector = r4.value.uint32;
    }
#elif defined (__arm__)
    DNBRegisterValue r0;
    DNBRegisterValue r1;
    if (DNBThreadGetRegisterValueByName(pid, tid, REGISTER_SET_ALL, "r0", &r0) &&
        DNBThreadGetRegisterValueByName(pid, tid, REGISTER_SET_ALL, "r1", &r1))
    {
        objc_super = r0.value.uint32;
        objc_selector = r1.value.uint32;
    }
#else
#error undefined architecture
#endif
    if (objc_selector != 0)
    {
        uint32_t isa = 0;
        if (objc_super == 0)
        {
            profile_objc->SelectorHit(0, objc_selector);
        }
        else
        if (DNBProcessMemoryRead(pid, (nub_addr_t)objc_super + 4, sizeof(isa), &isa) == sizeof(isa))
        {
            if (isa)
            {
                profile_objc->SelectorHit(isa, objc_selector);
            }
            else
            {
                profile_objc->SelectorHit(0, objc_selector);
            }
        }
    }

    // Dump stats if we are supposed to
    if (profile_objc->ShouldDumpStats())
    {
        profile_objc->DumpStats(pid, stdout);
        return true;
    }

    // Just let the target run again by returning false;
    return false;
}

void
ProfileObjectiveC::DumpStats(nub_process_t pid, FILE *f)
{
    if (f == NULL)
        return;

    if (m_hit_count == 0)
        return;

    ClassStatsMap::iterator class_pos;
    ClassStatsMap::iterator class_end = m_objcStats.end();

    struct timeval end_time;
    gettimeofday(&end_time, NULL);
    int64_t elapsed_usec = ((int64_t)(1000*1000))*((int64_t)end_time.tv_sec - (int64_t)m_begin_time.tv_sec) + ((int64_t)end_time.tv_usec - (int64_t)m_begin_time.tv_usec);
    fprintf(f, "%u probe hits for %.2f hits/sec)\n", m_hit_count, (double)m_hit_count / (((double)elapsed_usec)/(1000000.0)));

    for (class_pos = m_objcStats.begin(); class_pos != class_end; ++class_pos)
    {
        SelectorHitCount::iterator sel_pos;
        SelectorHitCount::iterator sel_end = class_pos->second.end();
        for (sel_pos = class_pos->second.begin(); sel_pos != sel_end; ++sel_pos)
        {
            struct objc_class objc_class;
            uint32_t isa = class_pos->first;
            uint32_t sel = sel_pos->first;
            uint32_t sel_hit_count = sel_pos->second;

            if (isa != 0 && DNBProcessMemoryRead(pid, isa, sizeof(objc_class), &objc_class) == sizeof(objc_class))
            {
            /*    fprintf(f, "%#.8x\n          isa = %p\n  super_class = %p\n         name = %p\n      version = %lx\n         info = %lx\ninstance_size = %lx\n        ivars = %p\n  methodLists = %p\n        cache = %p\n    protocols = %p\n",
                        arg1.value.pointer,
                        objc_class.isa,
                        objc_class.super_class,
                        objc_class.name,
                        objc_class.version,
                        objc_class.info,
                        objc_class.instance_size,
                        objc_class.ivars,
                        objc_class.methodLists,
                        objc_class.cache,
                        objc_class.protocols); */

                // Print the class name
                fprintf(f, "%6u hits for %c[", sel_hit_count, (objc_class.super_class == objc_class.isa ? '+' : '-'));
                DNBPrintf(pid, INVALID_NUB_THREAD, (nub_addr_t)objc_class.name, f, "%s ");
            }
            else
            {
                fprintf(f, "%6u hits for  [<nil> ", sel_hit_count);
            }
            DNBPrintf(pid, INVALID_NUB_THREAD, sel, f, "%s]\n");
        }
    }
}

