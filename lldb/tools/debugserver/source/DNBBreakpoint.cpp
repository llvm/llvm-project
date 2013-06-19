//===-- DNBBreakpoint.cpp ---------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  Created by Greg Clayton on 6/29/07.
//
//===----------------------------------------------------------------------===//

#include "DNBBreakpoint.h"
#include <algorithm>
#include <inttypes.h>
#include "DNBLog.h"


#pragma mark -- DNBBreakpoint
DNBBreakpoint::DNBBreakpoint(nub_addr_t addr, nub_size_t byte_size, nub_thread_t tid, bool hardware) :
    m_breakID(GetNextID()),
    m_tid(tid),
    m_byte_size(byte_size),
    m_opcode(),
    m_addr(addr),
    m_enabled(0),
    m_hw_preferred(hardware),
    m_is_watchpoint(0),
    m_watch_read(0),
    m_watch_write(0),
    m_hw_index(INVALID_NUB_HW_INDEX),
    m_hit_count(0),
    m_ignore_count(0),
    m_callback(NULL),
    m_callback_baton(NULL)
{
}

DNBBreakpoint::~DNBBreakpoint()
{
}

nub_break_t
DNBBreakpoint::GetNextID()
{
    static uint32_t g_nextBreakID = 0;
    return ++g_nextBreakID;
}

void
DNBBreakpoint::SetCallback(DNBCallbackBreakpointHit callback, void *callback_baton)
{
    m_callback = callback;
    m_callback_baton = callback_baton;
}


// RETURNS - true if we should stop at this breakpoint, false if we
// should continue.

bool
DNBBreakpoint::BreakpointHit(nub_process_t pid, nub_thread_t tid)
{
    m_hit_count++;

    if (m_hit_count > m_ignore_count)
    {
        if (m_callback)
            return m_callback(pid, tid, GetID(), m_callback_baton);
        return true;
    }
    return false;
}

void
DNBBreakpoint::Dump() const
{
    if (IsBreakpoint())
    {
        DNBLog ("DNBBreakpoint %u: tid = %8.8" PRIx64 "  addr = 0x%llx  state = %s  type = %s breakpoint  hw_index = %i  hit_count = %-4u  ignore_count = %-4u  callback = %p baton = %p",
                m_breakID,
                m_tid,
                (uint64_t)m_addr,
                m_enabled ? "enabled " : "disabled",
                IsHardware() ? "hardware" : "software",
                GetHardwareIndex(),
                GetHitCount(),
                GetIgnoreCount(),
                m_callback,
                m_callback_baton);
    }
    else
    {
        DNBLog ("DNBBreakpoint %u: tid = %8.8" PRIx64 "  addr = 0x%llx  size = %llu  state = %s  type = %s watchpoint (%s%s)  hw_index = %i  hit_count = %-4u  ignore_count = %-4u  callback = %p baton = %p",
                m_breakID,
                m_tid,
                (uint64_t)m_addr,
                (uint64_t)m_byte_size,
                m_enabled ? "enabled " : "disabled",
                IsHardware() ? "hardware" : "software",
                m_watch_read ? "r" : "",
                m_watch_write ? "w" : "",
                GetHardwareIndex(),
                GetHitCount(),
                GetIgnoreCount(),
                m_callback,
                m_callback_baton);
    }
}

#pragma mark -- DNBBreakpointList

DNBBreakpointList::DNBBreakpointList()
{
}

DNBBreakpointList::~DNBBreakpointList()
{
}


nub_break_t
DNBBreakpointList::Add(const DNBBreakpoint& bp)
{
    m_breakpoints.push_back(bp);
    return m_breakpoints.back().GetID();
}

bool
DNBBreakpointList::ShouldStop(nub_process_t pid, nub_thread_t tid, nub_break_t breakID)
{
    DNBBreakpoint *bp = FindByID (breakID);
    if (bp)
    {
        // Let the breakpoint decide if it should stop here (could not have
        // reached it's target hit count yet, or it could have a callback
        // that decided it shouldn't stop (shared library loads/unloads).
        return bp->BreakpointHit(pid, tid);
    }
    // We should stop here since this breakpoint isn't valid anymore or it
    // doesn't exist.
    return true;
}

nub_break_t
DNBBreakpointList::FindIDByAddress (nub_addr_t addr)
{
    DNBBreakpoint *bp = FindByAddress (addr);
    if (bp)
    {
        DNBLogThreadedIf(LOG_BREAKPOINTS, "DNBBreakpointList::%s ( addr = 0x%16.16llx ) => %u", __FUNCTION__, (uint64_t)addr, bp->GetID());
        return bp->GetID();
    }
    DNBLogThreadedIf(LOG_BREAKPOINTS, "DNBBreakpointList::%s ( addr = 0x%16.16llx ) => NONE", __FUNCTION__, (uint64_t)addr);
    return INVALID_NUB_BREAK_ID;
}

bool
DNBBreakpointList::Remove (nub_break_t breakID)
{
    iterator pos = GetBreakIDIterator(breakID);    // Predicate
    if (pos != m_breakpoints.end())
    {
        m_breakpoints.erase(pos);
        return true;
    }
    return false;
}


class BreakpointIDMatches
{
public:
    BreakpointIDMatches (nub_break_t breakID) : m_breakID(breakID) {}
    bool operator() (const DNBBreakpoint& bp) const
    {
        return m_breakID == bp.GetID();
    }
 private:
   const nub_break_t m_breakID;
};

class BreakpointAddressMatches
{
public:
    BreakpointAddressMatches (nub_addr_t addr) : m_addr(addr) {}
    bool operator() (const DNBBreakpoint& bp) const
    {
        return m_addr == bp.Address();
    }
 private:
   const nub_addr_t m_addr;
};

DNBBreakpointList::iterator
DNBBreakpointList::GetBreakIDIterator (nub_break_t breakID)
{
    return std::find_if(m_breakpoints.begin(), m_breakpoints.end(), // Search full range
                        BreakpointIDMatches(breakID));              // Predicate
}

DNBBreakpointList::const_iterator
DNBBreakpointList::GetBreakIDConstIterator (nub_break_t breakID) const
{
    return std::find_if(m_breakpoints.begin(), m_breakpoints.end(), // Search full range
                        BreakpointIDMatches(breakID));              // Predicate
}

DNBBreakpoint *
DNBBreakpointList::FindByID (nub_break_t breakID)
{
    iterator pos = GetBreakIDIterator(breakID);
    if (pos != m_breakpoints.end())
        return &(*pos);

    return NULL;
}

const DNBBreakpoint *
DNBBreakpointList::FindByID (nub_break_t breakID) const
{
    const_iterator pos = GetBreakIDConstIterator(breakID);
    if (pos != m_breakpoints.end())
        return &(*pos);

    return NULL;
}

DNBBreakpoint *
DNBBreakpointList::FindByAddress (nub_addr_t addr)
{
    iterator end = m_breakpoints.end();
    iterator pos = std::find_if(m_breakpoints.begin(), end,             // Search full range
                                BreakpointAddressMatches(addr));        // Predicate
    if (pos != end)
        return &(*pos);

    return NULL;
}

const DNBBreakpoint *
DNBBreakpointList::FindByAddress (nub_addr_t addr) const
{
    const_iterator end = m_breakpoints.end();
    const_iterator pos = std::find_if(m_breakpoints.begin(), end,       // Search full range
                                      BreakpointAddressMatches(addr));  // Predicate
    if (pos != end)
        return &(*pos);

    return NULL;
}

bool
DNBBreakpointList::SetCallback(nub_break_t breakID, DNBCallbackBreakpointHit callback, void *callback_baton)
{
    DNBBreakpoint *bp = FindByID (breakID);
    if (bp)
    {
        bp->SetCallback(callback, callback_baton);
        return true;
    }
    return false;
}


void
DNBBreakpointList::Dump() const
{
    const_iterator pos;
    const_iterator end = m_breakpoints.end();
    for (pos = m_breakpoints.begin(); pos != end; ++pos)
        (*pos).Dump();
}


DNBBreakpoint *
DNBBreakpointList::GetByIndex (uint32_t i)
{
    iterator end = m_breakpoints.end();
    iterator pos;
    uint32_t curr_i = 0;
    for (pos = m_breakpoints.begin(), curr_i = 0; pos != end; ++pos, ++curr_i)
    {
        if (curr_i == i)
            return &(*pos);
    }
    return NULL;
}

void
DNBBreakpointList::DisableAll ()
{
    iterator pos, end = m_breakpoints.end();
    for (pos = m_breakpoints.begin(); pos != end; ++pos)
        (*pos).SetEnabled(false);
}


const DNBBreakpoint *
DNBBreakpointList::GetByIndex (uint32_t i) const
{
    const_iterator end = m_breakpoints.end();
    const_iterator pos;
    uint32_t curr_i = 0;
    for (pos = m_breakpoints.begin(), curr_i = 0; pos != end; ++pos, ++curr_i)
    {
        if (curr_i == i)
            return &(*pos);
    }
    return NULL;
}

